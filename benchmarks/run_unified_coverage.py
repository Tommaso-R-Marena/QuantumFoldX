#!/usr/bin/env python3
"""
run_unified_coverage.py — Cross-dataset DSIB coverage benchmark (v2 vs v3).

Runs head-to-head comparison on all benchmark categories:
  - autoinhibited (24 targets)
  - foldswitch (12 targets)
  - multistate (13 targets)

Outputs per-category and combined statistics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.benchmark_dataset import (
    get_autoinhibited_benchmark, get_foldswitch_benchmark,
    get_multistate_benchmark, get_all_benchmarks, get_af3_baseline,
)
from src.scoring.qicess_v3 import create_dsib_scorer
from src.scoring.geometry_utils import transition_difficulty
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.quantum.dual_state_ising import compute_transition_complexity
from benchmarks.benchmark_utils import (
    parse_target_structures, get_domain_indices, find_common_residues,
    evaluate_ensemble_vs_state,
)
from benchmarks.compare_v2_v3_coverage import _coverage_metrics, _merge_v2_plus_bridge

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / 'results' / 'unified'
RAW_FILE = OUT_DIR / 'unified_coverage.csv'
STATS_FILE = OUT_DIR / 'unified_coverage_stats.json'

CONDITIONS = ['A_v2_full', 'B_v3_full', 'E_v2plus_bridge']
METRIC = 'ens_max_tm_state2'


def _process_target(target, scorer_v3, n_ens: int):
    from src.data.pdb_fetcher import compute_phi_psi
    from src.metrics.structural_metrics import tm_score

    s1, s2, status = parse_target_structures(target)
    if status != 'ok' or s1['n_residues'] > 1000:
        return None

    ci1, ci2, nc = find_common_residues(s1, s2)
    if nc < 20:
        return None

    baseline_tm = tm_score(s1['coords'][ci1[:nc]], s2['coords'][ci2[:nc]])
    fd_idx, im_idx = get_domain_indices(s1, target)
    diff = transition_difficulty(baseline_tm)
    phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])

    ens_v2 = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=42, phi_psi=phi_psi,
    )
    bridge = scorer_v3.build_bridge(
        s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx)
    tci = compute_transition_complexity(bridge)
    ens_v3 = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=42, phi_psi=phi_psi,
        coords_s2=s2['coords'], quantum_bridge=bridge,
        transition_difficulty=diff,
        common_idx_s1=ci1, common_idx_s2=ci2,
    )
    ens_v2_plus = _merge_v2_plus_bridge(ens_v2, ens_v3)
    for e in (ens_v2, ens_v3, ens_v2_plus):
        for c in e:
            c['phi_psi'] = phi_psi

    row = {
        'gene': target.gene_name,
        'category': target.category,
        'n_residues': s1['n_residues'],
        'n_common_residues': nc,
        'baseline_tm': baseline_tm,
        'tci': tci['tci'],
        'n_switch_contacts': len(bridge.switch_contacts),
        'n_bridge_confs': sum(1 for c in ens_v3 if c['method'] == 'quantum_bridge'),
        'n_manifold_confs': sum(1 for c in ens_v3 if c['method'] == 'manifold_bridge'),
    }

    cond_map = {
        'A_v2_full': ens_v2,
        'B_v3_full': ens_v3,
        'E_v2plus_bridge': ens_v2_plus,
    }
    for label, ens in cond_map.items():
        m = _coverage_metrics(ens, s1, s2, ci1, ci2)
        for k, v in m.items():
            row[f'{label}_{k}'] = v

    return row


def _analyze(df: pd.DataFrame, af3_base: dict) -> dict:
    stats = {'n_proteins': len(df), 'categories': {}}

    for cat in sorted(df['category'].unique()):
        sub = df[df['category'] == cat]
        cat_stats = {'n': int(len(sub))}
        for cond in CONDITIONS:
            col = f'{cond}_{METRIC}'
            if col in sub.columns:
                cat_stats[cond] = {
                    'mean_state2_tm': float(sub[col].mean()),
                    'dual_state_rate': float(sub[f'{cond}_dual_state_covered'].mean()),
                    'n_dual_covered': int(sub[f'{cond}_dual_state_covered'].sum()),
                }
        b_col = f'B_v3_full_{METRIC}'
        a_col = f'A_v2_full_{METRIC}'
        if b_col in sub.columns:
            diff = sub[b_col].values - sub[a_col].values
            try:
                _, p = scipy_stats.wilcoxon(diff, alternative='greater')
            except Exception:
                p = 1.0
            cat_stats['v3_vs_v2'] = {
                'mean_diff': float(np.mean(diff)),
                'wilcoxon_p': float(p),
                'significant': bool(p < 0.05),
            }
        stats['categories'][cat] = cat_stats

    # Combined
    combined = {'n': int(len(df))}
    for cond in CONDITIONS:
        col = f'{cond}_{METRIC}'
        combined[cond] = {
            'mean_state2_tm': float(df[col].mean()),
            'dual_state_rate': float(df[f'{cond}_dual_state_covered'].mean()),
            'n_dual_covered': int(df[f'{cond}_dual_state_covered'].sum()),
        }
    diff_all = df[f'B_v3_full_{METRIC}'].values - df[f'A_v2_full_{METRIC}'].values
    try:
        _, p_all = scipy_stats.wilcoxon(diff_all, alternative='greater')
    except Exception:
        p_all = 1.0
    combined['v3_vs_v2'] = {
        'mean_diff': float(np.mean(diff_all)),
        'wilcoxon_p': float(p_all),
        'significant': bool(p_all < 0.05),
    }

    # AF3 comparison (combined dual coverage)
    dsc = int(df['B_v3_full_dual_state_covered'].sum())
    n = len(df)
    af3_rates = {
        'autoinhibited': af3_base['autoinhibited']['fraction_both_states'],
        'foldswitch': af3_base['foldswitch']['success_rate'],
        'multistate': af3_base['multistate']['fraction_both_states_correct'],
    }
    weighted_af3 = np.mean(list(af3_rates.values()))
    combined['af3_weighted_mean_rate'] = float(weighted_af3)
    combined['p_vs_af3_weighted'] = float(
        scipy_stats.binomtest(dsc, n, weighted_af3, alternative='greater').pvalue
    )

    hard = df[df['baseline_tm'] < 0.5]
    if len(hard) >= 3:
        combined['hard_subset'] = {
            'n': int(len(hard)),
            'v2_dual': int(hard['A_v2_full_dual_state_covered'].sum()),
            'v3_dual': int(hard['B_v3_full_dual_state_covered'].sum()),
            'v3_mean_tm': float(hard[f'B_v3_full_{METRIC}'].mean()),
            'v2_mean_tm': float(hard[f'A_v2_full_{METRIC}'].mean()),
        }

    stats['combined'] = combined
    return stats


def run_unified(targets=None, resume: bool = True, n_ens: int = 80):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if targets is None:
        targets = get_all_benchmarks()

    scorer_v3 = create_dsib_scorer()
    af3_base = get_af3_baseline()

    results = []
    done = set()
    if resume and RAW_FILE.exists():
        existing = pd.read_csv(RAW_FILE)
        results = existing.to_dict('records')
        done = set(existing['gene'].tolist())

    for idx, target in enumerate(targets):
        if target.pdb_id_state1 == target.pdb_id_state2 or target.gene_name in done:
            continue

        logger.info("[%d/%d] %s (%s)", idx + 1, len(targets), target.gene_name, target.category)
        ens_n = n_ens if target.category != 'foldswitch' else min(n_ens, 60)
        try:
            row = _process_target(target, scorer_v3, ens_n)
            if row is None:
                logger.warning("  Skipped %s", target.gene_name)
                continue
            results.append(row)
            pd.DataFrame(results).to_csv(RAW_FILE, index=False)
        except Exception as e:
            logger.error("  ERROR %s: %s", target.gene_name, e)

    df = pd.DataFrame(results)
    stats = _analyze(df, af3_base)
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    _print_summary(stats)
    return df, stats


def _print_summary(stats: dict):
    logger.info("\n" + "=" * 70)
    logger.info("UNIFIED COVERAGE BENCHMARK (n=%d)", stats['n_proteins'])
    for cat, cs in stats.get('categories', {}).items():
        b = cs.get('B_v3_full', {})
        a = cs.get('A_v2_full', {})
        logger.info("  %s: v2=%d/%d v3=%d/%d p=%.4f",
                    cat,
                    a.get('n_dual_covered', 0), cs['n'],
                    b.get('n_dual_covered', 0), cs['n'],
                    cs.get('v3_vs_v2', {}).get('wilcoxon_p', 1.0))
    c = stats.get('combined', {})
    if c:
        logger.info("  COMBINED: v2=%d/%d v3=%d/%d p=%.4f",
                    c['A_v2_full']['n_dual_covered'], c['n'],
                    c['B_v3_full']['n_dual_covered'], c['n'],
                    c.get('v3_vs_v2', {}).get('wilcoxon_p', 1.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--category', choices=['auto', 'foldswitch', 'multistate', 'all'],
                        default='all')
    args = parser.parse_args()

    if args.category == 'auto':
        targets = get_autoinhibited_benchmark()
    elif args.category == 'foldswitch':
        targets = get_foldswitch_benchmark()
    elif args.category == 'multistate':
        targets = get_multistate_benchmark()
    else:
        targets = get_all_benchmarks()

    run_unified(targets=targets, resume=not args.no_resume)
