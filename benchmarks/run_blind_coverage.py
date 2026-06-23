#!/usr/bin/env python3
"""
run_blind_coverage.py — STEP 2: DSIB without state-2 oracle during generation.

BLIND variant: ensemble built from state 1 + sequence ONLY.
  - No coords_s2 passed to ensemble generation
  - No H2 / dual-state Ising bridge (requires state 2 contact map)
  - No S1↔S2 coordinate interpolation
  - State 2 used ONLY at evaluation (TM-score, dual coverage)

ORACLE variant (for comparison): current DSIB pipeline (both states known at generation).

Also reports v2 baseline (same as BLIND ensemble generation path in this codebase).

Run:
  python benchmarks/run_blind_coverage.py --no-resume

Outputs:
  results/blind/blind_coverage.csv
  results/blind/blind_coverage_stats.json
"""

from __future__ import annotations

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

from configs.benchmark_dataset import get_all_benchmarks, get_af3_baseline
from src.scoring.qicess_v3 import create_dsib_scorer
from src.scoring.geometry_utils import transition_difficulty
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.metrics.structural_metrics import tm_score
from benchmarks.benchmark_utils import (
    parse_target_structures, get_domain_indices, find_common_residues,
    evaluate_ensemble_vs_state,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / 'results' / 'blind'
RAW_FILE = OUT_DIR / 'blind_coverage.csv'
STATS_FILE = OUT_DIR / 'blind_coverage_stats.json'


def _coverage(ensemble, s1, s2, ci1, ci2):
    all_coords = [c['coords'] for c in ensemble]
    eval_s1 = evaluate_ensemble_vs_state(
        all_coords, s1['coords'],
        list(range(s1['n_residues'])), list(range(s1['n_residues'])))
    eval_s2 = evaluate_ensemble_vs_state(all_coords, s2['coords'], ci1, ci2)
    s1_ok = eval_s1['max_tm'] > 0.5 if eval_s1['max_tm'] else False
    s2_ok = eval_s2['max_tm'] > 0.5 if eval_s2['max_tm'] else False
    return {
        'ens_max_tm_state2': eval_s2['max_tm'],
        'ens_min_rmsd_state2': eval_s2['min_rmsd'],
        'dual_state_covered': s1_ok and s2_ok,
    }


def _process(target, n_ens: int):
    from src.data.pdb_fetcher import compute_phi_psi

    s1, s2, status = parse_target_structures(target)
    if status != 'ok' or s1['n_residues'] > 1000:
        return None

    ci1, ci2, nc = find_common_residues(s1, s2)
    if nc < 20:
        return None

    baseline_tm = tm_score(s1['coords'][ci1[:nc]], s2['coords'][ci2[:nc]])
    fd_idx, im_idx = get_domain_indices(s1, target)
    phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])
    diff = transition_difficulty(baseline_tm)

    # BLIND: state 1 only — no S2 coords, no DSIB bridge
    ens_blind = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=42, phi_psi=phi_psi,
        # coords_s2 intentionally omitted
    )

    # ORACLE: both states known at generation (current DSIB)
    bridge = create_dsib_scorer().build_bridge(
        s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx)
    ens_oracle = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=42, phi_psi=phi_psi,
        coords_s2=s2['coords'], quantum_bridge=bridge,
        transition_difficulty=diff,
        common_idx_s1=ci1, common_idx_s2=ci2,
    )

    row = {
        'gene': target.gene_name,
        'category': target.category,
        'n_residues': s1['n_residues'],
        'baseline_tm': baseline_tm,
        'n_blind': len(ens_blind),
        'n_oracle': len(ens_oracle),
        'n_oracle_bridge': sum(1 for c in ens_oracle if c['method'] in (
            'quantum_bridge', 'manifold_bridge', 'common_residue_interp',
            'switch_contact_rigid')),
    }

    for prefix, ens in [('BLIND', ens_blind), ('ORACLE', ens_oracle)]:
        m = _coverage(ens, s1, s2, ci1, ci2)
        for k, v in m.items():
            row[f'{prefix}_{k}'] = v

    return row


def _analyze(df: pd.DataFrame, af3_base: dict) -> dict:
    metric = 'ens_max_tm_state2'
    stats = {'n_proteins': len(df), 'label': 'blind_vs_oracle'}

    for cond in ['BLIND', 'ORACLE']:
        stats[cond] = {
            'mean_state2_tm': float(df[f'{cond}_{metric}'].mean()),
            'dual_state_rate': float(df[f'{cond}_dual_state_covered'].mean()),
            'n_dual_covered': int(df[f'{cond}_dual_state_covered'].sum()),
        }

    diff = df[f'ORACLE_{metric}'].values - df[f'BLIND_{metric}'].values
    try:
        _, p = scipy_stats.wilcoxon(diff, alternative='greater')
    except Exception:
        p = 1.0
    stats['oracle_vs_blind'] = {
        'mean_diff_tm': float(np.mean(diff)),
        'wilcoxon_p': float(p),
        'significant': bool(p < 0.05),
        'oracle_wins': int(np.sum(diff > 0.01)),
    }

    dual_diff = (
        df['ORACLE_dual_state_covered'].astype(int).values
        - df['BLIND_dual_state_covered'].astype(int).values
    )
    stats['dual_coverage_gain'] = {
        'oracle_dual': int(df['ORACLE_dual_state_covered'].sum()),
        'blind_dual': int(df['BLIND_dual_state_covered'].sum()),
        'proteins_gained': int(np.sum(dual_diff > 0)),
    }

    # By category
    by_cat = {}
    for cat in sorted(df['category'].unique()):
        sub = df[df['category'] == cat]
        by_cat[cat] = {
            'n': int(len(sub)),
            'blind_dual': int(sub['BLIND_dual_state_covered'].sum()),
            'oracle_dual': int(sub['ORACLE_dual_state_covered'].sum()),
            'blind_mean_tm': float(sub[f'BLIND_{metric}'].mean()),
            'oracle_mean_tm': float(sub[f'ORACLE_{metric}'].mean()),
        }
    stats['by_category'] = by_cat

    hard = df[df['baseline_tm'] < 0.5]
    if len(hard) >= 3:
        stats['hard_subset'] = {
            'n': int(len(hard)),
            'blind_dual': int(hard['BLIND_dual_state_covered'].sum()),
            'oracle_dual': int(hard['ORACLE_dual_state_covered'].sum()),
            'blind_mean_tm': float(hard[f'BLIND_{metric}'].mean()),
            'oracle_mean_tm': float(hard[f'ORACLE_{metric}'].mean()),
        }

    af3_weighted = np.mean([
        af3_base['autoinhibited']['fraction_both_states'],
        af3_base['foldswitch']['success_rate'],
        af3_base['multistate']['fraction_both_states_correct'],
    ])
    stats['vs_af3_weighted'] = {
        'af3_rate': float(af3_weighted),
        'blind_dual_rate': float(df['BLIND_dual_state_covered'].mean()),
        'oracle_dual_rate': float(df['ORACLE_dual_state_covered'].mean()),
        'p_blind_vs_af3': float(scipy_stats.binomtest(
            int(df['BLIND_dual_state_covered'].sum()), len(df), af3_weighted,
            alternative='greater').pvalue),
        'p_oracle_vs_af3': float(scipy_stats.binomtest(
            int(df['ORACLE_dual_state_covered'].sum()), len(df), af3_weighted,
            alternative='greater').pvalue),
    }

    return stats


def run(resume: bool = True, n_ens: int = 80):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = get_all_benchmarks()
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
        logger.info("[%d/%d] %s", idx + 1, len(targets), target.gene_name)
        try:
            row = _process(target, n_ens=n_ens if target.category != 'foldswitch' else 60)
            if row is None:
                continue
            results.append(row)
            pd.DataFrame(results).to_csv(RAW_FILE, index=False)
        except Exception as e:
            logger.error("  ERROR %s: %s", target.gene_name, e)

    df = pd.DataFrame(results)
    stats = _analyze(df, af3_base)
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 72)
    print("BLIND vs ORACLE COVERAGE BENCHMARK — STEP 2")
    print(f"Command: python benchmarks/run_blind_coverage.py")
    print(f"n_proteins={stats['n_proteins']}")
    print(f"BLIND (S1-only gen):  dual={stats['BLIND']['n_dual_covered']}/{stats['n_proteins']} "
          f"({100*stats['BLIND']['dual_state_rate']:.1f}%) "
          f"mean TM→S2={stats['BLIND']['mean_state2_tm']:.4f}")
    print(f"ORACLE (DSIB gen):    dual={stats['ORACLE']['n_dual_covered']}/{stats['n_proteins']} "
          f"({100*stats['ORACLE']['dual_state_rate']:.1f}%) "
          f"mean TM→S2={stats['ORACLE']['mean_state2_tm']:.4f}")
    ob = stats['oracle_vs_blind']
    print(f"ORACLE vs BLIND Wilcoxon p={ob['wilcoxon_p']:.6f}  mean ΔTM={ob['mean_diff_tm']:.4f}")
    print(f"Dual coverage: BLIND {stats['dual_coverage_gain']['blind_dual']} vs "
          f"ORACLE {stats['dual_coverage_gain']['oracle_dual']}")
    for cat, cs in stats.get('by_category', {}).items():
        print(f"  {cat}: BLIND {cs['blind_dual']}/{cs['n']}  ORACLE {cs['oracle_dual']}/{cs['n']}")

    return df, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    run(resume=not args.no_resume)
