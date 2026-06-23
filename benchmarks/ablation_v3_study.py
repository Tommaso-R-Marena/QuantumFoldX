#!/usr/bin/env python3
"""
ablation_v3_study.py — Does the Dual-State Quantum Bridge beat v2 and baselines?

Compares:
  1. QICESS-v3 (Dual-State Ising Bridge + exact enumeration)
  2. QICESS-v2 (single-state VQE — legacy)
  3. QICESS-Exact (single-state exact diag from ablation)
  4. Classical-MJ
  5. No-Quantum
  6. Random

Primary metric: top-10 mean TM-score to state 2.
"""

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pdb_fetcher import fetch_pdb, parse_pdb_ca_coords, compute_phi_psi
from src.scoring.qicess_v2 import QICESSv2Scorer
from src.scoring.qicess_v3 import QICESSv3Scorer
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from configs.benchmark_dataset import get_autoinhibited_benchmark
from benchmarks.ablation_study import (
    score_ensemble_classical_mj, score_ensemble_no_quantum,
    score_ensemble_random, score_ensemble_exact_diag, evaluate_ranking,
    find_common_residues,
)
from benchmarks.benchmark_utils import get_domain_indices, parse_target_structures

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results' / 'ablation'
RAW_FILE = RESULTS_DIR / 'ablation_v3_raw.csv'


def run_ablation_v3(resume: bool = True):
    logger.info("=" * 80)
    logger.info("QICESS v3 ablation — dual-state bridge vs baselines")
    logger.info("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = get_autoinhibited_benchmark()
    scorer_v2 = QICESSv2Scorer(vqe_layers=3, vqe_restarts=2, vqe_steps=50, use_qaoa=False)
    scorer_v3 = QICESSv3Scorer()

    methods = ['QICESS-v3', 'QICESS-v2', 'QICESS-Exact', 'Classical-MJ', 'No-Quantum', 'Random']
    all_results = []
    done_genes = set()

    if resume and RAW_FILE.exists():
        existing = pd.read_csv(RAW_FILE)
        all_results = existing.to_dict('records')
        done_genes = set(existing['gene'].tolist())
        logger.info("Resuming: %d proteins already done", len(done_genes))

    for idx, target in enumerate(targets):
        if target.pdb_id_state1 == target.pdb_id_state2:
            continue
        if target.gene_name in done_genes:
            logger.info("[%d/%d] %s — skip (done)", idx + 1, len(targets), target.gene_name)
            continue

        logger.info("\n[%d/%d] %s", idx + 1, len(targets), target.gene_name)

        s1, s2, status = parse_target_structures(target)
        if status != 'ok' or s1['n_residues'] > 1000:
            continue

        ci1, ci2, nc = find_common_residues(s1, s2)
        if nc < 20:
            continue

        fd_idx, im_idx = get_domain_indices(s1, target)
        n_ens = 80 if s1['n_residues'] < 400 else 50
        phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])

        bridge = scorer_v3.build_bridge(s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx)
        ensemble = generate_hybrid_ensemble(
            s1['coords'], s1['sequence'], fd_idx, im_idx,
            n_conformations=n_ens, seed=42, phi_psi=phi_psi,
            coords_s2=s2['coords'], quantum_bridge=bridge,
        )
        for conf in ensemble:
            conf['phi_psi'] = phi_psi

        rankings = {}

        t0 = time.time()
        rankings['QICESS-v3'] = scorer_v3.rank_ensemble(
            ensemble, s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx)
        v3_time = time.time() - t0

        t0 = time.time()
        rankings['QICESS-v2'] = scorer_v2.rank_ensemble(
            ensemble, s1['sequence'], s1['coords'], fd_idx, im_idx)
        v2_time = time.time() - t0

        rankings['QICESS-Exact'], _ = score_ensemble_exact_diag(
            ensemble, s1['sequence'], s1['coords'], fd_idx, im_idx, phi_psi, None)
        rankings['Classical-MJ'] = score_ensemble_classical_mj(
            ensemble, s1['sequence'], fd_idx, im_idx, phi_psi, None)
        rankings['No-Quantum'] = score_ensemble_no_quantum(
            ensemble, s1['sequence'], fd_idx, im_idx, phi_psi, None)
        rankings['Random'] = score_ensemble_random(ensemble, seed=42)

        row = {
            'gene': target.gene_name,
            'n_residues': s1['n_residues'],
            'n_switch_contacts': len(bridge.switch_contacts),
            'v3_time': v3_time,
            'v2_time': v2_time,
        }

        for method in methods:
            ev = evaluate_ranking(rankings[method], s2['coords'], ci1, ci2, k=10)
            if ev:
                for k, v in ev.items():
                    row[f'{method}_{k}'] = v

        all_results.append(row)

        for m in methods:
            key = f'{m}_top_k_mean_tm'
            if key in row:
                logger.info(f"    {m:18s}: {row[key]:.4f}")

        pd.DataFrame(all_results).to_csv(RAW_FILE, index=False)

    df = pd.DataFrame(all_results)
    stats = {'n_proteins': len(df), 'methods': methods}

    for m in methods:
        key = f'{m}_top_k_mean_tm'
        if key in df.columns:
            vals = df[key].dropna()
            stats[m] = {'top10_mean_tm': {'mean': float(vals.mean()), 'std': float(vals.std())}}

    paired = {}
    v3_key = 'QICESS-v3_top_k_mean_tm'
    for m in ['QICESS-v2', 'QICESS-Exact', 'Classical-MJ', 'No-Quantum', 'Random']:
        m_key = f'{m}_top_k_mean_tm'
        if v3_key in df.columns and m_key in df.columns:
            diff = df[v3_key].values - df[m_key].values
            try:
                _, p = scipy_stats.wilcoxon(diff, alternative='greater')
            except Exception:
                p = 1.0
            paired[f'v3_vs_{m}'] = {
                'mean_diff': float(np.mean(diff)),
                'wilcoxon_p': float(p),
                'significant': bool(p < 0.05),
                'v3_wins': int(np.sum(diff > 0.001)),
            }

    stats['paired_tests'] = paired
    stats['note'] = (
        'Top-10 TM ranking is an imperfect proxy for dual-state coverage; '
        'see per-protein ens_max_tm_state2 in the main benchmark.'
    )

    with open(RESULTS_DIR / 'ablation_v3_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("V3 ABLATION SUMMARY")
    for m in methods:
        if m in stats:
            s = stats[m]['top10_mean_tm']
            logger.info(f"  {m:18s}: {s['mean']:.4f} ± {s['std']:.4f}")
    logger.info("\nPaired tests (v3 vs baseline):")
    for name, t in paired.items():
        sig = "SIG" if t['significant'] else "NS"
        logger.info(f"  {name}: Δ={t['mean_diff']:+.4f} p={t['wilcoxon_p']:.4f} {sig}")

    return df, stats


if __name__ == '__main__':
    run_ablation_v3()
