#!/usr/bin/env python3
"""
run_softmode_improvement.py — Does soft-mode SUBSPACE sampling improve BLIND
alternate-state prediction, and does it generalize?

All conditions are strictly BLIND (state 1 only) and share the same conformation
budget and the same blind, radius-of-gyration-based amplitude schedule (no
state-2 information leaks into generation). We compare:

  baseline        : existing multi-scale sampler (NMA + rigid-body + torsion)
  single_mode     : scan the softest ANM modes one axis at a time
  subspace        : sample the softest-k ANM subspace (mode combinations)
  subspace_relax  : subspace + ENM-guided Calpha relaxation
  combo           : union of baseline and subspace_relax (practical best)

For each protein we also record state-1-only ANM mode-overlap metrics so the
result can be stratified by how "collective / low-mode" the transition is.
State-2-derived quantities (baseline_tm, transition_rmsd) are recorded ONLY for
post-hoc stratification, never used during generation.

Run:
  python benchmarks/run_softmode_improvement.py --no-resume --n-ens 56
  python benchmarks/run_softmode_improvement.py --genes AK1,SRC,KAI_B --n-ens 40

Outputs:
  results/rigorous/softmode_improvement.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.benchmark_dataset import get_all_benchmarks
from src.data.pdb_fetcher import compute_phi_psi
from src.metrics.structural_metrics import tm_score, radius_of_gyration
from src.ensemble.conformational_sampler import (
    generate_hybrid_ensemble, generate_anm_mode_scan_ensemble,
)
from src.ensemble.nm_guided import softmode_subspace_ensemble
from src.analysis.mode_overlap import analyze_transition
from benchmarks.benchmark_utils import (
    parse_target_structures, get_domain_indices, find_common_residues,
    evaluate_ensemble_vs_state,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / 'results' / 'rigorous'
RAW_FILE = OUT_DIR / 'softmode_improvement.csv'
TM_THRESHOLD = 0.5
SEED = 42


def _blind_amplitude(coords: np.ndarray) -> float:
    """Blind (state-1 only) max RMSD amplitude from the radius of gyration."""
    return float(np.clip(0.75 * radius_of_gyration(coords), 8.0, 30.0))


def _coords_list(ens):
    return [c['coords'] if isinstance(c, dict) else c for c in ens]


def _eval(coords_list, s1, s2, ci1, ci2):
    e_s1 = evaluate_ensemble_vs_state(
        coords_list, s1['coords'],
        list(range(s1['n_residues'])), list(range(s1['n_residues'])))
    e_s2 = evaluate_ensemble_vs_state(coords_list, s2['coords'], ci1, ci2)
    s1_ok = bool(e_s1['max_tm'] and e_s1['max_tm'] > TM_THRESHOLD)
    s2_ok = bool(e_s2['max_tm'] and e_s2['max_tm'] > TM_THRESHOLD)
    return e_s2['max_tm'], e_s2['min_rmsd'], (s1_ok and s2_ok)


def process_target(target, n_ens: int) -> dict:
    s1, s2, status = parse_target_structures(target)
    if status != 'ok' or s1['n_residues'] > 1000:
        return {'gene': target.gene_name, 'status': status or 'too_large'}
    ci1, ci2, nc = find_common_residues(s1, s2)
    if nc < 20:
        return {'gene': target.gene_name, 'status': 'insufficient_overlap'}

    fd_idx, im_idx = get_domain_indices(s1, target)
    phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])
    baseline_tm = float(tm_score(s1['coords'][ci1[:nc]], s2['coords'][ci2[:nc]]))
    mo = analyze_transition(s1['coords'], s2['coords'], ci1, ci2, n_modes=20)
    max_rmsd = _blind_amplitude(s1['coords'])           # BLIND

    row = {
        'gene': target.gene_name, 'category': target.category,
        'n_residues': s1['n_residues'], 'n_core': nc,
        'baseline_tm': baseline_tm, 'blind_max_rmsd': max_rmsd, 'status': 'ok',
    }
    if mo is not None:
        row.update({
            'transition_rmsd': mo.transition_magnitude,
            'best_single_overlap': mo.best_single_overlap,
            'cum_overlap_10': mo.cum_overlap_10,
        })

    orig = [s1['coords'].copy()]
    ensembles = {
        'baseline': _coords_list(generate_hybrid_ensemble(
            s1['coords'], s1['sequence'], fd_idx, im_idx,
            n_conformations=n_ens, seed=SEED, phi_psi=phi_psi)),
        'single_mode': orig + generate_anm_mode_scan_ensemble(
            s1['coords'], n_ens - 1, n_modes=6, max_rmsd=max_rmsd, seed=SEED),
        'subspace': orig + softmode_subspace_ensemble(
            s1['coords'], n_ens - 1, k_modes=10, max_rmsd=max_rmsd, seed=SEED),
        'subspace_relax': orig + softmode_subspace_ensemble(
            s1['coords'], n_ens - 1, k_modes=10, max_rmsd=max_rmsd, seed=SEED,
            relax=True, relax_iters=30),
    }
    ensembles['combo'] = ensembles['baseline'] + ensembles['subspace_relax']

    for name, cl in ensembles.items():
        tm, rm, dual = _eval(cl, s1, s2, ci1, ci2)
        row[f'{name}_max_tm_s2'] = tm
        row[f'{name}_min_rmsd_s2'] = rm
        row[f'{name}_dual'] = dual
    return row


def run(resume: bool = True, n_ens: int = 56, genes=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = get_all_benchmarks()
    if genes:
        gset = set(genes)
        targets = [t for t in targets if t.gene_name in gset]

    results, done = [], set()
    if resume and RAW_FILE.exists():
        existing = pd.read_csv(RAW_FILE)
        results = existing.to_dict('records')
        done = set(existing['gene'].tolist())

    for idx, target in enumerate(targets):
        if target.gene_name in done:
            continue
        logger.info("[%d/%d] %s (%s)", idx + 1, len(targets),
                    target.gene_name, target.category)
        t0 = time.time()
        try:
            row = process_target(target, n_ens=n_ens)
            row['runtime_s'] = round(time.time() - t0, 1)
            results.append(row)
            pd.DataFrame(results).to_csv(RAW_FILE, index=False)
            if row.get('status') == 'ok':
                logger.info(
                    "  base=%.3f single=%.3f sub=%.3f sub+relax=%.3f combo=%.3f | bestI=%.3f (%.1fs)",
                    row['baseline_max_tm_s2'], row['single_mode_max_tm_s2'],
                    row['subspace_max_tm_s2'], row['subspace_relax_max_tm_s2'],
                    row['combo_max_tm_s2'], row.get('best_single_overlap', 0) or 0,
                    row['runtime_s'])
        except Exception as e:
            logger.error("  ERROR %s: %s", target.gene_name, e)
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(results)
    logger.info("Done: %d rows", len(df))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--n-ens', type=int, default=56)
    ap.add_argument('--genes', type=str, default='')
    args = ap.parse_args()
    genes = [g.strip() for g in args.genes.split(',') if g.strip()] or None
    run(resume=not args.no_resume, n_ens=args.n_ens, genes=genes)
