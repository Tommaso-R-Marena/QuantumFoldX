#!/usr/bin/env python3
"""
run_rigorous_benchmark.py — Rigorous, honest benchmark of blind alternate-state
prediction with proper controls and elastic-network mode analysis.

For each of the 49 dual-state proteins we record:

  (1) Elastic-network mode-overlap metrics (state 1 only): how much of the
      observed state-1 -> state-2 change lies in the softest ANM modes.

  (2) Coverage / max-TM-to-state-2 for a panel of ensembles at a *fixed*
      conformation budget, split into honest categories:

      BLIND (state 1 only, genuinely predictive):
        - blind_baseline : existing multi-scale sampler (NMA + rigid + torsion)
        - blind_softmode : principled scan along the softest ANM modes
        - blind_union    : union of the two blind ensembles (2x budget)

      ORACLE (state 2 known at generation time; NOT predictive):
        - oracle_interp  : pure linear S1->S2 interpolation (NO Ising at all)
        - oracle_dsib    : full "quantum" Dual-State Ising Bridge pipeline

The oracle_interp vs oracle_dsib comparison isolates the contribution of the
Ising/quantum machinery; the blind_softmode vs blind_baseline comparison tests
whether physically-motivated soft-mode sampling improves genuine prediction.

Run:
  python benchmarks/run_rigorous_benchmark.py --no-resume
  python benchmarks/run_rigorous_benchmark.py --genes AK1,SRC,KAI_B --n-ens 40

Outputs:
  results/rigorous/rigorous_benchmark.csv
  results/rigorous/mode_overlap_curves.json
  results/rigorous/rigorous_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.benchmark_dataset import get_all_benchmarks, get_af3_baseline
from src.data.pdb_fetcher import compute_phi_psi
from src.scoring.qicess_v3 import create_dsib_scorer
from src.scoring.geometry_utils import (
    transition_difficulty, interpolate_coords_on_common,
)
from src.ensemble.conformational_sampler import (
    generate_hybrid_ensemble, generate_anm_mode_scan_ensemble,
)
from src.analysis.mode_overlap import analyze_transition
from benchmarks.benchmark_utils import (
    parse_target_structures, get_domain_indices, find_common_residues,
    evaluate_ensemble_vs_state,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / 'results' / 'rigorous'
RAW_FILE = OUT_DIR / 'rigorous_benchmark.csv'
CURVE_FILE = OUT_DIR / 'mode_overlap_curves.json'
STATS_FILE = OUT_DIR / 'rigorous_stats.json'

TM_THRESHOLD = 0.5
SEED = 42


def _coords_list(ensemble):
    return [c['coords'] if isinstance(c, dict) else c for c in ensemble]


def _evaluate(coords_list, s1, s2, ci1, ci2):
    """Max TM / min RMSD of an ensemble against state 1 and state 2."""
    e_s1 = evaluate_ensemble_vs_state(
        coords_list, s1['coords'],
        list(range(s1['n_residues'])), list(range(s1['n_residues'])))
    e_s2 = evaluate_ensemble_vs_state(coords_list, s2['coords'], ci1, ci2)
    s1_ok = bool(e_s1['max_tm'] and e_s1['max_tm'] > TM_THRESHOLD)
    s2_ok = bool(e_s2['max_tm'] and e_s2['max_tm'] > TM_THRESHOLD)
    return {
        'max_tm_s2': e_s2['max_tm'],
        'min_rmsd_s2': e_s2['min_rmsd'],
        'max_tm_s1': e_s1['max_tm'],
        'dual_covered': s1_ok and s2_ok,
        'n': len(coords_list),
    }


def _linear_interp_ensemble(s1, s2, ci1, ci2, n):
    """Pure geometric S1->S2 interpolation (oracle control, no Ising).

    Interpolation is capped at alpha=0.95 so the exact state-2 answer is never
    handed to the ensemble; this matches the DSIB oracle's interpolation range
    and makes the "Ising adds nothing beyond interpolation" comparison fair.
    """
    out = [s1['coords'].copy()]
    for alpha in np.linspace(0.05, 0.95, n - 1):
        out.append(interpolate_coords_on_common(
            s1['coords'], s2['coords'], float(alpha), ci1, ci2))
    return out


def process_target(target, n_ens: int, n_modes: int = 20, cutoff: float = 13.0) -> dict:
    s1, s2, status = parse_target_structures(target)
    if status != 'ok' or s1['n_residues'] > 1000:
        return {'gene': target.gene_name, 'status': status or 'too_large'}

    ci1, ci2, nc = find_common_residues(s1, s2)
    if nc < 20:
        return {'gene': target.gene_name, 'status': 'insufficient_overlap'}

    from src.metrics.structural_metrics import tm_score
    baseline_tm = float(tm_score(s1['coords'][ci1[:nc]], s2['coords'][ci2[:nc]]))
    fd_idx, im_idx = get_domain_indices(s1, target)
    phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])
    diff = transition_difficulty(baseline_tm)

    row = {
        'gene': target.gene_name, 'category': target.category,
        'n_residues': s1['n_residues'], 'n_core': nc,
        'baseline_tm': baseline_tm, 'status': 'ok',
    }

    # (1) Elastic-network mode overlap (state 1 only) --------------------------
    mo = analyze_transition(s1['coords'], s2['coords'], ci1, ci2,
                            n_modes=n_modes, cutoff=cutoff)
    curve = None
    if mo is not None:
        row.update({
            'transition_rmsd': mo.transition_magnitude,
            'softest_mode_overlap': mo.softest_mode_overlap,
            'best_single_overlap': mo.best_single_overlap,
            'best_mode_index': mo.best_mode_index,
            'cum_overlap_2': mo.cum_overlap_2,
            'cum_overlap_5': mo.cum_overlap_5,
            'cum_overlap_10': mo.cum_overlap_10,
            'n_modes_for_half': mo.n_modes_for_half,
            'best_mode_collectivity': mo.best_mode_collectivity,
        })
        curve = mo.cumulative_overlap

    # (2) Ensembles ------------------------------------------------------------
    # BLIND baseline (existing multi-scale sampler, no state 2)
    blind_base = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=SEED, phi_psi=phi_psi)
    # BLIND soft-mode scan (state 1 only)
    blind_soft = [s1['coords'].copy()] + generate_anm_mode_scan_ensemble(
        s1['coords'], n_conformations=n_ens - 1, n_modes=6,
        max_rmsd=max(6.0, 1.5 * (mo.transition_magnitude if mo else 8.0)),
        cutoff=cutoff, seed=SEED)

    base_c = _coords_list(blind_base)
    soft_c = _coords_list(blind_soft)
    union_c = base_c + soft_c

    # ORACLE interpolation (no Ising) and full DSIB
    interp_c = _linear_interp_ensemble(s1, s2, ci1, ci2, n_ens)
    bridge = create_dsib_scorer().build_bridge(
        s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx)
    dsib = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=SEED, phi_psi=phi_psi,
        coords_s2=s2['coords'], quantum_bridge=bridge,
        transition_difficulty=diff, common_idx_s1=ci1, common_idx_s2=ci2)
    dsib_c = _coords_list(dsib)

    conditions = {
        'blind_baseline': base_c,
        'blind_softmode': soft_c,
        'blind_union': union_c,
        'oracle_interp': interp_c,
        'oracle_dsib': dsib_c,
    }
    for name, cl in conditions.items():
        m = _evaluate(cl, s1, s2, ci1, ci2)
        row[f'{name}_max_tm_s2'] = m['max_tm_s2']
        row[f'{name}_min_rmsd_s2'] = m['min_rmsd_s2']
        row[f'{name}_dual'] = m['dual_covered']
        row[f'{name}_n'] = m['n']

    return row, curve


def run(resume: bool = True, n_ens: int = 64, genes=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = get_all_benchmarks()
    if genes:
        gset = set(genes)
        targets = [t for t in targets if t.gene_name in gset]

    results, curves, done = [], {}, set()
    if resume and RAW_FILE.exists():
        existing = pd.read_csv(RAW_FILE)
        results = existing.to_dict('records')
        done = set(existing['gene'].tolist())
        if CURVE_FILE.exists():
            curves = json.loads(CURVE_FILE.read_text())

    for idx, target in enumerate(targets):
        if target.gene_name in done:
            continue
        logger.info("[%d/%d] %s (%s)", idx + 1, len(targets),
                    target.gene_name, target.category)
        t0 = time.time()
        try:
            out = process_target(target, n_ens=n_ens)
            if isinstance(out, tuple):
                row, curve = out
                if curve is not None:
                    curves[target.gene_name] = {
                        'category': target.category,
                        'cumulative_overlap': curve,
                        'baseline_tm': row.get('baseline_tm'),
                    }
            else:
                row = out
            row['runtime_s'] = round(time.time() - t0, 1)
            results.append(row)
            pd.DataFrame(results).to_csv(RAW_FILE, index=False)
            CURVE_FILE.write_text(json.dumps(curves, indent=2))
            if row.get('status') == 'ok':
                logger.info(
                    "  blind base=%.3f soft=%.3f union=%.3f | oracle interp=%.3f dsib=%.3f | I1=%.3f (%.1fs)",
                    row.get('blind_baseline_max_tm_s2', 0) or 0,
                    row.get('blind_softmode_max_tm_s2', 0) or 0,
                    row.get('blind_union_max_tm_s2', 0) or 0,
                    row.get('oracle_interp_max_tm_s2', 0) or 0,
                    row.get('oracle_dsib_max_tm_s2', 0) or 0,
                    row.get('softest_mode_overlap', 0) or 0,
                    row['runtime_s'])
        except Exception as e:
            logger.error("  ERROR %s: %s", target.gene_name, e)
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(results)
    ok = df[df['status'] == 'ok'].copy() if 'status' in df.columns else df
    logger.info("Done: %d/%d ok", len(ok), len(df))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--n-ens', type=int, default=64)
    ap.add_argument('--genes', type=str, default='')
    args = ap.parse_args()
    genes = [g.strip() for g in args.genes.split(',') if g.strip()] or None
    run(resume=not args.no_resume, n_ens=args.n_ens, genes=genes)
