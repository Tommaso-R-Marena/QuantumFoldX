#!/usr/bin/env python3
"""
dsib_ablation_diagnostic.py — STEP 1: Why does condition E exactly match condition B?

Prints per-protein breakdown of:
  - Which bridge source wins max TM→S2 (interp / switch-rigid / manifold)
  - Whether v3 non-bridge conformations ever beat bridge max
  - Why E (v2 + bridge only) equals B (full v3) on max-TM coverage

Run:
  python benchmarks/dsib_ablation_diagnostic.py
  python benchmarks/dsib_ablation_diagnostic.py --genes SRC,WAS,CLIC1,PDGFRB,REV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.benchmark_dataset import get_all_benchmarks
from src.scoring.qicess_v3 import create_dsib_scorer
from src.scoring.geometry_utils import transition_difficulty
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.metrics.structural_metrics import tm_score
from benchmarks.benchmark_utils import (
    parse_target_structures, get_domain_indices, find_common_residues,
)
from benchmarks.compare_v2_v3_coverage import _merge_v2_plus_bridge

BRIDGE_METHODS = {
    'common_residue_interp',
    'switch_contact_rigid',
    'manifold_bridge',
    'quantum_bridge',  # legacy lumped tag if tag_bridge_sources=False
    'quantum_bridge_added',
}


def _classify_method(conf: dict) -> str:
    return conf.get('bridge_source') or conf.get('method', 'unknown')


def _is_bridge_member(conf: dict) -> bool:
    m = _classify_method(conf)
    return m in BRIDGE_METHODS or m.startswith('bridge_')


def _per_member_tm(ensemble, s2, ci1, ci2):
    """Return list of (index, method, tm_s2) for each ensemble member."""
    rows = []
    for idx, conf in enumerate(ensemble):
        coords = conf['coords']
        valid = [(i, j) for i, j in zip(ci1, ci2)
                 if i < len(coords) and j < len(s2['coords'])]
        if len(valid) < 10:
            continue
        ens_pts = coords[[p[0] for p in valid]]
        s2_pts = s2['coords'][[p[1] for p in valid]]
        try:
            t = float(tm_score(s2_pts, ens_pts))
        except Exception:
            continue
        rows.append({
            'idx': idx,
            'method': _classify_method(conf),
            'perturbation_id': conf.get('perturbation_id', ''),
            'tm_s2': t,
            'is_bridge': _is_bridge_member(conf),
        })
    return rows


def _summarize_group(rows, predicate):
    sub = [r for r in rows if predicate(r)]
    if not sub:
        return {'n': 0, 'max_tm': None, 'best_method': None, 'best_idx': None}
    best = max(sub, key=lambda r: r['tm_s2'])
    return {
        'n': len(sub),
        'max_tm': best['tm_s2'],
        'best_method': best['method'],
        'best_idx': best['idx'],
        'best_pert_id': best['perturbation_id'],
    }


def diagnose_target(target, n_ens: int = 80, seed: int = 42) -> dict:
    from src.data.pdb_fetcher import compute_phi_psi

    s1, s2, status = parse_target_structures(target)
    if status != 'ok':
        return {'gene': target.gene_name, 'status': status}

    ci1, ci2, nc = find_common_residues(s1, s2)
    if nc < 20:
        return {'gene': target.gene_name, 'status': 'insufficient_overlap'}

    fd_idx, im_idx = get_domain_indices(s1, target)
    baseline_tm = tm_score(s1['coords'][ci1[:nc]], s2['coords'][ci2[:nc]])
    phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])
    diff = transition_difficulty(baseline_tm)

    ens_v2 = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=seed, phi_psi=phi_psi,
    )
    bridge = create_dsib_scorer().build_bridge(
        s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx)
    ens_v3 = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'], fd_idx, im_idx,
        n_conformations=n_ens, seed=seed, phi_psi=phi_psi,
        coords_s2=s2['coords'], quantum_bridge=bridge,
        transition_difficulty=diff,
        common_idx_s1=ci1, common_idx_s2=ci2,
        tag_bridge_sources=True,
    )
    ens_e = _merge_v2_plus_bridge(ens_v2, ens_v3)

    rows_b = _per_member_tm(ens_v3, s2, ci1, ci2)
    rows_e = _per_member_tm(ens_e, s2, ci1, ci2)

    max_b = max(r['tm_s2'] for r in rows_b) if rows_b else None
    max_e = max(r['tm_s2'] for r in rows_e) if rows_e else None

    v3_non_bridge = _summarize_group(rows_b, lambda r: not r['is_bridge'])
    v3_bridge = _summarize_group(rows_b, lambda r: r['is_bridge'])
    v3_interp = _summarize_group(rows_b, lambda r: r['method'] == 'common_residue_interp')
    v3_switch = _summarize_group(rows_b, lambda r: r['method'] == 'switch_contact_rigid')
    v3_manifold = _summarize_group(rows_b, lambda r: r['method'] == 'manifold_bridge')

    winner_b = max(rows_b, key=lambda r: r['tm_s2']) if rows_b else None

    # Count bridge members by source in v3
    bridge_counts = {}
    for conf in ens_v3:
        if _is_bridge_member(conf):
            m = _classify_method(conf)
            bridge_counts[m] = bridge_counts.get(m, 0) + 1

    # E=B mechanism
    e_equals_b = (max_b is not None and max_e is not None
                  and abs(max_b - max_e) < 1e-9)
    bridge_dominates = (
        v3_bridge['max_tm'] is not None and v3_non_bridge['max_tm'] is not None
        and v3_bridge['max_tm'] >= v3_non_bridge['max_tm'] - 1e-9
    )

    return {
        'gene': target.gene_name,
        'category': target.category,
        'status': 'ok',
        'baseline_tm': baseline_tm,
        'n_v3': len(ens_v3),
        'n_v2': len(ens_v2),
        'n_bridge_in_v3': sum(bridge_counts.values()),
        'bridge_counts': bridge_counts,
        'B_max_tm_s2': max_b,
        'E_max_tm_s2': max_e,
        'E_equals_B': e_equals_b,
        'v3_non_bridge_max_tm': v3_non_bridge['max_tm'],
        'v3_non_bridge_best_method': v3_non_bridge['best_method'],
        'v3_bridge_max_tm': v3_bridge['max_tm'],
        'v3_bridge_best_method': v3_bridge['best_method'],
        'winner_B_method': winner_b['method'] if winner_b else None,
        'winner_B_tm': winner_b['tm_s2'] if winner_b else None,
        'winner_B_is_bridge': winner_b['is_bridge'] if winner_b else None,
        'by_source_interp': v3_interp,
        'by_source_switch_rigid': v3_switch,
        'by_source_manifold': v3_manifold,
        'bridge_dominates_non_bridge': bridge_dominates,
        'e_equals_b_explanation': (
            'E is v2+bridge; max-TM coverage uses ensemble max, not ranking. '
            'E==B when bridge max in v3 >= all v2-only and v3-only non-bridge members, '
            'and E contains the same bridge coordinates as v3.'
            if e_equals_b else 'E and B differ on max TM'
        ),
    }


def _print_report(d: dict):
    if d.get('status') != 'ok':
        print(f"\n{d['gene']}: SKIPPED ({d.get('status')})")
        return

    print(f"\n{'='*72}")
    print(f"{d['gene']} ({d['category']})  baseline S1↔S2 TM={d['baseline_tm']:.4f}")
    print(f"  Ensemble sizes: v2={d['n_v2']}  v3={d['n_v3']}  bridge_in_v3={d['n_bridge_in_v3']}")
    print(f"  Bridge member counts in v3: {d['bridge_counts']}")
    print(f"  B max TM→S2 = {d['B_max_tm_s2']:.6f}   E max TM→S2 = {d['E_max_tm_s2']:.6f}   E==B? {d['E_equals_B']}")
    print(f"  v3 NON-bridge max TM→S2 = {d['v3_non_bridge_max_tm']}  (best: {d['v3_non_bridge_best_method']})")
    print(f"  v3 BRIDGE max TM→S2     = {d['v3_bridge_max_tm']:.6f}  (best: {d['v3_bridge_best_method']})")
    print(f"  WINNER in B: method={d['winner_B_method']}  TM={d['winner_B_tm']:.6f}  is_bridge={d['winner_B_is_bridge']}")
    print(f"  Per-source max TM→S2 among bridge members in B:")
    for label, key in [
        ('(a) common_residue_interp', 'by_source_interp'),
        ('(b) switch_contact_rigid', 'by_source_switch_rigid'),
        ('(c) manifold_bridge', 'by_source_manifold'),
    ]:
        s = d[key]
        print(f"    {label}: n={s['n']} max_tm={s['max_tm']} best={s['best_method']}")
    print(f"  Bridge dominates non-bridge in v3? {d['bridge_dominates_non_bridge']}")
    print(f"  Mechanism: {d['e_equals_b_explanation']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genes', type=str, default='',
                        help='Comma-separated gene list; default = 6 across categories')
    parser.add_argument('--n-ens', type=int, default=80)
    args = parser.parse_args()

    default_genes = ['SRC', 'WAS', 'FYN', 'CLIC1', 'REV', 'PDGFRB']
    gene_list = [g.strip() for g in args.genes.split(',') if g.strip()] or default_genes

    by_gene = {t.gene_name: t for t in get_all_benchmarks()}
    print("DSIB ABLATION DIAGNOSTIC — STEP 1")
    print(f"Command: python benchmarks/dsib_ablation_diagnostic.py --genes {','.join(gene_list)}")
    print(f"Proteins: {gene_list}")

    results = []
    for gene in gene_list:
        if gene not in by_gene:
            print(f"\n{gene}: NOT IN BENCHMARK")
            continue
        d = diagnose_target(by_gene[gene], n_ens=args.n_ens)
        _print_report(d)
        results.append(d)

    n_eq = sum(1 for d in results if d.get('E_equals_B'))
    n_bridge_wins = sum(1 for d in results if d.get('winner_B_is_bridge'))
    n_interp_wins = sum(1 for d in results if d.get('winner_B_method') == 'common_residue_interp')
    n_switch_wins = sum(1 for d in results if d.get('winner_B_method') == 'switch_contact_rigid')
    n_manifold_wins = sum(1 for d in results if d.get('winner_B_method') == 'manifold_bridge')

    print(f"\n{'='*72}")
    print("AGGREGATE (this run)")
    print(f"  Proteins diagnosed: {len(results)}")
    print(f"  E==B count: {n_eq}/{len(results)}")
    print(f"  Winner is bridge member: {n_bridge_wins}/{len(results)}")
    print(f"  Winner by source: interp={n_interp_wins} switch_rigid={n_switch_wins} manifold={n_manifold_wins}")
    print("\nROOT CAUSE SUMMARY (if E==B on all proteins):")
    print("  1. Coverage metric = max TM over ALL ensemble members (no scoring/selection).")
    print("  2. Condition E = v2 ensemble + exact bridge coords copied from v3.")
    print("  3. Therefore E_max >= bridge_max always; E==B iff bridge member holds global max in v3.")
    print("  4. Extra v3 members (NMA/rigid-body/torsion) cannot raise max if bridge already wins.")


if __name__ == '__main__':
    main()
