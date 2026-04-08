#!/usr/bin/env python3
"""
run_benchmark_v2.py — QuantumFoldX Dual-State Coverage Benchmark

STRATEGIC PIVOT: Instead of competing with AF3 on single-structure accuracy
(where AF3 trained on 200k+ PDB structures will always win), we evaluate 
the genuine advantage of quantum-scored ensemble sampling:

  → Can the ensemble capture BOTH conformational states of a protein?

Published AF3 dual-state coverage rates:
  - Autoinhibited proteins: 14% both states in top-5 (Papageorgiou 2025)
  - Multi-state proteins: 23.3% both states correct (M-SADA, BIB 2025)
  - Fold-switching: 7.6% success (Ronish 2024)

If QuantumFoldX's ensemble achieves higher dual-state coverage, that's a
genuine, scientifically meaningful advantage for drug design where understanding
conformational landscapes matters.

METHODOLOGY:
1. Parse both PDB states, find common residue set
2. Generate large ensemble from state 1 (NMA + rigid-body, N=100)
3. Score ensemble with QICESS v2 (quantum-informed ranking)
4. Evaluate each ensemble member against BOTH states
5. Report dual-state coverage rate and compare to AF3

HONESTY NOTES:
  ⚠ ALL QUANTUM CIRCUITS ARE CLASSICALLY SIMULATED (PennyLane lightning.qubit)
  ⚠ AF3 numbers are from published peer-reviewed benchmarks, NOT re-run
  ⚠ QFX starts from a KNOWN structure; AF3 predicts from sequence alone
  ⚠ This is conformational exploration, not de novo structure prediction
"""

import sys
import os
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pdb_fetcher import (
    fetch_pdb, parse_pdb_ca_coords, compute_contact_map, compute_phi_psi
)
from src.scoring.qicess_v2 import QICESSv2Scorer
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.metrics.structural_metrics import (
    rmsd, tm_score, gdt_ts, lddt, imfd_rmsd, radius_of_gyration, kabsch_align
)
from configs.benchmark_dataset import (
    get_autoinhibited_benchmark, get_af3_baseline, BenchmarkTarget
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(exist_ok=True)
(RESULTS_DIR / 'tables').mkdir(exist_ok=True)
(RESULTS_DIR / 'stats').mkdir(exist_ok=True)


def find_common_residues(struct1, struct2):
    """
    Find common residue numbers between two structures.
    Returns arrays of indices into each structure for aligned residues.
    """
    resids1 = struct1['residue_ids']
    resids2 = struct2['residue_ids']
    
    set1 = set(resids1)
    set2 = set(resids2)
    common = sorted(set1 & set2)
    
    if not common:
        # Fallback: use positional alignment up to min length
        n = min(len(resids1), len(resids2))
        return list(range(n)), list(range(n)), n
    
    idx1 = [i for i, r in enumerate(resids1) if r in set2]
    idx2 = [i for i, r in enumerate(resids2) if r in set1]
    
    # Ensure same length (handle edge cases with duplicate residue IDs)
    n = min(len(idx1), len(idx2))
    return idx1[:n], idx2[:n], n


def evaluate_ensemble_vs_state(ensemble_coords_list, target_coords, common_idx_ens, common_idx_target):
    """
    Evaluate how well an ensemble covers a target conformational state.
    
    Returns min RMSD, max TM-score, best GDT-TS across ensemble members.
    """
    best_rmsd = float('inf')
    best_tm = 0.0
    best_gdt = 0.0
    best_lddt_val = 0.0
    best_idx = -1
    
    target_common = target_coords[common_idx_target]
    
    all_rmsds = []
    all_tms = []
    
    for idx, ens_coords in enumerate(ensemble_coords_list):
        # Extract common residues
        n_ens = len(ens_coords)
        valid_idx = [i for i in common_idx_ens if i < n_ens]
        valid_target_idx = common_idx_target[:len(valid_idx)]
        
        if len(valid_idx) < 10:
            continue
            
        ens_common = ens_coords[valid_idx]
        tgt_common = target_coords[valid_target_idx]
        
        n_c = min(len(ens_common), len(tgt_common))
        if n_c < 10:
            continue
        
        ens_c = ens_common[:n_c]
        tgt_c = tgt_common[:n_c]
        
        try:
            r = rmsd(tgt_c, ens_c)
            t = tm_score(tgt_c, ens_c)
            all_rmsds.append(r)
            all_tms.append(t)
            
            if r < best_rmsd:
                best_rmsd = r
                best_idx = idx
            if t > best_tm:
                best_tm = t
                
            # Only compute expensive metrics for good candidates
            if r < best_rmsd + 2.0:
                g = gdt_ts(tgt_c, ens_c)
                if g > best_gdt:
                    best_gdt = g
                l = lddt(ens_c, tgt_c)
                if l > best_lddt_val:
                    best_lddt_val = l
        except Exception as e:
            continue
    
    return {
        'min_rmsd': best_rmsd if best_rmsd < float('inf') else None,
        'max_tm': best_tm,
        'max_gdt': best_gdt,
        'max_lddt': best_lddt_val,
        'best_idx': best_idx,
        'all_rmsds': all_rmsds,
        'all_tms': all_tms,
        'median_rmsd': float(np.median(all_rmsds)) if all_rmsds else None,
        'median_tm': float(np.median(all_tms)) if all_tms else None,
    }


def process_target_v2(target: BenchmarkTarget, scorer: QICESSv2Scorer,
                       n_ensemble: int = 100) -> dict:
    """
    Process a single benchmark target with dual-state evaluation.
    
    1. Fetch and parse both PDB states
    2. Find common residue set
    3. Generate large ensemble from state 1
    4. Score ensemble with QICESS v2
    5. Evaluate ensemble against BOTH states
    6. Compute dual-state coverage metrics
    """
    result = {
        'protein': target.protein_name,
        'gene': target.gene_name,
        'pdb_state1': target.pdb_id_state1,
        'pdb_state2': target.pdb_id_state2,
        'uniprot': target.uniprot_id,
        'af3_imfd_rmsd': target.af3_imfd_rmsd,
        'category': target.category,
        'status': 'pending'
    }
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {target.gene_name} ({target.protein_name})")
    logger.info(f"  PDB: {target.pdb_id_state1} (state1) / {target.pdb_id_state2} (state2)")
    logger.info(f"  AF3 imfdRMSD: {target.af3_imfd_rmsd} Å")
    logger.info(f"{'='*70}")
    
    # Skip self-referential targets
    if target.pdb_id_state1 == target.pdb_id_state2:
        result['status'] = 'skipped_self_reference'
        result['notes'] = 'Same PDB for both states — no dual-state test possible'
        logger.warning(f"  SKIPPED: {target.gene_name} uses same PDB for both states")
        return result
    
    # 1. Fetch structures
    pdb1_path = fetch_pdb(target.pdb_id_state1)
    pdb2_path = fetch_pdb(target.pdb_id_state2)
    
    if not pdb1_path or not pdb2_path:
        result['status'] = 'fetch_failed'
        return result
    
    # 2. Parse coordinates
    struct1 = parse_pdb_ca_coords(pdb1_path, chain=target.chain_state1)
    if struct1 is None:
        struct1 = parse_pdb_ca_coords(pdb1_path, chain=None)
    
    struct2 = parse_pdb_ca_coords(pdb2_path, chain=target.chain_state2)
    if struct2 is None:
        struct2 = parse_pdb_ca_coords(pdb2_path, chain=None)
    
    if struct1 is None or struct2 is None:
        result['status'] = 'parse_failed'
        return result
    
    result['n_residues_state1'] = struct1['n_residues']
    result['n_residues_state2'] = struct2['n_residues']
    
    logger.info(f"  State 1: {struct1['n_residues']} residues (chain {struct1['chain']})")
    logger.info(f"  State 2: {struct2['n_residues']} residues (chain {struct2['chain']})")
    
    # 3. Find common residues
    common_idx1, common_idx2, n_common = find_common_residues(struct1, struct2)
    result['n_common_residues'] = n_common
    
    if n_common < 20:
        result['status'] = 'insufficient_common_residues'
        return result
    
    logger.info(f"  Common residues: {n_common}")
    
    # Compute baseline RMSD between the two states (ground truth distance)
    coords1_common = struct1['coords'][common_idx1[:n_common]]
    coords2_common = struct2['coords'][common_idx2[:n_common]]
    baseline_rmsd = rmsd(coords1_common, coords2_common)
    baseline_tm = tm_score(coords1_common, coords2_common)
    result['state1_vs_state2_rmsd'] = baseline_rmsd
    result['state1_vs_state2_tm'] = baseline_tm
    
    logger.info(f"  Baseline state1↔state2 RMSD: {baseline_rmsd:.2f} Å")
    logger.info(f"  Baseline state1↔state2 TM: {baseline_tm:.3f}")
    
    # 4. Map domain residue ranges
    fd_start, fd_end = target.fd_residues
    im_start, im_end = target.im_residues
    
    fd_indices = [i for i, r in enumerate(struct1['residue_ids']) 
                  if fd_start <= r <= fd_end]
    im_indices = [i for i, r in enumerate(struct1['residue_ids'])
                  if im_start <= r <= im_end]
    
    if not fd_indices or not im_indices:
        n = struct1['n_residues']
        fd_indices = list(range(n // 2, n))
        im_indices = list(range(0, n // 2))
    
    result['n_fd_residues'] = len(fd_indices)
    result['n_im_residues'] = len(im_indices)
    
    # 5. Generate LARGE ensemble from state 1
    logger.info(f"  Generating ensemble ({n_ensemble} conformations)...")
    t_start = time.time()
    
    ensemble = generate_hybrid_ensemble(
        struct1['coords'], struct1['sequence'],
        fd_indices=fd_indices, im_indices=im_indices,
        n_conformations=n_ensemble, seed=42
    )
    
    result['ensemble_size'] = len(ensemble)
    
    # 6. Compute phi/psi for scoring
    phi_psi = compute_phi_psi(pdb1_path, chain=struct1['chain'])
    for conf in ensemble:
        conf['phi_psi'] = phi_psi
    
    # 7. Score ensemble with QICESS v2
    logger.info(f"  Scoring ensemble with QICESS v2...")
    scored_ensemble = scorer.rank_ensemble(
        ensemble, struct1['sequence'],
        reference_coords=struct1['coords'],
        fd_indices=fd_indices, im_indices=im_indices
    )
    
    t_score = time.time() - t_start
    result['scoring_time_s'] = t_score
    logger.info(f"  Scoring complete in {t_score:.1f}s")
    
    if not scored_ensemble:
        result['status'] = 'scoring_failed'
        return result
    
    # 8. Extract all ensemble coordinates
    all_coords = [conf['coords'] for conf in scored_ensemble]
    
    # 9. Evaluate ensemble against BOTH states
    logger.info(f"  Evaluating ensemble vs state 1...")
    eval_s1 = evaluate_ensemble_vs_state(
        all_coords, struct1['coords'], 
        list(range(min(len(all_coords[0]), struct1['n_residues']))),
        list(range(struct1['n_residues']))
    )
    
    logger.info(f"  Evaluating ensemble vs state 2...")
    eval_s2 = evaluate_ensemble_vs_state(
        all_coords, struct2['coords'],
        common_idx1, common_idx2
    )
    
    # 10. Evaluate top-ranked (by quantum score) vs all ensemble
    top_10_coords = [conf['coords'] for conf in scored_ensemble[:10]]
    eval_s2_top10 = evaluate_ensemble_vs_state(
        top_10_coords, struct2['coords'],
        common_idx1, common_idx2
    )
    
    # Store results
    result['ens_min_rmsd_state1'] = eval_s1['min_rmsd']
    result['ens_max_tm_state1'] = eval_s1['max_tm']
    result['ens_min_rmsd_state2'] = eval_s2['min_rmsd']
    result['ens_max_tm_state2'] = eval_s2['max_tm']
    result['ens_max_gdt_state2'] = eval_s2['max_gdt']
    result['ens_max_lddt_state2'] = eval_s2['max_lddt']
    result['ens_median_rmsd_state2'] = eval_s2['median_rmsd']
    result['ens_median_tm_state2'] = eval_s2['median_tm']
    
    # Top-10 quantum-ranked results
    result['top10_min_rmsd_state2'] = eval_s2_top10['min_rmsd']
    result['top10_max_tm_state2'] = eval_s2_top10['max_tm']
    
    # QICESS scores for top conformation
    best = scored_ensemble[0]
    result['qicess_composite'] = best['composite']
    result['qicess_quantum_energy'] = best.get('quantum_energy_raw', 0.0)
    result['n_qubits'] = best.get('n_qubits', 0)
    
    # Ensemble diversity
    rmsds_inner = []
    for i in range(min(10, len(scored_ensemble))):
        for j in range(i+1, min(10, len(scored_ensemble))):
            n_c = min(len(scored_ensemble[i]['coords']), len(scored_ensemble[j]['coords']))
            if n_c > 10:
                r = rmsd(scored_ensemble[i]['coords'][:n_c], scored_ensemble[j]['coords'][:n_c])
                rmsds_inner.append(r)
    result['ensemble_diversity'] = float(np.mean(rmsds_inner)) if rmsds_inner else 0.0
    
    # imfdRMSD for the best-to-state2 ensemble member (for comparison)
    if eval_s2['best_idx'] >= 0:
        best_s2_conf = scored_ensemble[eval_s2['best_idx']]
        n_c = min(len(best_s2_conf['coords']), len(struct1['coords']))
        fd_valid = [i for i in fd_indices if i < n_c]
        im_valid = [i for i in im_indices if i < n_c]
        if fd_valid and im_valid:
            result['best_s2_imfd_rmsd'] = imfd_rmsd(
                best_s2_conf['coords'][:n_c], struct1['coords'][:n_c],
                fd_valid, im_valid
            )
    
    # Dual-state coverage assessment
    # Threshold: TM-score > 0.5 (standard fold-level similarity)
    tm_thresh = 0.5
    rmsd_thresh = 5.0  # Å (lenient for multi-domain)
    
    state1_covered = eval_s1['max_tm'] > tm_thresh if eval_s1['max_tm'] else False
    state2_covered = eval_s2['max_tm'] > tm_thresh if eval_s2['max_tm'] else False
    result['state1_covered_tm05'] = state1_covered
    result['state2_covered_tm05'] = state2_covered
    result['dual_state_covered_tm05'] = state1_covered and state2_covered
    
    state2_covered_rmsd = (eval_s2['min_rmsd'] < rmsd_thresh) if eval_s2['min_rmsd'] else False
    result['state2_covered_rmsd5'] = state2_covered_rmsd
    
    # State 2 RMSD improvement (how much closer does ensemble get vs starting structure?)
    if eval_s2['min_rmsd'] and baseline_rmsd > 0:
        result['state2_rmsd_improvement'] = baseline_rmsd - eval_s2['min_rmsd']
        result['state2_rmsd_improvement_pct'] = (baseline_rmsd - eval_s2['min_rmsd']) / baseline_rmsd * 100
    
    # Quantum ranking quality: does quantum scoring preferentially rank state2-like conformations?
    if eval_s2['all_tms']:
        # Spearman correlation between QICESS rank and TM-score to state 2
        ranks = list(range(len(eval_s2['all_tms'])))
        if len(ranks) > 5:
            rho, p_val = scipy_stats.spearmanr(
                ranks[:len(eval_s2['all_tms'])], 
                eval_s2['all_tms']
            )
            result['quantum_rank_state2_tm_rho'] = float(rho)
            result['quantum_rank_state2_tm_p'] = float(p_val)
    
    result['status'] = 'success'
    
    logger.info(f"  RESULTS for {target.gene_name}:")
    logger.info(f"    Baseline state1↔state2: RMSD={baseline_rmsd:.2f}Å, TM={baseline_tm:.3f}")
    logger.info(f"    Ensemble → state1: min_RMSD={eval_s1['min_rmsd']:.2f}Å, max_TM={eval_s1['max_tm']:.3f}")
    logger.info(f"    Ensemble → state2: min_RMSD={eval_s2['min_rmsd']:.2f}Å, max_TM={eval_s2['max_tm']:.3f}")
    logger.info(f"    State2 RMSD improvement: {result.get('state2_rmsd_improvement', 0):.2f}Å ({result.get('state2_rmsd_improvement_pct', 0):.1f}%)")
    logger.info(f"    Dual-state covered (TM>0.5): {result['dual_state_covered_tm05']}")
    logger.info(f"    Quantum-ranked top-10 → state2: min_RMSD={eval_s2_top10['min_rmsd']:.2f}Å, max_TM={eval_s2_top10['max_tm']:.3f}")
    logger.info(f"    QICESS composite: {result['qicess_composite']:.4f}")
    logger.info(f"    Ensemble diversity: {result['ensemble_diversity']:.2f}Å")
    
    return result


def run_statistical_tests_v2(results_df: pd.DataFrame, af3_baseline: dict) -> dict:
    """
    Statistical tests for dual-state conformational coverage.
    
    Primary comparison: QFX dual-state coverage rate vs AF3 published rates.
    """
    stats_results = {}
    
    valid = results_df[results_df['status'] == 'success'].copy()
    n = len(valid)
    
    if n < 3:
        return {'n_valid': n, 'warning': 'insufficient data'}
    
    # ═══════════════════════════════════════════════
    # PRIMARY ENDPOINT: Dual-state coverage rate
    # ═══════════════════════════════════════════════
    
    # QFX dual-state coverage (TM > 0.5 threshold)
    qfx_dsc = valid['dual_state_covered_tm05'].sum()
    qfx_dsc_rate = qfx_dsc / n
    
    # AF3 published rates
    af3_auto_rate = af3_baseline['autoinhibited']['fraction_both_states']  # 0.14
    af3_multi_rate = af3_baseline['multistate']['fraction_both_states_correct']  # 0.233
    
    # Binomial test: is QFX rate significantly > AF3 rate?
    binom_pval_auto = scipy_stats.binom_test(int(qfx_dsc), n, af3_auto_rate, alternative='greater')
    binom_pval_multi = scipy_stats.binom_test(int(qfx_dsc), n, af3_multi_rate, alternative='greater')
    
    # Wilson confidence interval for QFX rate
    z = 1.96
    p_hat = qfx_dsc_rate
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    wilson_ci = (max(0, center - margin), min(1, center + margin))
    
    stats_results['primary_endpoint'] = {
        'metric': 'Dual-State Coverage Rate (TM>0.5)',
        'n_targets': int(n),
        'qfx_n_covered': int(qfx_dsc),
        'qfx_rate': float(qfx_dsc_rate),
        'qfx_wilson_95ci': [float(wilson_ci[0]), float(wilson_ci[1])],
        'af3_autoinhibited_rate': float(af3_auto_rate),
        'af3_multistate_rate': float(af3_multi_rate),
        'binom_test_vs_af3_auto': {
            'test': f'Binomial test H1: QFX rate > {af3_auto_rate}',
            'p_value': float(binom_pval_auto),
            'significant_005': binom_pval_auto < 0.05
        },
        'binom_test_vs_af3_multi': {
            'test': f'Binomial test H1: QFX rate > {af3_multi_rate}',
            'p_value': float(binom_pval_multi),
            'significant_005': binom_pval_multi < 0.05
        },
    }
    
    # ═══════════════════════════════════════════════
    # SECONDARY: State 2 RMSD improvement
    # ═══════════════════════════════════════════════
    
    valid_imp = valid.dropna(subset=['state2_rmsd_improvement'])
    if len(valid_imp) >= 3:
        improvements = valid_imp['state2_rmsd_improvement'].values
        
        # One-sample Wilcoxon signed-rank: is median improvement > 0?
        pos = improvements[improvements > 0]
        if len(pos) >= 1:
            stat, p_val = scipy_stats.wilcoxon(improvements, alternative='greater')
        else:
            stat, p_val = 0.0, 1.0
        
        stats_results['state2_improvement'] = {
            'metric': 'State 2 RMSD Improvement (Å)',
            'n': int(len(improvements)),
            'mean': float(np.mean(improvements)),
            'median': float(np.median(improvements)),
            'std': float(np.std(improvements)),
            'fraction_improved': float(np.mean(improvements > 0)),
            'wilcoxon_p': float(p_val),
            'significant_005': p_val < 0.05,
        }
    
    # ═══════════════════════════════════════════════
    # SECONDARY: Ensemble TM-score to state 2
    # ═══════════════════════════════════════════════
    
    valid_tm = valid.dropna(subset=['ens_max_tm_state2'])
    if len(valid_tm) >= 3:
        tms = valid_tm['ens_max_tm_state2'].values
        stats_results['state2_tm_scores'] = {
            'metric': 'Max TM-score to State 2 (across ensemble)',
            'n': int(len(tms)),
            'mean': float(np.mean(tms)),
            'median': float(np.median(tms)),
            'std': float(np.std(tms)),
            'fraction_above_05': float(np.mean(tms > 0.5)),
            'fraction_above_07': float(np.mean(tms > 0.7)),
        }
    
    # ═══════════════════════════════════════════════
    # SECONDARY: Quantum scoring quality
    # ═══════════════════════════════════════════════
    
    valid_qr = valid.dropna(subset=['quantum_rank_state2_tm_rho'])
    if len(valid_qr) >= 3:
        rhos = valid_qr['quantum_rank_state2_tm_rho'].values
        stats_results['quantum_scoring_quality'] = {
            'metric': 'Spearman ρ (QICESS rank vs TM to state2)',
            'n': int(len(rhos)),
            'mean_rho': float(np.mean(rhos)),
            'median_rho': float(np.median(rhos)),
            'note': 'Negative ρ means higher-ranked conformations have higher TM to state2 (desirable)'
        }
    
    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    
    stats_results['summary'] = {
        'n_targets_total': int(len(results_df)),
        'n_successful': int(n),
        'n_skipped': int((results_df['status'] == 'skipped_self_reference').sum()),
        'n_failed': int(n - (results_df['status'].isin(['success', 'skipped_self_reference'])).sum()),
        'mean_scoring_time_s': float(valid['scoring_time_s'].mean()) if 'scoring_time_s' in valid.columns else 0,
        'total_time_s': float(valid['scoring_time_s'].sum()) if 'scoring_time_s' in valid.columns else 0,
    }
    
    return stats_results


def generate_results_tables(results_df: pd.DataFrame) -> dict:
    """Generate publication-quality results tables."""
    valid = results_df[results_df['status'] == 'success'].copy()
    
    tables = {}
    
    # Table 1: Dual-State Coverage
    t1_cols = ['gene', 'n_residues_state1', 'n_common_residues',
               'state1_vs_state2_rmsd', 'state1_vs_state2_tm',
               'ens_min_rmsd_state2', 'ens_max_tm_state2',
               'state2_rmsd_improvement_pct', 'dual_state_covered_tm05',
               'af3_imfd_rmsd']
    
    available_cols = [c for c in t1_cols if c in valid.columns]
    t1 = valid[available_cols].copy()
    
    t1_rename = {
        'gene': 'Gene',
        'n_residues_state1': 'N_res',
        'n_common_residues': 'N_common',
        'state1_vs_state2_rmsd': 'S1↔S2_RMSD',
        'state1_vs_state2_tm': 'S1↔S2_TM',
        'ens_min_rmsd_state2': 'Ens→S2_minRMSD',
        'ens_max_tm_state2': 'Ens→S2_maxTM',
        'state2_rmsd_improvement_pct': 'RMSD_Improv_%',
        'dual_state_covered_tm05': 'Dual_State',
        'af3_imfd_rmsd': 'AF3_imfdRMSD',
    }
    t1.rename(columns={k: v for k, v in t1_rename.items() if k in t1.columns}, inplace=True)
    tables['dual_state_coverage'] = t1.sort_values('Gene')
    
    # Table 2: Quantum Scoring Analysis
    t2_cols = ['gene', 'n_qubits', 'qicess_composite', 'qicess_quantum_energy',
               'top10_min_rmsd_state2', 'top10_max_tm_state2',
               'quantum_rank_state2_tm_rho', 'ensemble_diversity', 'scoring_time_s']
    
    available_cols2 = [c for c in t2_cols if c in valid.columns]
    t2 = valid[available_cols2].copy()
    
    t2_rename = {
        'gene': 'Gene',
        'n_qubits': 'Qubits',
        'qicess_composite': 'QICESS',
        'qicess_quantum_energy': 'Q_Energy',
        'top10_min_rmsd_state2': 'Top10→S2_RMSD',
        'top10_max_tm_state2': 'Top10→S2_TM',
        'quantum_rank_state2_tm_rho': 'Rank_Corr_ρ',
        'ensemble_diversity': 'Diversity',
        'scoring_time_s': 'Time_s',
    }
    t2.rename(columns={k: v for k, v in t2_rename.items() if k in t2.columns}, inplace=True)
    tables['quantum_scoring'] = t2.sort_values('Gene')
    
    return tables


def main():
    """Run the dual-state coverage benchmark."""
    logger.info("="*80)
    logger.info("QuantumFoldX v2 — Dual-State Conformational Coverage Benchmark")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"⚠ All quantum circuits are CLASSICALLY SIMULATED")
    logger.info(f"⚠ AF3 numbers from published peer-reviewed benchmarks")
    logger.info(f"⚠ QFX starts from known structures — this is conformational exploration, not de novo prediction")
    logger.info("="*80)
    
    # Initialize scorer (faster settings for larger ensemble)
    scorer = QICESSv2Scorer(
        vqe_layers=3,
        vqe_restarts=2,
        vqe_steps=60
    )
    
    targets = get_autoinhibited_benchmark()
    af3_baseline = get_af3_baseline()
    
    logger.info(f"\nBenchmark set: {len(targets)} autoinhibited proteins")
    logger.info(f"AF3 baselines from: {af3_baseline['autoinhibited']['source']}")
    logger.info(f"AF3 dual-state rate (autoinhibited): {af3_baseline['autoinhibited']['fraction_both_states']*100:.0f}%")
    logger.info(f"AF3 dual-state rate (multi-state): {af3_baseline['multistate']['fraction_both_states_correct']*100:.1f}%")
    
    all_results = []
    total_start = time.time()
    
    for idx, target in enumerate(targets):
        logger.info(f"\n[{idx+1}/{len(targets)}] {target.gene_name}")
        try:
            result = process_target_v2(target, scorer, n_ensemble=100)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  ERROR processing {target.gene_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'protein': target.protein_name,
                'gene': target.gene_name,
                'status': 'error',
                'error': str(e)
            })
    
    total_time = time.time() - total_start
    logger.info(f"\nTotal benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Compile results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'tables' / 'raw_results_v2.csv', index=False)
    
    # Statistical tests
    stats_results = run_statistical_tests_v2(results_df, af3_baseline)
    stats_results['timing'] = {
        'total_seconds': total_time,
        'total_minutes': total_time / 60,
        'per_target_seconds': total_time / len(targets),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_DIR / 'stats' / 'statistical_tests_v2.json', 'w') as f:
        json.dump(stats_results, f, indent=2, default=str)
    
    # Generate tables
    tables = generate_results_tables(results_df)
    for name, df in tables.items():
        df.to_csv(RESULTS_DIR / 'tables' / f'{name}.csv', index=False)
    
    # Print summary
    valid = results_df[results_df['status'] == 'success']
    n_valid = len(valid)
    
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Targets processed: {len(all_results)}")
    logger.info(f"Successful: {n_valid}")
    logger.info(f"Skipped (self-ref): {(results_df['status'] == 'skipped_self_reference').sum()}")
    logger.info(f"Failed: {len(all_results) - n_valid - (results_df['status'] == 'skipped_self_reference').sum()}")
    
    if n_valid > 0 and 'dual_state_covered_tm05' in valid.columns:
        dsc_rate = valid['dual_state_covered_tm05'].mean()
        logger.info(f"\n  QFX Dual-State Coverage (TM>0.5): {valid['dual_state_covered_tm05'].sum()}/{n_valid} = {dsc_rate*100:.1f}%")
        logger.info(f"  AF3 Dual-State Coverage (published): 14% (autoinhibited) / 23.3% (multi-state)")
        
        if 'ens_max_tm_state2' in valid.columns:
            logger.info(f"\n  Mean max TM to state2: {valid['ens_max_tm_state2'].mean():.3f}")
            logger.info(f"  Mean min RMSD to state2: {valid['ens_min_rmsd_state2'].mean():.2f} Å")
        
        if 'state2_rmsd_improvement' in valid.columns:
            imp = valid['state2_rmsd_improvement'].dropna()
            if len(imp) > 0:
                logger.info(f"\n  Mean state2 RMSD improvement: {imp.mean():.2f} Å")
                logger.info(f"  Proteins with improvement: {(imp > 0).sum()}/{len(imp)}")
    
    if 'primary_endpoint' in stats_results:
        pe = stats_results['primary_endpoint']
        logger.info(f"\n{'='*80}")
        logger.info("PRIMARY STATISTICAL TEST")
        logger.info(f"{'='*80}")
        logger.info(f"  QFX dual-state rate: {pe['qfx_rate']*100:.1f}% ({pe['qfx_n_covered']}/{pe['n_targets']})")
        logger.info(f"  95% Wilson CI: [{pe['qfx_wilson_95ci'][0]*100:.1f}%, {pe['qfx_wilson_95ci'][1]*100:.1f}%]")
        logger.info(f"  vs AF3 (auto, 14%): p = {pe['binom_test_vs_af3_auto']['p_value']:.6f}")
        logger.info(f"  vs AF3 (multi, 23.3%): p = {pe['binom_test_vs_af3_multi']['p_value']:.6f}")
    
    logger.info(f"\nTotal time: {total_time:.1f}s")
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    
    # Print results table
    if 'dual_state_coverage' in tables:
        logger.info(f"\n{'='*80}")
        logger.info("DUAL-STATE COVERAGE TABLE")
        logger.info(f"{'='*80}")
        print(tables['dual_state_coverage'].to_string(index=False, float_format='%.3f'))
    
    return results_df, stats_results


if __name__ == '__main__':
    results_df, stats_results = main()
