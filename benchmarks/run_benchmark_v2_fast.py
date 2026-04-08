#!/usr/bin/env python3
"""
run_benchmark_v2_fast.py — Streamlined dual-state coverage benchmark.
Saves results incrementally after each protein to avoid data loss on timeout.
Uses adaptive ensemble size based on protein length.
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

from src.data.pdb_fetcher import fetch_pdb, parse_pdb_ca_coords, compute_contact_map, compute_phi_psi
from src.scoring.qicess_v2 import QICESSv2Scorer
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.metrics.structural_metrics import rmsd, tm_score, gdt_ts, lddt, imfd_rmsd
from configs.benchmark_dataset import get_autoinhibited_benchmark, get_af3_baseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'stats').mkdir(parents=True, exist_ok=True)
INCREMENTAL_FILE = RESULTS_DIR / 'tables' / 'raw_results_v2.csv'


def find_common_residues(struct1, struct2):
    """Find common residues between two structures by residue numbering."""
    set1 = set(struct1['residue_ids'])
    set2 = set(struct2['residue_ids'])
    common = sorted(set1 & set2)
    if not common:
        n = min(struct1['n_residues'], struct2['n_residues'])
        return list(range(n)), list(range(n)), n
    idx1 = [i for i, r in enumerate(struct1['residue_ids']) if r in set2]
    idx2 = [i for i, r in enumerate(struct2['residue_ids']) if r in set1]
    n = min(len(idx1), len(idx2))
    return idx1[:n], idx2[:n], n


def evaluate_ensemble_vs_state(ensemble_coords_list, target_coords, common_idx_ens, common_idx_target):
    """Evaluate ensemble coverage of a target state."""
    best_rmsd = float('inf')
    best_tm = 0.0
    best_gdt = 0.0
    best_idx = -1
    all_rmsds = []
    all_tms = []
    
    for idx, ens_coords in enumerate(ensemble_coords_list):
        n_ens = len(ens_coords)
        valid_idx = [i for i in common_idx_ens if i < n_ens]
        valid_target_idx = common_idx_target[:len(valid_idx)]
        n_c = min(len(valid_idx), len(valid_target_idx))
        if n_c < 10:
            continue
        
        ens_c = ens_coords[valid_idx[:n_c]]
        tgt_c = target_coords[valid_target_idx[:n_c]]
        
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
            if r < best_rmsd + 2.0:
                g = gdt_ts(tgt_c, ens_c)
                if g > best_gdt:
                    best_gdt = g
        except:
            continue
    
    return {
        'min_rmsd': best_rmsd if best_rmsd < float('inf') else None,
        'max_tm': best_tm,
        'max_gdt': best_gdt,
        'best_idx': best_idx,
        'all_tms': all_tms,
        'median_rmsd': float(np.median(all_rmsds)) if all_rmsds else None,
    }


def process_single_target(target, scorer):
    """Process one protein with adaptive parameters."""
    result = {
        'protein': target.protein_name, 'gene': target.gene_name,
        'pdb_state1': target.pdb_id_state1, 'pdb_state2': target.pdb_id_state2,
        'uniprot': target.uniprot_id, 'af3_imfd_rmsd': target.af3_imfd_rmsd,
        'status': 'pending'
    }
    
    if target.pdb_id_state1 == target.pdb_id_state2:
        result['status'] = 'skipped_self_reference'
        return result
    
    # Fetch and parse structures
    pdb1 = fetch_pdb(target.pdb_id_state1)
    pdb2 = fetch_pdb(target.pdb_id_state2)
    if not pdb1 or not pdb2:
        result['status'] = 'fetch_failed'
        return result
    
    s1 = parse_pdb_ca_coords(pdb1, chain=target.chain_state1)
    if s1 is None:
        s1 = parse_pdb_ca_coords(pdb1, chain=None)
    s2 = parse_pdb_ca_coords(pdb2, chain=target.chain_state2)
    if s2 is None:
        s2 = parse_pdb_ca_coords(pdb2, chain=None)
    
    if s1 is None or s2 is None:
        result['status'] = 'parse_failed'
        return result
    
    # Skip very large proteins (>1000 residues) to avoid timeout
    if s1['n_residues'] > 1000:
        result['status'] = 'skipped_too_large'
        result['n_residues_state1'] = s1['n_residues']
        return result
    
    result['n_residues_state1'] = s1['n_residues']
    result['n_residues_state2'] = s2['n_residues']
    
    # Common residues
    ci1, ci2, nc = find_common_residues(s1, s2)
    result['n_common_residues'] = nc
    if nc < 20:
        result['status'] = 'insufficient_overlap'
        return result
    
    # Baseline state1↔state2
    c1c = s1['coords'][ci1[:nc]]
    c2c = s2['coords'][ci2[:nc]]
    baseline_rmsd_val = rmsd(c1c, c2c)
    baseline_tm_val = tm_score(c1c, c2c)
    result['state1_vs_state2_rmsd'] = baseline_rmsd_val
    result['state1_vs_state2_tm'] = baseline_tm_val
    
    logger.info(f"  S1↔S2: RMSD={baseline_rmsd_val:.2f}Å TM={baseline_tm_val:.3f} (n_common={nc})")
    
    # Domain indices
    fd_start, fd_end = target.fd_residues
    im_start, im_end = target.im_residues
    fd_idx = [i for i, r in enumerate(s1['residue_ids']) if fd_start <= r <= fd_end]
    im_idx = [i for i, r in enumerate(s1['residue_ids']) if im_start <= r <= im_end]
    if not fd_idx or not im_idx:
        n = s1['n_residues']
        fd_idx = list(range(n // 2, n))
        im_idx = list(range(0, n // 2))
    result['n_fd_residues'] = len(fd_idx)
    result['n_im_residues'] = len(im_idx)
    
    # Adaptive ensemble size: 80 for normal, 50 for large proteins
    n_ens = 80 if s1['n_residues'] < 400 else 50
    
    t_start = time.time()
    ensemble = generate_hybrid_ensemble(
        s1['coords'], s1['sequence'],
        fd_indices=fd_idx, im_indices=im_idx,
        n_conformations=n_ens, seed=42
    )
    result['ensemble_size'] = len(ensemble)
    
    # Phi/psi
    phi_psi = compute_phi_psi(pdb1, chain=s1['chain'])
    for conf in ensemble:
        conf['phi_psi'] = phi_psi
    
    # QICESS scoring
    scored = scorer.rank_ensemble(
        ensemble, s1['sequence'],
        reference_coords=s1['coords'],
        fd_indices=fd_idx, im_indices=im_idx
    )
    
    t_total = time.time() - t_start
    result['scoring_time_s'] = t_total
    
    if not scored:
        result['status'] = 'scoring_failed'
        return result
    
    all_coords = [c['coords'] for c in scored]
    
    # Evaluate vs state 1
    eval_s1 = evaluate_ensemble_vs_state(
        all_coords, s1['coords'],
        list(range(min(len(all_coords[0]), s1['n_residues']))),
        list(range(s1['n_residues']))
    )
    
    # Evaluate vs state 2
    eval_s2 = evaluate_ensemble_vs_state(all_coords, s2['coords'], ci1, ci2)
    
    # Top-10 quantum-ranked
    top10 = [c['coords'] for c in scored[:10]]
    eval_s2_t10 = evaluate_ensemble_vs_state(top10, s2['coords'], ci1, ci2)
    
    result['ens_min_rmsd_state1'] = eval_s1['min_rmsd']
    result['ens_max_tm_state1'] = eval_s1['max_tm']
    result['ens_min_rmsd_state2'] = eval_s2['min_rmsd']
    result['ens_max_tm_state2'] = eval_s2['max_tm']
    result['ens_max_gdt_state2'] = eval_s2['max_gdt']
    result['ens_median_rmsd_state2'] = eval_s2['median_rmsd']
    result['top10_min_rmsd_state2'] = eval_s2_t10['min_rmsd']
    result['top10_max_tm_state2'] = eval_s2_t10['max_tm']
    
    # QICESS scores
    best = scored[0]
    result['qicess_composite'] = best['composite']
    result['qicess_quantum_energy'] = best.get('quantum_energy_raw', 0.0)
    result['n_qubits'] = best.get('n_qubits', 0)
    
    # Ensemble diversity
    rmsds_inner = []
    for i in range(min(10, len(scored))):
        for j in range(i+1, min(10, len(scored))):
            n_c = min(len(scored[i]['coords']), len(scored[j]['coords']))
            if n_c > 10:
                rmsds_inner.append(rmsd(scored[i]['coords'][:n_c], scored[j]['coords'][:n_c]))
    result['ensemble_diversity'] = float(np.mean(rmsds_inner)) if rmsds_inner else 0.0
    
    # State 2 improvement
    if eval_s2['min_rmsd'] and baseline_rmsd_val > 0:
        result['state2_rmsd_improvement'] = baseline_rmsd_val - eval_s2['min_rmsd']
        result['state2_rmsd_improvement_pct'] = (baseline_rmsd_val - eval_s2['min_rmsd']) / baseline_rmsd_val * 100
    
    # Dual-state coverage
    tm_thresh = 0.5
    s1_cov = eval_s1['max_tm'] > tm_thresh if eval_s1['max_tm'] else False
    s2_cov = eval_s2['max_tm'] > tm_thresh if eval_s2['max_tm'] else False
    result['state1_covered_tm05'] = s1_cov
    result['state2_covered_tm05'] = s2_cov
    result['dual_state_covered_tm05'] = s1_cov and s2_cov
    
    # Quantum rank correlation with state2 TM
    if eval_s2['all_tms'] and len(eval_s2['all_tms']) > 5:
        ranks = list(range(len(eval_s2['all_tms'])))
        rho, p_val = scipy_stats.spearmanr(ranks, eval_s2['all_tms'])
        result['quantum_rank_corr_rho'] = float(rho)
        result['quantum_rank_corr_p'] = float(p_val)
    
    result['status'] = 'success'
    
    logger.info(f"  Ens→S2: minRMSD={eval_s2['min_rmsd']:.2f}Å maxTM={eval_s2['max_tm']:.3f}")
    logger.info(f"  Improvement: {result.get('state2_rmsd_improvement', 0):.2f}Å ({result.get('state2_rmsd_improvement_pct', 0):.1f}%)")
    logger.info(f"  Dual-state(TM>0.5): {result['dual_state_covered_tm05']} | Time: {t_total:.1f}s")
    
    return result


def run_stats(df, af3_base):
    """Statistical analysis."""
    v = df[df['status'] == 'success'].copy()
    n = len(v)
    stats = {'n_valid': n}
    
    if n < 3:
        stats['warning'] = 'insufficient data'
        return stats
    
    # Primary: Dual-state coverage rate
    dsc = int(v['dual_state_covered_tm05'].sum())
    rate = dsc / n
    af3_auto = af3_base['autoinhibited']['fraction_both_states']
    af3_multi = af3_base['multistate']['fraction_both_states_correct']
    
    # Binomial tests
    p_auto = float(scipy_stats.binom_test(dsc, n, af3_auto, alternative='greater'))
    p_multi = float(scipy_stats.binom_test(dsc, n, af3_multi, alternative='greater'))
    
    # Wilson CI
    z = 1.96
    p_hat = rate
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
    
    stats['primary'] = {
        'metric': 'Dual-State Coverage (TM>0.5)',
        'n': n, 'n_covered': dsc, 'rate': rate,
        'wilson_95ci': [max(0, center-margin), min(1, center+margin)],
        'af3_auto_rate': af3_auto, 'af3_multi_rate': af3_multi,
        'p_vs_af3_auto': p_auto, 'p_vs_af3_multi': p_multi,
        'sig_vs_auto': p_auto < 0.05, 'sig_vs_multi': p_multi < 0.05,
    }
    
    # State 2 TM scores
    tms = v['ens_max_tm_state2'].dropna().values
    if len(tms) > 0:
        stats['state2_tm'] = {
            'mean': float(np.mean(tms)), 'median': float(np.median(tms)),
            'std': float(np.std(tms)),
            'frac_above_05': float(np.mean(tms > 0.5)),
            'frac_above_07': float(np.mean(tms > 0.7)),
        }
    
    # State 2 RMSD improvement
    imp = v['state2_rmsd_improvement'].dropna().values
    if len(imp) >= 3:
        try:
            stat, p = scipy_stats.wilcoxon(imp, alternative='greater')
        except:
            stat, p = 0, 1
        stats['improvement'] = {
            'mean': float(np.mean(imp)), 'median': float(np.median(imp)),
            'frac_improved': float(np.mean(imp > 0)),
            'wilcoxon_p': float(p), 'sig': p < 0.05,
        }
    
    # Quantum scoring quality
    rhos = v['quantum_rank_corr_rho'].dropna().values
    if len(rhos) > 0:
        stats['quantum_ranking'] = {
            'mean_rho': float(np.mean(rhos)),
            'n_negative': int(np.sum(rhos < 0)),
            'note': 'Negative rho = quantum ranking favors state2-like conformations'
        }
    
    return stats


def main():
    logger.info("="*80)
    logger.info("QuantumFoldX v2 — Dual-State Coverage Benchmark (Fast)")
    logger.info("="*80)
    
    scorer = QICESSv2Scorer(vqe_layers=3, vqe_restarts=2, vqe_steps=50)
    targets = get_autoinhibited_benchmark()
    af3_base = get_af3_baseline()
    
    all_results = []
    
    # Check for existing incremental results
    if INCREMENTAL_FILE.exists():
        existing = pd.read_csv(INCREMENTAL_FILE)
        done_genes = set(existing['gene'].tolist())
        all_results = existing.to_dict('records')
        logger.info(f"Resuming: {len(done_genes)} proteins already done")
    else:
        done_genes = set()
    
    total_start = time.time()
    
    for idx, target in enumerate(targets):
        if target.gene_name in done_genes:
            logger.info(f"[{idx+1}/{len(targets)}] {target.gene_name} — ALREADY DONE")
            continue
        
        logger.info(f"\n[{idx+1}/{len(targets)}] {target.gene_name}")
        try:
            result = process_single_target(target, scorer)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_results.append({
                'protein': target.protein_name, 'gene': target.gene_name,
                'status': 'error', 'error': str(e)
            })
        
        # INCREMENTAL SAVE after each protein
        df = pd.DataFrame(all_results)
        df.to_csv(INCREMENTAL_FILE, index=False)
    
    total_time = time.time() - total_start
    
    # Final analysis
    df = pd.DataFrame(all_results)
    stats = run_stats(df, af3_base)
    stats['timing'] = {'total_s': total_time, 'timestamp': datetime.now().isoformat()}
    
    with open(RESULTS_DIR / 'stats' / 'statistical_tests_v2.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Summary table
    valid = df[df['status'] == 'success']
    if len(valid) > 0:
        cols = ['gene', 'n_residues_state1', 'n_common_residues',
                'state1_vs_state2_rmsd', 'state1_vs_state2_tm',
                'ens_min_rmsd_state2', 'ens_max_tm_state2',
                'state2_rmsd_improvement_pct', 'dual_state_covered_tm05',
                'af3_imfd_rmsd', 'scoring_time_s']
        avail = [c for c in cols if c in valid.columns]
        summary = valid[avail].sort_values('gene')
        summary.to_csv(RESULTS_DIR / 'tables' / 'dual_state_coverage.csv', index=False)
    
    # Print summary
    n_valid = len(valid)
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK COMPLETE — {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {n_valid} | Skipped: {(df['status'].str.startswith('skipped')).sum()} | Failed: {(df['status'] == 'error').sum()}")
    
    if 'primary' in stats:
        p = stats['primary']
        logger.info(f"\nPRIMARY: Dual-State Coverage (TM>0.5)")
        logger.info(f"  QFX: {p['n_covered']}/{p['n']} = {p['rate']*100:.1f}%")
        logger.info(f"  95% CI: [{p['wilson_95ci'][0]*100:.1f}%, {p['wilson_95ci'][1]*100:.1f}%]")
        logger.info(f"  AF3 (auto): {p['af3_auto_rate']*100:.0f}% | p={p['p_vs_af3_auto']:.4f} sig={p['sig_vs_auto']}")
        logger.info(f"  AF3 (multi): {p['af3_multi_rate']*100:.1f}% | p={p['p_vs_af3_multi']:.4f} sig={p['sig_vs_multi']}")
    
    if 'state2_tm' in stats:
        t = stats['state2_tm']
        logger.info(f"\nState 2 TM-scores: mean={t['mean']:.3f} median={t['median']:.3f}")
        logger.info(f"  Above 0.5: {t['frac_above_05']*100:.1f}% | Above 0.7: {t['frac_above_07']*100:.1f}%")
    
    if 'improvement' in stats:
        im = stats['improvement']
        logger.info(f"\nState 2 RMSD Improvement: mean={im['mean']:.2f}Å | {im['frac_improved']*100:.0f}% improved")
        logger.info(f"  Wilcoxon p={im['wilcoxon_p']:.4f} sig={im['sig']}")
    
    return df, stats


if __name__ == '__main__':
    main()
