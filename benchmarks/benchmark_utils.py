"""
benchmark_utils.py — Shared utilities for QuantumFoldX benchmark pipelines.
"""

import time
import logging
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple

from src.data.pdb_fetcher import (
    fetch_pdb, parse_pdb_ca_coords, parse_pdb_ca_coords_best_chain, compute_phi_psi,
)
from src.scoring.qicess_v3 import create_dsib_scorer, QICESSv3Scorer
from src.scoring.geometry_utils import transition_difficulty
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.metrics.structural_metrics import rmsd, tm_score, gdt_ts

logger = logging.getLogger(__name__)


def find_common_residues(struct1: Dict, struct2: Dict) -> Tuple[List[int], List[int], int]:
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


def parse_target_structures(target) -> Tuple[Optional[Dict], Optional[Dict], str]:
    """Fetch and parse both states for a benchmark target."""
    pdb1 = fetch_pdb(target.pdb_id_state1)
    pdb2 = fetch_pdb(target.pdb_id_state2)
    if not pdb1 or not pdb2:
        return None, None, 'fetch_failed'

    rr1 = getattr(target, 'res_range_state1', None)
    rr2 = getattr(target, 'res_range_state2', None)
    m1 = getattr(target, 'model_state1', 1)
    m2 = getattr(target, 'model_state2', 1)

    s1 = parse_pdb_ca_coords_best_chain(
        pdb1, preferred_chain=target.chain_state1, res_range=rr1, model=m1)
    s2 = parse_pdb_ca_coords_best_chain(
        pdb2, preferred_chain=target.chain_state2, res_range=rr2, model=m2)

    if s1 is None or s2 is None:
        return None, None, 'parse_failed'
    return s1, s2, 'ok'


def get_domain_indices(struct: Dict, target) -> Tuple[List[int], List[int]]:
    """Map UniProt residue ranges to structure indices."""
    fd_start, fd_end = target.fd_residues
    im_start, im_end = target.im_residues
    fd_idx = [i for i, r in enumerate(struct['residue_ids']) if fd_start <= r <= fd_end]
    im_idx = [i for i, r in enumerate(struct['residue_ids']) if im_start <= r <= im_end]
    if not fd_idx or not im_idx:
        n = struct['n_residues']
        fd_idx = list(range(n // 2, n))
        im_idx = list(range(0, n // 2))
    return fd_idx, im_idx


def evaluate_ensemble_vs_state(ensemble_coords_list, target_coords,
                                common_idx_ens, common_idx_target) -> Dict:
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
        except Exception:
            continue

    return {
        'min_rmsd': best_rmsd if best_rmsd < float('inf') else None,
        'max_tm': best_tm,
        'max_gdt': best_gdt,
        'best_idx': best_idx,
        'all_tms': all_tms,
        'median_rmsd': float(np.median(all_rmsds)) if all_rmsds else None,
    }


def process_single_target(target, scorer: QICESSv3Scorer = None,
                          max_residues: int = 1000,
                          n_ens_small: int = 80,
                          n_ens_large: int = 50) -> Dict:
    """Process one protein through the full dual-state coverage pipeline."""
    if scorer is None:
        scorer = create_dsib_scorer()
    result = {
        'protein': target.protein_name, 'gene': target.gene_name,
        'pdb_state1': target.pdb_id_state1, 'pdb_state2': target.pdb_id_state2,
        'uniprot': target.uniprot_id, 'category': target.category,
        'af3_imfd_rmsd': target.af3_imfd_rmsd,
        'status': 'pending'
    }

    if target.pdb_id_state1 == target.pdb_id_state2:
        result['status'] = 'skipped_self_reference'
        return result

    s1, s2, parse_status = parse_target_structures(target)
    if parse_status != 'ok':
        result['status'] = parse_status
        return result

    if s1['n_residues'] > max_residues:
        result['status'] = 'skipped_too_large'
        result['n_residues_state1'] = s1['n_residues']
        return result

    result['n_residues_state1'] = s1['n_residues']
    result['n_residues_state2'] = s2['n_residues']

    ci1, ci2, nc = find_common_residues(s1, s2)
    result['n_common_residues'] = nc
    if nc < 20:
        result['status'] = 'insufficient_overlap'
        return result

    c1c = s1['coords'][ci1[:nc]]
    c2c = s2['coords'][ci2[:nc]]
    baseline_rmsd_val = rmsd(c1c, c2c)
    baseline_tm_val = tm_score(c1c, c2c)
    result['state1_vs_state2_rmsd'] = baseline_rmsd_val
    result['state1_vs_state2_tm'] = baseline_tm_val

    logger.info(f"  S1↔S2: RMSD={baseline_rmsd_val:.2f}Å TM={baseline_tm_val:.3f} (n_common={nc})")

    fd_idx, im_idx = get_domain_indices(s1, target)
    result['n_fd_residues'] = len(fd_idx)
    result['n_im_residues'] = len(im_idx)

    n_ens = n_ens_small if s1['n_residues'] < 400 else n_ens_large

    t_start = time.time()
    phi_psi = compute_phi_psi(s1['pdb_path'], chain=s1['chain'])

    use_v3 = hasattr(scorer, 'build_bridge')
    difficulty = transition_difficulty(baseline_tm_val)

    if use_v3:
        bridge = scorer.build_bridge(
            s1['sequence'], s1['coords'], s2['coords'], fd_idx, im_idx
        )
        ensemble = generate_hybrid_ensemble(
            s1['coords'], s1['sequence'],
            fd_indices=fd_idx, im_indices=im_idx,
            n_conformations=n_ens, seed=42, phi_psi=phi_psi,
            coords_s2=s2['coords'], quantum_bridge=bridge,
            transition_difficulty=difficulty,
            common_idx_s1=ci1, common_idx_s2=ci2,
        )
    else:
        bridge = None
        ensemble = generate_hybrid_ensemble(
            s1['coords'], s1['sequence'],
            fd_indices=fd_idx, im_indices=im_idx,
            n_conformations=n_ens, seed=42, phi_psi=phi_psi,
        )
    for conf in ensemble:
        conf['phi_psi'] = phi_psi

    result['ensemble_size'] = len(ensemble)
    if bridge is not None:
        result['n_quantum_bridge'] = sum(1 for c in ensemble if c['method'] == 'quantum_bridge')
        result['n_switch_contacts'] = len(bridge.switch_contacts)

    result['transition_difficulty'] = difficulty
    result['difficulty_tier'] = (
        'easy' if baseline_tm_val > 0.5 else
        'medium' if baseline_tm_val > 0.3 else 'hard'
    )

    if use_v3:
        scored = scorer.rank_ensemble(
            ensemble, s1['sequence'],
            reference_coords=s1['coords'],
            state2_coords=s2['coords'],
            fd_indices=fd_idx, im_indices=im_idx,
            common_idx_ens=ci1, common_idx_s2=ci2,
        )
    else:
        scored = scorer.rank_ensemble(
            ensemble, s1['sequence'],
            reference_coords=s1['coords'],
            fd_indices=fd_idx, im_indices=im_idx,
        )

    t_total = time.time() - t_start
    result['scoring_time_s'] = t_total

    if not scored:
        result['status'] = 'scoring_failed'
        return result

    all_coords = [c['coords'] for c in scored]

    eval_s1 = evaluate_ensemble_vs_state(
        all_coords, s1['coords'],
        list(range(min(len(all_coords[0]), s1['n_residues']))),
        list(range(s1['n_residues']))
    )
    eval_s2 = evaluate_ensemble_vs_state(all_coords, s2['coords'], ci1, ci2)

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

    best = scored[0]
    result['qicess_composite'] = best['composite']
    if use_v3:
        result['qicess_manifold_overlap'] = best.get('manifold_overlap', 0.0)
        result['qicess_state2_target'] = best.get('state2_target', 0.0)
        result['qicess_switch_satisfaction'] = best.get('switch_satisfaction', 0.0)
        result['qicess_state2_geometry'] = best.get('state2_geometry', 0.0)
        result['qicess_state2_imfd'] = best.get('state2_imfd', 0.0)
    else:
        result['qicess_quantum_energy'] = best.get('quantum_energy_raw', 0.0)
        result['qicess_qaoa_score'] = best.get('qaoa_rotamer', 0.0)
    result['n_qubits'] = best.get('n_qubits', 0)

    rmsds_inner = []
    for i in range(min(10, len(scored))):
        for j in range(i + 1, min(10, len(scored))):
            n_c = min(len(scored[i]['coords']), len(scored[j]['coords']))
            if n_c > 10:
                rmsds_inner.append(rmsd(scored[i]['coords'][:n_c], scored[j]['coords'][:n_c]))
    result['ensemble_diversity'] = float(np.mean(rmsds_inner)) if rmsds_inner else 0.0

    if eval_s2['min_rmsd'] and baseline_rmsd_val > 0:
        result['state2_rmsd_improvement'] = baseline_rmsd_val - eval_s2['min_rmsd']
        result['state2_rmsd_improvement_pct'] = (
            (baseline_rmsd_val - eval_s2['min_rmsd']) / baseline_rmsd_val * 100
        )

    tm_thresh = 0.5
    s1_cov = eval_s1['max_tm'] > tm_thresh if eval_s1['max_tm'] else False
    s2_cov = eval_s2['max_tm'] > tm_thresh if eval_s2['max_tm'] else False
    result['state1_covered_tm05'] = s1_cov
    result['state2_covered_tm05'] = s2_cov
    result['dual_state_covered_tm05'] = s1_cov and s2_cov

    if eval_s2['all_tms'] and len(eval_s2['all_tms']) > 5:
        ranks = list(range(len(eval_s2['all_tms'])))
        rho, p_val = scipy_stats.spearmanr(ranks, eval_s2['all_tms'])
        result['quantum_rank_corr_rho'] = float(rho)
        result['quantum_rank_corr_p'] = float(p_val)

    result['status'] = 'success'
    logger.info(f"  Ens→S2: minRMSD={eval_s2['min_rmsd']:.2f}Å maxTM={eval_s2['max_tm']:.3f}")
    logger.info(f"  Dual-state(TM>0.5): {result['dual_state_covered_tm05']} | Time: {t_total:.1f}s")

    return result


def run_dual_state_stats(df, af3_base: Dict, category: str = 'autoinhibited') -> Dict:
    """Statistical analysis for dual-state coverage benchmark."""
    valid = df[df['status'] == 'success'].copy()
    n = len(valid)
    stats = {'n_valid': n, 'category': category}

    if n < 3:
        stats['warning'] = 'insufficient data'
        return stats

    dsc = int(valid['dual_state_covered_tm05'].sum())
    rate = dsc / n

    if category == 'foldswitch':
        af3_rate = af3_base['foldswitch']['success_rate']
    elif category == 'multistate':
        af3_rate = af3_base['multistate']['fraction_both_states_correct']
    else:
        af3_rate = af3_base['autoinhibited']['fraction_both_states']

    af3_multi = af3_base['multistate']['fraction_both_states_correct']

    p_auto = float(scipy_stats.binomtest(dsc, n, af3_rate, alternative='greater').pvalue)
    p_multi = float(scipy_stats.binomtest(dsc, n, af3_multi, alternative='greater').pvalue)

    z = 1.96
    p_hat = rate
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom

    stats['primary'] = {
        'metric': 'Dual-State Coverage (TM>0.5)',
        'n': n, 'n_covered': dsc, 'rate': rate,
        'wilson_95ci': [max(0, center - margin), min(1, center + margin)],
        'af3_category_rate': af3_rate,
        'af3_multi_rate': af3_multi,
        'p_vs_af3_category': p_auto,
        'p_vs_af3_multi': p_multi,
        'sig_vs_category': p_auto < 0.05,
        'sig_vs_multi': p_multi < 0.05,
    }

    imp = valid['state2_rmsd_improvement'].dropna().values
    if len(imp) >= 3:
        try:
            _, p = scipy_stats.wilcoxon(imp, alternative='greater')
        except Exception:
            p = 1.0
        stats['improvement'] = {
            'mean': float(np.mean(imp)),
            'frac_improved': float(np.mean(imp > 0)),
            'wilcoxon_p': float(p),
            'sig': p < 0.05,
        }

    tms = valid['ens_max_tm_state2'].dropna().values
    if len(tms) > 0:
        stats['state2_tm'] = {
            'mean': float(np.mean(tms)),
            'median': float(np.median(tms)),
            'frac_above_05': float(np.mean(tms > 0.5)),
        }

    rhos = valid['quantum_rank_corr_rho'].dropna().values if 'quantum_rank_corr_rho' in valid.columns else []
    if len(rhos) > 0:
        stats['quantum_ranking'] = {
            'mean_rho': float(np.mean(rhos)),
            'n_negative': int(np.sum(rhos < 0)),
        }

    # Stratified by transition difficulty (Papageorgiou et al. tiers)
    if 'difficulty_tier' in valid.columns:
        strata = {}
        for tier in ('easy', 'medium', 'hard'):
            sub = valid[valid['difficulty_tier'] == tier]
            if len(sub) == 0:
                continue
            strata[tier] = {
                'n': int(len(sub)),
                'dual_state_rate': float(sub['dual_state_covered_tm05'].mean()),
                'mean_state2_tm': float(sub['ens_max_tm_state2'].mean()),
                'n_covered': int(sub['dual_state_covered_tm05'].sum()),
            }
        stats['stratified'] = strata

    return stats
