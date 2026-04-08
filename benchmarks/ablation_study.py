#!/usr/bin/env python3
"""
ablation_study.py — Does the quantum scoring layer actually matter?

THE CENTRAL QUESTION:
Does QICESS v2's VQE-derived contact pattern provide any ranking advantage
over purely classical scoring for identifying state-2-like conformations?

METHODS COMPARED:
1. QICESS-VQE:    Full QICESS v2 (35% VQE agreement + 65% classical terms)
2. QICESS-Exact:  Same as above but replace VQE with exact diagonalization
                   (2^16 = 65,536 states — trivial classically)
3. Classical-MJ:  Replace quantum_agreement with raw MJ contact energy sum
4. No-Quantum:    Drop quantum term entirely, renormalize remaining weights
5. Random:        Random ranking (null baseline)

METRIC:
For each protein, rank the ensemble with each scorer.
Then evaluate: which ranking best identifies conformations closest to state 2?

Evaluation metrics:
- Top-10 mean TM-score to state 2 (practical: how good are the top picks?)
- Spearman ρ between rank and TM-score to state 2 (overall ranking quality)
- NDCG@10 (normalized discounted cumulative gain for state-2 proximity)

HYPOTHESIS:
If VQE scoring provides genuine value, QICESS-VQE should outperform
Classical-MJ and No-Quantum on these metrics. If QICESS-Exact matches
QICESS-VQE, the quantum optimization is unnecessary (exact diag suffices).

⚠ HONEST EXPECTATION: At 16 qubits, exact diag is trivially cheap.
The quantum advantage argument would only hold at >50 qubits.
"""

import sys
import os
import time
import json
import logging
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pdb_fetcher import (
    fetch_pdb, parse_pdb_ca_coords, compute_contact_map, compute_phi_psi
)
from src.quantum.ising_vqe import (
    build_ising_hamiltonian, IsingVQESolver,
    MJ_POTENTIALS, MJ_AA_TO_IDX
)
from src.scoring.qicess_v2 import (
    QICESSv2Scorer, ramachandran_score, compactness_score,
    contact_order_score, interdomain_contact_density, vqe_contact_agreement
)
from src.ensemble.conformational_sampler import generate_hybrid_ensemble
from src.metrics.structural_metrics import rmsd, tm_score
from configs.benchmark_dataset import get_autoinhibited_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
(RESULTS_DIR / 'ablation').mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# EXACT DIAGONALIZATION (the 16-qubit elephant in the room)
# ═══════════════════════════════════════════════════════════

def exact_diag_ground_state(H, n_qubits: int, selected_contacts) -> dict:
    """
    Find the EXACT ground state of the Ising Hamiltonian by brute force.
    
    KEY INSIGHT: The Ising Hamiltonian H = Σ J_ij Z_i Z_j + Σ h_i Z_i
    is DIAGONAL in the computational basis. Each Pauli Z operator has
    eigenvalues ±1 depending on the qubit state. So we can compute
    all 2^n energies without building any matrix.
    
    For 16 qubits: 65,536 scalar energy evaluations. Takes milliseconds.
    
    This is the definitive comparison: if VQE finds the same bitstring,
    the quantum circuit adds zero value over classical enumeration.
    """
    n_states = 2 ** n_qubits
    
    # Parse Hamiltonian coefficients
    # H = Σ coeff * (product of PauliZ on specific wires)
    # In computational basis |b⟩: Z_i |b_i⟩ = (-1)^b_i |b_i⟩
    
    # Extract terms from PennyLane Hamiltonian
    coeffs = H.coeffs
    ops = H.ops
    
    # Compute energy for each basis state
    energies = np.zeros(n_states)
    
    for state_idx in range(n_states):
        bits = format(state_idx, f'0{n_qubits}b')
        e = 0.0
        for coeff, op in zip(coeffs, ops):
            coeff_val = float(coeff)
            # Determine which wires this operator acts on
            if isinstance(op, qml.Identity):
                e += coeff_val
            elif isinstance(op, qml.PauliZ):
                wire = op.wires[0]
                sign = (-1) ** int(bits[wire])
                e += coeff_val * sign
            elif isinstance(op, qml.ops.op_math.Prod):
                # Product of PauliZ operators (ZZ term)
                sign = 1
                for sub_op in op.operands:
                    if isinstance(sub_op, qml.PauliZ):
                        wire = sub_op.wires[0]
                        sign *= (-1) ** int(bits[wire])
                e += coeff_val * sign
            else:
                # Try to get wires for composite operators
                try:
                    wires = op.wires
                    sign = 1
                    for w in wires:
                        sign *= (-1) ** int(bits[w])
                    e += coeff_val * sign
                except:
                    e += coeff_val  # Identity-like
        energies[state_idx] = e
    
    # Find ground state
    ground_idx = np.argmin(energies)
    ground_energy = float(energies[ground_idx])
    ground_bitstring = format(ground_idx, f'0{n_qubits}b')
    
    # Top 5 lowest energy states
    top_indices = np.argsort(energies)[:5]
    top_bitstrings = []
    for idx in top_indices:
        bs = format(idx, f'0{n_qubits}b')
        top_bitstrings.append((bs, float(energies[idx])))
    
    return {
        'ground_energy': ground_energy,
        'ground_bitstring': ground_bitstring,
        'top_bitstrings': top_bitstrings,
        'method': 'Exact Enumeration [CLASSICAL]',
        'n_states_evaluated': n_states,
    }


# ═══════════════════════════════════════════════════════════
# CLASSICAL SCORERS
# ═══════════════════════════════════════════════════════════

def classical_mj_energy(coords, sequence, threshold=8.0):
    """
    Sum Miyazawa-Jernigan potentials for all contacts.
    No Ising model, no optimization — just raw statistical potential energy.
    
    This is the classical version of what the Ising Hamiltonian encodes.
    """
    n = min(len(coords), len(sequence))
    contact_map = compute_contact_map(coords[:n], threshold=threshold)
    
    energy = 0.0
    n_contacts = 0
    for i in range(n):
        for j in range(i + 3, n):
            if contact_map[i, j] > 0:
                aa_i = MJ_AA_TO_IDX.get(sequence[i], 0)
                aa_j = MJ_AA_TO_IDX.get(sequence[j], 0)
                energy += MJ_POTENTIALS[aa_i, aa_j]
                n_contacts += 1
    
    # Normalize by protein length
    return energy / max(n, 1), n_contacts


def score_ensemble_classical_mj(ensemble, sequence, fd_indices, im_indices,
                                  phi_psi, ref_rg):
    """Score ensemble using classical MJ energy replacing quantum_agreement."""
    scored = []
    
    # Compute MJ energies for all conformations
    mj_energies = []
    for conf in ensemble:
        e, nc = classical_mj_energy(conf['coords'], sequence)
        mj_energies.append(e)
    
    # Normalize MJ energy to [0, 1] (lower energy = better = higher score)
    e_min = min(mj_energies) if mj_energies else 0
    e_max = max(mj_energies) if mj_energies else 1
    e_range = max(e_max - e_min, 1e-8)
    
    for idx, conf in enumerate(ensemble):
        coords = conf['coords']
        
        # Classical MJ score (normalized, inverted: lower energy = higher score)
        mj_score = 1.0 - (mj_energies[idx] - e_min) / e_range
        
        rama = ramachandran_score(phi_psi) if phi_psi else 0.5
        compact = compactness_score(coords, ref_rg)
        co = contact_order_score(coords, sequence)
        idc = interdomain_contact_density(coords, fd_indices, im_indices) if fd_indices and im_indices else 0.0
        
        # Same weights as QICESS but replace quantum with MJ
        composite = 0.35 * mj_score + 0.10 * rama + 0.15 * compact + 0.10 * co + 0.30 * idc
        
        scored.append({
            **conf,
            'composite': composite,
            'mj_score': mj_score,
            'mj_energy': mj_energies[idx],
            'original_idx': idx,
        })
    
    scored.sort(key=lambda x: x['composite'], reverse=True)
    return scored


def score_ensemble_no_quantum(ensemble, sequence, fd_indices, im_indices,
                                phi_psi, ref_rg):
    """Score ensemble dropping quantum term entirely, renormalizing weights."""
    scored = []
    
    for idx, conf in enumerate(ensemble):
        coords = conf['coords']
        
        rama = ramachandran_score(phi_psi) if phi_psi else 0.5
        compact = compactness_score(coords, ref_rg)
        co = contact_order_score(coords, sequence)
        idc = interdomain_contact_density(coords, fd_indices, im_indices) if fd_indices and im_indices else 0.0
        
        # Renormalized weights (remove 0.35 quantum, rescale rest to sum to 1)
        # Original: rama=0.10, compact=0.15, co=0.10, idc=0.30 → sum=0.65
        composite = (0.10/0.65) * rama + (0.15/0.65) * compact + \
                   (0.10/0.65) * co + (0.30/0.65) * idc
        
        scored.append({
            **conf,
            'composite': composite,
            'original_idx': idx,
        })
    
    scored.sort(key=lambda x: x['composite'], reverse=True)
    return scored


def score_ensemble_random(ensemble, seed=42):
    """Random ranking — null baseline."""
    rng = np.random.default_rng(seed)
    scored = []
    for idx, conf in enumerate(ensemble):
        scored.append({
            **conf,
            'composite': rng.random(),
            'original_idx': idx,
        })
    scored.sort(key=lambda x: x['composite'], reverse=True)
    return scored


def score_ensemble_exact_diag(ensemble, sequence, reference_coords,
                                fd_indices, im_indices, phi_psi, ref_rg):
    """
    Same as QICESS-VQE but use exact diagonalization instead of VQE.
    
    If this matches QICESS-VQE results, the variational quantum circuit
    is unnecessary overhead.
    """
    # Build Ising Hamiltonian
    contact_map = compute_contact_map(reference_coords, threshold=8.0)
    interface_res = (fd_indices, im_indices) if fd_indices and im_indices else None
    
    result = build_ising_hamiltonian(sequence, contact_map, interface_res)
    if result is None or not isinstance(result, tuple) or len(result) < 3 or result[1] == 0:
        # Fallback to no-quantum
        return score_ensemble_no_quantum(ensemble, sequence, fd_indices, im_indices, phi_psi, ref_rg), None
    
    H, n_qubits, selected_contacts = result
    
    # EXACT classical ground state
    t_start = time.time()
    exact = exact_diag_ground_state(H, n_qubits, selected_contacts)
    exact_time = time.time() - t_start
    
    logger.info(f"    Exact diag: {2**n_qubits:,} states in {exact_time*1000:.1f}ms, "
                f"E_ground={exact['ground_energy']:.4f}")
    
    # Score conformations using exact ground state bitstring
    scored = []
    for idx, conf in enumerate(ensemble):
        coords = conf['coords']
        conf_contacts = compute_contact_map(coords, threshold=8.0)
        
        qa = vqe_contact_agreement(conf_contacts, selected_contacts, exact['ground_bitstring'])
        rama = ramachandran_score(phi_psi) if phi_psi else 0.5
        compact = compactness_score(coords, ref_rg)
        co = contact_order_score(coords, sequence)
        idc = interdomain_contact_density(coords, fd_indices, im_indices) if fd_indices and im_indices else 0.0
        
        composite = 0.35 * qa + 0.10 * rama + 0.15 * compact + 0.10 * co + 0.30 * idc
        
        scored.append({
            **conf,
            'composite': composite,
            'quantum_agreement': qa,
            'original_idx': idx,
        })
    
    scored.sort(key=lambda x: x['composite'], reverse=True)
    return scored, exact


# ═══════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════

def evaluate_ranking(scored_ensemble, state2_coords, common_idx_ens, common_idx_target, k=10):
    """
    Evaluate a ranking's quality at identifying state-2-like conformations.
    
    Returns dict with:
    - top_k_mean_tm: Mean TM-score to state 2 of top-K ranked conformations
    - spearman_rho: Rank correlation between ranking position and TM to state 2
    - ndcg_k: NDCG@K for state 2 proximity
    """
    n = len(scored_ensemble)
    nc = min(len(common_idx_ens), len(common_idx_target))
    
    if nc < 10:
        return None
    
    # Compute TM-score to state 2 for each conformation
    tms = []
    for conf in scored_ensemble:
        coords = conf['coords']
        n_c = min(len(coords), len(common_idx_ens))
        valid_ens = [i for i in common_idx_ens[:nc] if i < n_c]
        valid_tgt = common_idx_target[:len(valid_ens)]
        
        if len(valid_ens) < 10:
            tms.append(0.0)
            continue
        
        try:
            t = tm_score(state2_coords[valid_tgt], coords[valid_ens])
            tms.append(t)
        except:
            tms.append(0.0)
    
    if not tms:
        return None
    
    tms = np.array(tms)
    
    # Top-K mean TM
    k_actual = min(k, len(tms))
    top_k_tms = tms[:k_actual]
    
    # Spearman ρ (negative means higher rank = higher TM, which is good)
    ranks = np.arange(len(tms))
    if len(set(tms)) > 1:
        rho, p_rho = scipy_stats.spearmanr(ranks, tms)
    else:
        rho, p_rho = 0.0, 1.0
    
    # NDCG@K
    # Relevance = TM-score to state 2
    ideal_order = np.sort(tms)[::-1][:k_actual]
    dcg = sum(tms[i] / np.log2(i + 2) for i in range(k_actual))
    idcg = sum(ideal_order[i] / np.log2(i + 2) for i in range(k_actual))
    ndcg = dcg / max(idcg, 1e-10)
    
    # Best in top-10 vs best overall
    best_top10_tm = float(np.max(top_k_tms)) if len(top_k_tms) > 0 else 0.0
    best_overall_tm = float(np.max(tms))
    
    return {
        'top_k_mean_tm': float(np.mean(top_k_tms)),
        'top_k_max_tm': best_top10_tm,
        'best_overall_tm': best_overall_tm,
        'spearman_rho': float(rho),
        'spearman_p': float(p_rho),
        'ndcg_k': float(ndcg),
        'n_evaluated': len(tms),
        'mean_tm_all': float(np.mean(tms)),
    }


def find_common_residues(struct1, struct2):
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


# ═══════════════════════════════════════════════════════════
# MAIN ABLATION PIPELINE
# ═══════════════════════════════════════════════════════════

def run_ablation():
    logger.info("="*80)
    logger.info("QUANTUM SCORING ABLATION STUDY")
    logger.info("Does the VQE layer actually contribute to ranking quality?")
    logger.info("="*80)
    
    targets = get_autoinhibited_benchmark()
    scorer_vqe = QICESSv2Scorer(vqe_layers=3, vqe_restarts=2, vqe_steps=50)
    
    methods = ['QICESS-VQE', 'QICESS-Exact', 'Classical-MJ', 'No-Quantum', 'Random']
    all_results = []
    vqe_vs_exact_bitstrings = []
    
    for idx, target in enumerate(targets):
        if target.pdb_id_state1 == target.pdb_id_state2:
            continue
        
        logger.info(f"\n[{idx+1}/{len(targets)}] {target.gene_name}")
        
        # Parse structures
        pdb1 = fetch_pdb(target.pdb_id_state1)
        pdb2 = fetch_pdb(target.pdb_id_state2)
        if not pdb1 or not pdb2:
            continue
        
        s1 = parse_pdb_ca_coords(pdb1, chain=target.chain_state1)
        if s1 is None:
            s1 = parse_pdb_ca_coords(pdb1, chain=None)
        s2 = parse_pdb_ca_coords(pdb2, chain=target.chain_state2)
        if s2 is None:
            s2 = parse_pdb_ca_coords(pdb2, chain=None)
        
        if s1 is None or s2 is None:
            continue
        if s1['n_residues'] > 1000:
            logger.info(f"  Skipping {target.gene_name} (too large: {s1['n_residues']})")
            continue
        
        ci1, ci2, nc = find_common_residues(s1, s2)
        if nc < 20:
            continue
        
        # Domain indices
        fd_start, fd_end = target.fd_residues
        im_start, im_end = target.im_residues
        fd_idx = [i for i, r in enumerate(s1['residue_ids']) if fd_start <= r <= fd_end]
        im_idx = [i for i, r in enumerate(s1['residue_ids']) if im_start <= r <= im_end]
        if not fd_idx or not im_idx:
            n = s1['n_residues']
            fd_idx = list(range(n // 2, n))
            im_idx = list(range(0, n // 2))
        
        # Generate ensemble (same for all scorers)
        n_ens = 80 if s1['n_residues'] < 400 else 50
        ensemble = generate_hybrid_ensemble(
            s1['coords'], s1['sequence'],
            fd_indices=fd_idx, im_indices=im_idx,
            n_conformations=n_ens, seed=42
        )
        
        phi_psi = compute_phi_psi(pdb1, chain=s1['chain'])
        for conf in ensemble:
            conf['phi_psi'] = phi_psi
        
        ref_rg = None  # Let compactness_score use Flory scaling
        
        logger.info(f"  Ensemble: {len(ensemble)} conformations, {s1['n_residues']} residues")
        
        # ─── Score with all methods ───
        rankings = {}
        
        # 1. QICESS-VQE (the quantum method)
        t0 = time.time()
        scored_vqe = scorer_vqe.rank_ensemble(
            ensemble, s1['sequence'],
            reference_coords=s1['coords'],
            fd_indices=fd_idx, im_indices=im_idx
        )
        vqe_time = time.time() - t0
        rankings['QICESS-VQE'] = scored_vqe
        
        # Capture VQE bitstring for comparison
        vqe_bs = None
        cache_key = s1['sequence'][:20] + str(len(s1['sequence']))
        if cache_key in scorer_vqe._vqe_cache:
            vqe_bs = scorer_vqe._vqe_cache[cache_key].get('ground_bitstring')
        
        # 2. QICESS-Exact (same Ising model, exact classical solution)
        t0 = time.time()
        scored_exact, exact_result = score_ensemble_exact_diag(
            ensemble, s1['sequence'], s1['coords'],
            fd_idx, im_idx, phi_psi, ref_rg
        )
        exact_time = time.time() - t0
        rankings['QICESS-Exact'] = scored_exact
        
        # Compare VQE vs Exact bitstrings
        exact_bs = exact_result['ground_bitstring'] if exact_result else None
        if vqe_bs and exact_bs:
            hamming = sum(a != b for a, b in zip(vqe_bs, exact_bs))
            agree = vqe_bs == exact_bs
            vqe_vs_exact_bitstrings.append({
                'gene': target.gene_name,
                'vqe_bitstring': vqe_bs,
                'exact_bitstring': exact_bs,
                'hamming_distance': hamming,
                'identical': agree,
                'vqe_time': vqe_time,
                'exact_time': exact_time,
                'speedup_factor': vqe_time / max(exact_time, 0.001),
            })
            logger.info(f"    VQE vs Exact: Hamming={hamming}, Match={'YES' if agree else 'NO'}")
            logger.info(f"    VQE: {vqe_time:.1f}s | Exact: {exact_time:.1f}s | "
                       f"VQE is {vqe_time/max(exact_time,0.001):.0f}x SLOWER")
        
        # 3. Classical-MJ
        t0 = time.time()
        scored_mj = score_ensemble_classical_mj(
            ensemble, s1['sequence'], fd_idx, im_idx, phi_psi, ref_rg
        )
        mj_time = time.time() - t0
        rankings['Classical-MJ'] = scored_mj
        
        # 4. No-Quantum
        scored_noq = score_ensemble_no_quantum(
            ensemble, s1['sequence'], fd_idx, im_idx, phi_psi, ref_rg
        )
        rankings['No-Quantum'] = scored_noq
        
        # 5. Random
        scored_rand = score_ensemble_random(ensemble, seed=42)
        rankings['Random'] = scored_rand
        
        # ─── Evaluate all rankings ───
        row = {'gene': target.gene_name, 'n_residues': s1['n_residues'],
               'n_ensemble': len(ensemble), 'vqe_time': vqe_time, 'exact_time': exact_time}
        
        for method_name in methods:
            eval_result = evaluate_ranking(
                rankings[method_name], s2['coords'], ci1, ci2, k=10
            )
            if eval_result:
                for metric, value in eval_result.items():
                    row[f'{method_name}_{metric}'] = value
        
        all_results.append(row)
        
        # Log comparison for this protein
        logger.info(f"  Ranking quality (top-10 mean TM to state 2):")
        for m in methods:
            key = f'{m}_top_k_mean_tm'
            if key in row:
                logger.info(f"    {m:20s}: {row[key]:.4f}")
        
        # Incremental save
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'ablation' / 'ablation_raw.csv', index=False)
    
    # ═══════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    df = pd.DataFrame(all_results)
    n = len(df)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION RESULTS ({n} proteins)")
    logger.info(f"{'='*80}")
    
    stats = {'n_proteins': n, 'methods': methods}
    
    # Mean metrics per method
    for m in methods:
        key_tm = f'{m}_top_k_mean_tm'
        key_rho = f'{m}_spearman_rho'
        key_ndcg = f'{m}_ndcg_k'
        
        if key_tm in df.columns:
            vals = df[key_tm].dropna()
            stats[m] = {
                'top10_mean_tm': {'mean': float(vals.mean()), 'std': float(vals.std()), 'n': int(len(vals))},
            }
            if key_rho in df.columns:
                rho_vals = df[key_rho].dropna()
                stats[m]['spearman_rho'] = {'mean': float(rho_vals.mean()), 'std': float(rho_vals.std())}
            if key_ndcg in df.columns:
                ndcg_vals = df[key_ndcg].dropna()
                stats[m]['ndcg_10'] = {'mean': float(ndcg_vals.mean()), 'std': float(ndcg_vals.std())}
    
    # Pairwise comparisons: VQE vs each classical baseline
    paired_tests = {}
    vqe_key = 'QICESS-VQE_top_k_mean_tm'
    
    for m in ['QICESS-Exact', 'Classical-MJ', 'No-Quantum', 'Random']:
        m_key = f'{m}_top_k_mean_tm'
        if vqe_key in df.columns and m_key in df.columns:
            vqe_vals = df[vqe_key].dropna()
            m_vals = df[m_key].dropna()
            
            n_common = min(len(vqe_vals), len(m_vals))
            if n_common >= 3:
                diff = vqe_vals.values[:n_common] - m_vals.values[:n_common]
                
                # Paired Wilcoxon signed-rank
                try:
                    stat, p = scipy_stats.wilcoxon(diff, alternative='greater')
                except:
                    stat, p = 0, 1
                
                # How many proteins does VQE win?
                n_vqe_wins = int(np.sum(diff > 0.001))
                n_ties = int(np.sum(np.abs(diff) <= 0.001))
                n_m_wins = int(np.sum(diff < -0.001))
                
                paired_tests[f'VQE_vs_{m}'] = {
                    'mean_diff': float(np.mean(diff)),
                    'median_diff': float(np.median(diff)),
                    'wilcoxon_p': float(p),
                    'significant': p < 0.05,
                    'vqe_wins': n_vqe_wins,
                    'ties': n_ties,
                    'other_wins': n_m_wins,
                    'effect_size': float(np.mean(diff) / max(np.std(diff), 1e-10)),
                }
    
    stats['paired_tests'] = paired_tests
    
    # VQE vs Exact diagonalization comparison
    bs_df = pd.DataFrame(vqe_vs_exact_bitstrings) if vqe_vs_exact_bitstrings else pd.DataFrame()
    if len(bs_df) > 0:
        stats['vqe_vs_exact'] = {
            'n_proteins': int(len(bs_df)),
            'n_identical': int(bs_df['identical'].sum()),
            'match_rate': float(bs_df['identical'].mean()),
            'mean_hamming': float(bs_df['hamming_distance'].mean()),
            'mean_speedup_factor': float(bs_df['speedup_factor'].mean()),
            'note': f'Exact diag is {bs_df["speedup_factor"].mean():.0f}x faster than VQE on average'
        }
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TOP-10 MEAN TM TO STATE 2 (higher = better ranking)")
    logger.info(f"{'='*80}")
    for m in methods:
        if m in stats and 'top10_mean_tm' in stats[m]:
            s = stats[m]['top10_mean_tm']
            logger.info(f"  {m:20s}: {s['mean']:.4f} ± {s['std']:.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("PAIRED TESTS: VQE vs EACH BASELINE")
    logger.info(f"{'='*80}")
    for name, test in paired_tests.items():
        sig = "✓ SIG" if test['significant'] else "✗ NS"
        logger.info(f"  {name:30s}: Δ={test['mean_diff']:+.4f}, "
                    f"p={test['wilcoxon_p']:.4f} {sig}, "
                    f"W/T/L={test['vqe_wins']}/{test['ties']}/{test['other_wins']}")
    
    if 'vqe_vs_exact' in stats:
        ve = stats['vqe_vs_exact']
        logger.info(f"\n{'='*80}")
        logger.info("VQE vs EXACT DIAGONALIZATION GROUND STATE")
        logger.info(f"{'='*80}")
        logger.info(f"  Bitstring match rate: {ve['match_rate']*100:.1f}% ({ve['n_identical']}/{ve['n_proteins']})")
        logger.info(f"  Mean Hamming distance: {ve['mean_hamming']:.1f}")
        logger.info(f"  ⚠ Exact diag is {ve['mean_speedup_factor']:.0f}x FASTER than VQE")
    
    # Save everything
    with open(RESULTS_DIR / 'ablation' / 'ablation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    if len(bs_df) > 0:
        bs_df.to_csv(RESULTS_DIR / 'ablation' / 'vqe_vs_exact_bitstrings.csv', index=False)
    
    logger.info(f"\nResults saved to {RESULTS_DIR / 'ablation'}")
    
    return df, stats


if __name__ == '__main__':
    df, stats = run_ablation()
