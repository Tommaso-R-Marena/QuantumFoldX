"""
structural_metrics.py — Standard structural biology metrics for protein comparison.

Implements RMSD, TM-score, GDT-TS, lDDT, and domain-specific metrics
for autoinhibited protein evaluation.

All metrics follow standard definitions used in CASP, CAMEO, and 
published benchmarks (Papageorgiou et al. 2025, Ronish et al. 2024).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def rmsd(coords1: np.ndarray, coords2: np.ndarray, align: bool = True) -> float:
    """
    Compute RMSD between two coordinate sets after optimal superposition.
    
    Uses Kabsch algorithm for alignment when align=True.
    
    Parameters:
        coords1, coords2: np.array of shape (N, 3) — Cα coordinates
        align: if True, perform Kabsch alignment first
    
    Returns:
        RMSD in Å
    """
    assert coords1.shape == coords2.shape, f"Shape mismatch: {coords1.shape} vs {coords2.shape}"
    
    if align:
        coords2_aligned = kabsch_align(coords1, coords2)
    else:
        coords2_aligned = coords2
    
    diff = coords1 - coords2_aligned
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: find optimal rotation to align mobile onto reference.
    
    Returns aligned version of mobile coordinates.
    """
    # Center both structures
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    
    ref_centered = reference - ref_center
    mob_centered = mobile - mob_center
    
    # Covariance matrix
    H = mob_centered.T @ ref_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    
    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T
    
    # Apply rotation and translation
    aligned = (mob_centered @ R.T) + ref_center
    
    return aligned


def tm_score(coords1: np.ndarray, coords2: np.ndarray, 
             L_target: int = None) -> float:
    """
    Compute TM-score between two structures.
    
    TM-score normalizes by target length and is length-independent,
    unlike RMSD. TM > 0.5 indicates same fold; TM > 0.17 is significant.
    
    Follows Zhang & Skolnick (2004) Proteins 57:702-710.
    
    Parameters:
        coords1: reference structure (N, 3)
        coords2: predicted structure (N, 3)
        L_target: target length for normalization (default: len(coords1))
    """
    assert coords1.shape == coords2.shape
    n = len(coords1)
    
    if L_target is None:
        L_target = n
    
    # d0 normalization factor (Zhang & Skolnick formula)
    d0 = 1.24 * (max(L_target, 15) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)  # Floor at 0.5
    
    # Align structures
    coords2_aligned = kabsch_align(coords1, coords2)
    
    # Compute per-residue distances
    distances = np.sqrt(np.sum((coords1 - coords2_aligned) ** 2, axis=1))
    
    # TM-score formula
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    tm = float(np.sum(scores) / L_target)
    
    return min(tm, 1.0)


def gdt_ts(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute GDT-TS (Global Distance Test - Total Score).
    
    GDT-TS = (GDT_1 + GDT_2 + GDT_4 + GDT_8) / 4
    where GDT_d = fraction of residues within d Å after alignment.
    
    Standard CASP metric.
    """
    coords2_aligned = kabsch_align(coords1, coords2)
    distances = np.sqrt(np.sum((coords1 - coords2_aligned) ** 2, axis=1))
    n = len(distances)
    
    gdt_1 = np.sum(distances <= 1.0) / n
    gdt_2 = np.sum(distances <= 2.0) / n
    gdt_4 = np.sum(distances <= 4.0) / n
    gdt_8 = np.sum(distances <= 8.0) / n
    
    return float((gdt_1 + gdt_2 + gdt_4 + gdt_8) / 4.0) * 100.0


def lddt(coords_pred: np.ndarray, coords_true: np.ndarray,
         cutoff: float = 15.0, thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]) -> float:
    """
    Compute lDDT (local Distance Difference Test).
    
    Evaluates local distance preservation without requiring global alignment.
    This makes it more robust for multi-domain proteins where domains
    may be correctly folded but mispositioned relative to each other.
    
    Mariani et al. (2013) Bioinformatics 29:2722-2728.
    """
    n = len(coords_true)
    assert n == len(coords_pred)
    
    # Distance matrices
    D_true = np.sqrt(np.sum((coords_true[:, None, :] - coords_true[None, :, :]) ** 2, axis=-1))
    D_pred = np.sqrt(np.sum((coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=-1))
    
    # Consider only pairs within cutoff in reference
    mask = (D_true < cutoff) & (D_true > 0)  # Exclude self
    
    if not np.any(mask):
        return 0.0
    
    # Compute fraction preserved at each threshold
    diff = np.abs(D_true - D_pred)
    
    total_score = 0.0
    for thresh in thresholds:
        preserved = np.sum((diff < thresh) & mask)
        total = np.sum(mask)
        total_score += preserved / total if total > 0 else 0.0
    
    return float(total_score / len(thresholds)) * 100.0


def imfd_rmsd(coords_pred: np.ndarray, coords_true: np.ndarray,
              fd_indices: List[int], im_indices: List[int]) -> float:
    """
    Inter-module/functional-domain RMSD (imfdRMSD).
    
    Primary metric for autoinhibited proteins (Papageorgiou et al. 2025).
    
    Procedure:
    1. Align on the functional domain (FD) only
    2. Compute RMSD of the inhibitory module (IM) after FD alignment
    
    This measures the accuracy of inter-domain positioning,
    the specific failure mode of AF3 for autoinhibited proteins.
    
    Parameters:
        coords_pred, coords_true: full structure coordinates (N, 3)
        fd_indices: residue indices belonging to functional domain
        im_indices: residue indices belonging to inhibitory module
    """
    # Validate indices
    fd_indices = [i for i in fd_indices if i < len(coords_pred) and i < len(coords_true)]
    im_indices = [i for i in im_indices if i < len(coords_pred) and i < len(coords_true)]
    
    if not fd_indices or not im_indices:
        return float('nan')
    
    # Extract FD coordinates
    fd_pred = coords_pred[fd_indices]
    fd_true = coords_true[fd_indices]
    
    # Compute optimal rotation aligning FD
    fd_ref_center = fd_true.mean(axis=0)
    fd_mob_center = fd_pred.mean(axis=0)
    
    fd_ref_c = fd_true - fd_ref_center
    fd_mob_c = fd_pred - fd_mob_center
    
    H = fd_mob_c.T @ fd_ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    
    # Apply same transformation to full structure
    full_pred_centered = coords_pred - fd_mob_center
    full_pred_aligned = (full_pred_centered @ R.T) + fd_ref_center
    
    # Compute RMSD of IM after FD alignment
    im_pred_aligned = full_pred_aligned[im_indices]
    im_true = coords_true[im_indices]
    
    diff = im_pred_aligned - im_true
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def radius_of_gyration(coords: np.ndarray) -> float:
    """Compute radius of gyration (Rg) in Å."""
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def contact_map_accuracy(pred_contacts: np.ndarray, true_contacts: np.ndarray,
                          top_L: int = None) -> Dict[str, float]:
    """
    Evaluate contact map prediction accuracy.
    
    Returns precision, recall, F1 for top-L/5, top-L/2, top-L contacts.
    """
    n = len(true_contacts)
    if top_L is None:
        top_L = n
    
    # Get predicted contact strengths (upper triangle)
    pred_upper = []
    true_upper = []
    for i in range(n):
        for j in range(i + 3, n):  # Skip short-range
            pred_upper.append((pred_contacts[i, j], i, j))
            true_upper.append(true_contacts[i, j])
    
    pred_upper.sort(reverse=True)
    true_upper_dict = {(i, j): true_contacts[i, j] for i in range(n) for j in range(i + 3, n)}
    
    results = {}
    for label, k in [('L/5', max(1, top_L // 5)), ('L/2', max(1, top_L // 2)), ('L', top_L)]:
        top_k_pred = pred_upper[:k]
        tp = sum(1 for score, i, j in top_k_pred if true_upper_dict.get((i, j), 0) > 0)
        
        precision = tp / k if k > 0 else 0
        total_true = sum(1 for v in true_upper_dict.values() if v > 0)
        recall = tp / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[f'precision_{label}'] = precision
        results[f'recall_{label}'] = recall
        results[f'f1_{label}'] = f1
    
    return results


def dockq_score(coords_pred: np.ndarray, coords_true: np.ndarray,
                interface_residues: List[int], ligand_residues: List[int]) -> float:
    """
    Simplified DockQ score approximation.
    
    DockQ = (fnat + 1/(1+(irms/1.5)^2) + 1/(1+(Lrms/8.5)^2)) / 3
    
    Full DockQ requires all-atom representation; this uses Cα approximation.
    """
    if not interface_residues or not ligand_residues:
        return 0.0
    
    # Interface RMSD (irms): RMSD of interface residues after alignment
    iface_pred = coords_pred[interface_residues]
    iface_true = coords_true[interface_residues]
    irms = rmsd(iface_true, iface_pred, align=True)
    
    # Ligand RMSD (Lrms): RMSD of ligand (IM) after receptor (FD) alignment
    lig_pred = coords_pred[ligand_residues]
    lig_true = coords_true[ligand_residues]
    lrms = rmsd(lig_true, lig_pred, align=True)
    
    # Fraction of native contacts (fnat): approximated from Cα contacts
    true_contacts = np.sqrt(np.sum((coords_true[:, None, :] - coords_true[None, :, :]) ** 2, axis=-1)) < 8.0
    pred_contacts = np.sqrt(np.sum((coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=-1)) < 8.0
    
    native_count = 0
    preserved_count = 0
    for i in interface_residues:
        for j in ligand_residues:
            if i < len(true_contacts) and j < len(true_contacts):
                if true_contacts[i, j]:
                    native_count += 1
                    if pred_contacts[i, j]:
                        preserved_count += 1
    
    fnat = preserved_count / native_count if native_count > 0 else 0.0
    
    # DockQ formula
    dockq = (fnat + 1.0 / (1.0 + (irms / 1.5) ** 2) + 1.0 / (1.0 + (lrms / 8.5) ** 2)) / 3.0
    
    return float(dockq)
