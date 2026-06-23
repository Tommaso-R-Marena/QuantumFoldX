"""
geometry_utils.py — Residue-aligned geometry scores for dual-state evaluation.

Structures in different PDB entries often use different chain lengths and
residue numbering. Scoring must map common residues before comparing to
state 2, especially for imfdRMSD (Papageorgiou et al. 2025).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..metrics.structural_metrics import imfd_rmsd, tm_score


def map_common_residue_indices(
    struct1: Dict,
    struct2: Dict,
) -> Tuple[List[int], List[int], int]:
    """Map struct1 indices to struct2 indices by shared residue number."""
    id_to_idx2 = {r: i for i, r in enumerate(struct2.get('residue_ids', []))}
    idx1, idx2 = [], []
    for i, r in enumerate(struct1.get('residue_ids', [])):
        if r in id_to_idx2:
            idx1.append(i)
            idx2.append(id_to_idx2[r])
    n = min(len(idx1), len(idx2))
    return idx1[:n], idx2[:n], n


def align_coords_on_subset(
    mobile: np.ndarray,
    reference: np.ndarray,
    mobile_idx: List[int],
    ref_idx: List[int],
) -> np.ndarray:
    """Kabsch-align full mobile structure using paired residue subsets."""
    pairs = [(m, r) for m, r in zip(mobile_idx, ref_idx)
             if m < len(mobile) and r < len(reference)]
    if len(pairs) < 3:
        return mobile.copy()

    mob_pts = mobile[[p[0] for p in pairs]]
    ref_pts = reference[[p[1] for p in pairs]]
    mob_center = mob_pts.mean(axis=0)
    ref_center = ref_pts.mean(axis=0)
    mob_c = mob_pts - mob_center
    ref_c = ref_pts - ref_center
    H = mob_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    return (mobile - mob_center) @ R.T + ref_center


def state2_imfd_score(
    coords: np.ndarray,
    state2_coords: np.ndarray,
    fd_indices: List[int],
    im_indices: List[int],
    common_idx_ens: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> float:
    """imfdRMSD-based score in [0, 1]: lower imfdRMSD to state 2 → higher score."""
    pairs = _common_pairs(coords, state2_coords, common_idx_ens, common_idx_s2)
    if len(pairs) < 20:
        return 0.0

    c_ens = coords[[p[0] for p in pairs]]
    c_s2 = state2_coords[[p[1] for p in pairs]]
    fd_set, im_set = set(fd_indices), set(im_indices)
    fd = [k for k, (e, _) in enumerate(pairs) if e in fd_set]
    im = [k for k, (e, _) in enumerate(pairs) if e in im_set]

    if len(fd) < 5 or len(im) < 5:
        return 0.0
    try:
        val = imfd_rmsd(c_ens, c_s2, fd, im)
        return 0.0 if np.isnan(val) else float(np.exp(-val / 8.0))
    except Exception:
        return 0.0


def _common_pairs(
    coords: np.ndarray,
    state2_coords: np.ndarray,
    common_idx_ens: Optional[List[int]],
    common_idx_s2: Optional[List[int]],
) -> List[Tuple[int, int]]:
    if common_idx_ens and common_idx_s2:
        return [(e, s) for e, s in zip(common_idx_ens, common_idx_s2)
                if e < len(coords) and s < len(state2_coords)]
    n = min(len(coords), len(state2_coords))
    return [(i, i) for i in range(n)]


def state2_aligned_tm_score(
    coords: np.ndarray,
    state2_coords: np.ndarray,
    fd_indices: Optional[List[int]] = None,
    common_idx_ens: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> float:
    """TM-score to state 2 on common residues, FD-aligned when possible."""
    pairs = _common_pairs(coords, state2_coords, common_idx_ens, common_idx_s2)
    if len(pairs) < 10:
        return 0.0

    if fd_indices:
        fd_ens = [p[0] for p in pairs if p[0] in fd_indices]
        fd_s2 = [p[1] for p in pairs if p[0] in fd_indices]
        if len(fd_ens) >= 3:
            ens_aligned = align_coords_on_subset(coords, state2_coords, fd_ens, fd_s2)
            ens_pts = ens_aligned[[p[0] for p in pairs]]
            s2_pts = state2_coords[[p[1] for p in pairs]]
            try:
                return float(tm_score(s2_pts, ens_pts))
            except Exception:
                pass

    ens_pts = coords[[p[0] for p in pairs]]
    s2_pts = state2_coords[[p[1] for p in pairs]]
    try:
        return float(tm_score(s2_pts, ens_pts))
    except Exception:
        return 0.0


def transition_difficulty(baseline_tm: float) -> float:
    """0 = easy (similar states), 1 = hard (very different states)."""
    return float(np.clip(1.0 - baseline_tm, 0.0, 1.0))


def common_residue_pairs(
    common_idx_s1: Optional[List[int]],
    common_idx_s2: Optional[List[int]],
    n_s1: int,
    n_s2: int,
) -> List[Tuple[int, int]]:
    """Return (s1_idx, s2_idx) pairs for aligned interpolation."""
    if common_idx_s1 and common_idx_s2:
        return [(i, j) for i, j in zip(common_idx_s1, common_idx_s2)
                if i < n_s1 and j < n_s2]
    n = min(n_s1, n_s2)
    return [(i, i) for i in range(n)]


def interpolate_coords_on_common(
    coords_s1: np.ndarray,
    coords_s2: np.ndarray,
    alpha: float,
    common_idx_s1: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Linear S1→S2 blend on paired residues; unpaired residues stay at state 1.

    Critical when state 1 and state 2 PDBs have different chain lengths
    (e.g. WAS: 107 vs 59 residues).
    """
    blended = coords_s1.copy()
    pairs = common_residue_pairs(common_idx_s1, common_idx_s2, len(coords_s1), len(coords_s2))
    for i, j in pairs:
        blended[i] = (1.0 - alpha) * coords_s1[i] + alpha * coords_s2[j]
    return blended
