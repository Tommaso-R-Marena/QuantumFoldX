"""
mode_overlap.py — Elastic-network normal-mode analysis of conformational transitions.

Central question: *without knowing state 2*, how much of the observed
state-1 -> state-2 conformational change is captured by the low-frequency
modes of an elastic network built on state 1 alone?

This is the standard framework for testing whether a conformational
transition is "collective" / low-mode (and therefore, in principle,
predictable from a single structure via linear response) versus a
localized or fold-switching rearrangement that soft modes cannot describe.

Metrics (all textbook-standard for ENM vs. conformational change):
  - Per-mode overlap  I_j = |a_j . dr| / (|a_j| |dr|)          (Marques & Sanejouand 1995)
  - Cumulative overlap CO(m) = sqrt( sum_{j<=m} I_j^2 )         (fraction of dr spanned)
  - Collectivity      kappa (Bruschweiler 1995)                (how global a mode is)

The elastic network (anisotropic network model, ANM) is built on the
Calpha atoms of the *shared structural core* of state 1, so the modes and
the displacement vector live in the same 3*N_core space and the mode set
is orthonormal — making cumulative overlap and RMSIP well defined.

References:
  Tirion (1996) PRL 77:1905; Atilgan et al. (2001) Biophys J 80:505;
  Marques & Sanejouand (1995) Proteins 23:557; Tama & Sanejouand (2001)
  Protein Eng 14:1; Bahar et al. (2010) Chem Rev 110:1463.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Rigid-body degrees of freedom (3 translations + 3 rotations) that carry
# ~zero eigenvalue and must be skipped.
N_RIGID_BODY = 6


def kabsch_rotation(mobile: np.ndarray, reference: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal rotation aligning `mobile` onto `reference` (both N x 3).

    Returns (R, mobile_centroid, reference_centroid) such that
    aligned = (mobile - mobile_centroid) @ R.T + reference_centroid.
    """
    mob_c = mobile.mean(axis=0)
    ref_c = reference.mean(axis=0)
    H = (mobile - mob_c).T @ (reference - ref_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mob_c, ref_c


def superpose(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return `mobile` optimally superposed onto `reference`."""
    R, mob_c, ref_c = kabsch_rotation(mobile, reference)
    return (mobile - mob_c) @ R.T + ref_c


def build_anm_hessian(coords: np.ndarray, cutoff: float = 13.0,
                      spring: str = "inverse_square") -> np.ndarray:
    """Anisotropic Network Model Hessian (3N x 3N) for Calpha coordinates.

    A spring connects residue pairs within `cutoff` Angstrom. With
    `spring='inverse_square'` the force constant scales as 1/d^2 (distance-
    weighted ANM, Yang et al. 2009); `spring='uniform'` uses gamma=1.
    """
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    diff = coords[:, None, :] - coords[None, :, :]          # (N, N, 3)
    dist2 = np.sum(diff ** 2, axis=-1)
    mask = (dist2 < cutoff ** 2) & (dist2 > 1e-6)
    pairs = np.argwhere(np.triu(mask, k=1))
    for i, j in pairs:
        d2 = dist2[i, j]
        gamma = 1.0 / d2 if spring == "inverse_square" else 1.0
        # 3x3 super-element = -gamma * (outer product of unit vector) * |dr|^0
        block = gamma * np.outer(diff[i, j], diff[i, j]) / d2
        bi, bj = 3 * i, 3 * j
        H[bi:bi + 3, bj:bj + 3] -= block
        H[bj:bj + 3, bi:bi + 3] -= block
        H[bi:bi + 3, bi:bi + 3] += block
        H[bj:bj + 3, bj:bj + 3] += block
    return H


def compute_anm_modes(coords: np.ndarray, n_modes: int = 20,
                      cutoff: float = 13.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return the softest `n_modes` non-trivial ANM modes for `coords`.

    Returns (eigenvalues, eigenvectors) where eigenvectors[:, k] is mode k
    (shape 3N), sorted by increasing eigenvalue, with the 6 rigid-body
    modes removed.
    """
    H = build_anm_hessian(coords, cutoff=cutoff)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = eigvals[N_RIGID_BODY:]
    eigvecs = eigvecs[:, N_RIGID_BODY:]
    n_keep = min(n_modes, eigvecs.shape[1])
    return eigvals[:n_keep], eigvecs[:, :n_keep]


def collectivity(mode: np.ndarray) -> float:
    """Degree of collectivity kappa of a mode (Bruschweiler 1995), in (0, 1].

    kappa = (1/N) exp(-sum p_i log p_i), p_i = |u_i|^2 / sum|u_j|^2.
    kappa ~ 1 => whole-molecule collective motion; ~1/N => localized.
    """
    disp = mode.reshape(-1, 3)
    sq = np.sum(disp ** 2, axis=1)
    total = sq.sum()
    if total <= 0:
        return 0.0
    p = sq / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    n = len(sq)
    return float(np.exp(entropy) / n)


@dataclass
class ModeOverlapResult:
    """Result of comparing state-1 ANM modes to the observed transition."""
    n_core: int
    transition_magnitude: float          # RMSD of internal displacement (A)
    per_mode_overlap: List[float]         # I_j for softest modes
    cumulative_overlap: List[float]       # CO(m) for m = 1..n_modes
    best_single_overlap: float
    best_mode_index: int                  # 0-based among non-trivial modes
    softest_mode_overlap: float           # I_1 (mode 6+1)
    cum_overlap_2: float
    cum_overlap_5: float
    cum_overlap_10: float
    softest_mode_collectivity: float
    best_mode_collectivity: float
    n_modes_for_half: int                 # modes needed to reach CO >= 0.5


def analyze_transition(
    coords_s1: np.ndarray,
    coords_s2: np.ndarray,
    common_idx_s1: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
    n_modes: int = 20,
    cutoff: float = 13.0,
) -> Optional[ModeOverlapResult]:
    """Compare the softest ANM modes of state 1 to the state-1 -> state-2 change.

    The ANM is built on the shared core of state 1; the observed displacement
    is computed on the same core after optimal rigid-body superposition (so
    only internal conformational change is scored).
    """
    if common_idx_s1 is not None and common_idx_s2 is not None:
        pairs = [(i, j) for i, j in zip(common_idx_s1, common_idx_s2)
                 if i < len(coords_s1) and j < len(coords_s2)]
    else:
        n = min(len(coords_s1), len(coords_s2))
        pairs = [(i, i) for i in range(n)]

    if len(pairs) < 12:
        return None

    core1 = coords_s1[[p[0] for p in pairs]]
    core2 = coords_s2[[p[1] for p in pairs]]

    # Internal displacement: superpose state 1 core onto state 2 core, so the
    # residual is genuine conformational change, not overall translation/rotation.
    core1_aligned = superpose(core1, core2)
    dr = (core2 - core1_aligned).reshape(-1)                # (3N_core,)
    dr_norm = np.linalg.norm(dr)
    n_core = len(pairs)
    rmsd_internal = float(dr_norm / np.sqrt(n_core))
    if dr_norm < 1e-8:
        return None
    dr_hat = dr / dr_norm

    eigvals, eigvecs = compute_anm_modes(core1, n_modes=n_modes, cutoff=cutoff)
    n_avail = eigvecs.shape[1]
    if n_avail == 0:
        return None

    overlaps = []
    for k in range(n_avail):
        a = eigvecs[:, k]
        a_norm = np.linalg.norm(a)
        overlaps.append(abs(float(a @ dr_hat)) / a_norm if a_norm > 0 else 0.0)
    overlaps = np.array(overlaps)

    cumulative = np.sqrt(np.cumsum(overlaps ** 2))
    cumulative = np.clip(cumulative, 0.0, 1.0)

    best_idx = int(np.argmax(overlaps))
    n_half = int(np.searchsorted(cumulative, 0.5) + 1)

    def _co(m: int) -> float:
        return float(cumulative[min(m, n_avail) - 1]) if n_avail >= 1 else 0.0

    return ModeOverlapResult(
        n_core=n_core,
        transition_magnitude=rmsd_internal,
        per_mode_overlap=[float(x) for x in overlaps],
        cumulative_overlap=[float(x) for x in cumulative],
        best_single_overlap=float(overlaps[best_idx]),
        best_mode_index=best_idx,
        softest_mode_overlap=float(overlaps[0]),
        cum_overlap_2=_co(2),
        cum_overlap_5=_co(5),
        cum_overlap_10=_co(10),
        softest_mode_collectivity=collectivity(eigvecs[:, 0]),
        best_mode_collectivity=collectivity(eigvecs[:, best_idx]),
        n_modes_for_half=min(n_half, n_avail),
    )
