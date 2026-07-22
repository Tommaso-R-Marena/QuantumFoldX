"""
nm_guided.py — Normal-mode-guided BLIND conformational sampling.

Motivation (from results/rigorous/FINDINGS.md): the amount a blind ensemble can
move toward the alternate state is significantly governed by how well the
softest elastic-network modes overlap the transition (Spearman rho=0.46,
independent of transition size). Real conformational transitions, however, are
*combinations* of low modes (cumulative overlap only reaches ~0.5 by ~10-20
modes), so scanning single mode axes is too restrictive.

This module therefore samples the low-frequency **subspace** (mode combinations),
and optionally applies an elastic-network-guided Calpha relaxation that removes
the finite-amplitude distortion of linear mode displacement while leaving the
soft collective motion intact (the collective motion lives in the near-null
space of the network stiffness, so relaxation does not undo it).

Everything here uses STATE 1 ONLY — no state-2 information leaks into generation.

References: Atilgan et al. (2001) Biophys J 80:505 (ANM); Tama & Sanejouand
(2001) Protein Eng 14:1 (low modes vs conformational change); Bahar et al.
(2010) Chem Rev 110:1463.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..analysis.mode_overlap import compute_anm_modes


def _native_network(coords: np.ndarray, cutoff: float = 10.0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elastic-network pair list and native Calpha-Calpha distances of state 1."""
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    mask = np.triu((dist2 < cutoff ** 2) & (dist2 > 1e-6), k=1)
    ij = np.argwhere(mask)
    d0 = np.sqrt(dist2[ij[:, 0], ij[:, 1]])
    # per-atom degree, for stable averaged gradient steps
    degree = np.ones(n)
    for a, b in ij:
        degree[a] += 1
        degree[b] += 1
    return ij, d0, degree


def relax_ca_geometry(
    coords: np.ndarray,
    ref_coords: np.ndarray,
    iters: int = 30,
    step: float = 0.4,
    cutoff: float = 10.0,
    network: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """ENM-guided Calpha relaxation toward state-1 local geometry.

    Minimises E = sum_{(i,j) in native network} (|r_i - r_j| - d0_ij)^2 by
    averaged gradient descent. Because collective low-frequency motions lie in
    the (near-)null space of this network, relaxation removes finite-amplitude
    bond/contact distortion without cancelling the mode displacement.
    """
    if network is None:
        ij, d0, degree = _native_network(ref_coords, cutoff=cutoff)
    else:
        ij, d0, degree = network
    if len(ij) == 0:
        return coords.copy()

    r = coords.copy()
    ii, jj = ij[:, 0], ij[:, 1]
    inv_deg = (step / degree)[:, None]
    for _ in range(iters):
        diff = r[ii] - r[jj]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        d = np.where(d < 1e-6, 1e-6, d)
        f = ((d - d0) / d)[:, None] * diff          # gradient per pair (toward target)
        grad = np.zeros_like(r)
        np.add.at(grad, ii, f)
        np.add.at(grad, jj, -f)
        r = r - inv_deg * grad
    return r


def softmode_subspace_ensemble(
    coords: np.ndarray,
    n_conformations: int = 56,
    k_modes: int = 10,
    max_rmsd: float = 12.0,
    cutoff: float = 13.0,
    seed: int = 42,
    relax: bool = False,
    relax_iters: int = 30,
    thermal_weighting: bool = True,
) -> List[np.ndarray]:
    """Blind ensemble sampling the softest-``k_modes`` ANM subspace of state 1.

    Budget: ~40% deterministic single-mode +/- scans over the softest modes,
    ~60% random unit vectors drawn from the softest-``k_modes`` subspace at a
    graded set of target Calpha RMSD amplitudes. With ``thermal_weighting`` the
    combination coefficients favour softer modes (equipartition-like), matching
    the expectation that low modes carry functional motion. Optionally each
    conformation is ENM-relaxed toward state-1 local geometry.

    Displacing ``coords`` by ``c * v`` (v a unit 3N-vector) yields Calpha RMSD
    ``c / sqrt(N)``; we invert this to hit target amplitudes.
    """
    rng = np.random.default_rng(seed)
    n = len(coords)
    if n < 8:
        return [coords.copy() for _ in range(n_conformations)]

    eigvals, eigvecs = compute_anm_modes(coords, n_modes=k_modes, cutoff=cutoff)
    n_avail = eigvecs.shape[1]
    if n_avail == 0:
        return [coords.copy() for _ in range(n_conformations)]

    sqrt_n = np.sqrt(n)
    amps = np.linspace(max_rmsd / 5.0, max_rmsd, 5)
    network = _native_network(coords, cutoff=10.0) if relax else None

    def _finish(vec_flat, r):
        cand = coords + (r * sqrt_n) * vec_flat.reshape(n, 3)
        if relax:
            cand = relax_ca_geometry(cand, coords, iters=relax_iters, network=network)
        return cand

    ensemble: List[np.ndarray] = []
    n_single = int(round(0.4 * n_conformations))

    # (a) single-mode +/- scans over the softest modes
    n_axis = min(6, n_avail)
    made = 0
    for k in range(n_axis):
        for r in amps:
            for sign in (+1.0, -1.0):
                ensemble.append(_finish(sign * eigvecs[:, k], r))
                made += 1
                if made >= n_single or len(ensemble) >= n_conformations:
                    break
            if made >= n_single or len(ensemble) >= n_conformations:
                break
        if made >= n_single or len(ensemble) >= n_conformations:
            break

    # (b) random combinations spanning the softest-k subspace
    if thermal_weighting:
        # thermal amplitude ~ 1/sqrt(eigval); guard tiny eigenvalues
        w = 1.0 / np.sqrt(np.clip(eigvals[:n_avail], 1e-6, None))
        w = w / w.sum()
    else:
        w = np.ones(n_avail) / n_avail
    while len(ensemble) < n_conformations:
        coeffs = rng.normal(0, 1, n_avail) * w
        v = eigvecs @ coeffs
        nv = np.linalg.norm(v)
        if nv < 1e-8:
            continue
        v /= nv
        r = float(rng.choice(amps))
        ensemble.append(_finish(v, r))

    return ensemble[:n_conformations]
