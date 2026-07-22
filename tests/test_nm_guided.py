"""Tests for normal-mode-guided blind sampling (src/ensemble/nm_guided.py)."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble.nm_guided import (
    softmode_subspace_ensemble, relax_ca_geometry, _native_network,
)
from src.analysis.mode_overlap import compute_anm_modes


def _globular_protein(n=60, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.cumsum(rng.normal(0, 1, (n, 3)), axis=0) * 3.8
    coords += rng.normal(0, 2.0, (n, 3))
    return coords


def _enm_energy(coords, ref):
    ij, d0, _ = _native_network(ref, cutoff=10.0)
    d = np.linalg.norm(coords[ij[:, 0]] - coords[ij[:, 1]], axis=1)
    return float(np.sum((d - d0) ** 2))


class TestSubspaceSampler:
    def test_size_and_shape(self):
        coords = _globular_protein(50, 1)
        ens = softmode_subspace_ensemble(coords, n_conformations=40, k_modes=10, seed=1)
        assert len(ens) == 40
        assert all(c.shape == coords.shape for c in ens)

    def test_deterministic(self):
        coords = _globular_protein(50, 2)
        a = softmode_subspace_ensemble(coords, n_conformations=30, seed=7)
        b = softmode_subspace_ensemble(coords, n_conformations=30, seed=7)
        assert all(np.allclose(x, y) for x, y in zip(a, b))

    def test_reaches_large_amplitude(self):
        coords = _globular_protein(60, 3)
        ens = softmode_subspace_ensemble(coords, n_conformations=40, max_rmsd=12.0, seed=3)
        rmsds = [np.sqrt(np.mean(np.sum((c - coords) ** 2, axis=1))) for c in ens]
        assert max(rmsds) > 5.0

    def test_uses_only_state1(self):
        # The signature takes a single coordinate set; there is no way to leak
        # state 2. This guards against accidental signature changes.
        import inspect
        params = inspect.signature(softmode_subspace_ensemble).parameters
        assert 'coords_s2' not in params and 'state2' not in params

    def test_subspace_differs_from_single_axes(self):
        # Random-combination members should not all lie on single mode axes.
        coords = _globular_protein(60, 5)
        _, vecs = compute_anm_modes(coords, n_modes=10)
        ens = softmode_subspace_ensemble(coords, n_conformations=56, k_modes=10, seed=5)
        # take a late (combination) member and check it has weight on multiple modes
        disp = (ens[-1] - coords).reshape(-1)
        disp /= np.linalg.norm(disp)
        proj = np.abs(vecs.T @ disp)
        # more than one mode carries appreciable projection
        assert np.sum(proj > 0.1) >= 2


class TestRelaxation:
    def test_relax_reduces_distortion_but_keeps_motion(self):
        coords = _globular_protein(60, 4)
        _, vecs = compute_anm_modes(coords, n_modes=10)
        n = len(coords)
        # large linear displacement along a mid mode -> introduces distortion
        disp = 8.0 * np.sqrt(n) * vecs[:, 4].reshape(n, 3)
        moved = coords + disp
        relaxed = relax_ca_geometry(moved, coords, iters=40)
        e_before = _enm_energy(moved, coords)
        e_after = _enm_energy(relaxed, coords)
        assert e_after < e_before                      # distortion removed
        rmsd_kept = np.sqrt(np.mean(np.sum((relaxed - coords) ** 2, axis=1)))
        assert rmsd_kept > 2.0                         # collective motion preserved

    def test_relax_is_noop_on_native(self):
        coords = _globular_protein(50, 6)
        relaxed = relax_ca_geometry(coords.copy(), coords, iters=30)
        assert np.sqrt(np.mean(np.sum((relaxed - coords) ** 2, axis=1))) < 0.2

    def test_relaxed_ensemble_size(self):
        coords = _globular_protein(55, 8)
        ens = softmode_subspace_ensemble(coords, n_conformations=30, seed=2,
                                         relax=True, relax_iters=20)
        assert len(ens) == 30
