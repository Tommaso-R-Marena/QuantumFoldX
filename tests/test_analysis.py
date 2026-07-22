"""Tests for the mode-overlap and statistics analysis modules."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.mode_overlap import (
    build_anm_hessian, compute_anm_modes, collectivity, analyze_transition,
    superpose, N_RIGID_BODY,
)
from src.analysis import statistics as st
from src.ensemble.conformational_sampler import generate_anm_mode_scan_ensemble


def _random_protein(n=40, seed=0):
    rng = np.random.default_rng(seed)
    # A compact, roughly linear chain so the ANM is connected.
    steps = rng.normal(0, 1, (n, 3))
    coords = np.cumsum(steps, axis=0) * 3.8
    coords += rng.normal(0, 1.5, (n, 3))
    return coords


class TestModeOverlap:
    def test_hessian_symmetric_with_six_zero_modes(self):
        coords = _random_protein(30, seed=1)
        H = build_anm_hessian(coords, cutoff=13.0)
        assert H.shape == (90, 90)
        assert np.allclose(H, H.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(H)
        # Exactly 6 rigid-body modes near zero.
        assert np.sum(np.abs(eigvals) < 1e-6) >= N_RIGID_BODY

    def test_collectivity_bounds(self):
        _, vecs = compute_anm_modes(_random_protein(35, seed=2), n_modes=5)
        for k in range(vecs.shape[1]):
            kappa = collectivity(vecs[:, k])
            assert 0.0 < kappa <= 1.0 + 1e-9

    def test_displacement_along_mode_has_unit_overlap(self):
        # Projection math: a displacement equal to a single (orthonormal) mode
        # must project onto that mode with overlap 1 and zero onto the others.
        coords = _random_protein(40, seed=3)
        _, vecs = compute_anm_modes(coords, n_modes=10)
        dr_hat = vecs[:, 2] / np.linalg.norm(vecs[:, 2])
        overlaps = np.abs(vecs.T @ dr_hat) / np.linalg.norm(vecs, axis=0)
        assert overlaps[2] > 0.999
        assert np.max(np.delete(overlaps, 2)) < 1e-6

    def test_analyze_identifies_dominant_mode(self):
        # Through the full pipeline, a displacement dominated by one internal
        # mode should be identified as the best-overlapping mode.
        coords = _random_protein(60, seed=8)
        _, vecs = compute_anm_modes(coords, n_modes=12)
        n = len(coords)
        target_mode = 6
        disp = 0.4 * np.sqrt(n) * vecs[:, target_mode].reshape(n, 3)
        res = analyze_transition(coords, coords + disp, n_modes=12)
        assert res is not None
        assert res.best_mode_index == target_mode
        assert res.best_single_overlap == max(res.per_mode_overlap)

    def test_overlaps_in_range_and_cumulative_monotonic(self):
        coords1 = _random_protein(45, seed=4)
        coords2 = coords1 + np.random.default_rng(9).normal(0, 2.0, coords1.shape)
        res = analyze_transition(coords1, coords2, n_modes=15)
        assert res is not None
        assert all(0.0 <= o <= 1.0 + 1e-9 for o in res.per_mode_overlap)
        cum = res.cumulative_overlap
        assert all(cum[i] <= cum[i + 1] + 1e-9 for i in range(len(cum) - 1))
        assert cum[-1] <= 1.0 + 1e-9

    def test_identical_structures_return_none(self):
        coords = _random_protein(30, seed=5)
        assert analyze_transition(coords, coords.copy(), n_modes=5) is None

    def test_superpose_removes_rigid_body(self):
        coords = _random_protein(30, seed=6)
        theta = 0.5
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        moved = coords @ R.T + np.array([10.0, -5.0, 3.0])
        aligned = superpose(moved, coords)
        assert np.allclose(aligned, coords, atol=1e-6)

    def test_softmode_sampler_reaches_large_amplitude(self):
        coords = _random_protein(50, seed=7)
        ens = generate_anm_mode_scan_ensemble(coords, n_conformations=30,
                                               n_modes=6, max_rmsd=15.0)
        assert len(ens) == 30
        rmsds = [np.sqrt(np.mean(np.sum((c - coords) ** 2, axis=1))) for c in ens]
        assert max(rmsds) > 5.0


class TestStatistics:
    def test_wilson_contains_estimate(self):
        lo, hi = st.wilson_interval(11, 49)
        assert 0.0 <= lo <= 11 / 49 <= hi <= 1.0

    def test_bootstrap_ci_brackets_mean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, 200)
        res = st.bootstrap_ci(x, np.mean, n_boot=2000, seed=1)
        assert res['ci'][0] < res['point'] < res['ci'][1]
        assert abs(res['point'] - 5.0) < 0.3

    def test_mcnemar_symmetric_is_ns(self):
        assert st.mcnemar_exact(5, 5)['p_value'] == 1.0
        assert st.mcnemar_exact(20, 2)['p_value'] < 0.05

    def test_permutation_paired_detects_shift(self):
        rng = np.random.default_rng(2)
        b = rng.normal(0, 1, 40)
        a = b + 1.0
        res = st.permutation_paired(a, b, n_perm=5000, alternative='greater')
        assert res['p_value'] < 0.01
        null = st.permutation_paired(b, b.copy(), n_perm=2000)
        assert null['p_value'] > 0.2

    def test_cliffs_delta_large(self):
        a = np.arange(0, 50) + 100
        b = np.arange(0, 50)
        d = st.cliffs_delta(a, b)
        assert d['delta'] > 0.9 and d['magnitude'] == 'large'

    def test_holm_monotonic(self):
        res = st.holm_bonferroni({'x': 0.001, 'y': 0.02, 'z': 0.5})
        assert res['x']['p_adjusted'] <= res['y']['p_adjusted'] <= res['z']['p_adjusted']
        assert res['x']['reject_h0'] is True

    def test_paired_diff_bootstrap(self):
        rng = np.random.default_rng(3)
        a = rng.normal(1.0, 1.0, 60)
        b = rng.normal(0.0, 1.0, 60)
        res = st.paired_diff_bootstrap(a, b, n_boot=2000, seed=4)
        assert res['ci'][0] < res['mean_diff'] < res['ci'][1]
