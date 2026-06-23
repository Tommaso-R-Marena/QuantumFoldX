"""Tests for QuantumFoldX core components."""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pdb_fetcher import (
    fetch_pdb, parse_pdb_ca_coords, parse_pdb_ca_coords_best_chain,
    list_pdb_chains, compute_contact_map, compute_phi_psi,
)
from src.ensemble.conformational_sampler import (
    generate_nma_ensemble, generate_torsion_ensemble, generate_hybrid_ensemble
)
from src.metrics.structural_metrics import rmsd, tm_score, imfd_rmsd
from src.quantum.ising_vqe import build_ising_hamiltonian, IsingVQESolver
from src.quantum.qaoa_rotamer import build_rotamer_qubo, QAOARotamerOptimizer
from src.scoring.qicess_v2 import QICESSv2Scorer, ramachandran_score
from src.scoring.qicess_v3 import QICESSv3Scorer, state2_geometry_score
from src.quantum.exact_ising import IsingModel, IsingTerm, exact_ground_state, interpolate_ising
from src.quantum.dual_state_ising import (
    build_dual_state_bridge, contacts_to_bitstring, manifold_overlap_score,
    select_shared_contact_qubits,
)
from configs.benchmark_dataset import (
    get_autoinhibited_benchmark, get_foldswitch_benchmark,
    get_multistate_benchmark, get_all_benchmarks
)
from benchmarks.benchmark_utils import find_common_residues, parse_target_structures


class TestPDBFetcher:
    def test_fetch_and_parse_abl1(self):
        path = fetch_pdb('2HYY')
        assert path is not None
        struct = parse_pdb_ca_coords(path, chain='A')
        assert struct is not None
        assert struct['n_residues'] > 100
        assert len(struct['sequence']) == struct['n_residues']

    def test_best_chain_fallback(self):
        path = fetch_pdb('2HYY')
        chains = list_pdb_chains(path)
        assert len(chains) >= 1
        struct = parse_pdb_ca_coords_best_chain(path, preferred_chain='Z')
        assert struct is not None
        assert struct['n_residues'] > 100

    def test_nmr_model_parsing(self):
        path = fetch_pdb('1EJ5')
        struct = parse_pdb_ca_coords(path, chain='A', model=1)
        assert struct is not None
        assert struct['n_residues'] == 107

    def test_contact_map(self):
        coords = np.random.randn(50, 3) * 5
        cm = compute_contact_map(coords, threshold=8.0)
        assert cm.shape == (50, 50)
        assert cm[0, 0] == 0.0


class TestEnsemble:
    def test_nma_ensemble(self):
        coords = np.random.randn(30, 3) * 10
        ens = generate_nma_ensemble(coords, n_conformations=5, seed=42)
        assert len(ens) == 5
        for c in ens:
            assert c.shape == coords.shape

    def test_torsion_ensemble(self):
        coords = np.random.randn(20, 3) * 10
        phi_psi = [( -60.0, -40.0)] * 20
        ens = generate_torsion_ensemble(coords, phi_psi, n_conformations=3, seed=42)
        assert len(ens) == 3

    def test_hybrid_ensemble_includes_quantum_bridge(self):
        n = 40
        coords1 = np.random.randn(n, 3) * 10
        coords2 = coords1 + np.random.randn(n, 3) * 3
        seq = 'A' * n
        phi_psi = [(-60.0, -40.0)] * n
        ens = generate_hybrid_ensemble(
            coords1, seq, fd_indices=list(range(20)), im_indices=list(range(20, 40)),
            n_conformations=20, seed=42, phi_psi=phi_psi,
            coords_s2=coords2,
        )
        methods = {c['method'] for c in ens}
        assert 'quantum_bridge' in methods
        assert 'original' in methods
        assert len(ens) >= 20


class TestMetrics:
    def test_rmsd_identity(self):
        coords = np.random.randn(30, 3)
        assert rmsd(coords, coords) < 1e-6

    def test_tm_score_identity(self):
        coords = np.random.randn(30, 3)
        assert tm_score(coords, coords) > 0.99

    def test_imfd_rmsd(self):
        coords = np.random.randn(50, 3)
        fd = list(range(25, 50))
        im = list(range(0, 25))
        val = imfd_rmsd(coords, coords, fd, im)
        assert val < 1e-6


class TestQuantum:
    def test_ising_hamiltonian(self):
        seq = 'ACDEFGHIKLMNPQRSTVWY' * 2
        n = len(seq)
        coords = np.random.randn(n, 3) * 5
        cm = compute_contact_map(coords)
        result = build_ising_hamiltonian(seq, cm)
        assert result is not None
        H, n_qubits, contacts = result
        assert n_qubits > 0
        assert n_qubits <= 16

    def test_vqe_solver(self):
        seq = 'ACDEFGHIKLMN' * 3
        coords = np.random.randn(len(seq), 3) * 5
        cm = compute_contact_map(coords)
        result = build_ising_hamiltonian(seq, cm)
        H, n_qubits, _ = result
        solver = IsingVQESolver(n_qubits, n_layers=2)
        out = solver.solve(H, n_restarts=1, max_steps=10)
        assert 'ground_energy' in out
        assert out['ground_bitstring'] is not None

    def test_qaoa_rotamer(self):
        seq = 'ACDEFGHIKLMNPQRSTVWY'
        coords = np.random.randn(len(seq), 3) * 5
        cm = compute_contact_map(coords)
        Q, mapping = build_rotamer_qubo(seq, coords, cm)
        if Q.shape[0] >= 2:
            opt = QAOARotamerOptimizer(Q.shape[0], p_layers=2)
            out = opt.optimize(Q, max_steps=10)
            assert 'optimal_bitstring' in out


class TestDualStateBridge:
    def test_dual_state_hamiltonian(self):
        n = 40
        seq = 'A' * n
        coords1 = np.random.randn(n, 3) * 10
        coords2 = coords1 + np.random.randn(n, 3) * 3
        cm1 = compute_contact_map(coords1)
        cm2 = compute_contact_map(coords2)
        bridge = build_dual_state_bridge(
            seq, cm1, cm2,
            fd_indices=list(range(20)), im_indices=list(range(20, 40)),
            max_qubits=12,
        )
        assert bridge.n_qubits > 0
        assert len(bridge.ground_states) == 5
        assert bridge.s1_ground is not None
        assert bridge.s2_ground is not None

    def test_manifold_overlap(self):
        n = 30
        seq = 'A' * n
        coords1 = np.random.randn(n, 3) * 8
        coords2 = np.random.randn(n, 3) * 8
        cm1 = compute_contact_map(coords1)
        cm2 = compute_contact_map(coords2)
        bridge = build_dual_state_bridge(seq, cm1, cm2, max_qubits=10)
        bs = contacts_to_bitstring(cm1, bridge.qubits)
        score = manifold_overlap_score(bs, bridge)
        assert 0.0 <= score <= 1.0

    def test_exact_ising_solver(self):
        model = IsingModel(n_qubits=4, terms=[
            IsingTerm(-1.0, (0,)), IsingTerm(-0.5, (1,)),
            IsingTerm(0.3, (0, 1)),
        ])
        result = exact_ground_state(model)
        assert result['ground_bitstring'] is not None
        assert result['n_states_evaluated'] == 16

    def test_interpolation(self):
        H1 = IsingModel(3, [IsingTerm(-1.0, (0,))])
        H2 = IsingModel(3, [IsingTerm(1.0, (0,))])
        H_mid = interpolate_ising(H1, H2, 0.5)
        assert H_mid.n_qubits == 3


class TestQICESSv3:
    def test_state2_geometry_score(self):
        n = 30
        c1 = np.random.randn(n, 3) * 10
        c2 = c1 + np.random.randn(n, 3) * 0.5
        score = state2_geometry_score(c2, c1, list(range(15)), list(range(15, 30)))
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_v3_ranking_dual_state(self):
        n = 35
        coords1 = np.random.randn(n, 3) * 10
        coords2 = coords1 + np.random.randn(n, 3) * 2
        seq = 'A' * n
        ensemble = [
            {'coords': coords1 + np.random.randn(n, 3) * 0.5,
             'method': 'test', 'perturbation_id': f't_{i}'}
            for i in range(6)
        ]
        ensemble[0]['coords'] = coords1.copy()

        scorer = QICESSv3Scorer(max_qubits=12)
        ranked = scorer.rank_ensemble(
            ensemble, seq,
            reference_coords=coords1,
            state2_coords=coords2,
            fd_indices=list(range(15)),
            im_indices=list(range(15, 35)),
        )
        assert len(ranked) == 6
        assert ranked[0]['rank'] == 1
        assert 'manifold_overlap' in ranked[0]
        assert 'state2_target' in ranked[0]
        assert 'state2_geometry' in ranked[0]
        assert ranked[0]['n_qubits'] > 0


class TestQICESS:
    def test_scorer_ranking(self):
        n = 30
        coords = np.random.randn(n, 3) * 10
        seq = 'A' * n
        ensemble = [{'coords': coords + np.random.randn(n, 3) * 0.5,
                     'method': 'test', 'perturbation_id': f't_{i}'}
                    for i in range(5)]
        ensemble[0]['coords'] = coords.copy()

        scorer = QICESSv2Scorer(vqe_layers=2, vqe_restarts=1, vqe_steps=10, use_qaoa=True)
        ranked = scorer.rank_ensemble(
            ensemble, seq, reference_coords=coords,
            fd_indices=list(range(15)), im_indices=list(range(15, 30))
        )
        assert len(ranked) == 5
        assert ranked[0]['rank'] == 1
        assert 'composite' in ranked[0]

    def test_ramachandran_score(self):
        phi_psi = [(-60.0, -40.0), (-120.0, 120.0), (0.0, 0.0)]
        score = ramachandran_score(phi_psi)
        assert 0.0 <= score <= 1.0


class TestBenchmarkDataset:
    def test_all_benchmarks_no_self_reference(self):
        for target in get_all_benchmarks():
            assert target.pdb_id_state1 != target.pdb_id_state2, (
                f"{target.gene_name} has self-referencing PDB IDs"
            )

    def test_autoinhibited_count(self):
        assert len(get_autoinhibited_benchmark()) == 16

    def test_foldswitch_count(self):
        assert len(get_foldswitch_benchmark()) == 6

    def test_multistate_count(self):
        assert len(get_multistate_benchmark()) == 6

    def test_ptk2_was_parse(self):
        targets = {t.gene_name: t for t in get_autoinhibited_benchmark()}
        ptk2 = targets['PTK2']
        s1, s2, status = parse_target_structures(ptk2)
        assert status == 'ok'
        assert s1['n_residues'] < 1000
        assert s2['n_residues'] < 1000
        _, _, nc = find_common_residues(s1, s2)
        assert nc >= 20

        was = targets['WAS']
        s1, s2, status = parse_target_structures(was)
        assert status == 'ok'
        assert s1['n_residues'] == 107
        assert s2['n_residues'] == 59
