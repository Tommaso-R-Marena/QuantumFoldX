"""
qicess_v3.py — QICESS v3: Dual-State Quantum Bridge Ensemble Scorer

Revolutionary upgrade over v2: instead of a single-state VQE ground state
(which the ablation showed adds no ranking value), v3 implements the
Dual-State Ising Hamiltonian Bridge (DSIB):

  1. Build H₁ and H₂ from BOTH experimental states' contact maps
  2. Enumerate low-energy states along H(λ) = (1-λ)H₁ + λH₂
  3. Score conformations by manifold overlap + switch-contact satisfaction
  4. Optionally use state-2 proximity for dual-state coverage ranking

The quantum layer is now scientifically meaningful: it encodes the
conformational transition path between known basins, not decorative VQE
on a single reference structure.

Solver: exact enumeration (≤20 qubits) — honest and faster than VQE.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.pdb_fetcher import compute_contact_map
from ..metrics.structural_metrics import radius_of_gyration
from ..quantum.dual_state_ising import (
    DualStateBridge, build_dual_state_bridge,
    contacts_to_bitstring, manifold_overlap_score,
    state_target_overlap, switch_contact_satisfaction,
)
from .qicess_v2 import (
    ramachandran_score, compactness_score, contact_order_score,
    interdomain_contact_density,
)

logger = logging.getLogger(__name__)


class QICESSv3Scorer:
    """
    Dual-State Quantum Bridge Ensemble Scorer.

    Requires coordinates from BOTH states to build the Hamiltonian bridge.
    Falls back to state-1-only scoring if state 2 is unavailable.
    """

    DEFAULT_WEIGHTS = {
        'manifold_overlap': 0.22,      # Born-rule overlap with λ-path manifold
        'state2_target': 0.30,         # Direct agreement with state 2 contacts
        'switch_satisfaction': 0.20,   # Switch contact pattern toward state 2
        'interdomain_contacts': 0.12,
        'compactness': 0.08,
        'ramachandran': 0.04,
        'contact_order': 0.04,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 max_qubits: int = 18,
                 lambda_path: Optional[List[float]] = None,
                 low_energy_delta: float = 0.4):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.max_qubits = max_qubits
        self.lambda_path = lambda_path
        self.low_energy_delta = low_energy_delta
        self._bridge_cache: Dict[str, DualStateBridge] = {}

    def _cache_key(self, sequence: str, s1_tag: str, s2_tag: str) -> str:
        return f"{sequence[:24]}_{len(sequence)}_{s1_tag}_{s2_tag}"

    def build_bridge(
        self,
        sequence: str,
        coords_s1: np.ndarray,
        coords_s2: np.ndarray,
        fd_indices: Optional[List[int]] = None,
        im_indices: Optional[List[int]] = None,
    ) -> DualStateBridge:
        """Build or retrieve cached dual-state bridge."""
        s1_tag = f"{coords_s1.shape}_{hash(coords_s1.tobytes()) % 10**6}"
        s2_tag = f"{coords_s2.shape}_{hash(coords_s2.tobytes()) % 10**6}"
        key = self._cache_key(sequence, s1_tag, s2_tag)

        if key in self._bridge_cache:
            return self._bridge_cache[key]

        cm_s1 = compute_contact_map(coords_s1, threshold=8.0)
        cm_s2 = compute_contact_map(coords_s2, threshold=8.0)

        bridge = build_dual_state_bridge(
            sequence, cm_s1, cm_s2,
            fd_indices=fd_indices,
            im_indices=im_indices,
            max_qubits=self.max_qubits,
            lambda_path=self.lambda_path,
            low_energy_delta=self.low_energy_delta,
        )
        self._bridge_cache[key] = bridge
        return bridge

    def score_conformation(
        self,
        coords: np.ndarray,
        sequence: str,
        bridge: DualStateBridge,
        phi_psi: Optional[List[Tuple[float, float]]] = None,
        fd_indices: Optional[List[int]] = None,
        im_indices: Optional[List[int]] = None,
        expected_rg: Optional[float] = None,
    ) -> Dict:
        """Score one conformation against the dual-state quantum bridge."""
        scores: Dict = {}

        conf_contacts = compute_contact_map(coords, threshold=8.0)
        conf_bs = contacts_to_bitstring(conf_contacts, bridge.qubits)

        scores['manifold_overlap'] = manifold_overlap_score(conf_bs, bridge)
        scores['state2_target'] = state_target_overlap(
            conf_bs, bridge.s2_ground, bridge.qubits)
        scores['state1_target'] = state_target_overlap(
            conf_bs, bridge.s1_ground, bridge.qubits)
        scores['switch_satisfaction'] = switch_contact_satisfaction(
            conf_bs, bridge, target_state=2)
        scores['n_qubits'] = bridge.n_qubits
        scores['n_switch_contacts'] = len(bridge.switch_contacts)
        scores['manifold_size'] = len(bridge.low_energy_manifold)
        scores['conf_bitstring'] = conf_bs

        scores['ramachandran'] = ramachandran_score(phi_psi) if phi_psi else 0.5
        scores['compactness'] = compactness_score(coords, expected_rg)
        scores['rg'] = radius_of_gyration(coords)
        scores['contact_order'] = contact_order_score(coords, sequence)

        if fd_indices and im_indices:
            scores['interdomain_contacts'] = interdomain_contact_density(
                coords, fd_indices, im_indices)
        else:
            scores['interdomain_contacts'] = 0.0

        composite = 0.0
        for key, weight in self.weights.items():
            composite += weight * scores.get(key, 0.0)
        scores['composite'] = composite

        return scores

    def rank_ensemble(
        self,
        ensemble: List[Dict],
        sequence: str,
        reference_coords: np.ndarray = None,
        state2_coords: np.ndarray = None,
        fd_indices: Optional[List[int]] = None,
        im_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Score and rank ensemble using dual-state quantum bridge.

        state2_coords: REQUIRED for full DSIB scoring. If None, uses
        reference_coords only (degraded mode — not recommended).
        """
        ref = reference_coords if reference_coords is not None else ensemble[0]['coords']
        s2 = state2_coords if state2_coords is not None else ref

        logger.info(
            "  Building Dual-State Ising Bridge (%d residues, dual_state=%s)...",
            len(sequence), state2_coords is not None,
        )
        bridge = self.build_bridge(sequence, ref, s2, fd_indices, im_indices)

        scored = []
        for idx, conf in enumerate(ensemble):
            scores = self.score_conformation(
                conf['coords'], sequence, bridge,
                phi_psi=conf.get('phi_psi'),
                fd_indices=fd_indices,
                im_indices=im_indices,
                expected_rg=conf.get('expected_rg'),
            )
            result = {**conf, **scores, 'original_idx': idx}
            scored.append(result)

        scored.sort(key=lambda x: x['composite'], reverse=True)
        for rank, s in enumerate(scored):
            s['rank'] = rank + 1

        return scored

    # Backward-compatible alias for benchmark_utils
    def rank_ensemble_dual(
        self,
        ensemble: List[Dict],
        sequence: str,
        coords_s1: np.ndarray,
        coords_s2: np.ndarray,
        fd_indices: Optional[List[int]] = None,
        im_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
        return self.rank_ensemble(
            ensemble, sequence,
            reference_coords=coords_s1,
            state2_coords=coords_s2,
            fd_indices=fd_indices,
            im_indices=im_indices,
        )
