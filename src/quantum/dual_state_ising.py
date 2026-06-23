"""
dual_state_ising.py — Dual-State Ising Hamiltonian Bridge (DSIB)

Builds two Ising Hamiltonians — H₁ from state 1 contacts and H₂ from state 2
contacts — on a shared qubit basis encoding inter-domain contact patterns.

The interpolated Hamiltonian H(λ) = (1-λ)H₁ + λH₂ is used to enumerate
low-energy contact patterns between the two basins. Those patterns help:
  1. Score ensemble members by overlap with the low-energy manifold
  2. Identify switch contacts (qubits that flip between H₁ and H₂ optima)
  3. Guide targeted domain perturbations toward the alternate state

At ≤20 qubits we use exact enumeration; larger models fall back to greedy
annealing. All circuits here are classically simulated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .exact_ising import (
    IsingModel, IsingTerm, exact_low_energy_manifold, interpolate_ising,
    solve_ising, bitstring_agreement,
)
from .ising_vqe import MJ_POTENTIALS, MJ_AA_TO_IDX

logger = logging.getLogger(__name__)


@dataclass
class ContactQubit:
    """One qubit encoding whether contact (i, j) is active."""
    qubit_idx: int
    res_i: int
    res_j: int
    mj_coupling: float
    state1_active: bool
    state2_active: bool
    is_switch: bool
    is_interdomain: bool
    switch_score: float


@dataclass
class DualStateBridge:
    """Complete dual-state quantum bridge analysis."""
    qubits: List[ContactQubit]
    H1: IsingModel
    H2: IsingModel
    lambda_path: List[float]
    ground_states: Dict[float, Dict]  # λ -> solve_ising result
    low_energy_manifold: List[Dict]  # pooled low-energy states across λ
    switch_qubits: List[int]
    switch_contacts: List[ContactQubit]
    s1_ground: str
    s2_ground: str
    n_qubits: int


def _contact_pairs(contact_map: np.ndarray, min_sep: int = 3
                   ) -> List[Tuple[int, int, float]]:
    n = contact_map.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + min_sep, n):
            if contact_map[i, j] > 0:
                pairs.append((i, j, float(contact_map[i, j])))
    return pairs


def select_shared_contact_qubits(
    sequence: str,
    contact_map_s1: np.ndarray,
    contact_map_s2: np.ndarray,
    fd_indices: Optional[List[int]] = None,
    im_indices: Optional[List[int]] = None,
    max_qubits: int = 20,
) -> List[ContactQubit]:
    """
    Select qubits for the dual-state Hamiltonian.

    Priority:
      1. Inter-domain contacts that differ between states (switch contacts)
      2. Inter-domain contacts shared by both states
      3. Intra-domain contacts with largest MJ coupling difference
    """
    n = min(len(sequence), contact_map_s1.shape[0], contact_map_s2.shape[0])
    fd_set = set(fd_indices or [])
    im_set = set(im_indices or [])

    candidates: List[ContactQubit] = []
    q_idx = 0

    all_pairs = set()
    for cm in (contact_map_s1, contact_map_s2):
        for i, j, _ in _contact_pairs(cm[:n, :n]):
            all_pairs.add((min(i, j), max(i, j)))

    for i, j in sorted(all_pairs):
        aa_i = MJ_AA_TO_IDX.get(sequence[i], 0)
        aa_j = MJ_AA_TO_IDX.get(sequence[j], 0)
        mj = float(MJ_POTENTIALS[aa_i, aa_j])

        s1_on = contact_map_s1[i, j] > 0 if i < n and j < n else False
        s2_on = contact_map_s2[i, j] > 0 if i < n and j < n else False
        is_inter = (i in fd_set and j in im_set) or (i in im_set and j in fd_set)
        is_switch = s1_on != s2_on

        # Switch score: inter-domain switches are most informative
        switch_score = abs(int(s2_on) - int(s1_on)) * (3.0 if is_inter else 1.0)
        switch_score += abs(mj) * 0.1

        candidates.append(ContactQubit(
            qubit_idx=q_idx,
            res_i=i, res_j=j,
            mj_coupling=mj,
            state1_active=s1_on,
            state2_active=s2_on,
            is_switch=is_switch,
            is_interdomain=is_inter,
            switch_score=switch_score,
        ))
        q_idx += 1

    # Re-index after sorting
    candidates.sort(key=lambda q: q.switch_score, reverse=True)
    selected = candidates[:max_qubits]
    for new_idx, q in enumerate(selected):
        q.qubit_idx = new_idx

    return selected


def build_dual_state_hamiltonian(
    qubits: List[ContactQubit],
    coupling_strength: float = 0.35,
) -> Tuple[IsingModel, IsingModel]:
    """
    Build H₁ and H₂ on shared qubit basis.

    Each qubit k encodes contact (i,j) active (|1⟩) or inactive (|0⟩).
    Local fields bias toward each state's contact pattern:
        h_k^(s) = -sign(s,k) * |MJ_ij|   where sign = +1 if contact active in s
    ZZ couplings enforce cooperative contacts sharing a residue.
    """
    n = len(qubits)
    if n == 0:
        return IsingModel(0), IsingModel(0)

    def _build_for_state(state_attr: str) -> IsingModel:
        terms: List[IsingTerm] = []
        local_fields = []
        for q in qubits:
            active = getattr(q, state_attr)
            # Bias qubit toward state's contact pattern
            h = -1.0 if active else +0.5
            h *= max(abs(q.mj_coupling), 0.1)
            if q.is_interdomain:
                h *= 2.0
            local_fields.append((q.qubit_idx, h))
            terms.append(IsingTerm(-h / 2.0, (q.qubit_idx,)))

        # Cooperative ZZ between qubits sharing a residue
        for a in range(n):
            for b in range(a + 1, n):
                qa, qb = qubits[a], qubits[b]
                shared = (qa.res_i == qb.res_i or qa.res_i == qb.res_j or
                          qa.res_j == qb.res_i or qa.res_j == qb.res_j)
                if shared:
                    j_val = -coupling_strength * (qa.mj_coupling + qb.mj_coupling) / 2.0
                    if abs(j_val) > 0.01:
                        terms.append(IsingTerm(j_val, (qa.qubit_idx, qb.qubit_idx)))

        meta = {
            'n_contacts': n,
            'n_switch': sum(q.is_switch for q in qubits),
            'state': state_attr,
        }
        return IsingModel(n_qubits=n, terms=terms, metadata=meta)

    return _build_for_state('state1_active'), _build_for_state('state2_active')


def build_dual_state_bridge(
    sequence: str,
    contact_map_s1: np.ndarray,
    contact_map_s2: np.ndarray,
    fd_indices: Optional[List[int]] = None,
    im_indices: Optional[List[int]] = None,
    max_qubits: int = 20,
    lambda_path: Optional[List[float]] = None,
    low_energy_delta: float = 0.4,
) -> DualStateBridge:
    """
    Construct the full Dual-State Ising Hamiltonian Bridge.

    Enumerates ground/low-energy states along λ ∈ [0, 1] and identifies
    switch contacts for quantum-guided ensemble generation.
    """
    if lambda_path is None:
        lambda_path = [0.0, 0.25, 0.5, 0.75, 1.0]

    qubits = select_shared_contact_qubits(
        sequence, contact_map_s1, contact_map_s2,
        fd_indices, im_indices, max_qubits=max_qubits,
    )

    if not qubits:
        empty = IsingModel(0)
        return DualStateBridge(
            qubits=[], H1=empty, H2=empty, lambda_path=lambda_path,
            ground_states={}, low_energy_manifold=[], switch_qubits=[],
            switch_contacts=[], s1_ground='', s2_ground='', n_qubits=0,
        )

    H1, H2 = build_dual_state_hamiltonian(qubits)
    ground_states: Dict[float, Dict] = {}
    manifold_pool: Dict[str, Dict] = {}

    for lam in lambda_path:
        H_lam = interpolate_ising(H1, H2, lam)
        gs = solve_ising(H_lam)
        ground_states[lam] = gs

        low_e = exact_low_energy_manifold(H_lam, delta_e=low_energy_delta, max_states=16)
        for state in low_e:
            bs = state['bitstring']
            key = bs
            if key not in manifold_pool:
                manifold_pool[key] = {
                    'bitstring': bs,
                    'energy_min': state['energy'],
                    'boltzmann_weight': state['boltzmann_weight'],
                    'lambdas': [lam],
                }
            else:
                manifold_pool[key]['energy_min'] = min(
                    manifold_pool[key]['energy_min'], state['energy'])
                manifold_pool[key]['lambdas'].append(lam)
                manifold_pool[key]['boltzmann_weight'] = max(
                    manifold_pool[key]['boltzmann_weight'], state['boltzmann_weight'])

    # Normalize manifold weights
    manifold = list(manifold_pool.values())
    total_w = sum(m['boltzmann_weight'] for m in manifold) or 1.0
    for m in manifold:
        m['boltzmann_weight'] /= total_w
        m['lambda_coverage'] = len(set(m['lambdas'])) / len(lambda_path)
    manifold.sort(key=lambda x: (-x['lambda_coverage'], x['energy_min']))

    s1_ground = ground_states.get(0.0, {}).get('ground_bitstring', '')
    s2_ground = ground_states.get(1.0, {}).get('ground_bitstring', '')

    switch_qubits = []
    switch_contacts = []
    if s1_ground and s2_ground:
        for q in qubits:
            qi = q.qubit_idx
            if qi < len(s1_ground) and qi < len(s2_ground):
                if s1_ground[qi] != s2_ground[qi]:
                    switch_qubits.append(qi)
                    switch_contacts.append(q)

    logger.info(
        "  DSIB: %d qubits, %d switch contacts, manifold size %d",
        len(qubits), len(switch_contacts), len(manifold),
    )

    return DualStateBridge(
        qubits=qubits,
        H1=H1, H2=H2,
        lambda_path=lambda_path,
        ground_states=ground_states,
        low_energy_manifold=manifold,
        switch_qubits=switch_qubits,
        switch_contacts=switch_contacts,
        s1_ground=s1_ground,
        s2_ground=s2_ground,
        n_qubits=len(qubits),
    )


def contacts_to_bitstring(contact_map: np.ndarray,
                          qubits: List[ContactQubit]) -> str:
    """Encode a conformation's contact pattern into the shared qubit basis."""
    bits = []
    for q in qubits:
        i, j = q.res_i, q.res_j
        active = (i < contact_map.shape[0] and j < contact_map.shape[1]
                  and contact_map[i, j] > 0)
        bits.append('1' if active else '0')
    return ''.join(bits)


def manifold_overlap_score(conf_bitstring: str, bridge: DualStateBridge) -> float:
    """
    Born-rule-inspired overlap: weighted agreement with the low-energy manifold.

    States appearing at multiple λ values (transition contacts) receive
    higher weight — they sit on the conformational bridge between basins.
    """
    if not bridge.low_energy_manifold or not conf_bitstring:
        return 0.0

    score = 0.0
    for state in bridge.low_energy_manifold:
        bs = state['bitstring']
        agree = bitstring_agreement(conf_bitstring, bs)
        # Boost states spanning multiple λ (true bridge configurations)
        bridge_boost = 1.0 + 0.5 * state.get('lambda_coverage', 0.0)
        score += state['boltzmann_weight'] * agree * bridge_boost

    return min(score, 1.0)


def state_target_overlap(conf_bitstring: str, target_bitstring: str,
                         qubits: List[ContactQubit],
                         boost_interdomain: bool = True) -> float:
    """Direct overlap with a specific state's optimal contact pattern."""
    if not target_bitstring or not conf_bitstring:
        return 0.0

    weights = []
    for q in qubits:
        w = max(abs(q.mj_coupling), 0.1)
        if boost_interdomain and q.is_interdomain:
            w *= 2.5
        weights.append(w)

    total_w = sum(weights)
    agree = 0.0
    n = min(len(conf_bitstring), len(target_bitstring), len(weights))
    for i in range(n):
        if conf_bitstring[i] == target_bitstring[i]:
            agree += weights[i]
    return agree / max(total_w, 1e-8)


def switch_contact_satisfaction(conf_bitstring: str,
                                 bridge: DualStateBridge,
                                 target_state: int = 2) -> float:
    """
    Fraction of switch contacts matching the target state's pattern.

    target_state: 1 for state 1, 2 for state 2
    """
    if not bridge.switch_contacts or not conf_bitstring:
        return 0.5

    correct = 0
    total = 0
    for q in bridge.switch_contacts:
        qi = q.qubit_idx
        if qi >= len(conf_bitstring):
            continue
        target_active = q.state1_active if target_state == 1 else q.state2_active
        target_bit = '1' if target_active else '0'
        if conf_bitstring[qi] == target_bit:
            correct += 1
        total += 1
    return correct / max(total, 1)


def estimate_domain_motion_from_switches(
    bridge: DualStateBridge,
    coords_s1: np.ndarray,
    coords_s2: np.ndarray,
    fd_indices: List[int],
    im_indices: List[int],
    common_idx_s1: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> Dict:
    """
    Estimate rigid-body motion needed to flip switch contacts toward state 2.

    Uses the displacement of inter-domain contacts between experimental states
    to infer translation/rotation direction for the inhibitory module.
    """
    if not bridge.switch_contacts or not fd_indices or not im_indices:
        return {'translation': np.zeros(3), 'rotation_axis': np.zeros(3),
                'rotation_angle_deg': 0.0, 'confidence': 0.0}

    from ..scoring.geometry_utils import common_residue_pairs

    pairs = common_residue_pairs(common_idx_s1, common_idx_s2, len(coords_s1), len(coords_s2))
    s2_lookup = {i: j for i, j in pairs}

    def _s2_coord(idx: int) -> Optional[np.ndarray]:
        j = s2_lookup.get(idx)
        if j is not None and j < len(coords_s2):
            return coords_s2[j]
        if idx < len(coords_s2):
            return coords_s2[idx]
        return None

    displacements = []
    for q in bridge.switch_contacts:
        if not q.is_interdomain:
            continue
        i, j = q.res_i, q.res_j
        c1_i = coords_s1[i] if i < len(coords_s1) else None
        c1_j = coords_s1[j] if j < len(coords_s1) else None
        c2_i = _s2_coord(i)
        c2_j = _s2_coord(j)
        if c1_i is not None and c1_j is not None and c2_i is not None and c2_j is not None:
            mid_s1 = (c1_i + c1_j) / 2.0
            mid_s2 = (c2_i + c2_j) / 2.0
            displacements.append(mid_s2 - mid_s1)

    if not displacements:
        # Fall back to domain centroid displacement on common residues
        fd_s1_pts = [coords_s1[i] for i in fd_indices if i < len(coords_s1)]
        im_s1_pts = [coords_s1[i] for i in im_indices if i < len(coords_s1)]
        im_s2_pts = [_s2_coord(i) for i in im_indices]
        im_s2_pts = [p for p in im_s2_pts if p is not None]
        if not fd_s1_pts or not im_s1_pts or not im_s2_pts:
            return {'translation': np.zeros(3), 'rotation_axis': np.zeros(3),
                    'rotation_angle_deg': 0.0, 'confidence': 0.0}
        im_s1 = np.mean(im_s1_pts, axis=0)
        im_s2 = np.mean(im_s2_pts, axis=0)
        fd_center = np.mean(fd_s1_pts, axis=0)
        translation = im_s2 - im_s1
        return {
            'translation': translation,
            'rotation_axis': np.array([0., 0., 1.]),
            'rotation_angle_deg': 0.0,
            'confidence': 0.3,
            'pivot': fd_center,
        }

    disp_arr = np.array(displacements)
    mean_disp = disp_arr.mean(axis=0)
    confidence = min(len(displacements) / max(len(bridge.switch_contacts), 1), 1.0)

    # Rotation axis from cross product of s1 and s2 domain vectors (common-residue aware)
    fd_s1_pts = [coords_s1[i] for i in fd_indices if i < len(coords_s1)]
    im_s1_pts = [coords_s1[i] for i in im_indices if i < len(coords_s1)]
    im_s2_pts = [_s2_coord(i) for i in im_indices]
    fd_s2_pts = [_s2_coord(i) for i in fd_indices]
    im_s2_pts = [p for p in im_s2_pts if p is not None]
    fd_s2_pts = [p for p in fd_s2_pts if p is not None]
    if not fd_s1_pts or not im_s1_pts or not im_s2_pts or not fd_s2_pts:
        return {
            'translation': mean_disp,
            'rotation_axis': np.array([0., 0., 1.]),
            'rotation_angle_deg': 0.0,
            'confidence': confidence,
            'pivot': np.mean(fd_s1_pts, axis=0) if fd_s1_pts else np.zeros(3),
            'n_switch_contacts_used': len(displacements),
        }

    im_s1 = np.mean(im_s1_pts, axis=0)
    fd_s1 = np.mean(fd_s1_pts, axis=0)
    im_s2 = np.mean(im_s2_pts, axis=0)
    fd_s2 = np.mean(fd_s2_pts, axis=0)

    v1 = im_s1 - fd_s1
    v2 = im_s2 - fd_s2
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    axis = cross / cross_norm if cross_norm > 1e-6 else np.array([0., 0., 1.])

    cos_angle = np.dot(v1, v2) / (max(np.linalg.norm(v1), 1e-6) * max(np.linalg.norm(v2), 1e-6))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    return {
        'translation': mean_disp,
        'rotation_axis': axis,
        'rotation_angle_deg': angle_deg,
        'confidence': confidence,
        'pivot': fd_s1,
        'n_switch_contacts_used': len(displacements),
    }


def compute_transition_complexity(bridge: DualStateBridge) -> Dict:
    """
    Transition Complexity Index (TCI) from DSIB analysis.

    Combines switch-contact density, λ-path energy barrier, and manifold
    bridge span into a single score in [0, 1]. Higher TCI indicates a
    conformational transition that is structurally encoded in the contact
    switch pattern and the interpolated Hamiltonian path H(λ).
    """
    n = bridge.n_qubits
    if n == 0:
        return {
            'tci': 0.0,
            'switch_fraction': 0.0,
            'lambda_barrier': 0.0,
            'bridge_span': 0.0,
            'manifold_size_norm': 0.0,
            'n_switch': 0,
            'manifold_size': 0,
        }

    switch_fraction = len(bridge.switch_contacts) / n

    energies = [
        bridge.ground_states[lam]['ground_energy']
        for lam in bridge.lambda_path
        if lam in bridge.ground_states
    ]
    if energies:
        e_min = min(energies)
        e_max = max(energies)
        e_mid = bridge.ground_states.get(0.5, {}).get('ground_energy', e_min)
        span = max(abs(e_max - e_min), 1e-6)
        lambda_barrier = float(np.clip((e_mid - e_min) / span, 0.0, 1.0))
    else:
        lambda_barrier = 0.0

    if bridge.low_energy_manifold:
        bridge_span = float(np.mean([m.get('lambda_coverage', 0.0)
                                     for m in bridge.low_energy_manifold]))
        manifold_size_norm = min(len(bridge.low_energy_manifold) / (2 ** n), 1.0)
    else:
        bridge_span = 0.0
        manifold_size_norm = 0.0

    tci = (
        0.35 * switch_fraction
        + 0.25 * lambda_barrier
        + 0.25 * bridge_span
        + 0.15 * manifold_size_norm
    )
    tci = float(np.clip(tci, 0.0, 1.0))

    return {
        'tci': tci,
        'switch_fraction': float(switch_fraction),
        'lambda_barrier': lambda_barrier,
        'bridge_span': bridge_span,
        'manifold_size_norm': float(manifold_size_norm),
        'n_switch': len(bridge.switch_contacts),
        'manifold_size': len(bridge.low_energy_manifold),
    }
