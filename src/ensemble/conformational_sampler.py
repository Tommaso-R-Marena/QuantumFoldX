"""
conformational_sampler.py — Generate conformational ensembles from protein structures.

Methods:
1. Normal Mode Analysis (NMA) perturbation — physics-based backbone flexibility
2. Torsion angle perturbation — sample backbone φ/ψ variations
3. Domain rigid-body perturbation — explore inter-domain arrangements

These generate the INPUT ensemble that QICESS v2 then scores and ranks.
The idea: generate many plausible conformations, then use quantum-enhanced
scoring to select the best ones.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from ..data.pdb_fetcher import compute_contact_map

logger = logging.getLogger(__name__)


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def generate_quantum_bridge_ensemble(
    coords_s1: np.ndarray,
    coords_s2: np.ndarray,
    domain1_indices: List[int],
    domain2_indices: List[int],
    motion_hint: Dict,
    n_conformations: int = 15,
    seed: int = 42,
    common_idx_s1: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Generate conformations along the quantum-identified conformational bridge.

    Uses switch-contact-derived domain motion to interpolate between states
    and explore targeted rigid-body trajectories toward state 2.
    """
    rng = np.random.default_rng(seed)
    ensemble = []
    n = len(coords_s1)

    translation = motion_hint.get('translation', np.zeros(3))
    axis = motion_hint.get('rotation_axis', np.array([0., 0., 1.]))
    base_angle = motion_hint.get('rotation_angle_deg', 0.0)
    pivot = motion_hint.get('pivot', coords_s1[domain1_indices].mean(axis=0))
    confidence = motion_hint.get('confidence', 0.5)

    d2_coords = coords_s1[domain2_indices].copy()
    d2_center = d2_coords.mean(axis=0)

    from ..scoring.geometry_utils import interpolate_coords_on_common

    # Linear interpolation on common residues (handles different chain lengths)
    n_interp = max(2, n_conformations // 3)
    for t_idx in range(n_interp):
        alpha = (t_idx + 1) / (n_interp + 1)
        blended = interpolate_coords_on_common(
            coords_s1, coords_s2, alpha,
            common_idx_s1, common_idx_s2,
        )
        ensemble.append(blended)

    # Targeted rigid-body motions along switch-contact direction
    n_targeted = n_conformations - n_interp
    for i in range(n_targeted):
        new_coords = coords_s1.copy()
        progress = (i + 1) / max(n_targeted, 1)

        # Scale motion by confidence and progress along bridge
        trans = translation * progress * (0.5 + 0.5 * confidence)
        trans += rng.normal(0, 0.5, 3)  # small noise

        angle_deg = base_angle * progress + rng.uniform(-5, 5)
        angle_rad = angle_deg * np.pi / 180.0
        R = _rotation_matrix(axis, angle_rad)

        d2_centered = d2_coords - d2_center
        d2_moved = (d2_centered @ R.T) + d2_center + trans

        for j, idx in enumerate(domain2_indices):
            if idx < n:
                new_coords[idx] = d2_moved[j]

        ensemble.append(new_coords)

    return ensemble


def generate_switch_guided_ensemble(
    coords: np.ndarray,
    coords_s2: np.ndarray,
    domain1_indices: List[int],
    domain2_indices: List[int],
    bridge,
    n_conformations: int = 10,
    seed: int = 42,
    common_idx_s1: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Generate conformations biased toward flipping switch contacts to state 2.

    Uses the DualStateBridge switch contact analysis to drive domain motion.
    """
    from ..quantum.dual_state_ising import estimate_domain_motion_from_switches

    motion = estimate_domain_motion_from_switches(
        bridge, coords, coords_s2, domain1_indices, domain2_indices,
        common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2,
    )
    return generate_quantum_bridge_ensemble(
        coords, coords_s2, domain1_indices, domain2_indices,
        motion, n_conformations=n_conformations, seed=seed,
        common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2,
    )


def generate_manifold_bridge_ensemble(
    coords_s1: np.ndarray,
    coords_s2: np.ndarray,
    bridge,
    n_conformations: int = 8,
    seed: int = 42,
    common_idx_s1: Optional[List[int]] = None,
    common_idx_s2: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Generate conformations at λ values suggested by the DSIB low-energy manifold.

    Each manifold state encodes a contact pattern observed at specific λ values
    along H(λ); we map those back to geometric interpolants on common residues.
    """
    from ..scoring.geometry_utils import interpolate_coords_on_common

    if bridge is None or not getattr(bridge, 'low_energy_manifold', None):
        return []

    rng = np.random.default_rng(seed)
    ranked = sorted(
        bridge.low_energy_manifold,
        key=lambda m: (-m.get('lambda_coverage', 0.0), -m.get('boltzmann_weight', 0.0)),
    )

    ensemble = []
    for state in ranked[:n_conformations]:
        lams = state.get('lambdas', [0.5])
        lam = float(np.clip(np.mean(lams), 0.05, 0.95))
        blended = interpolate_coords_on_common(
            coords_s1, coords_s2, lam,
            common_idx_s1, common_idx_s2,
        )
        noise = rng.normal(0, 0.15, blended.shape)
        ensemble.append(blended + noise)

    return ensemble


def generate_nma_ensemble(coords: np.ndarray, n_conformations: int = 20,
                           amplitude: float = 2.0, n_modes: int = 10,
                           seed: int = 42) -> List[np.ndarray]:
    """
    Generate conformational ensemble using Elastic Network Model (ENM)
    normal mode analysis.
    
    The lowest-frequency normal modes capture the largest collective
    motions — exactly the inter-domain movements relevant for
    autoinhibited proteins.
    
    Parameters:
        coords: Cα coordinates (N, 3)
        n_conformations: number of conformations to generate
        amplitude: perturbation amplitude in Å
        n_modes: number of lowest-frequency modes to use
        seed: random seed for reproducibility
    
    Returns:
        list of np.array (N, 3) — perturbed coordinates
    """
    rng = np.random.default_rng(seed)
    n = len(coords)
    
    # Build Kirchhoff/Hessian matrix for Gaussian Network Model (GNM)
    cutoff = 10.0  # Å
    dist_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    
    # Kirchhoff matrix (connectivity)
    gamma = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < cutoff:
                gamma[i, j] = -1.0
                gamma[j, i] = -1.0
                gamma[i, i] += 1.0
                gamma[j, j] += 1.0
    
    # Anisotropic Network Model (ANM) Hessian
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < cutoff:
                diff = coords[j] - coords[i]
                d = dist_matrix[i, j]
                k = 1.0 / (d * d)  # Spring constant
                
                for a in range(3):
                    for b in range(3):
                        val = k * diff[a] * diff[b] / (d * d)
                        H[3*i+a, 3*j+b] = -val
                        H[3*j+b, 3*i+a] = -val
                        H[3*i+a, 3*i+b] += val
                        H[3*j+a, 3*j+b] += val
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Skip first 6 modes (rigid body: 3 translation + 3 rotation)
    # Use modes 6 to 6+n_modes (lowest frequency non-trivial modes)
    mode_start = 6
    mode_end = min(mode_start + n_modes, len(eigenvalues))
    
    ensemble = []
    for conf_idx in range(n_conformations):
        perturbation = np.zeros(3 * n)
        
        for mode_idx in range(mode_start, mode_end):
            if eigenvalues[mode_idx] < 1e-6:
                continue
            
            # Random amplitude along this mode
            amp = rng.normal(0, amplitude / np.sqrt(eigenvalues[mode_idx]))
            perturbation += amp * eigenvectors[:, mode_idx]
        
        # Scale to desired amplitude
        pert_magnitude = np.linalg.norm(perturbation)
        if pert_magnitude > 0:
            perturbation *= (amplitude / pert_magnitude) * rng.uniform(0.5, 1.5)
        
        new_coords = coords + perturbation.reshape(n, 3)
        ensemble.append(new_coords)
    
    return ensemble


def generate_domain_rigid_body_ensemble(coords: np.ndarray,
                                         domain1_indices: List[int],
                                         domain2_indices: List[int],
                                         n_conformations: int = 20,
                                         max_translation: float = 5.0,
                                         max_rotation: float = 15.0,
                                         seed: int = 42) -> List[np.ndarray]:
    """
    Generate ensemble by rigid-body perturbation of one domain relative to another.
    
    This directly models the inter-domain positioning problem
    that AF3 struggles with for autoinhibited proteins.
    
    Parameters:
        coords: full structure coordinates (N, 3)
        domain1_indices: fixed domain (FD) residue indices
        domain2_indices: mobile domain (IM) residue indices
        n_conformations: number to generate
        max_translation: max translation in Å
        max_rotation: max rotation in degrees
    """
    rng = np.random.default_rng(seed)
    
    ensemble = []
    d2_coords = coords[domain2_indices].copy()
    d2_center = d2_coords.mean(axis=0)
    
    for _ in range(n_conformations):
        new_coords = coords.copy()
        
        # Random rotation of domain 2
        angle_x = rng.uniform(-max_rotation, max_rotation) * np.pi / 180
        angle_y = rng.uniform(-max_rotation, max_rotation) * np.pi / 180
        angle_z = rng.uniform(-max_rotation, max_rotation) * np.pi / 180
        
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], 
                        [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0],
                        [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                        [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        # Random translation
        translation = rng.uniform(-max_translation, max_translation, 3)
        
        # Apply to domain 2
        d2_centered = d2_coords - d2_center
        d2_rotated = (d2_centered @ R.T) + d2_center + translation
        
        new_coords[domain2_indices] = d2_rotated
        ensemble.append(new_coords)
    
    return ensemble


def generate_torsion_ensemble(coords: np.ndarray,
                               phi_psi: List[Tuple[float, float]],
                               n_conformations: int = 20,
                               max_delta: float = 15.0,
                               seed: int = 42) -> List[np.ndarray]:
    """
    Generate ensemble by perturbing backbone φ/ψ dihedral angles.
    
    Uses a simplified Cα-only reconstruction: each residue is displaced
    based on cumulative dihedral perturbations along the chain. This captures
    local backbone flexibility complementary to NMA and rigid-body sampling.
    """
    rng = np.random.default_rng(seed)
    n = len(coords)
    if n < 4 or not phi_psi or len(phi_psi) != n:
        return [coords.copy() for _ in range(n_conformations)]
    
    ensemble = []
    for _ in range(n_conformations):
        new_coords = coords.copy()
        cumulative_shift = np.zeros(3)
        
        for i in range(1, n - 1):
            phi, psi = phi_psi[i]
            if np.isnan(phi) or np.isnan(psi):
                continue
            
            dphi = rng.normal(0, max_delta / 3)
            dpsi = rng.normal(0, max_delta / 3)
            
            # Displacement direction from local backbone geometry
            v1 = new_coords[i] - new_coords[i - 1]
            v2 = new_coords[i + 1] - new_coords[i]
            v1_norm = np.linalg.norm(v1)
            if v1_norm < 1e-6:
                continue
            v1 /= v1_norm
            
            # Perturbation magnitude scales with dihedral change
            mag = 0.15 * (abs(dphi) + abs(dpsi)) / 180.0 * 3.8  # ~Cα-Cα distance
            shift = mag * np.cross(v1, v2 / max(np.linalg.norm(v2), 1e-6))
            if np.linalg.norm(shift) < 1e-8:
                shift = mag * rng.normal(size=3)
            
            cumulative_shift += shift
            new_coords[i:] += shift
        
        ensemble.append(new_coords)
    
    return ensemble


def generate_hybrid_ensemble(coords: np.ndarray,
                              sequence: str,
                              fd_indices: List[int] = None,
                              im_indices: List[int] = None,
                              n_conformations: int = 50,
                              seed: int = 42,
                              phi_psi: List[Tuple[float, float]] = None,
                              use_qaoa: bool = False,
                              coords_s2: np.ndarray = None,
                              quantum_bridge=None,
                              transition_difficulty: float = 0.0,
                              common_idx_s1: Optional[List[int]] = None,
                              common_idx_s2: Optional[List[int]] = None) -> List[Dict]:
    """
    Generate comprehensive ensemble using multiple methods at multiple scales.
    
    Combines:
    - NMA perturbations at conservative amplitude (local flexibility)
    - NMA perturbations at large amplitude (conformational transitions)
    - Domain rigid-body perturbations at conservative scale
    - Domain rigid-body perturbations at large scale (state transitions)
    - Torsion angle perturbations (backbone φ/ψ sampling)
    - Include original structure as reference
    
    The multi-scale approach is critical for dual-state coverage:
    autoinhibited proteins undergo 10-30Å domain displacements, so we
    need both subtle and dramatic perturbations.
    
    Returns list of dicts with 'coords', 'method', 'perturbation_id'.
    """
    ensemble = []
    
    # Original structure (always included)
    ensemble.append({
        'coords': coords.copy(),
        'method': 'original',
        'perturbation_id': 'orig_0'
    })
    
    if fd_indices is not None and im_indices is not None:
        # MULTI-SCALE SAMPLING for dual-state exploration
        # Allocate: 20% conservative NMA, 10% large NMA, 25% conservative RB,
        #           25% large RB, 20% torsion
        n_nma_cons = max(2, n_conformations // 5)
        n_nma_large = max(2, n_conformations // 10)
        n_rb_cons = max(2, n_conformations // 4)
        n_rb_large = max(2, n_conformations // 4)
        n_torsion = max(2, n_conformations - n_nma_cons - n_nma_large - n_rb_cons - n_rb_large - 1)
        
        # Conservative NMA (local backbone flexibility)
        nma_coords = generate_nma_ensemble(
            coords, n_nma_cons, amplitude=2.0, n_modes=10, seed=seed)
        for i, c in enumerate(nma_coords):
            ensemble.append({
                'coords': c,
                'method': 'nma_conservative',
                'perturbation_id': f'nma_c_{i}'
            })
        
        # Large-amplitude NMA (conformational transitions)
        nma_large = generate_nma_ensemble(
            coords, n_nma_large, amplitude=6.0, n_modes=6, seed=seed + 10)
        for i, c in enumerate(nma_large):
            ensemble.append({
                'coords': c,
                'method': 'nma_large',
                'perturbation_id': f'nma_l_{i}'
            })
        
        # Conservative rigid-body (moderate domain rearrangement)
        rb_cons = generate_domain_rigid_body_ensemble(
            coords, fd_indices, im_indices, n_rb_cons,
            max_translation=5.0, max_rotation=20.0, seed=seed + 1)
        for i, c in enumerate(rb_cons):
            ensemble.append({
                'coords': c,
                'method': 'rigid_body_conservative',
                'perturbation_id': f'rb_c_{i}'
            })
        
        # Large-scale rigid-body (full state transitions: 10-30Å displacement)
        rb_large = generate_domain_rigid_body_ensemble(
            coords, fd_indices, im_indices, n_rb_large,
            max_translation=15.0, max_rotation=45.0, seed=seed + 2)
        for i, c in enumerate(rb_large):
            ensemble.append({
                'coords': c,
                'method': 'rigid_body_large',
                'perturbation_id': f'rb_l_{i}'
            })
        
        # Torsion angle perturbations
        if phi_psi:
            torsion_coords = generate_torsion_ensemble(
                coords, phi_psi, n_torsion, max_delta=20.0, seed=seed + 3)
            for i, c in enumerate(torsion_coords):
                ensemble.append({
                    'coords': c,
                    'method': 'torsion',
                    'perturbation_id': f'tor_{i}'
                })

        # Bridge conformations: scale with transition difficulty and TCI
        if coords_s2 is not None:
            tci_val = 0.0
            if quantum_bridge is not None:
                from ..quantum.dual_state_ising import compute_transition_complexity
                tci_val = compute_transition_complexity(quantum_bridge).get('tci', 0.0)
            base_frac = 0.20 + 0.20 * float(np.clip(transition_difficulty, 0.0, 1.0))
            base_frac += 0.10 * float(np.clip(tci_val, 0.0, 1.0))
            n_bridge = max(6, int(n_conformations * base_frac))
            n_manifold = max(3, n_bridge // 3)

            if quantum_bridge is not None:
                bridge_coords = generate_switch_guided_ensemble(
                    coords, coords_s2, fd_indices, im_indices,
                    quantum_bridge, n_conformations=n_bridge, seed=seed + 20,
                    common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2)
                manifold_coords = generate_manifold_bridge_ensemble(
                    coords, coords_s2, quantum_bridge,
                    n_conformations=n_manifold, seed=seed + 30,
                    common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2,
                )
            else:
                from ..quantum.dual_state_ising import (
                    build_dual_state_bridge, estimate_domain_motion_from_switches,
                )
                cm1 = compute_contact_map(coords, threshold=8.0)
                cm2 = compute_contact_map(coords_s2, threshold=8.0)
                bridge = build_dual_state_bridge(
                    sequence, cm1, cm2, fd_indices, im_indices, max_qubits=20)
                motion = estimate_domain_motion_from_switches(
                    bridge, coords, coords_s2, fd_indices, im_indices,
                    common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2)
                bridge_coords = generate_quantum_bridge_ensemble(
                    coords, coords_s2, fd_indices, im_indices,
                    motion, n_conformations=n_bridge, seed=seed + 20,
                    common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2)
                manifold_coords = generate_manifold_bridge_ensemble(
                    coords, coords_s2, bridge,
                    n_conformations=n_manifold, seed=seed + 30,
                    common_idx_s1=common_idx_s1, common_idx_s2=common_idx_s2,
                )
            for i, c in enumerate(bridge_coords):
                ensemble.append({
                    'coords': c,
                    'method': 'quantum_bridge',
                    'perturbation_id': f'qbridge_{i}'
                })
            for i, c in enumerate(manifold_coords):
                ensemble.append({
                    'coords': c,
                    'method': 'manifold_bridge',
                    'perturbation_id': f'mbridge_{i}'
                })
    else:
        # No domain info — use NMA only at multiple scales
        n_nma = n_conformations * 2 // 3
        n_extra = n_conformations - n_nma - 1
        
        nma_coords = generate_nma_ensemble(
            coords, n_nma, amplitude=2.0, seed=seed)
        for i, c in enumerate(nma_coords):
            ensemble.append({
                'coords': c,
                'method': 'nma',
                'perturbation_id': f'nma_{i}'
            })
        
        extra_coords = generate_nma_ensemble(
            coords, n_extra, amplitude=5.0, n_modes=15, seed=seed + 1)
        for i, c in enumerate(extra_coords):
            ensemble.append({
                'coords': c,
                'method': 'nma_extended',
                'perturbation_id': f'nma_ext_{i}'
            })
    
    return ensemble
