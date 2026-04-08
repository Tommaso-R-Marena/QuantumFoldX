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


def generate_hybrid_ensemble(coords: np.ndarray,
                              sequence: str,
                              fd_indices: List[int] = None,
                              im_indices: List[int] = None,
                              n_conformations: int = 50,
                              seed: int = 42) -> List[Dict]:
    """
    Generate comprehensive ensemble using multiple methods at multiple scales.
    
    Combines:
    - NMA perturbations at conservative amplitude (local flexibility)
    - NMA perturbations at large amplitude (conformational transitions)
    - Domain rigid-body perturbations at conservative scale
    - Domain rigid-body perturbations at large scale (state transitions)
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
        # Allocate: 25% conservative NMA, 15% large NMA, 30% conservative RB, 30% large RB
        n_nma_cons = max(2, n_conformations // 4)
        n_nma_large = max(2, n_conformations * 15 // 100)
        n_rb_cons = max(2, n_conformations * 3 // 10)
        n_rb_large = n_conformations - n_nma_cons - n_nma_large - n_rb_cons - 1
        
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
