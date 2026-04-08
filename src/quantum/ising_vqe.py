"""
ising_vqe.py — Quantum Ising Hamiltonian solver for protein contact energy landscapes.

Uses VQE (Variational Quantum Eigensolver) to find low-energy configurations of
a spin-glass Hamiltonian derived from Miyazawa-Jernigan statistical potentials.

The key advantage over classical greedy/SA methods: VQE naturally explores
quantum superposition states, finding multiple low-energy basins through
the variational landscape rather than getting trapped in local minima.

⚠ ALL QUANTUM CIRCUITS ARE CLASSICALLY SIMULATED using PennyLane lightning.qubit.
No quantum processing unit (QPU) hardware was used.
"""

import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Miyazawa-Jernigan contact potentials (20x20 matrix)
# Rows/cols: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
MJ_POTENTIALS = np.array([
    [-1.40,-0.73,-0.58,-0.64,-2.13,-0.72,-0.62,-0.47,-0.77,-1.95,-2.05,-0.64,-1.76,-2.21,-0.48,-0.53,-0.64,-1.99,-1.68,-1.72],
    [-0.73,-1.14,-0.55,-0.61,-1.30,-0.71,-0.64,-0.21,-0.80,-1.14,-1.19,-0.55,-0.80,-0.92,-0.61,-0.58,-0.69,-1.02,-0.82,-0.96],
    [-0.58,-0.55,-0.90,-0.75,-0.99,-0.85,-0.68,-0.47,-0.71,-0.87,-0.94,-0.62,-0.78,-0.93,-0.42,-0.66,-0.69,-1.07,-0.97,-0.83],
    [-0.64,-0.61,-0.75,-0.90,-0.94,-0.81,-0.77,-0.44,-0.66,-0.91,-0.98,-0.60,-0.74,-0.89,-0.55,-0.73,-0.78,-0.96,-0.94,-0.88],
    [-2.13,-1.30,-0.99,-0.94,-4.11,-1.12,-1.00,-1.22,-1.40,-2.37,-2.53,-1.29,-2.15,-2.54,-1.14,-1.27,-1.42,-2.94,-2.52,-2.22],
    [-0.72,-0.71,-0.85,-0.81,-1.12,-0.87,-0.82,-0.49,-0.79,-1.06,-1.16,-0.65,-0.89,-1.00,-0.44,-0.64,-0.72,-1.15,-1.07,-0.94],
    [-0.62,-0.64,-0.68,-0.77,-1.00,-0.82,-0.73,-0.40,-0.66,-0.98,-1.07,-0.60,-0.80,-0.91,-0.53,-0.64,-0.72,-1.03,-0.99,-0.87],
    [-0.47,-0.21,-0.47,-0.44,-1.22,-0.49,-0.40,-0.26,-0.43,-0.86,-0.90,-0.29,-0.73,-0.99,-0.24,-0.49,-0.46,-0.84,-0.84,-0.76],
    [-0.77,-0.80,-0.71,-0.66,-1.40,-0.79,-0.66,-0.43,-1.45,-1.30,-1.37,-0.72,-1.17,-1.46,-0.59,-0.72,-0.82,-1.80,-1.59,-1.17],
    [-1.95,-1.14,-0.87,-0.91,-2.37,-1.06,-0.98,-0.86,-1.30,-3.12,-3.30,-1.08,-2.72,-3.39,-1.07,-1.13,-1.41,-3.45,-2.89,-2.80],
    [-2.05,-1.19,-0.94,-0.98,-2.53,-1.16,-1.07,-0.90,-1.37,-3.30,-3.56,-1.17,-2.85,-3.65,-1.10,-1.20,-1.49,-3.62,-3.01,-2.96],
    [-0.64,-0.55,-0.62,-0.60,-1.29,-0.65,-0.60,-0.29,-0.72,-1.08,-1.17,-0.50,-0.80,-0.96,-0.55,-0.65,-0.73,-1.13,-0.91,-0.91],
    [-1.76,-0.80,-0.78,-0.74,-2.15,-0.89,-0.80,-0.73,-1.17,-2.72,-2.85,-0.80,-3.22,-3.30,-0.87,-1.02,-1.27,-3.47,-2.75,-2.54],
    [-2.21,-0.92,-0.93,-0.89,-2.54,-1.00,-0.91,-0.99,-1.46,-3.39,-3.65,-0.96,-3.30,-4.36,-0.99,-1.23,-1.57,-3.89,-3.30,-3.10],
    [-0.48,-0.61,-0.42,-0.55,-1.14,-0.44,-0.53,-0.24,-0.59,-1.07,-1.10,-0.55,-0.87,-0.99,-0.85,-0.47,-0.58,-1.24,-0.98,-0.97],
    [-0.53,-0.58,-0.66,-0.73,-1.27,-0.64,-0.64,-0.49,-0.72,-1.13,-1.20,-0.65,-1.02,-1.23,-0.47,-0.53,-0.71,-1.41,-1.19,-1.00],
    [-0.64,-0.69,-0.69,-0.78,-1.42,-0.72,-0.72,-0.46,-0.82,-1.41,-1.49,-0.73,-1.27,-1.57,-0.58,-0.71,-0.82,-1.74,-1.45,-1.23],
    [-1.99,-1.02,-1.07,-0.96,-2.94,-1.15,-1.03,-0.84,-1.80,-3.45,-3.62,-1.13,-3.47,-3.89,-1.24,-1.41,-1.74,-5.06,-3.58,-3.26],
    [-1.68,-0.82,-0.97,-0.94,-2.52,-1.07,-0.99,-0.84,-1.59,-2.89,-3.01,-0.91,-2.75,-3.30,-0.98,-1.19,-1.45,-3.58,-3.40,-2.75],
    [-1.72,-0.96,-0.83,-0.88,-2.22,-0.94,-0.87,-0.76,-1.17,-2.80,-2.96,-0.91,-2.54,-3.10,-0.97,-1.00,-1.23,-3.26,-2.75,-2.68],
])

MJ_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
MJ_AA_TO_IDX = {aa: i for i, aa in enumerate(MJ_AA_ORDER)}


def build_ising_hamiltonian(sequence: str, contact_map: np.ndarray,
                            interface_residues: Tuple[List[int], List[int]] = None):
    """
    Build an Ising spin-glass Hamiltonian from protein sequence and contact map.
    
    H = -∑_{i<j} J_{ij} Z_i Z_j + ∑_i h_i Z_i
    
    where:
        J_{ij} = MJ potential between residues i,j (weighted by contact probability)
        h_i = local field biasing interface residues
    
    For autoinhibited proteins, interface_residues = (FD_residues, IM_residues)
    specifies the functional domain and inhibitory module residue indices.
    
    Returns:
        pennylane.Hamiltonian: The Ising Hamiltonian
        n_qubits: Number of qubits (= number of interface-relevant residue pairs)
    """
    n = len(sequence)
    
    # Build coupling matrix J_{ij} from MJ potentials × contact probability
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            aa_i = MJ_AA_TO_IDX.get(sequence[i], 0)
            aa_j = MJ_AA_TO_IDX.get(sequence[j], 0)
            J[i, j] = MJ_POTENTIALS[aa_i, aa_j] * contact_map[i, j]
    
    # If interface residues specified, boost inter-domain contacts
    if interface_residues is not None:
        fd_res, im_res = interface_residues
        boost = np.ones((n, n))
        n_boosted = 0
        for i in fd_res:
            for j in im_res:
                if i < n and j < n:
                    ii, jj = min(i, j), max(i, j)
                    boost[ii, jj] = 3.0  # 3x weight for inter-domain
                    n_boosted += 1
        if n_boosted > 0:
            J = J * boost
    
    # Select top-k couplings for qubit encoding (limit to manageable circuit size)
    # Sort by absolute coupling strength
    couplings = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 0.01:  # Only significant contacts
                couplings.append((i, j, J[i, j]))
    
    couplings.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Map top contacts to qubits
    # Each qubit represents whether a contact pair is "active" in this conformation
    max_qubits = min(len(couplings), 16)  # Cap at 16 qubits for tractability
    selected = couplings[:max_qubits]
    
    if not selected:
        logger.warning("No significant contacts found for Ising Hamiltonian")
        return None, 0
    
    n_qubits = len(selected)
    
    # Build PennyLane Hamiltonian
    coeffs = []
    ops = []
    
    # ZZ couplings between qubit pairs that share a residue
    for qi in range(n_qubits):
        for qj in range(qi + 1, n_qubits):
            ri1, ri2, ji = selected[qi]
            rj1, rj2, jj = selected[qj]
            
            # Coupling if they share a residue (cooperative/anticooperative contacts)
            if ri1 == rj1 or ri1 == rj2 or ri2 == rj1 or ri2 == rj2:
                coupling = -0.5 * (ji + jj) / 2.0  # Average coupling
                if abs(coupling) > 0.001:
                    coeffs.append(coupling)
                    ops.append(qml.PauliZ(qi) @ qml.PauliZ(qj))
    
    # Single-qubit Z terms (local fields from contact energies)
    for qi in range(n_qubits):
        _, _, ji = selected[qi]
        coeffs.append(-ji / 2.0)
        ops.append(qml.PauliZ(qi))
    
    # Add identity term for energy offset
    if not coeffs:
        coeffs.append(0.0)
        ops.append(qml.Identity(0))
    
    H = qml.Hamiltonian(coeffs, ops)
    
    return H, n_qubits, selected


class IsingVQESolver:
    """
    VQE solver for protein Ising Hamiltonian.
    
    Uses hardware-efficient ansatz with PennyLane lightning.qubit simulator.
    Finds multiple low-energy configurations for ensemble generation.
    
    ⚠ CLASSICALLY SIMULATED — no QPU hardware used.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 4, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        self.rng = np.random.default_rng(seed)
        
    def _ansatz(self, params, wires):
        """Hardware-efficient ansatz: Ry-Rz layers + CNOT entangling."""
        n = len(wires)
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(n):
                qml.RY(params[layer, i, 0], wires=wires[i])
                qml.RZ(params[layer, i, 1], wires=wires[i])
            # Entangling layer (linear connectivity)
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            # Circular connectivity for last-to-first
            if n > 2:
                qml.CNOT(wires=[wires[-1], wires[0]])
    
    def solve(self, hamiltonian, n_restarts: int = 5,
              max_steps: int = 200, lr: float = 0.1) -> Dict:
        """
        Run VQE to find ground state of the Ising Hamiltonian.
        
        Uses multiple random restarts to avoid local minima.
        
        Returns:
            dict with:
                'ground_energy': float — lowest energy found
                'ground_params': np.array — optimal circuit parameters
                'ground_bitstring': str — most likely computational basis state
                'energies': list — energy trajectory
                'all_solutions': list — all restart solutions sorted by energy
        """
        wires = list(range(self.n_qubits))
        
        @qml.qnode(self.dev, diff_method="adjoint")
        def cost_fn(params):
            self._ansatz(params, wires)
            return qml.expval(hamiltonian)
        
        @qml.qnode(self.dev)
        def sample_fn(params):
            self._ansatz(params, wires)
            return qml.probs(wires=wires)
        
        all_solutions = []
        
        for restart in range(n_restarts):
            # Random initialization (different seed each restart)
            params = pnp.array(self.rng.uniform(-np.pi, np.pi, 
                                      size=(self.n_layers, self.n_qubits, 2)),
                              requires_grad=True)
            
            opt = qml.AdamOptimizer(stepsize=lr)
            energies = []
            best_energy = float('inf')
            best_params = params.copy()
            patience_counter = 0
            
            for step in range(max_steps):
                params, energy = opt.step_and_cost(cost_fn, params)
                energy_val = float(energy)
                energies.append(energy_val)
                
                if energy_val < best_energy - 1e-6:
                    best_energy = energy_val
                    best_params = pnp.array(params, requires_grad=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter > 30:
                    break
            
            # Get probability distribution
            probs = sample_fn(best_params)
            probs_np = np.array(probs)
            top_states = np.argsort(probs_np)[::-1][:5]
            
            bitstrings = []
            for state_idx in top_states:
                bs = format(state_idx, f'0{self.n_qubits}b')
                bitstrings.append((bs, float(probs_np[state_idx])))
            
            all_solutions.append({
                'energy': best_energy,
                'params': best_params,
                'bitstrings': bitstrings,
                'trajectory': energies,
                'n_steps': len(energies)
            })
            
            logger.debug(f"  Restart {restart + 1}/{n_restarts}: E = {best_energy:.4f} "
                        f"({len(energies)} steps)")
        
        # Sort by energy
        all_solutions.sort(key=lambda x: x['energy'])
        
        best = all_solutions[0]
        return {
            'ground_energy': best['energy'],
            'ground_params': best['params'],
            'ground_bitstring': best['bitstrings'][0][0] if best['bitstrings'] else None,
            'top_bitstrings': best['bitstrings'],
            'energies': best['trajectory'],
            'all_solutions': all_solutions,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'method': 'VQE [CLASSICALLY SIMULATED]'
        }
    
    def generate_ensemble(self, hamiltonian, n_conformations: int = 10,
                          temperature: float = 1.0) -> List[Dict]:
        """
        Generate a diverse ensemble of low-energy configurations.
        
        Uses VQE with different initializations to find multiple
        low-energy basins, mimicking quantum sampling across
        the energy landscape.
        
        This is the key advantage: classical methods (SA, greedy) tend to
        find the same local minimum repeatedly. VQE with random 
        initializations explores different regions of Hilbert space.
        
        Returns: list of dicts with 'bitstring', 'energy', 'probability'
        """
        wires = list(range(self.n_qubits))
        
        @qml.qnode(self.dev, diff_method="adjoint")
        def cost_fn(params):
            self._ansatz(params, wires)
            return qml.expval(hamiltonian)
        
        @qml.qnode(self.dev)
        def sample_fn(params):
            self._ansatz(params, wires)
            return qml.probs(wires=wires)
        
        ensemble = []
        seen_bitstrings = set()
        
        for trial in range(n_conformations * 3):  # Extra trials for diversity
            params = pnp.array(self.rng.uniform(-np.pi, np.pi,
                                      size=(self.n_layers, self.n_qubits, 2)),
                              requires_grad=True)
            
            opt = qml.AdamOptimizer(stepsize=0.1)
            
            # Shorter optimization for ensemble diversity
            for step in range(100):
                params = opt.step(cost_fn, params)
            
            energy = float(cost_fn(params))
            probs = np.array(sample_fn(params))
            
            # Extract top bitstring
            top_idx = np.argmax(probs)
            bs = format(top_idx, f'0{self.n_qubits}b')
            
            if bs not in seen_bitstrings:
                seen_bitstrings.add(bs)
                ensemble.append({
                    'bitstring': bs,
                    'energy': energy,
                    'probability': float(probs[top_idx]),
                    'params': pnp.array(params, requires_grad=False)
                })
            
            if len(ensemble) >= n_conformations:
                break
        
        # Sort by energy and assign Boltzmann weights
        ensemble.sort(key=lambda x: x['energy'])
        
        if ensemble:
            e_min = ensemble[0]['energy']
            Z = sum(np.exp(-(e['energy'] - e_min) / temperature) for e in ensemble)
            for e in ensemble:
                e['boltzmann_weight'] = np.exp(-(e['energy'] - e_min) / temperature) / Z
        
        return ensemble


def compute_ising_energy_classical(sequence: str, contact_map: np.ndarray,
                                    spin_config: np.ndarray) -> float:
    """
    Classical computation of Ising energy for comparison.
    
    E = -∑_{i<j} J_{ij} s_i s_j
    
    where s_i ∈ {-1, +1} from spin_config and J_{ij} from MJ potentials.
    """
    n = len(sequence)
    energy = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            if contact_map[i, j] > 0:
                aa_i = MJ_AA_TO_IDX.get(sequence[i], 0)
                aa_j = MJ_AA_TO_IDX.get(sequence[j], 0)
                J_ij = MJ_POTENTIALS[aa_i, aa_j]
                energy -= J_ij * spin_config[i] * spin_config[j] * contact_map[i, j]
    
    return energy
