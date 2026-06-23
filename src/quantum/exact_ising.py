"""
exact_ising.py — Exact classical enumeration for Ising Hamiltonians.

At ≤20 qubits the ground state and low-energy manifold are found by evaluating
all 2^n computational basis states. This is the honest baseline for small
problems and the production solver used by QICESS v3.

For larger encodings the same API falls back to greedy bit-flip annealing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml

logger = logging.getLogger(__name__)


@dataclass
class IsingTerm:
    """Single Ising term: coeff * (product of Z on wires)."""
    coeff: float
    wires: Tuple[int, ...]  # empty tuple = identity offset


@dataclass
class IsingModel:
    """Sparse Ising model H = offset + Σ coeff * Π Z_w."""
    n_qubits: int
    terms: List[IsingTerm] = field(default_factory=list)
    offset: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def energy(self, bitstring: str) -> float:
        """Energy of a computational basis state."""
        e = self.offset
        for term in self.terms:
            sign = 1
            for w in term.wires:
                sign *= (-1) ** int(bitstring[w])
            e += term.coeff * sign
        return e

    def to_pennylane(self) -> qml.Hamiltonian:
        coeffs, ops = [], []
        if self.offset != 0.0:
            coeffs.append(self.offset)
            ops.append(qml.Identity(0))
        for term in self.terms:
            if not term.wires:
                continue
            if len(term.wires) == 1:
                ops.append(qml.PauliZ(term.wires[0]))
            else:
                prod = qml.PauliZ(term.wires[0])
                for w in term.wires[1:]:
                    prod = prod @ qml.PauliZ(w)
                ops.append(prod)
            coeffs.append(term.coeff)
        if not coeffs:
            coeffs, ops = [0.0], [qml.Identity(0)]
        return qml.Hamiltonian(coeffs, ops)

    @classmethod
    def from_pennylane(cls, H: qml.Hamiltonian, n_qubits: int,
                       metadata: Optional[Dict] = None) -> "IsingModel":
        terms: List[IsingTerm] = []
        offset = 0.0
        for coeff, op in zip(H.coeffs, H.ops):
            c = float(coeff)
            if isinstance(op, qml.Identity):
                offset += c
            elif isinstance(op, qml.PauliZ):
                terms.append(IsingTerm(c, (op.wires[0],)))
            elif isinstance(op, qml.ops.op_math.Prod):
                wires = tuple(
                    sub.wires[0] for sub in op.operands if isinstance(sub, qml.PauliZ)
                )
                terms.append(IsingTerm(c, wires))
            else:
                try:
                    wires = tuple(op.wires)
                    terms.append(IsingTerm(c, wires))
                except Exception:
                    offset += c
        return cls(n_qubits=n_qubits, terms=terms, offset=offset,
                   metadata=metadata or {})


def interpolate_ising(H1: IsingModel, H2: IsingModel, lam: float) -> IsingModel:
    """Linear interpolation H(λ) = (1-λ) H₁ + λ H₂ on aligned qubit indices."""
    assert H1.n_qubits == H2.n_qubits
    lam = float(np.clip(lam, 0.0, 1.0))

    term_map: Dict[Tuple[int, ...], float] = {}
    for model, weight in ((H1, 1.0 - lam), (H2, lam)):
        for term in model.terms:
            key = term.wires
            term_map[key] = term_map.get(key, 0.0) + weight * term.coeff

    terms = [IsingTerm(c, w) for w, c in term_map.items() if abs(c) > 1e-12]
    offset = (1.0 - lam) * H1.offset + lam * H2.offset
    meta = {**H1.metadata, 'lambda': lam}
    return IsingModel(n_qubits=H1.n_qubits, terms=terms, offset=offset, metadata=meta)


def exact_ground_state(model: IsingModel, top_k: int = 5) -> Dict:
    """Brute-force ground state for n_qubits ≤ 22."""
    n = model.n_qubits
    n_states = 2 ** n
    energies = np.empty(n_states, dtype=np.float64)

    for idx in range(n_states):
        bs = format(idx, f'0{n}b')
        energies[idx] = model.energy(bs)

    order = np.argsort(energies)
    ground_idx = int(order[0])
    ground_bs = format(ground_idx, f'0{n}b')

    top = []
    for idx in order[:top_k]:
        bs = format(int(idx), f'0{n}b')
        top.append((bs, float(energies[idx])))

    return {
        'ground_energy': float(energies[ground_idx]),
        'ground_bitstring': ground_bs,
        'top_bitstrings': top,
        'method': 'Exact Enumeration',
        'n_states_evaluated': n_states,
    }


def exact_low_energy_manifold(model: IsingModel, delta_e: float = 0.5,
                               max_states: int = 32) -> List[Dict]:
    """All basis states within delta_e of the ground state."""
    n = model.n_qubits
    n_states = 2 ** n
    energies = np.empty(n_states, dtype=np.float64)
    bitstrings: List[str] = []

    for idx in range(n_states):
        bs = format(idx, f'0{n}b')
        bitstrings.append(bs)
        energies[idx] = model.energy(bs)

    e_min = float(np.min(energies))
    mask = energies <= e_min + delta_e
    indices = np.where(mask)[0]
    if len(indices) > max_states:
        indices = indices[np.argsort(energies[indices])[:max_states]]

    boltzmann = np.exp(-(energies[indices] - e_min))
    Z = float(np.sum(boltzmann))
    results = []
    for idx in indices:
        bs = bitstrings[int(idx)]
        results.append({
            'bitstring': bs,
            'energy': float(energies[idx]),
            'boltzmann_weight': float(np.exp(-(energies[idx] - e_min)) / max(Z, 1e-12)),
        })
    results.sort(key=lambda x: x['energy'])
    return results


def greedy_anneal(model: IsingModel, n_restarts: int = 8,
                  max_steps: int = 200, seed: int = 42) -> Dict:
    """Greedy single-bit-flip annealing for models too large for exact enum."""
    rng = np.random.default_rng(seed)
    n = model.n_qubits
    best_e = float('inf')
    best_bs = '0' * n

    for _ in range(n_restarts):
        bs_list = list(rng.integers(0, 2, size=n, dtype=int))
        bs = ''.join(str(b) for b in bs_list)
        e = model.energy(bs)

        for _ in range(max_steps):
            improved = False
            order = rng.permutation(n)
            for q in order:
                flipped = list(bs)
                flipped[q] = '1' if flipped[q] == '0' else '0'
                bs_new = ''.join(flipped)
                e_new = model.energy(bs_new)
                if e_new < e - 1e-9:
                    bs, e = bs_new, e_new
                    improved = True
            if not improved:
                break

        if e < best_e:
            best_e, best_bs = e, bs

    return {
        'ground_energy': best_e,
        'ground_bitstring': best_bs,
        'top_bitstrings': [(best_bs, best_e)],
        'method': 'Greedy Annealing',
        'n_states_evaluated': n_restarts * max_steps * n,
    }


def solve_ising(model: IsingModel, top_k: int = 5,
                exact_limit: int = 22) -> Dict:
    """Exact enumeration when tractable, otherwise greedy annealing."""
    if model.n_qubits <= exact_limit:
        return exact_ground_state(model, top_k=top_k)
    logger.warning(
        "Ising model has %d qubits; using greedy annealing (exact limit=%d)",
        model.n_qubits, exact_limit,
    )
    return greedy_anneal(model)


def bitstring_agreement(bs_a: str, bs_b: str) -> float:
    """Fraction of matching bits."""
    if not bs_a or not bs_b:
        return 0.0
    n = min(len(bs_a), len(bs_b))
    return sum(a == b for a, b in zip(bs_a[:n], bs_b[:n])) / n


def bitstring_overlap(conf_bits: Sequence[int], target_bs: str,
                      weights: Optional[Sequence[float]] = None) -> float:
    """Weighted agreement between a contact pattern and a target bitstring."""
    if not target_bs:
        return 0.0
    n = min(len(conf_bits), len(target_bs))
    if weights is None:
        weights = [1.0] * n
    total_w = 0.0
    agree = 0.0
    for i in range(n):
        w = weights[i] if i < len(weights) else 1.0
        total_w += w
        if int(conf_bits[i]) == int(target_bs[i]):
            agree += w
    return agree / max(total_w, 1e-8)
