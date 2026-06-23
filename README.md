# QuantumFoldX: Dual-State Conformational Ensemble Analysis

[![Benchmark](https://img.shields.io/badge/Benchmark-16%20proteins-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Quantum](https://img.shields.io/badge/Quantum-Exact%20Ising%20Enumeration-orange)]()

## Overview

QuantumFoldX explores protein conformational landscapes using hybrid quantum-classical methods. It does not try to beat AlphaFold 3 (AF3) on single-structure accuracy — AF3's training data makes that a difficult target. Instead, it focuses on **conformational state coverage**: whether an ensemble can represent more than one biologically relevant state.

AF3 typically predicts one conformation per protein. For autoinhibited kinases, fold-switching proteins, and other multi-state systems, that limitation is well documented in the literature.

## QICESS v3 (current default)

**QICESS v2 did not hold up in ablation.** Single-state VQE scoring did not beat random ranking (VQE 0.391 vs Random 0.394, p=0.25), and the variational solver never matched exact diagonalization (0/16 proteins). We kept those results and rebuilt the scoring layer.

**QICESS v3** uses a Dual-State Ising Hamiltonian Bridge (DSIB):

1. Build H₁ and H₂ from state 1 and state 2 contact maps on a shared qubit basis
2. Enumerate low-energy states along H(λ) = (1−λ)H₁ + λH₂ for several λ values
3. Identify switch contacts (qubits whose optimal value differs between H₁ and H₂)
4. Generate bridge conformations via S1↔S2 interpolation and switch-guided domain motion
5. Score conformations by manifold overlap, state-2 contact agreement, switch-contact satisfaction, and (when available) inter-domain TM-score to state 2

The Ising layer is classically simulated: exact enumeration for ≤20 qubits, greedy annealing beyond that. No QPU is required for the current benchmark scale.

### Reported results (mixed; read the caveats)

| Metric | QuantumFoldX (v2 era) | AlphaFold 3 | Notes |
|--------|----------------------|-------------|-------|
| Dual-state coverage (autoinhibited, TM>0.5) | 37.5% (6/16) | 14% (published) | p=0.017 vs AF3 autoinhibited rate |
| Ensemble RMSD vs state 2 | improved on all proteins tested | N/A | p=0.00006 (Wilcoxon) |
| Hard cases (baseline TM<0.5) | often still fails | often fails | perturbation alone is not enough |

Early v3 checks on individual proteins showed gains on some hard targets (e.g. FYN TM→S2 0.098→0.456, BRAF dual coverage unlocked in spot checks), but **ranking ablation on top-10 TM is not clearly better than v2 or random across the full set**. We report both. Run `python benchmarks/ablation_v3_study.py` to reproduce on your machine.

### Limitations

- Exact Ising enumeration is classical simulation, not hardware quantum advantage
- AF3 comparison numbers come from published benchmarks; we do not re-run AF3
- Both experimental structures must be available — this is conformational exploration, not de novo folding
- Easy dual-state cases (baseline TM>0.5) dominate coverage statistics; hard cases remain hard
- DisorderNet and ChiralFold (sibling projects) use classical methods, not quantum circuits

## Architecture

```
QuantumFoldX v3
├── PDB fetching (RCSB) — both states required
├── Dual-State Ising Hamiltonian Bridge (DSIB)
│   ├── Shared contact qubit basis (inter-domain + switch contacts)
│   ├── H₁, H₂ from Miyazawa–Jernigan potentials
│   ├── λ-path exact enumeration
│   └── Switch-contact extraction
├── Ensemble generation
│   ├── NMA (2 Å and 6 Å amplitude)
│   ├── Domain rigid-body perturbation (conservative + large scale)
│   ├── Torsion sampling
│   └── Bridge conformations (interpolation + switch-guided motion)
├── QICESS v3 scoring
│   ├── Manifold overlap, state-2 contacts, switch satisfaction
│   ├── State-2 geometry (inter-domain TM)
│   └── Ramachandran, compactness, contact order, inter-domain density
└── Metrics + statistics (RMSD, TM, dual-state coverage, ablation)
```

## v2 → v3 changes

| v2 issue | v3 approach |
|----------|-------------|
| VQE on one reference structure | H₁ + H₂ from both states |
| VQE missed ground state (0/16) | Exact enumeration where tractable |
| Ranking ≈ random | Dual-state contact + geometry terms (mixed results) |
| Ensemble started from state 1 only | Bridge conformations toward state 2 |

## Installation

```bash
pip install -r requirements.txt
```

```bash
# Autoinhibited dual-state benchmark (v3)
python benchmarks/run_benchmark_v2_fast.py

# Full suite
python benchmarks/run_all_benchmarks.py

# v3 vs v2 vs classical ablation
python benchmarks/ablation_v3_study.py

# Legacy v2 VQE ablation
python benchmarks/ablation_study.py

# Tests
python -m pytest tests/ -v
```

## Project structure

```
QuantumFoldX/
├── src/quantum/
│   ├── exact_ising.py           # Exact enumeration + interpolation
│   ├── dual_state_ising.py      # Dual-State Ising Hamiltonian Bridge
│   ├── ising_vqe.py             # Legacy VQE (ablation only)
│   └── qaoa_rotamer.py
├── src/scoring/
│   ├── qicess_v3.py             # Default scorer
│   └── qicess_v2.py             # Legacy scorer
├── src/ensemble/conformational_sampler.py
├── src/data/pdb_fetcher.py
├── benchmarks/
└── tests/
```

## Related work

Useful alongside:

- **[DisorderNet](https://github.com/Tommaso-R-Marena/DisorderNet)** — disorder prediction (classical)
- **[ChiralFold](https://github.com/Tommaso-R-Marena/ChiralFold)** — stereochemistry checks (geometric)

## References

```
- Papageorgiou et al. (2025) Communications Chemistry. https://doi.org/10.1038/s42004-025-01763-0
- Peng et al. (2025) Briefings in Bioinformatics. https://doi.org/10.1093/bib/bbaf170
- Ronish et al. (2024) Nature Communications. https://doi.org/10.1038/s41467-024-51801-z
- Jumper et al. (2021) Nature — AlphaFold2
```

## License

MIT
