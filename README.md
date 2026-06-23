# QuantumFoldX: Dual-State Conformational Ensemble Analysis

[![Benchmark](https://img.shields.io/badge/Benchmark-16%20proteins-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Quantum](https://img.shields.io/badge/Quantum-Exact%20Ising%20Enumeration-orange)]()

## Overview

QuantumFoldX explores protein conformational landscapes using a Dual-State Ising Hamiltonian Bridge (DSIB). It targets **conformational state coverage** — whether an ensemble represents more than one biologically relevant state — rather than single-structure accuracy against AlphaFold 3.

AF3 typically predicts one conformation per protein. For autoinhibited kinases and other multi-state systems, that limitation is documented in the literature (Papageorgiou et al. 2025).

## Core idea: DSIB

When two experimental structures exist (state 1 and state 2):

1. Build Ising Hamiltonians H₁ and H₂ from each state's contact map on a shared qubit basis
2. Enumerate low-energy contact patterns along H(λ) = (1−λ)H₁ + λH₂
3. Identify **switch contacts** — qubits whose optimal value differs between H₁ and H₂
4. Generate **bridge conformations** via S1↔S2 interpolation and switch-contact-guided domain motion
5. Score the ensemble (contact overlap, geometry, imfdRMSD)

The Ising layer uses exact classical enumeration (≤20 qubits). No QPU is required at benchmark scale.

## Evidence (head-to-head v2 vs v3, n=16 autoinhibited proteins)

Run: `python benchmarks/compare_v2_v3_coverage.py`

| Condition | Dual-state coverage (TM>0.5) | Mean TM to state 2 |
|-----------|------------------------------|---------------------|
| v2 (baseline ensemble + VQE scoring) | 6/16 (37.5%) | 0.449 |
| **v3 (DSIB ensemble + v3 scoring)** | **11/16 (68.8%)** | **0.657** |
| v3 ensemble + v2 scoring | 11/16 (68.8%) | 0.657 |
| v2 ensemble + v3 scoring | 6/16 (37.5%) | 0.449 |

**Paired Wilcoxon (v3 vs v2 TM→S2): p = 0.006.** Dual coverage vs published AF3 autoinhibited rate (14%): 11/16, p < 0.001 (binomial).

The decomposition shows that **bridge ensemble generation** drives the gain, not the scoring layer alone. When the DSIB ensemble is used, v2 and v3 scorers perform identically. When the v2 ensemble is used, v3 scoring does not help.

### Hard subset (baseline S1↔S2 TM < 0.5, n=10)

| Condition | Dual-state coverage | Mean TM to state 2 |
|-----------|--------------------|--------------------|
| v2 | 0/10 (0%) | 0.229 |
| **v3 (DSIB)** | **5/10 (50%)** | **0.541** |

This is where the method matters most: proteins whose conformational states differ substantially.

### What did not work

QICESS v2's single-state VQE scoring did not beat random ranking (VQE 0.391 vs Random 0.394, p=0.25) and never matched exact diagonalization (0/16). We report that failure and replaced it.

Top-10 ranking ablation remains a weak proxy; dual-state coverage is the primary endpoint.

## Limitations

- Classically simulated Ising enumeration, not hardware quantum advantage
- AF3 numbers are from published benchmarks; we do not re-run AF3
- Both experimental structures must be available
- Easy cases (baseline TM > 0.5) are often covered by any reasonable ensemble
- Scoring refinements beyond ensemble generation show little additional benefit in our decomposition

## Installation

```bash
pip install -r requirements.txt
```

```bash
# Head-to-head v2 vs v3 evidence (recommended first)
python benchmarks/compare_v2_v3_coverage.py

# Full autoinhibited benchmark (v3)
python benchmarks/run_benchmark_v2_fast.py

# Ranking ablation (v3 vs v2 vs classical)
python benchmarks/ablation_v3_study.py

# Tests
python -m pytest tests/ -v
```

## Architecture

```
QuantumFoldX v3
├── PDB fetching (both states)
├── DSIB: H₁, H₂, λ-path enumeration, switch contacts
├── Ensemble: NMA + rigid-body + torsion + bridge conformations
├── Scoring: manifold overlap, state-2 contacts/geometry/imfdRMSD
└── Metrics: dual-state coverage, stratified by difficulty
```

## Project structure

```
src/quantum/dual_state_ising.py   # DSIB core
src/quantum/exact_ising.py        # Exact enumeration
src/scoring/qicess_v3.py          # Default scorer
src/scoring/geometry_utils.py     # Residue-aligned geometry scores
src/ensemble/conformational_sampler.py
benchmarks/compare_v2_v3_coverage.py  # Primary evidence script
results/evidence/                 # Comparison outputs
```

## References

- Papageorgiou et al. (2025) Communications Chemistry. https://doi.org/10.1038/s42004-025-01763-0
- Peng et al. (2025) Briefings in Bioinformatics. https://doi.org/10.1093/bib/bbaf170

## License

MIT
