# QuantumFoldX: Dual-State Conformational Ensemble Analysis

[![Benchmark](https://img.shields.io/badge/Benchmark-49%20proteins-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Quantum](https://img.shields.io/badge/Quantum-Exact%20Ising%20Enumeration-orange)]()

## Overview

QuantumFoldX explores protein conformational landscapes using a Dual-State Ising Hamiltonian Bridge (DSIB). It targets **conformational state coverage** — whether an ensemble represents more than one biologically relevant state — rather than single-structure accuracy against AlphaFold 3.

AF3 typically predicts one conformation per protein. For autoinhibited kinases, fold-switchers, and other multi-state systems, that limitation is documented in the literature (Papageorgiou et al. 2025; Ronish et al. 2024; Peng et al. 2025).

## Core idea: DSIB

When two experimental structures exist (state 1 and state 2):

1. Build Ising Hamiltonians H₁ and H₂ from each state's contact map on a shared qubit basis (20 qubits)
2. Enumerate low-energy contact patterns along H(λ) = (1−λ)H₁ + λH₂ on a fine λ-path (9 points)
3. Identify **switch contacts** — qubits whose optimal value differs between H₁ and H₂
4. Generate **bridge conformations** via:
   - Common-residue S1↔S2 interpolation
   - Switch-contact-guided domain motion
   - **Manifold bridges** — geometric interpolants at λ values encoded in the low-energy manifold
5. Score the ensemble (contact overlap, geometry, imfdRMSD)

The Ising layer uses exact classical enumeration (≤20 qubits). No QPU is required at benchmark scale.

### Transition Complexity Index (TCI)

DSIB computes a **Transition Complexity Index** from switch-contact density, the λ-path energy barrier, and manifold bridge span. TCI is reported alongside coverage; it does not yet predict per-protein gain on small samples.

## Benchmark datasets (49 proteins)

| Category | n | Source | Published AF3 dual-state rate |
|----------|---|--------|----------------------------|
| Autoinhibited | 24 | Papageorgiou et al. 2025 | 14% |
| Fold-switching | 12 | Ronish et al. 2024 | 7.6% |
| Multi-state | 13 | M-SADA (Peng et al. 2025) | 23.3% |

## Evidence (autoinhibited subset, n=16 head-to-head)

Run: `python benchmarks/compare_v2_v3_coverage.py`

| Condition | Dual-state coverage (TM>0.5) | Mean TM to state 2 |
|-----------|------------------------------|---------------------|
| v2 (baseline ensemble + VQE scoring) | 6/16 (37.5%) | 0.449 |
| **v3 (DSIB ensemble + v3 scoring)** | **16/16 (100%)** | **0.798** |
| v2 + bridge conformations only | 16/16 (100%) | 0.798 |

**Paired Wilcoxon (v3 vs v2 TM→S2): p = 0.0005.** Bridge-only ablation matches full v3 exactly (E = B).

Hard subset (baseline TM < 0.5): 0/10 → 10/10 dual coverage.

### Cross-dataset benchmark

Run: `python benchmarks/run_unified_coverage.py`

Evaluates all 49 proteins across autoinhibited, fold-switch, and multi-state categories with v2 vs v3 comparison. Results in `results/unified/`.

## What did not work

QICESS v2's single-state VQE scoring did not beat random ranking (VQE 0.391 vs Random 0.394, p=0.25). We report that failure and replaced it with DSIB bridge generation.

Top-10 ranking ablation remains a weak proxy; dual-state coverage is the primary endpoint.

## Limitations

- Classically simulated Ising enumeration, not hardware quantum advantage
- AF3 numbers are from published benchmarks; we do not re-run AF3
- Both experimental structures must be available
- Scoring refinements beyond ensemble generation show little additional benefit

## Installation

```bash
pip install -r requirements.txt
```

```bash
# Unified cross-dataset benchmark (49 proteins)
python benchmarks/run_unified_coverage.py

# Autoinhibited head-to-head v2 vs v3
python benchmarks/compare_v2_v3_coverage.py

# Full autoinhibited benchmark
python benchmarks/run_benchmark_v2_fast.py

# All categories + ablation
python benchmarks/run_all_benchmarks.py

# Tests
python -m pytest tests/ -v
```

## Architecture

```
QuantumFoldX v3 (DSIB)
├── PDB fetching (both states, chain fallback)
├── DSIB: H₁, H₂, 9-point λ-path, switch contacts, TCI
├── Ensemble: NMA + rigid-body + torsion + quantum_bridge + manifold_bridge
├── Scoring: manifold overlap, state-2 contacts/geometry/imfdRMSD
└── Metrics: dual-state coverage, stratified by category and difficulty
```

## Project structure

```
src/quantum/dual_state_ising.py   # DSIB core + TCI
src/scoring/qicess_v3.py          # create_dsib_scorer() factory
src/ensemble/conformational_sampler.py
benchmarks/run_unified_coverage.py  # 49-protein cross-dataset benchmark
benchmarks/compare_v2_v3_coverage.py
configs/benchmark_dataset.py      # 49 curated targets
results/unified/                  # Cross-dataset outputs
results/evidence/                 # Autoinhibited head-to-head
```

## References

- Papageorgiou et al. (2025) Communications Chemistry. https://doi.org/10.1038/s42004-025-01763-0
- Ronish et al. (2024) Nature Communications. https://doi.org/10.1038/s41467-024-51801-z
- Peng et al. (2025) Briefings in Bioinformatics. https://doi.org/10.1093/bib/bbaf170

## License

MIT
