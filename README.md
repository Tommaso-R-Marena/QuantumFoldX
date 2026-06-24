# QuantumFoldX: Dual-State Conformational Ensemble Analysis

[![Benchmark](https://img.shields.io/badge/Benchmark-49%20proteins-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Method](https://img.shields.io/badge/Method-Conformational%20bridging%20%2B%20Ising--guided%20sampling-orange)]()

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

## Evidence: oracle vs blind (n=49)

Run: `python benchmarks/run_blind_coverage.py`

Diagnostic (Step 1, bridge-source breakdown): `python benchmarks/dsib_ablation_diagnostic.py`

| Setting | Dual coverage (n=49) | Mean TM→S2 | vs AF3 (15%) |
|---------|----------------------|------------|--------------|
| BLIND (state 1 only at generation) | 11/49 (22.4%) | 0.331 | p=0.106 (not significant) |
| ORACLE (both states known at generation) | 48/49 (98.0%) | 0.881 | p~1.06e-38 |

ORACLE vs BLIND: Wilcoxon p=1.20e-09.

BLIND performance matches the existing v2 baseline. The 98% ORACLE figure reflects conformational interpolation toward an already-known target structure, not de novo prediction. This benchmark cannot demonstrate predictive or generative capability, because it requires the answer (state 2) as input.

### By category

| Category | BLIND dual | ORACLE dual | BLIND mean TM→S2 | ORACLE mean TM→S2 |
|----------|------------|-------------|------------------|-------------------|
| Autoinhibited (24) | 6/24 | 24/24 | 0.341 | 0.903 |
| Fold-switch (12) | 1/12 | 11/12 | 0.250 | 0.796 |
| Multi-state (13) | 4/13 | 13/13 | 0.388 | 0.919 |

Hard subset (baseline TM < 0.5, n=38): BLIND 0/38 dual coverage → ORACLE 37/38 dual coverage.

### Bridge-source diagnostic (Step 1, n=6 proteins across all categories)

On SRC, WAS, FYN, CLIC1, REV, and PDGFRB, the max-TM winner in the ORACLE ensemble was a bridge member on 6/6 proteins. By source: **common_residue_interp** won 5/6; **manifold_bridge** won 1/6 (PDGFRB); **switch_contact_rigid** won 0/6. Switch-contact-guided rigid-body motion (the Ising H₁/H₂ output) generates conformations but did not produce the max-TM winner in this sample. Condition E (v2 + bridge only) matched condition B (full v3) on 6/6 because coverage is max TM over the full ensemble, not a ranked selection step.

## What did not work

QICESS v2's single-state VQE scoring did not beat random ranking (VQE 0.391 vs Random 0.394, p=0.25) and never matched exact diagonalization (0/16). We report that failure and replaced it.

The Ising switch-contact machinery generates conformations but does not produce the max-TM winner in the Step 1 diagnostic sample (switch_contact_rigid won 0/6; common_residue_interp won 5/6; manifold_bridge won 1/6 and still interpolates toward state 2 coordinates). DSIB's apparent success under the ORACLE setting is attributable to direct S1↔S2 interpolation, confirmed by the BLIND/ORACLE gap: BLIND 11/49 (22.4%) vs ORACLE 48/49 (98.0%), Wilcoxon p=1.20e-09.

Top-10 ranking ablation remains a weak proxy; dual-state coverage is the primary endpoint.

## Limitations

- Classically simulated Ising enumeration, not hardware quantum advantage
- AF3 numbers are from published benchmarks; we do not re-run AF3
- **Both experimental structures must be available at generation time for ORACLE performance.** Without state 2, dual coverage falls to 11/49 (22.4%) — see the oracle-vs-blind table above. This is not de novo prediction; it is pathway modeling between two known endpoints.
- Scoring refinements beyond ensemble generation show little additional benefit

## Installation

```bash
pip install -r requirements.txt
```

```bash
# Blind vs oracle benchmark (49 proteins) — primary evidence
python benchmarks/run_blind_coverage.py

# Bridge-source ablation diagnostic (Step 1)
python benchmarks/dsib_ablation_diagnostic.py

# Unified cross-dataset benchmark (v2 vs v3)
python benchmarks/run_unified_coverage.py

# Autoinhibited head-to-head v2 vs v3
python benchmarks/compare_v2_v3_coverage.py

# Tests
python -m pytest tests/ -v
```

## Architecture

```
QuantumFoldX v3 (DSIB)
├── PDB fetching (both states, chain fallback)
├── DSIB: H₁, H₂, 9-point λ-path, switch contacts, TCI
├── Ensemble: NMA + rigid-body + torsion + bridge_interp + switch_rigid + manifold_bridge
├── Scoring: manifold overlap, state-2 contacts/geometry/imfdRMSD
└── Metrics: dual-state coverage (oracle vs blind)
```

## Project structure

```
src/quantum/dual_state_ising.py   # DSIB core + TCI
src/scoring/qicess_v3.py          # create_dsib_scorer() factory
src/ensemble/conformational_sampler.py
benchmarks/run_blind_coverage.py       # Primary evidence: blind vs oracle
benchmarks/dsib_ablation_diagnostic.py # Bridge-source breakdown
benchmarks/run_unified_coverage.py
configs/benchmark_dataset.py           # 49 curated targets
results/blind/                         # Blind vs oracle outputs
results/unified/                       # Cross-dataset v2 vs v3 outputs
```

## References

- Papageorgiou et al. (2025) Communications Chemistry. https://doi.org/10.1038/s42004-025-01763-0
- Ronish et al. (2024) Nature Communications. https://doi.org/10.1038/s41467-024-51801-z
- Peng et al. (2025) Briefings in Bioinformatics. https://doi.org/10.1093/bib/bbaf170

## License

MIT
