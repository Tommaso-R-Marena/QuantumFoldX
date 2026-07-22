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

## Rigorous re-analysis: mode overlap + honest controls (n=49)

A fully controlled re-analysis with confidence intervals, paired permutation /
exact tests, effect sizes, and Holm–Bonferroni correction. See
[`results/rigorous/FINDINGS.md`](results/rigorous/FINDINGS.md) and figures in
`results/rigorous/figures/`.

Run:
```bash
python benchmarks/run_rigorous_benchmark.py --no-resume --n-ens 56   # generate (needs network)
python benchmarks/analyze_rigorous.py                               # statistics + figures
```

Headline findings (all paired over the same 49 proteins):

1. **The Ising/quantum bridge adds nothing.** The full DSIB pipeline is
   *significantly worse* than plain S1→S2 interpolation with no Ising at all
   (ΔTM = −0.101, 95% CI [−0.126, −0.077]; DSIB better on 1/49; Holm-adj
   p = 4×10⁻⁴).
2. **The oracle result is a trivial artifact.** Pure interpolation toward the
   known state 2 covers 49/49 (100%). This is not prediction.
3. **Blind prediction is not significantly above AF3** (11/49 = 22.4%, 95% CI
   13–36%; p = 0.11 vs 15%). 0/38 hard proteins (baseline TM < 0.5) are covered
   blindly; 10/11 "covered" proteins already have similar states.
4. **Blind gain toward state 2 is governed by elastic-network soft-mode
   overlap** (Spearman ρ = 0.46, p = 0.001; independent of transition size:
   partial ρ = 0.49, p = 4×10⁻⁴). Adenylate kinase (highest single-mode overlap)
   is the only genuine blind conformational-change success.

The honest contribution is a mechanistic characterization of *what is
predictable and why* — and a clean refutation of the quantum-advantage claim —
not a state-of-the-art prediction result.

### Follow-up: soft-mode subspace sampling

Acting on finding 4, a fully blind soft-mode **subspace** sampler with
ENM-guided Cα relaxation (`src/ensemble/nm_guided.py`,
`benchmarks/run_softmode_improvement.py`, `benchmarks/analyze_softmode.py`)
gives a small but statistically robust improvement in blind max-TM to state 2
(ΔTM +0.013, 95% CI [+0.006, +0.029]; Holm-adj p = 0.008), and the improvement
is significantly concentrated on high-overlap collective/hinge transitions
(high vs low stratum +0.025 vs +0.002, interaction p = 0.004). The combined
ensemble is never worse than baseline (30 improved, 0 worsened), and adenylate
kinase rises 0.58 → 0.80. **It does not, however, change the binary dual-state
coverage rate (11/49 for every sampler)** — the gains rarely cross TM > 0.5 for
hard cases, and fold-switchers gain nothing. Details in
[`results/rigorous/FINDINGS.md`](results/rigorous/FINDINGS.md).

```bash
python benchmarks/run_softmode_improvement.py --no-resume --n-ens 56
python benchmarks/analyze_softmode.py
```

### Bridge-source diagnostic (Step 1, n=6 proteins across all categories)

On SRC, WAS, FYN, CLIC1, REV, and PDGFRB, the max-TM winner in the ORACLE ensemble was a bridge member on 6/6 proteins. By source: **common_residue_interp** won 5/6; **manifold_bridge** won 1/6 (PDGFRB); **switch_contact_rigid** won 0/6. Switch-contact-guided rigid-body motion (the Ising H₁/H₂ output) generates conformations but did not produce the max-TM winner in this sample. Condition E (v2 + bridge only) matched condition B (full v3) on 6/6 because coverage is max TM over the full ensemble, not a ranked selection step.

## What did not work

**QICESS v2 ranking (n=16 autoinhibited proteins, stored ablation).** Metric: top-10 mean TM to state 2. QICESS-VQE 0.349 vs Random 0.356 (Wilcoxon p=0.345, not significant). QICESS-Exact 0.346 vs Random 0.356 (p=0.345). VQE never matched exact diagonalization (0/16; mean Hamming 5.1/16). An earlier 14-protein run in commit `1df6b3a` gave VQE 0.391 vs Random 0.394 (p=0.25) — same metric, smaller cohort; the README previously cited that stale figure while `results/ablation/ablation_raw.csv` was re-run at n=16 (`45f3f6a`).

**Contact-Ising energy model (standalone negative result).** Follow-up diagnostic (`benchmarks/qicess_discrimination_diagnostic.py`) found this is not a sign or normalization bug. On six test kinases, reference contact bitstrings matched the exact ground state in **0/6** (mean Hamming distance **5.0/16**): the optimization target is disconnected from the reference structure it is meant to represent.

- **SRC, per-conformation Ising energy vs TM→S2:** Spearman ρ = **−0.479**, p = **0.0004**, n=40. Real signal, wrong direction.
- **Deca-alanine Ramachandran basins (known order):** the full Hamiltonian with ZZ cooperative terms ranks C7ax below C7eq (wrong order). Removing ZZ terms restores the correct ordering (C7eq lowest energy).
- **Principled fix tried once (n=14 held-out kinases):** remove ZZ terms; rank by per-conformation independent-qubit energy. Result: **significantly worse than random** (mean Spearman ρ = **−0.3726**, Wilcoxon p vs random = **0.0192**), not better.

Meaningful dual-state discrimination from contact-Ising alone would require different collective coordinates (domain packing angle, hinge distance) or a dual-state Hamiltonian. DSIB explored the latter separately, with its own honestly documented BLIND/ORACLE gap (see table above).

The Ising switch-contact machinery generates conformations but does not produce the max-TM winner in the DSIB bridge-source diagnostic (switch_contact_rigid won 0/6; common_residue_interp won 5/6; manifold_bridge won 1/6 and still interpolates toward state 2 coordinates). DSIB's apparent success under the ORACLE setting is attributable to direct S1↔S2 interpolation, confirmed by the BLIND/ORACLE gap: BLIND 11/49 (22.4%) vs ORACLE 48/49 (98.0%), Wilcoxon p=1.20e-09.

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
