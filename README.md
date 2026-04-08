# QuantumFoldX: Quantum-Scored Conformational Ensemble Analysis for Protein Structure Prediction

[![Benchmark](https://img.shields.io/badge/Benchmark-14%20proteins-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Quantum](https://img.shields.io/badge/Quantum-Classically%20Simulated-orange)]()

## Overview

QuantumFoldX is a hybrid quantum-classical framework for exploring protein conformational landscapes. Rather than competing with AlphaFold 3 (AF3) on single-structure prediction accuracy — where AF3's training on 200k+ PDB structures makes it nearly unbeatable — QuantumFoldX targets a documented weakness: **conformational state coverage**.

AF3 typically predicts a single conformational state per protein. For drug design and mechanistic biology, understanding the full conformational landscape (including inactive/autoinhibited states, disorder, and chirality) matters significantly.

### Key Results

| Metric | QuantumFoldX | AlphaFold 3 | Significance |
|--------|-------------|-------------|--------------|
| Dual-state coverage (autoinhibited) | **35.7%** (5/14) | 14% | p=0.036 (binomial) |
| Ensemble RMSD improvement | **100%** of proteins | N/A (single prediction) | p=0.00006 (Wilcoxon) |
| Disorder prediction AUC | **0.831** (DisorderNet) | 0.747 | +0.084 AUC |
| D-peptide chirality violations | **0%** (ChiralFold) | 51% | -51pp |
| Quantum scoring × state 2 | **71%** correlation | N/A | 10/14 proteins |

### Honest Limitations

- ⚠ **All quantum circuits are classically simulated** (PennyLane `lightning.qubit`, 16 qubits)
- ⚠ **AF3 numbers are from published benchmarks**, not re-run by us
- ⚠ **QFX starts from known experimental structures** — this is conformational exploration, not de novo prediction
- ⚠ Dual-state coverage at TM>0.5 is driven by proteins with already-similar states (100% easy, 0% hard)
- ⚠ For proteins with genuinely different conformational states (baseline TM<0.5), ensemble perturbation alone does not bridge the gap
- ⚠ DisorderNet uses classical ML (LightGBM/XGBoost), not quantum circuits
- ⚠ ChiralFold uses geometric corrections, not quantum circuits

## Architecture

```
QuantumFoldX Pipeline
├── 1. PDB Structure Fetching (RCSB)
├── 2. Conformational Ensemble Generation
│   ├── Normal Mode Analysis (multi-scale: 2Å + 6Å amplitude)
│   └── Domain Rigid-Body Perturbation (multi-scale: 5Å/20° + 15Å/45°)
├── 3. Quantum-Enhanced Scoring (QICESS v2)
│   ├── Ising Hamiltonian from Miyazawa-Jernigan potentials
│   ├── VQE ground state via PennyLane (16 qubits, 3 layers)
│   └── Conformation ranking by quantum contact agreement
├── 4. Structural Metrics
│   ├── RMSD, TM-score, GDT-TS, lDDT
│   ├── imfdRMSD (inter-module functional domain RMSD)
│   └── Dual-state coverage evaluation
└── 5. Statistical Analysis
    ├── Binomial test vs AF3 published rates
    ├── Wilcoxon signed-rank for RMSD improvement
    └── Bootstrap confidence intervals
```

## Benchmark Results

### Dual-State Conformational Coverage

Evaluated on 14 autoinhibited proteins from [Papageorgiou et al. 2025](https://www.nature.com/articles/s42004-025-01763-0):

| Gene | N_res | S1↔S2 RMSD (Å) | S1↔S2 TM | Ens→S2 minRMSD (Å) | Ens→S2 maxTM | RMSD Improv. | Dual-State |
|------|-------|-----------------|-----------|---------------------|--------------|--------------|------------|
| ABL1 | 263 | 4.97 | 0.888 | 4.88 | 0.890 | 0.08Å (1.7%) | ✓ |
| AK1 | 214 | 7.12 | 0.565 | 6.77 | 0.584 | 0.35Å (4.9%) | ✓ |
| BRAF | 289 | 13.16 | 0.323 | 12.80 | 0.334 | 0.35Å (2.7%) | ✗ |
| CBS | 498 | 14.45 | 0.369 | 12.55 | 0.431 | 1.89Å (13.1%) | ✗ |
| EGFR | 311 | 3.93 | 0.859 | 3.90 | 0.862 | 0.03Å (0.7%) | ✓ |
| FGFR1 | 149 | 18.19 | 0.091 | 16.42 | 0.129 | 1.78Å (9.8%) | ✗ |
| FYN | 262 | 18.04 | 0.067 | 16.37 | 0.098 | 1.67Å (9.2%) | ✗ |
| HCK | 437 | 19.29 | 0.170 | 19.27 | 0.170 | 0.02Å (0.1%) | ✗ |
| JAK1 | 281 | 19.50 | 0.157 | 19.18 | 0.160 | 0.31Å (1.6%) | ✗ |
| KIT | 297 | 5.47 | 0.853 | 5.44 | 0.857 | 0.03Å (0.6%) | ✓ |
| LCK | 288 | 15.25 | 0.220 | 15.10 | 0.241 | 0.15Å (1.0%) | ✗ |
| PTPN6 | 504 | 23.33 | 0.191 | 17.73 | 0.387 | 5.61Å (24.0%) | ✗ |
| SRC | 449 | 23.20 | 0.198 | 22.84 | 0.218 | 0.36Å (1.5%) | ✗ |
| STAT3 | 558 | 0.85 | 0.989 | 0.84 | 0.989 | 0.01Å (1.1%) | ✓ |

### Stratified Analysis

| Difficulty | Baseline TM | Coverage Rate | N |
|-----------|------------|---------------|---|
| Easy | > 0.5 | **100%** (5/5) | 5 |
| Medium | 0.3–0.5 | 0% (0/2) | 2 |
| Hard | < 0.3 | 0% (0/7) | 7 |
| **Overall** | — | **35.7%** (5/14) | 14 |

### Statistical Tests

| Test | Statistic | p-value | Significant |
|------|-----------|---------|-------------|
| QFX rate vs AF3 autoinhibited (14%) | Binomial | 0.036 | ✓ (α=0.05) |
| QFX rate vs AF3 multi-state (23.3%) | Binomial | 0.210 | ✗ |
| RMSD improvement > 0 | Wilcoxon signed-rank | 0.000061 | ✓ (α=0.001) |

## Installation

```bash
pip install pennylane pennylane-lightning numpy scipy pandas matplotlib requests
```

## Usage

```python
# Run the full benchmark
cd QuantumFoldX
python benchmarks/run_benchmark_v2_fast.py

# Analyze results and generate figures
python benchmarks/analyze_results.py
```

## Project Structure

```
QuantumFoldX/
├── src/
│   ├── quantum/
│   │   ├── ising_vqe.py          # VQE Ising Hamiltonian solver (PennyLane)
│   │   └── qaoa_rotamer.py       # QAOA side-chain optimizer
│   ├── scoring/
│   │   └── qicess_v2.py          # QICESS v2 ensemble scorer
│   ├── ensemble/
│   │   └── conformational_sampler.py  # NMA + rigid-body ensemble generation
│   ├── metrics/
│   │   └── structural_metrics.py  # RMSD, TM-score, GDT-TS, lDDT, imfdRMSD
│   └── data/
│       └── pdb_fetcher.py        # Real PDB structure fetching from RCSB
├── configs/
│   └── benchmark_dataset.py      # 16 autoinhibited proteins with AF3 baselines
├── benchmarks/
│   ├── run_benchmark_v2_fast.py  # Main benchmark pipeline
│   └── analyze_results.py       # Statistical analysis & figure generation
├── results/
│   ├── tables/                   # CSV results
│   ├── stats/                    # Statistical tests (JSON)
│   └── figures/                  # Publication figures
└── data/
    └── pdb_cache/                # Downloaded PDB files
```

## Complementary Modules

QuantumFoldX results are most meaningful when combined with:

- **[DisorderNet](https://github.com/Tommaso-R-Marena/DisorderNet)** — AUC 0.831 vs AF3's 0.747 on DisProt (CAID3 benchmark)
- **[ChiralFold](https://github.com/Tommaso-R-Marena/ChiralFold)** — 0% chirality violation vs AF3's 51% for D-peptides

Together, these three modules address the three principal failure modes of current protein structure prediction: conformational diversity, intrinsic disorder, and stereochemistry.

## Citation

```
AF3 baseline sources:
- Papageorgiou et al. (2025) Communications Chemistry. https://doi.org/10.1038/s42004-025-01763-0
- Peng et al. (2025) Briefings in Bioinformatics. https://doi.org/10.1093/bib/bbaf170
- Ronish et al. (2024) Nature Communications. https://doi.org/10.1038/s41467-024-51801-z
- CAID3 (2024) disorder prediction benchmark
- Jumper et al. (2021) Nature. AlphaFold2 original paper
```

## License

MIT
