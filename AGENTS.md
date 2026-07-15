# AGENTS.md

## Cursor Cloud specific instructions

QuantumFoldX is a **Python-only scientific CLI toolkit** (no server, frontend, or database). It fetches experimental protein structures from public PDB/AlphaFold servers, builds Ising Hamiltonians, generates conformational ensembles, and scores dual-state coverage. See `README.md` for the method and the list of runnable benchmark commands.

### Environment
- Dependencies are installed into a virtualenv at `.venv/` (the update script refreshes it from `requirements.txt`). Run everything with `.venv/bin/python ...` (e.g. `.venv/bin/python -m pytest tests/ -v`), or activate with `source .venv/bin/activate`.
- Requires outbound HTTPS to `files.rcsb.org` (required) and `alphafold.ebi.ac.uk` (optional). Downloaded structures are cached under `data/pdb_cache/` (gitignored); after the first run, reruns of the same targets work offline.

### Test / run / lint
- Tests: `.venv/bin/python -m pytest tests/ -v` — the suite hits the live PDB server (some tests download real structures), so it needs network and takes ~1–2 min.
- Run benchmarks: entry points live in `benchmarks/` and are documented in `README.md` (e.g. `.venv/bin/python benchmarks/run_blind_coverage.py`, `.venv/bin/python benchmarks/dsib_ablation_diagnostic.py`).
- No linter/formatter is configured in this repo (no ruff/flake8/black config). Use `.venv/bin/python -m compileall src benchmarks configs tests` for a quick syntax check.

### Gotchas
- Benchmarks are compute-heavy: per protein the pipeline classically enumerates Ising states and generates 80+ conformations, so a single protein can take a few minutes. The full 49-protein `run_blind_coverage.py` is long-running — scope with `--genes` (on `dsib_ablation_diagnostic.py`) or expect a lengthy run.
- `run_blind_coverage.py` resumes from `results/blind/blind_coverage.csv` by default; pass `--no-resume` to force a fresh run.
