#!/usr/bin/env python3
"""
run_all_benchmarks.py — Master benchmark runner for QuantumFoldX.

Runs all benchmark suites:
  1. Autoinhibited proteins (v2 dual-state coverage) — 16 targets
  2. Fold-switching proteins — 6 targets
  3. Multi-state proteins (M-SADA subset) — 6 targets
  4. Quantum scoring ablation study
  5. Statistical analysis and figure generation

Usage:
  python benchmarks/run_all_benchmarks.py              # Full suite
  python benchmarks/run_all_benchmarks.py --quick      # Reduced VQE steps
  python benchmarks/run_all_benchmarks.py --only auto  # Single category
"""

import sys
import os
import time
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.qicess_v3 import QICESSv3Scorer
from configs.benchmark_dataset import (
    get_autoinhibited_benchmark, get_foldswitch_benchmark,
    get_multistate_benchmark, get_af3_baseline
)
from benchmarks.benchmark_utils import process_single_target, run_dual_state_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'stats').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'ablation').mkdir(parents=True, exist_ok=True)


BENCHMARK_CONFIGS = {
    'autoinhibited': {
        'getter': get_autoinhibited_benchmark,
        'output': 'raw_results_v2.csv',
        'stats_output': 'statistical_tests_v2.json',
        'category': 'autoinhibited',
    },
    'foldswitch': {
        'getter': get_foldswitch_benchmark,
        'output': 'raw_results_foldswitch.csv',
        'stats_output': 'statistical_tests_foldswitch.json',
        'category': 'foldswitch',
    },
    'multistate': {
        'getter': get_multistate_benchmark,
        'output': 'raw_results_multistate.csv',
        'stats_output': 'statistical_tests_multistate.json',
        'category': 'multistate',
    },
}


def run_benchmark_suite(name: str, config: dict, scorer: QICESSv3Scorer,
                        af3_base: dict, resume: bool = True) -> tuple:
    """Run a single benchmark category with incremental saves."""
    targets = config['getter']()
    output_file = RESULTS_DIR / 'tables' / config['output']

    all_results = []
    done_genes = set()

    if resume and output_file.exists():
        existing = pd.read_csv(output_file)
        done_genes = set(existing['gene'].tolist())
        all_results = existing.to_dict('records')
        logger.info(f"  Resuming {name}: {len(done_genes)} already done")

    suite_start = time.time()

    for idx, target in enumerate(targets):
        if target.gene_name in done_genes:
            logger.info(f"  [{idx+1}/{len(targets)}] {target.gene_name} — SKIP (done)")
            continue

        logger.info(f"\n  [{idx+1}/{len(targets)}] {target.gene_name} ({name})")
        try:
            result = process_single_target(target, scorer)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_results.append({
                'protein': target.protein_name, 'gene': target.gene_name,
                'category': target.category, 'status': 'error', 'error': str(e)
            })

        pd.DataFrame(all_results).to_csv(output_file, index=False)

    elapsed = time.time() - suite_start
    df = pd.DataFrame(all_results)
    stats = run_dual_state_stats(df, af3_base, category=config['category'])
    stats['timing'] = {'suite_s': elapsed, 'timestamp': datetime.now().isoformat()}

    stats_file = RESULTS_DIR / 'stats' / config['stats_output']
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    valid = df[df['status'] == 'success']
    n_valid = len(valid)
    dsc = int(valid['dual_state_covered_tm05'].sum()) if n_valid > 0 else 0

    logger.info(f"\n  {name.upper()} COMPLETE: {n_valid}/{len(targets)} success, "
                f"{dsc}/{n_valid} dual-state covered, {elapsed:.0f}s")

    if 'primary' in stats:
        p = stats['primary']
        logger.info(f"  Coverage: {p['n_covered']}/{p['n']} = {p['rate']*100:.1f}% "
                    f"(AF3 baseline: {p['af3_category_rate']*100:.1f}%, "
                    f"p={p['p_vs_af3_category']:.4f})")

    return df, stats


def run_ablation(scorer):
    """Run the quantum scoring ablation study."""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING ABLATION STUDY")
    logger.info("=" * 80)
    from benchmarks.ablation_study import run_ablation
    return run_ablation()


def generate_figures():
    """Generate all analysis figures."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 80)
    import benchmarks.analyze_results  # noqa: F401 — runs on import
    import benchmarks.ablation_figures  # noqa: F401


def print_summary(all_stats: dict):
    """Print cross-suite summary."""
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-SUITE SUMMARY")
    logger.info("=" * 80)

    for name, stats in all_stats.items():
        if 'primary' in stats:
            p = stats['primary']
            sig = "✓" if p.get('sig_vs_category', False) else "✗"
            logger.info(f"  {name:15s}: {p['n_covered']}/{p['n']} = {p['rate']*100:.1f}% "
                        f"vs AF3 {p['af3_category_rate']*100:.1f}% "
                        f"(p={p['p_vs_af3_category']:.4f} {sig})")
        elif 'warning' in stats:
            logger.info(f"  {name:15s}: {stats['warning']}")


def main():
    parser = argparse.ArgumentParser(description='QuantumFoldX Full Benchmark Suite')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced VQE steps for faster runs')
    parser.add_argument('--only', choices=['auto', 'foldswitch', 'multistate', 'ablation', 'figures'],
                        help='Run only a specific component')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore existing results')
    parser.add_argument('--no-ablation', action='store_true',
                        help='Skip ablation study')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    args = parser.parse_args()

    vqe_steps = 30 if args.quick else 50
    vqe_restarts = 1 if args.quick else 2

    logger.info("=" * 80)
    logger.info("QuantumFoldX v3 — FULL BENCHMARK SUITE (Dual-State Quantum Bridge)")
    logger.info("=" * 80)
    logger.info(f"Scorer: QICESS v3 (exact Ising enumeration, dual-state bridge)")

    scorer = QICESSv3Scorer()
    af3_base = get_af3_baseline()
    all_stats = {}
    total_start = time.time()

    suites_to_run = list(BENCHMARK_CONFIGS.keys())
    if args.only == 'auto':
        suites_to_run = ['autoinhibited']
    elif args.only == 'foldswitch':
        suites_to_run = ['foldswitch']
    elif args.only == 'multistate':
        suites_to_run = ['multistate']
    elif args.only in ('ablation', 'figures'):
        suites_to_run = []

    for name in suites_to_run:
        logger.info(f"\n{'='*80}")
        logger.info(f"BENCHMARK: {name.upper()}")
        logger.info(f"{'='*80}")
        _, stats = run_benchmark_suite(
            name, BENCHMARK_CONFIGS[name], scorer, af3_base,
            resume=not args.no_resume
        )
        all_stats[name] = stats

    if not args.no_ablation and args.only not in ('auto', 'foldswitch', 'multistate', 'figures'):
        if args.only == 'ablation' or args.only is None:
            run_ablation(scorer)

    if not args.no_figures and args.only not in ('auto', 'foldswitch', 'multistate', 'ablation'):
        if args.only == 'figures' or args.only is None:
            generate_figures()

    total_time = time.time() - total_start

    summary = {
        'suites': all_stats,
        'total_time_s': total_time,
        'timestamp': datetime.now().isoformat(),
        'config': {'scorer': 'QICESS-v3', 'dual_state_bridge': True},
    }
    with open(RESULTS_DIR / 'stats' / 'full_benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print_summary(all_stats)
    logger.info(f"\nTotal benchmark time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Results saved to {RESULTS_DIR}")

    return all_stats


if __name__ == '__main__':
    main()
