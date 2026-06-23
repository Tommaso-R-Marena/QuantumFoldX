#!/usr/bin/env python3
"""
run_benchmark_v2_fast.py — Streamlined dual-state coverage benchmark.
Saves results incrementally after each protein to avoid data loss on timeout.
Uses adaptive ensemble size based on protein length.
"""

import sys
import time
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.qicess_v2 import QICESSv2Scorer
from configs.benchmark_dataset import get_autoinhibited_benchmark, get_af3_baseline
from benchmarks.benchmark_utils import process_single_target, run_dual_state_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'stats').mkdir(parents=True, exist_ok=True)
INCREMENTAL_FILE = RESULTS_DIR / 'tables' / 'raw_results_v2.csv'


def main():
    logger.info("="*80)
    logger.info("QuantumFoldX v2 — Dual-State Coverage Benchmark (Fast)")
    logger.info("="*80)
    
    scorer = QICESSv2Scorer(vqe_layers=3, vqe_restarts=2, vqe_steps=50, use_qaoa=True)
    targets = get_autoinhibited_benchmark()
    af3_base = get_af3_baseline()
    
    all_results = []
    
    # Check for existing incremental results
    if INCREMENTAL_FILE.exists():
        existing = pd.read_csv(INCREMENTAL_FILE)
        done_genes = set(existing['gene'].tolist())
        all_results = existing.to_dict('records')
        logger.info(f"Resuming: {len(done_genes)} proteins already done")
    else:
        done_genes = set()
    
    total_start = time.time()
    
    for idx, target in enumerate(targets):
        if target.gene_name in done_genes:
            logger.info(f"[{idx+1}/{len(targets)}] {target.gene_name} — ALREADY DONE")
            continue
        
        logger.info(f"\n[{idx+1}/{len(targets)}] {target.gene_name}")
        try:
            result = process_single_target(target, scorer)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_results.append({
                'protein': target.protein_name, 'gene': target.gene_name,
                'status': 'error', 'error': str(e)
            })
        
        # INCREMENTAL SAVE after each protein
        df = pd.DataFrame(all_results)
        df.to_csv(INCREMENTAL_FILE, index=False)
    
    total_time = time.time() - total_start
    
    # Final analysis
    df = pd.DataFrame(all_results)
    stats = run_dual_state_stats(df, af3_base, category='autoinhibited')
    # Backward-compatible keys for analyze_results.py
    if 'primary' in stats:
        stats['primary']['af3_auto_rate'] = stats['primary']['af3_category_rate']
        stats['primary']['p_vs_af3_auto'] = stats['primary']['p_vs_af3_category']
        stats['primary']['sig_vs_auto'] = stats['primary']['sig_vs_category']
    stats['timing'] = {'total_s': total_time, 'timestamp': datetime.now().isoformat()}
    
    with open(RESULTS_DIR / 'stats' / 'statistical_tests_v2.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Summary table
    valid = df[df['status'] == 'success']
    if len(valid) > 0:
        cols = ['gene', 'n_residues_state1', 'n_common_residues',
                'state1_vs_state2_rmsd', 'state1_vs_state2_tm',
                'ens_min_rmsd_state2', 'ens_max_tm_state2',
                'state2_rmsd_improvement_pct', 'dual_state_covered_tm05',
                'af3_imfd_rmsd', 'scoring_time_s']
        avail = [c for c in cols if c in valid.columns]
        summary = valid[avail].sort_values('gene')
        summary.to_csv(RESULTS_DIR / 'tables' / 'dual_state_coverage.csv', index=False)
    
    # Print summary
    n_valid = len(valid)
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK COMPLETE — {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {n_valid} | Skipped: {(df['status'].str.startswith('skipped')).sum()} | Failed: {(df['status'] == 'error').sum()}")
    
    if 'primary' in stats:
        p = stats['primary']
        logger.info(f"\nPRIMARY: Dual-State Coverage (TM>0.5)")
        logger.info(f"  QFX: {p['n_covered']}/{p['n']} = {p['rate']*100:.1f}%")
        logger.info(f"  95% CI: [{p['wilson_95ci'][0]*100:.1f}%, {p['wilson_95ci'][1]*100:.1f}%]")
        logger.info(f"  AF3 (auto): {p['af3_auto_rate']*100:.0f}% | p={p['p_vs_af3_auto']:.4f} sig={p['sig_vs_auto']}")
        logger.info(f"  AF3 (multi): {p['af3_multi_rate']*100:.1f}% | p={p['p_vs_af3_multi']:.4f} sig={p['sig_vs_multi']}")
    
    if 'state2_tm' in stats:
        t = stats['state2_tm']
        logger.info(f"\nState 2 TM-scores: mean={t['mean']:.3f} median={t['median']:.3f}")
        logger.info(f"  Above 0.5: {t['frac_above_05']*100:.1f}% | Above 0.7: {t['frac_above_07']*100:.1f}%")
    
    if 'improvement' in stats:
        im = stats['improvement']
        logger.info(f"\nState 2 RMSD Improvement: mean={im['mean']:.2f}Å | {im['frac_improved']*100:.0f}% improved")
        logger.info(f"  Wilcoxon p={im['wilcoxon_p']:.4f} sig={im['sig']}")
    
    return df, stats


if __name__ == '__main__':
    main()
