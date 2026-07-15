#!/usr/bin/env python3
"""
analyze_rigorous.py — Statistics + publication figures for the rigorous benchmark.

Reads results/rigorous/rigorous_benchmark.csv (+ mode_overlap_curves.json) and:

  1. Computes honest statistics with confidence intervals, paired tests,
     multiple-comparison control, and effect sizes (src/analysis/statistics.py).
  2. Emits publication-quality figures to results/rigorous/figures/.

Key questions answered:
  Q1  Is blind alternate-state prediction better than AF3? (coverage + Wilson CI)
  Q2  Does principled soft-mode sampling beat the baseline blind sampler?
      (paired: McNemar on coverage, bootstrap + permutation on max-TM)
  Q3  Does the "quantum" Ising bridge add anything beyond plain interpolation?
      (oracle_dsib vs oracle_interp, paired)
  Q4  Is blind predictability explained by elastic-network mode overlap?
      (Spearman correlation of overlap vs blind max-TM, with permutation p)

Run:
  python benchmarks/analyze_rigorous.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import statistics as st
from configs.benchmark_dataset import get_af3_baseline

RES_DIR = PROJECT_ROOT / 'results' / 'rigorous'
CSV = RES_DIR / 'rigorous_benchmark.csv'
CURVES = RES_DIR / 'mode_overlap_curves.json'
FIG_DIR = RES_DIR / 'figures'
STATS_OUT = RES_DIR / 'rigorous_stats.json'

CAT_COLORS = {'autoinhibited': '#1f77b4', 'foldswitch': '#d62728',
              'multistate': '#2ca02c'}
CAT_LABEL = {'autoinhibited': 'Autoinhibited', 'foldswitch': 'Fold-switching',
             'multistate': 'Multi-state'}


def _af3_weighted():
    b = get_af3_baseline()
    return float(np.mean([
        b['autoinhibited']['fraction_both_states'],
        b['foldswitch']['success_rate'],
        b['multistate']['fraction_both_states_correct'],
    ]))


def compute_stats(df: pd.DataFrame) -> dict:
    ok = df[df['status'] == 'ok'].copy()
    n = len(ok)
    out = {'n_proteins': int(n), 'af3_weighted_rate': _af3_weighted()}

    conditions = ['blind_baseline', 'blind_softmode', 'blind_union',
                  'oracle_interp', 'oracle_dsib']
    cov = {}
    for c in conditions:
        k = int(ok[f'{c}_dual'].sum())
        lo, hi = st.wilson_interval(k, n)
        tm = ok[f'{c}_max_tm_s2'].astype(float)
        cov[c] = {
            'n_covered': k, 'rate': k / n,
            'wilson_95ci': [lo, hi],
            'mean_max_tm_s2': float(tm.mean()),
            'median_max_tm_s2': float(tm.median()),
        }
    out['coverage'] = cov

    # Q1: blind vs AF3 (one-sided binomial)
    af3 = out['af3_weighted_rate']
    for c in ['blind_baseline', 'blind_softmode', 'blind_union']:
        k = cov[c]['n_covered']
        out.setdefault('vs_af3', {})[c] = {
            'rate': k / n, 'af3': af3,
            'p_binom_greater': float(sp.binomtest(k, n, af3, alternative='greater').pvalue),
        }

    # Q2: soft-mode vs baseline (paired)
    a = ok['blind_softmode_max_tm_s2'].astype(float).values
    b = ok['blind_baseline_max_tm_s2'].astype(float).values
    cov_soft = ok['blind_softmode_dual'].astype(bool).values
    cov_base = ok['blind_baseline_dual'].astype(bool).values
    gained = int(np.sum(cov_soft & ~cov_base))
    lost = int(np.sum(~cov_soft & cov_base))
    out['softmode_vs_baseline'] = {
        'mean_tm_softmode': float(a.mean()), 'mean_tm_baseline': float(b.mean()),
        'paired_bootstrap': st.paired_diff_bootstrap(a, b, seed=1),
        'permutation_greater': st.permutation_paired(a, b, alternative='greater', seed=2),
        'mcnemar_coverage': st.mcnemar_exact(gained, lost),
        'rank_biserial': st.rank_biserial_paired(a, b),
        'cohens_d': st.cohens_d_paired(a, b),
        'n_improved': int(np.sum(a > b + 1e-6)),
        'n_worsened': int(np.sum(a < b - 1e-6)),
    }

    # Q3: DSIB vs plain interpolation (paired) — does Ising add anything?
    d = ok['oracle_dsib_max_tm_s2'].astype(float).values
    e = ok['oracle_interp_max_tm_s2'].astype(float).values
    out['dsib_vs_interp'] = {
        'mean_tm_dsib': float(d.mean()), 'mean_tm_interp': float(e.mean()),
        'paired_bootstrap_dsib_minus_interp': st.paired_diff_bootstrap(d, e, seed=3),
        'permutation_two_sided': st.permutation_paired(d, e, alternative='two-sided', seed=4),
        'n_dsib_better': int(np.sum(d > e + 1e-6)),
        'n_interp_better_or_equal': int(np.sum(d <= e + 1e-6)),
    }

    # Q4: mode overlap explains blind predictability
    q4 = {}
    blind_best = ok[['blind_baseline_max_tm_s2', 'blind_softmode_max_tm_s2']].max(axis=1).astype(float).values
    for feat in ['best_single_overlap', 'cum_overlap_5', 'cum_overlap_10',
                 'softest_mode_overlap']:
        if feat not in ok.columns:
            continue
        x = ok[feat].astype(float).values
        m = ~(np.isnan(x) | np.isnan(blind_best))
        if m.sum() < 5:
            continue
        rho, p = sp.spearmanr(x[m], blind_best[m])
        # permutation p on Spearman rho
        rng = np.random.default_rng(7)
        perm = np.array([sp.spearmanr(x[m], rng.permutation(blind_best[m]))[0]
                         for _ in range(5000)])
        p_perm = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)
        q4[feat] = {'spearman_rho': float(rho), 'p_analytic': float(p),
                    'p_permutation': float(p_perm), 'n': int(m.sum())}
    out['overlap_vs_blind'] = q4

    # By category
    bycat = {}
    for cat in sorted(ok['category'].unique()):
        sub = ok[ok['category'] == cat]
        nc = len(sub)
        bycat[cat] = {
            'n': nc,
            'mean_best_overlap': float(sub['best_single_overlap'].mean())
                if 'best_single_overlap' in sub.columns else None,
            'mean_cum_overlap_10': float(sub['cum_overlap_10'].mean())
                if 'cum_overlap_10' in sub.columns else None,
            'blind_baseline_dual': int(sub['blind_baseline_dual'].sum()),
            'blind_softmode_dual': int(sub['blind_softmode_dual'].sum()),
            'blind_union_dual': int(sub['blind_union_dual'].sum()),
            'oracle_interp_dual': int(sub['oracle_interp_dual'].sum()),
            'oracle_dsib_dual': int(sub['oracle_dsib_dual'].sum()),
            'blind_baseline_mean_tm': float(sub['blind_baseline_max_tm_s2'].mean()),
            'blind_softmode_mean_tm': float(sub['blind_softmode_max_tm_s2'].mean()),
        }
    out['by_category'] = bycat

    # Holm-Bonferroni across the primary confirmatory tests
    family = {
        'softmode>baseline (perm)': out['softmode_vs_baseline']['permutation_greater']['p_value'],
        'dsib!=interp (perm)': out['dsib_vs_interp']['permutation_two_sided']['p_value'],
        'blind_union>af3 (binom)': out['vs_af3']['blind_union']['p_binom_greater'],
    }
    for feat, r in q4.items():
        family[f'overlap~blind:{feat} (perm)'] = r['p_permutation']
    out['holm_bonferroni'] = st.holm_bonferroni(family)

    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.grid': True,
                         'grid.alpha': 0.3, 'figure.dpi': 130,
                         'savefig.bbox': 'tight'})
    return plt


def fig_overlap_vs_blind(df, plt):
    ok = df[df['status'] == 'ok']
    blind_best = ok[['blind_baseline_max_tm_s2', 'blind_softmode_max_tm_s2']].max(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cat, g in ok.groupby('category'):
        yy = g[['blind_baseline_max_tm_s2', 'blind_softmode_max_tm_s2']].max(axis=1)
        ax.scatter(g['best_single_overlap'], yy, s=55, alpha=0.8,
                   color=CAT_COLORS[cat], label=CAT_LABEL[cat], edgecolor='k', lw=0.4)
    rho, p = sp.spearmanr(ok['best_single_overlap'], blind_best)
    ax.axhline(0.5, ls='--', c='gray', lw=1)
    ax.text(0.02, 0.52, 'TM = 0.5 (fold match)', color='gray', fontsize=9)
    ax.set_xlabel('ANM best single-mode overlap with observed transition (state 1 only)')
    ax.set_ylabel('Best BLIND max TM-score to state 2')
    ax.set_title(f'Blind predictability tracks soft-mode overlap\nSpearman '
                 rf'$\rho$={rho:.2f} (p={p:.1e}), n={len(ok)}')
    ax.legend(loc='upper left', framealpha=0.9)
    fig.savefig(FIG_DIR / 'fig1_overlap_vs_blind_tm.png')
    plt.close(fig)


def fig_cumulative_curves(plt):
    if not CURVES.exists():
        return
    curves = json.loads(CURVES.read_text())
    by_cat = {}
    for gene, d in curves.items():
        by_cat.setdefault(d['category'], []).append(d['cumulative_overlap'])
    fig, ax = plt.subplots(figsize=(7, 5))
    for cat, arrs in by_cat.items():
        L = min(len(a) for a in arrs)
        M = np.array([a[:L] for a in arrs])
        x = np.arange(1, L + 1)
        mean = M.mean(0)
        sem = M.std(0) / np.sqrt(len(arrs))
        ax.plot(x, mean, color=CAT_COLORS[cat], lw=2,
                label=f'{CAT_LABEL[cat]} (n={len(arrs)})')
        ax.fill_between(x, mean - sem, mean + sem, color=CAT_COLORS[cat], alpha=0.2)
    ax.set_xlabel('Number of softest ANM modes included')
    ax.set_ylabel('Cumulative overlap with observed transition')
    ax.set_title('How many soft modes span the conformational change?')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(FIG_DIR / 'fig2_cumulative_overlap_curves.png')
    plt.close(fig)


def fig_softmode_vs_baseline(df, plt):
    ok = df[df['status'] == 'ok'].sort_values('best_single_overlap')
    fig, ax = plt.subplots(figsize=(7, 5.5))
    b = ok['blind_baseline_max_tm_s2'].values
    s = ok['blind_softmode_max_tm_s2'].values
    ax.plot([0, 1], [0, 1], ls='--', c='gray', lw=1)
    for cat, g in ok.groupby('category'):
        ax.scatter(g['blind_baseline_max_tm_s2'], g['blind_softmode_max_tm_s2'],
                   s=55, alpha=0.8, color=CAT_COLORS[cat], label=CAT_LABEL[cat],
                   edgecolor='k', lw=0.4)
    ax.axhline(0.5, ls=':', c='k', lw=0.8, alpha=0.5)
    ax.axvline(0.5, ls=':', c='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Baseline blind sampler — max TM to state 2')
    ax.set_ylabel('Soft-mode blind sampler — max TM to state 2')
    ax.set_title('Soft-mode sampling vs baseline (points above diagonal = improved)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG_DIR / 'fig3_softmode_vs_baseline.png')
    plt.close(fig)


def fig_dsib_vs_interp(df, plt):
    ok = df[df['status'] == 'ok']
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], ls='--', c='gray', lw=1)
    for cat, g in ok.groupby('category'):
        ax.scatter(g['oracle_interp_max_tm_s2'], g['oracle_dsib_max_tm_s2'],
                   s=55, alpha=0.8, color=CAT_COLORS[cat], label=CAT_LABEL[cat],
                   edgecolor='k', lw=0.4)
    ax.set_xlabel('Plain linear interpolation (no Ising) — max TM to state 2')
    ax.set_ylabel('Full "quantum" DSIB pipeline — max TM to state 2')
    ax.set_title('The Ising/quantum layer adds nothing beyond interpolation')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG_DIR / 'fig4_dsib_vs_interp.png')
    plt.close(fig)


def fig_coverage_bars(stats, plt):
    order = ['AF3 (published)', 'Blind baseline', 'Blind soft-mode',
             'Blind union', 'Oracle interp', 'Oracle DSIB']
    cov = stats['coverage']
    n = stats['n_proteins']
    rates = [stats['af3_weighted_rate'],
             cov['blind_baseline']['rate'], cov['blind_softmode']['rate'],
             cov['blind_union']['rate'], cov['oracle_interp']['rate'],
             cov['oracle_dsib']['rate']]
    err_lo, err_hi = [], []
    for key in ['blind_baseline', 'blind_softmode', 'blind_union',
                'oracle_interp', 'oracle_dsib']:
        lo, hi = cov[key]['wilson_95ci']
        err_lo.append(max(0.0, cov[key]['rate'] - lo))
        err_hi.append(max(0.0, hi - cov[key]['rate']))
    yerr = [[0] + err_lo, [0] + err_hi]
    colors = ['gray', '#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, rates, color=colors, yerr=yerr, capsize=4,
                  edgecolor='k', lw=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02, f'{r:.0%}',
                ha='center', fontsize=10)
    ax.axvspan(-0.5, 3.5, color='#1f77b4', alpha=0.05)
    ax.axvspan(3.5, 5.5, color='#ff7f0e', alpha=0.05)
    ax.text(1.5, 0.93, 'BLIND (predictive)', ha='center', color='#1f77b4', fontsize=10)
    ax.text(4.5, 0.93, 'ORACLE (state 2 known)', ha='center', color='#d95f0e', fontsize=10)
    ax.set_ylabel(f'Dual-state coverage rate (TM>0.5), n={n}')
    ax.set_title('Dual-state coverage: honest blind prediction vs oracle controls')
    ax.set_ylim(0, 1.02)
    plt.xticks(rotation=20, ha='right')
    fig.savefig(FIG_DIR / 'fig5_coverage_bars.png')
    plt.close(fig)


def fig_overlap_by_category(df, plt):
    ok = df[df['status'] == 'ok']
    cats = ['autoinhibited', 'multistate', 'foldswitch']
    data = [ok[ok['category'] == c]['best_single_overlap'].dropna().values for c in cats]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=[CAT_LABEL[c] for c in cats],
                    patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], cats):
        patch.set_facecolor(CAT_COLORS[c]); patch.set_alpha(0.5)
    for i, c in enumerate(cats):
        y = ok[ok['category'] == c]['best_single_overlap'].dropna().values
        x = np.random.default_rng(i).normal(i + 1, 0.05, len(y))
        ax.scatter(x, y, s=22, color=CAT_COLORS[c], edgecolor='k', lw=0.3, zorder=3)
    ax.set_ylabel('ANM best single-mode overlap with transition')
    ax.set_title('Soft-mode overlap by transition type')
    fig.savefig(FIG_DIR / 'fig6_overlap_by_category.png')
    plt.close(fig)


def main():
    if not CSV.exists():
        raise SystemExit(f'Missing {CSV}; run run_rigorous_benchmark.py first.')
    df = pd.read_csv(CSV)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stats = compute_stats(df)
    STATS_OUT.write_text(json.dumps(stats, indent=2))

    plt = _setup_mpl()
    fig_overlap_vs_blind(df, plt)
    fig_cumulative_curves(plt)
    fig_softmode_vs_baseline(df, plt)
    fig_dsib_vs_interp(df, plt)
    fig_coverage_bars(stats, plt)
    fig_overlap_by_category(df, plt)

    # Console summary
    n = stats['n_proteins']
    cov = stats['coverage']
    print('=' * 74)
    print(f'RIGOROUS BENCHMARK SUMMARY  (n={n} proteins)')
    print('=' * 74)
    print(f"AF3 published weighted rate: {stats['af3_weighted_rate']:.1%}")
    for c in ['blind_baseline', 'blind_softmode', 'blind_union',
              'oracle_interp', 'oracle_dsib']:
        ci = cov[c]['wilson_95ci']
        print(f"  {c:16s} coverage {cov[c]['n_covered']:2d}/{n} "
              f"({cov[c]['rate']:.1%}, 95%CI {ci[0]:.1%}-{ci[1]:.1%})  "
              f"mean TM={cov[c]['mean_max_tm_s2']:.3f}")
    sm = stats['softmode_vs_baseline']
    print(f"\nQ2 soft-mode vs baseline: ΔTM={sm['paired_bootstrap']['mean_diff']:+.3f} "
          f"95%CI [{sm['paired_bootstrap']['ci'][0]:+.3f},{sm['paired_bootstrap']['ci'][1]:+.3f}]  "
          f"perm p={sm['permutation_greater']['p_value']:.4f}  "
          f"McNemar p={sm['mcnemar_coverage']['p_value']:.4f}  "
          f"(improved {sm['n_improved']}, worsened {sm['n_worsened']})")
    dv = stats['dsib_vs_interp']
    print(f"Q3 DSIB vs interp: ΔTM={dv['paired_bootstrap_dsib_minus_interp']['mean_diff']:+.3f} "
          f"95%CI [{dv['paired_bootstrap_dsib_minus_interp']['ci'][0]:+.3f},"
          f"{dv['paired_bootstrap_dsib_minus_interp']['ci'][1]:+.3f}]  "
          f"(DSIB better on {dv['n_dsib_better']}/{n})")
    print('Q4 overlap vs blind max-TM (Spearman):')
    for feat, r in stats['overlap_vs_blind'].items():
        print(f"    {feat:22s} rho={r['spearman_rho']:+.3f}  perm p={r['p_permutation']:.4f}")
    print('\nBy category:')
    for cat, cs in stats['by_category'].items():
        print(f"  {cat:14s} n={cs['n']:2d}  overlap={cs['mean_best_overlap']:.3f}  "
              f"blind base/soft/union dual = {cs['blind_baseline_dual']}/"
              f"{cs['blind_softmode_dual']}/{cs['blind_union_dual']}")
    print(f'\nFigures written to {FIG_DIR}')


if __name__ == '__main__':
    main()
