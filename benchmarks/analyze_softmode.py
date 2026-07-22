#!/usr/bin/env python3
"""
analyze_softmode.py — Rigorous, honest evaluation of soft-mode subspace sampling
for BLIND alternate-state prediction (reads softmode_improvement.csv).

Questions:
  A  Does subspace / relaxation improve blind max-TM over the baseline sampler?
     (paired bootstrap ΔTM, permutation, McNemar on coverage, effect sizes)
  B  Does the improvement concentrate where soft-mode overlap is high?
     (per-protein gain vs overlap; high vs low stratum interaction)
  C  Does it generalize to more covered proteins? (coverage per condition)

All with confidence intervals and Holm-Bonferroni control. Figures ->
results/rigorous/figures/.
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

RES = PROJECT_ROOT / 'results' / 'rigorous'
CSV = RES / 'softmode_improvement.csv'
FIG = RES / 'figures'
OUT = RES / 'softmode_stats.json'

CONDS = ['baseline', 'single_mode', 'subspace', 'subspace_relax', 'combo']
CAT_COLORS = {'autoinhibited': '#1f77b4', 'foldswitch': '#d62728',
              'multistate': '#2ca02c'}
CAT_LABEL = {'autoinhibited': 'Autoinhibited', 'foldswitch': 'Fold-switching',
             'multistate': 'Multi-state'}


def _paired(ok, a, b, seed):
    x = ok[f'{a}_max_tm_s2'].astype(float).values
    y = ok[f'{b}_max_tm_s2'].astype(float).values
    ca = ok[f'{a}_dual'].astype(bool).values
    cb = ok[f'{b}_dual'].astype(bool).values
    return {
        'mean_a': float(x.mean()), 'mean_b': float(y.mean()),
        'paired_bootstrap': st.paired_diff_bootstrap(x, y, seed=seed),
        'permutation_greater': st.permutation_paired(x, y, alternative='greater', seed=seed + 1),
        'mcnemar_coverage': st.mcnemar_exact(int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))),
        'rank_biserial': st.rank_biserial_paired(x, y),
        'cohens_d': st.cohens_d_paired(x, y),
        'n_improved': int(np.sum(x > y + 1e-3)),
        'n_worsened': int(np.sum(x < y - 1e-3)),
    }


def compute(df):
    ok = df[df['status'] == 'ok'].copy()
    n = len(ok)
    out = {'n_proteins': n}

    cov = {}
    for c in CONDS:
        k = int(ok[f'{c}_dual'].sum())
        lo, hi = st.wilson_interval(k, n)
        cov[c] = {'n_covered': k, 'rate': k / n, 'wilson_95ci': [lo, hi],
                  'mean_max_tm_s2': float(ok[f'{c}_max_tm_s2'].mean()),
                  'median_max_tm_s2': float(ok[f'{c}_max_tm_s2'].median())}
    out['coverage'] = cov

    # A. paired comparisons
    out['comparisons'] = {
        'subspace_vs_baseline': _paired(ok, 'subspace', 'baseline', 10),
        'subspace_relax_vs_baseline': _paired(ok, 'subspace_relax', 'baseline', 20),
        'subspace_vs_single_mode': _paired(ok, 'subspace', 'single_mode', 30),
        'subspace_relax_vs_subspace': _paired(ok, 'subspace_relax', 'subspace', 40),
        'combo_vs_baseline': _paired(ok, 'combo', 'baseline', 50),
    }

    # B. gain vs overlap
    gain = (ok['subspace_relax_max_tm_s2'] - ok['baseline_max_tm_s2']).values
    ov = ok['best_single_overlap'].astype(float).values
    tr = ok['transition_rmsd'].astype(float).values
    rho, p = sp.spearmanr(ov, gain)
    rng = np.random.default_rng(7)
    perm = np.array([sp.spearmanr(ov, rng.permutation(gain))[0] for _ in range(10000)])
    p_perm = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)

    def _partial(x, y, z):
        rx, ry, rz = (sp.rankdata(v) for v in (x, y, z))
        def res(a, b):
            B = np.vstack([b, np.ones_like(b)]).T
            c, *_ = np.linalg.lstsq(B, a, rcond=None)
            return a - B @ c
        r, pp = sp.spearmanr(res(rx, rz), res(ry, rz))
        return {'partial_rho': float(r), 'p': float(pp)}

    med = np.median(ov)
    hi_mask = ov > med
    hi_gain, lo_gain = gain[hi_mask], gain[~hi_mask]
    # interaction: is mean gain larger in high-overlap stratum than chance?
    obs = hi_gain.mean() - lo_gain.mean()
    rng2 = np.random.default_rng(11)
    perm_int = np.empty(10000)
    allg = gain.copy()
    nh = hi_mask.sum()
    for i in range(10000):
        rng2.shuffle(allg)
        perm_int[i] = allg[:nh].mean() - allg[nh:].mean()
    p_int = (np.sum(perm_int >= obs) + 1) / (len(perm_int) + 1)

    out['gain_vs_overlap'] = {
        'spearman_rho': float(rho), 'p_permutation': float(p_perm),
        'partial_ctrl_transition': _partial(ov, gain, tr),
        'high_overlap_mean_gain': float(hi_gain.mean()),
        'low_overlap_mean_gain': float(lo_gain.mean()),
        'high_gain_bootstrap': st.bootstrap_ci(hi_gain, np.mean, seed=1),
        'low_gain_bootstrap': st.bootstrap_ci(lo_gain, np.mean, seed=2),
        'stratum_interaction_perm_p': float(p_int),
        'median_overlap_split': float(med),
    }

    # per-category
    bycat = {}
    for cat, g in ok.groupby('category'):
        gg = (g['subspace_relax_max_tm_s2'] - g['baseline_max_tm_s2']).values
        bycat[cat] = {'n': int(len(g)), 'mean_gain': float(gg.mean()),
                      'mean_overlap': float(g['best_single_overlap'].mean()),
                      'baseline_dual': int(g['baseline_dual'].sum()),
                      'subspace_relax_dual': int(g['subspace_relax_dual'].sum())}
    out['by_category'] = bycat

    # Holm across confirmatory family
    family = {
        'subspace_relax>baseline': out['comparisons']['subspace_relax_vs_baseline']['permutation_greater']['p_value'],
        'subspace>single': out['comparisons']['subspace_vs_single_mode']['permutation_greater']['p_value'],
        'relax>subspace': out['comparisons']['subspace_relax_vs_subspace']['permutation_greater']['p_value'],
        'gain~overlap': out['gain_vs_overlap']['p_permutation'],
        'overlap_stratum_interaction': out['gain_vs_overlap']['stratum_interaction_perm_p'],
    }
    out['holm_bonferroni'] = st.holm_bonferroni(family)
    return out, ok


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                         'figure.dpi': 130, 'savefig.bbox': 'tight'})
    return plt


def figures(ok, stats, plt):
    FIG.mkdir(parents=True, exist_ok=True)
    ok = ok.copy()
    ok['gain'] = ok['subspace_relax_max_tm_s2'] - ok['baseline_max_tm_s2']

    # Fig 1: gain vs overlap
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for cat, g in ok.groupby('category'):
        ax.scatter(g['best_single_overlap'], g['gain'], s=55, alpha=0.85,
                   color=CAT_COLORS[cat], label=CAT_LABEL[cat], edgecolor='k', lw=0.4)
    ak = ok[ok['gene'] == 'AK1']
    if len(ak):
        ax.annotate('AK1 (hinge)', (float(ak['best_single_overlap'].iloc[0]),
                    float(ak['gain'].iloc[0])), textcoords='offset points',
                    xytext=(-60, -6), fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.axhline(0, ls='--', c='gray', lw=1)
    r = stats['gain_vs_overlap']
    ax.set_xlabel('ANM best single-mode overlap with transition (state 1 only)')
    ax.set_ylabel('Blind TM gain from subspace+relaxation\n(subspace_relax $-$ baseline)')
    ax.set_title(f"Improvement concentrates where soft modes carry signal\n"
                 rf"Spearman $\rho$={r['spearman_rho']:.2f} (perm p={r['p_permutation']:.3f}), n={stats['n_proteins']}")
    ax.legend(loc='upper left')
    fig.savefig(FIG / 'fig8_softmode_gain_vs_overlap.png'); plt.close(fig)

    # Fig 2: paired subspace_relax vs baseline
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.plot([0, 1], [0, 1], ls='--', c='gray', lw=1)
    for cat, g in ok.groupby('category'):
        ax.scatter(g['baseline_max_tm_s2'], g['subspace_relax_max_tm_s2'], s=55,
                   alpha=0.85, color=CAT_COLORS[cat], label=CAT_LABEL[cat],
                   edgecolor='k', lw=0.4)
    ak = ok[ok['gene'] == 'AK1']
    if len(ak):
        ax.annotate('AK1', (float(ak['baseline_max_tm_s2'].iloc[0]),
                    float(ak['subspace_relax_max_tm_s2'].iloc[0])),
                    textcoords='offset points', xytext=(-28, 6), fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.axhline(0.5, ls=':', c='k', lw=0.8, alpha=0.5)
    ax.axvline(0.5, ls=':', c='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Baseline sampler — max TM to state 2')
    ax.set_ylabel('Subspace + relaxation — max TM to state 2')
    ax.set_title('Soft-mode subspace + relaxation vs baseline (points above line = improved)')
    ax.legend(loc='lower right'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / 'fig9_subspace_relax_vs_baseline.png'); plt.close(fig)

    # Fig 3: mean gain by overlap tercile
    fig, ax = plt.subplots(figsize=(7, 5))
    ov = ok['best_single_overlap'].values
    terc = pd.qcut(ok['best_single_overlap'], 3,
                   labels=['low', 'mid', 'high'])
    means, los, his, labs = [], [], [], []
    for t in ['low', 'mid', 'high']:
        gg = ok['gain'][terc == t].values
        b = st.bootstrap_ci(gg, np.mean, seed=3)
        means.append(b['point']); los.append(b['point'] - b['ci'][0])
        his.append(b['ci'][1] - b['point'])
        labs.append(f'{t}\n(n={len(gg)})')
    ax.bar(labs, means, yerr=[np.maximum(0, los), np.maximum(0, his)], capsize=5,
           color=['#c6dbef', '#6baed6', '#2171b5'], edgecolor='k')
    ax.axhline(0, c='gray', lw=1)
    ax.set_ylabel('Mean blind TM gain (subspace_relax $-$ baseline)')
    ax.set_xlabel('Soft-mode overlap tercile')
    ax.set_title('Gain from soft-mode sampling by overlap tercile (95% bootstrap CI)')
    fig.savefig(FIG / 'fig10_gain_by_overlap_tercile.png'); plt.close(fig)

    # Fig 4: mean max TM per condition
    fig, ax = plt.subplots(figsize=(7.5, 5))
    vals = [stats['coverage'][c]['mean_max_tm_s2'] for c in CONDS]
    bars = ax.bar([c.replace('_', '\n') for c in CONDS], vals,
                  color=['gray', '#9ecae1', '#6baed6', '#2171b5', '#08519c'],
                  edgecolor='k')
    for bar, v, c in zip(bars, vals, CONDS):
        cc = stats['coverage'][c]['n_covered']
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f'{v:.3f}\ncov {cc}/{stats["n_proteins"]}', ha='center', fontsize=8.5)
    ax.set_ylabel('Mean max TM to state 2 (blind, n=49)')
    ax.set_title('Blind samplers: mean best-TM to state 2 and dual coverage')
    ax.set_ylim(0, max(vals) * 1.25)
    fig.savefig(FIG / 'fig11_condition_means.png'); plt.close(fig)


def main():
    if not CSV.exists():
        raise SystemExit(f'Missing {CSV}; run run_softmode_improvement.py first.')
    df = pd.read_csv(CSV)
    stats, ok = compute(df)
    OUT.write_text(json.dumps(stats, indent=2))
    plt = _mpl()
    figures(ok, stats, plt)

    n = stats['n_proteins']
    print('=' * 74)
    print(f'SOFT-MODE SUBSPACE IMPROVEMENT — BLIND prediction (n={n})')
    print('=' * 74)
    for c in CONDS:
        cc = stats['coverage'][c]
        print(f"  {c:16s} mean TM={cc['mean_max_tm_s2']:.3f}  "
              f"coverage {cc['n_covered']}/{n} ({cc['rate']:.1%})")
    print('\nPaired comparisons (ΔTM = A - B, 95% CI, permutation p, McNemar):')
    for name, r in stats['comparisons'].items():
        pb = r['paired_bootstrap']
        print(f"  {name:32s} ΔTM={pb['mean_diff']:+.4f} "
              f"[{pb['ci'][0]:+.4f},{pb['ci'][1]:+.4f}] "
              f"perm={r['permutation_greater']['p_value']:.4f} "
              f"McNemar={r['mcnemar_coverage']['p_value']:.3f} "
              f"(+{r['n_improved']}/-{r['n_worsened']})")
    g = stats['gain_vs_overlap']
    print(f"\nGain vs overlap: rho={g['spearman_rho']:.3f} perm p={g['p_permutation']:.4f} "
          f"| partial|transition rho={g['partial_ctrl_transition']['partial_rho']:.3f} "
          f"p={g['partial_ctrl_transition']['p']:.4f}")
    print(f"  high-overlap mean gain={g['high_overlap_mean_gain']:+.4f} "
          f"vs low={g['low_overlap_mean_gain']:+.4f}  interaction perm p={g['stratum_interaction_perm_p']:.4f}")
    print('\nBy category (mean gain, coverage base->relax):')
    for cat, cs in stats['by_category'].items():
        print(f"  {cat:14s} n={cs['n']:2d} overlap={cs['mean_overlap']:.3f} "
              f"gain={cs['mean_gain']:+.4f} cov {cs['baseline_dual']}->{cs['subspace_relax_dual']}")
    print('\nHolm-Bonferroni:')
    for k, v in stats['holm_bonferroni'].items():
        print(f"  {k:32s} p_raw={v['p_raw']:.4f} p_adj={v['p_adjusted']:.4f} reject={v['reject_h0']}")
    print(f'\nFigures -> {FIG}')


if __name__ == '__main__':
    main()
