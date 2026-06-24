#!/usr/bin/env python3
"""Generate ablation study figures."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

df = pd.read_csv(RESULTS_DIR / 'ablation' / 'ablation_raw.csv')
with open(RESULTS_DIR / 'ablation' / 'ablation_stats.json') as f:
    stats = json.load(f)

bs_df = pd.read_csv(RESULTS_DIR / 'ablation' / 'vqe_vs_exact_bitstrings.csv')

methods = ['QICESS-VQE', 'QICESS-Exact', 'Classical-MJ', 'No-Quantum', 'Random']
colors = {'QICESS-VQE': '#2196F3', 'QICESS-Exact': '#9C27B0', 
          'Classical-MJ': '#FF9800', 'No-Quantum': '#4CAF50', 'Random': '#9E9E9E'}
labels = {'QICESS-VQE': 'QICESS-VQE\n(quantum)', 'QICESS-Exact': 'QICESS-Exact\n(classical diag)', 
          'Classical-MJ': 'Classical MJ\n(sum potentials)', 'No-Quantum': 'No Quantum\n(renormalized)', 
          'Random': 'Random\n(null baseline)'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300,
})

# ═══════════════════════════════════════
# Figure 4: Ablation Results
# ═══════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# Panel A: Mean Top-10 TM by method
ax = axes[0]
means = []
stds = []
for m in methods:
    key = f'{m}_top_k_mean_tm'
    vals = df[key].dropna()
    means.append(vals.mean())
    stds.append(vals.std() / np.sqrt(len(vals)))  # SEM

x_pos = range(len(methods))
bars = ax.bar(x_pos, means, yerr=stds, color=[colors[m] for m in methods],
              edgecolor='black', linewidth=0.8, width=0.6, capsize=4)

ax.set_xticks(x_pos)
ax.set_xticklabels([labels[m] for m in methods], fontsize=8)
ax.set_ylabel('Top-10 Mean TM to State 2')
ax.set_title('A) Ranking Quality by Method')

# Add significance annotations
for i, m in enumerate(means):
    ax.text(i, m + stds[i] + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Annotate NS
ax.text(2.5, max(means) + max(stds) + 0.05, 'All pairwise: p > 0.05 (NS)',
        ha='center', fontsize=9, style='italic', color='red')

# Panel B: Per-protein VQE vs Random comparison
ax = axes[1]
genes = df['gene'].values
vqe_tms = df['QICESS-VQE_top_k_mean_tm'].values
rand_tms = df['Random_top_k_mean_tm'].values

# Sort by VQE - Random difference
diff = vqe_tms - rand_tms
sort_idx = np.argsort(diff)
genes_sorted = genes[sort_idx]
diff_sorted = diff[sort_idx]

bar_colors = ['#4CAF50' if d > 0.001 else '#FF5722' if d < -0.001 else '#9E9E9E' for d in diff_sorted]
bars = ax.barh(range(len(genes_sorted)), diff_sorted, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(genes_sorted)))
ax.set_yticklabels(genes_sorted, fontsize=9)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('VQE - Random (TM difference)')
ax.set_title('B) VQE vs Random per Protein')
ax.invert_yaxis()

n_vqe_wins = sum(1 for d in diff if d > 0.001)
n_rand_wins = sum(1 for d in diff if d < -0.001)
ax.text(0.95, 0.05, f'VQE wins: {n_vqe_wins}\nRandom wins: {n_rand_wins}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel C: VQE vs Exact diag bitstring comparison
ax = axes[2]
hamming_dists = bs_df['hamming_distance'].values
genes_bs = bs_df['gene'].values
vqe_times = bs_df['vqe_time'].values
exact_times = bs_df['exact_time'].values

sort_idx = np.argsort(hamming_dists)[::-1]
ax.barh(range(len(genes_bs)), hamming_dists[sort_idx], color='#E53935', 
        edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_yticks(range(len(genes_bs)))
ax.set_yticklabels(genes_bs[sort_idx], fontsize=8)
ax.set_xlabel('Hamming Distance (bits out of 16)')
ax.set_title('C) VQE vs Exact Ground State')
ax.invert_yaxis()

# Annotate match rate from stored bitstring comparison
ax.text(0.95, 0.05, f'Match rate: {int(bs_df["identical"].sum())}/{len(bs_df)}\nMean Hamming: {hamming_dists.mean():.1f}/16\n'
        f'Exact diag: {exact_times.mean():.1f}s\nVQE: {vqe_times.mean():.1f}s',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9))

plt.suptitle('Ablation Study: Does the Quantum Scoring Layer Matter?', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURES_DIR / 'fig4_ablation.png'}")

# ═══════════════════════════════════════
# Figure 5: Verdict summary
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

n_prot = stats.get('n_proteins', len(df))
vqe_tm = stats['QICESS-VQE']['top10_mean_tm']['mean']
rand_tm = stats['Random']['top10_mean_tm']['mean']
vqe_rand = stats['paired_tests']['VQE_vs_Random']
exact_tm = stats['QICESS-Exact']['top10_mean_tm']['mean']
vqe_exact = stats['paired_tests']['VQE_vs_QICESS-Exact']
vqe_noq = stats['paired_tests']['VQE_vs_No-Quantum']
ve = stats.get('vqe_vs_exact', {})
n_match = ve.get('n_identical', 0)
mean_h = ve.get('mean_hamming', 0.0)
mean_vqe_t = bs_df['vqe_time'].mean() if len(bs_df) else 0.0
mean_ex_t = bs_df['exact_time'].mean() if len(bs_df) else 0.0

verdict_data = [
    ['Finding', 'Detail', 'Implication'],
    [f'VQE never finds exact\nground state', f'{n_match}/{n_prot} proteins match\n(mean Hamming = {mean_h:.1f}/16)',
     'VQE is trapped in local minima\nat 16 qubits with 3 layers'],
    ['VQE does not outperform\nrandom ranking',
     f'VQE: {vqe_tm:.3f} vs Random: {rand_tm:.3f}\n(p = {vqe_rand["wilcoxon_p"]:.2f}, NS)',
     'Quantum scoring adds no\nmeasurable ranking value'],
    ['VQE \u2248 Exact diag\nfor ranking',
     f'\u0394 = {vqe_exact["mean_diff"]:+.3f} (p = {vqe_exact["wilcoxon_p"]:.2f}, NS)',
     'Even correct ground state\ndoes not improve ranking'],
    ['Exact diag faster\nthan VQE',
     f'VQE: ~{mean_vqe_t:.0f}s vs Exact: ~{mean_ex_t:.0f}s\nper protein',
     'Quantum circuit overhead\nwith no compensating benefit'],
    ['Marginal VQE advantage\nover no-quantum',
     f'\u0394 = {vqe_noq["mean_diff"]:+.3f} (p = {vqe_noq["wilcoxon_p"]:.2f}, NS)',
     'Any signal is within noise;\ncohort insufficient power'],
]

table = ax.table(cellText=verdict_data[1:], colLabels=verdict_data[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.4)

for j in range(3):
    table[0, j].set_facecolor('#E53935')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, 6):
    table[i, 2].set_facecolor('#FFF3E0')

ax.set_title('Ablation Verdict: The Quantum Layer is Currently Decorative',
             fontsize=14, fontweight='bold', pad=20, color='#B71C1C')

plt.savefig(FIGURES_DIR / 'fig5_ablation_verdict.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURES_DIR / 'fig5_ablation_verdict.png'}")
