# -*- coding: utf-8 -*-
"""
plot_label_refinement.py
疑似ラベル改善実験の比較棒グラフを生成するスクリプト。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = "/home/shota/work/year1"
csv_path = os.path.join(PROJECT_ROOT, "進捗報告用/20261125/tables/exp3d_label_refinement_results.csv")
df = pd.read_csv(csv_path)

core4_df = df[df['feature_condition'] == 'Core4特徴量'].copy()

strategies = [
    '1. Baseline (Likes)',
    '2. Composite (Likes+Comments)',
    '3. Sentiment (Likes+Comm+Sent)',
    '4. Clear Margin (Top/Bottom 35%)',
    '5. Comp. Margin (Top/Bottom 35%)'
]

svm_accs = core4_df[core4_df['model'] == 'Linear SVM']['acc_val'].values * 100
lr_accs = core4_df[core4_df['model'] == 'Logistic Regression']['acc_val'].values * 100

x = np.arange(len(strategies))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, svm_accs, width, label='Linear SVM', color='steelblue')
bars2 = ax.bar(x + width/2, lr_accs, width, label='Logistic Regression', color='coral')

ax.axhline(50.0, color='gray', linestyle=':', label='Chance Level (50%)')
ax.axhline(73.9, color='darkgreen', linestyle='--', alpha=0.7, label='Current Best (73.9%)')

ax.set_title('Pseudo-Label Refinement Transfer Comparison (Core4 Features)', fontsize=13)
ax.set_ylabel('Transfer Accuracy to Survey 23 (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9, rotation=15)
ax.set_ylim(20, 85)
ax.legend(fontsize=9, loc='lower left')
ax.bar_label(bars1, fmt='%.1f%%', padding=3)
ax.bar_label(bars2, fmt='%.1f%%', padding=3)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

figures_dir = os.path.join(PROJECT_ROOT, "進捗報告用/20261125/figures")
paper_fig_dir = os.path.join(PROJECT_ROOT, "docs/paper/figures")

for d in [figures_dir, paper_fig_dir]:
    plt.savefig(os.path.join(d, "exp3d_label_refinement_comparison.png"), dpi=300)
    plt.savefig(os.path.join(d, "exp3d_label_refinement_comparison.pdf"))
plt.close()
print("Figure saved successfully.")
