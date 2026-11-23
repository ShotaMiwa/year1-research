# -*- coding: utf-8 -*-
"""
plot_bayesian_composite_results.py
案1（信頼度付き加重合成スコア）のalpha変化による転移精度の推移グラフを生成。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = "/home/shota/work/year1"
csv_path = os.path.join(PROJECT_ROOT, "進捗報告用/20261125/tables/exp3e_bayesian_composite_results.csv")
df = pd.read_csv(csv_path)

alphas = sorted(df['alpha'].unique())

core4_svm = df[(df['feature_condition'] == 'Core4特徴量') & (df['model'] == 'Linear SVM')].sort_values('alpha')['acc_val'].values * 100
core4_lr  = df[(df['feature_condition'] == 'Core4特徴量') & (df['model'] == 'Logistic Regression')].sort_values('alpha')['acc_val'].values * 100

all12_svm = df[(df['feature_condition'] == '全12特徴量') & (df['model'] == 'Linear SVM')].sort_values('alpha')['acc_val'].values * 100
all12_lr  = df[(df['feature_condition'] == '全12特徴量') & (df['model'] == 'Logistic Regression')].sort_values('alpha')['acc_val'].values * 100

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(alphas, core4_svm, marker='o', linewidth=2.5, color='steelblue', label='Core4 Features (Linear SVM)')
ax.plot(alphas, core4_lr,  marker='s', linewidth=2.5, linestyle='--', color='royalblue', label='Core4 Features (Logistic Regression)')

ax.plot(alphas, all12_svm, marker='^', linewidth=2.0, color='darkorange', label='All 12 Features (Linear SVM)')
ax.plot(alphas, all12_lr,  marker='v', linewidth=2.0, linestyle='--', color='coral', label='All 12 Features (Logistic Regression)')

ax.axhline(50.0, color='gray', linestyle=':', label='Chance Level (50%)')
ax.axhline(73.9, color='darkgreen', linestyle=':', alpha=0.7, label='Core4 Best (73.9%)')

ax.set_title('Transfer Accuracy vs. Comment Sentiment Weight (alpha)\n[Bayesian Reliability-Weighted Composite Label]', fontsize=13)
ax.set_xlabel('Weight of Shorts Comment Sentiment (alpha)\n(alpha=0.0: Likes Only -> alpha>0: Adding Comment Sentiment)', fontsize=11)
ax.set_ylabel('Transfer Accuracy to Survey 23 (%)', fontsize=12)
ax.set_xticks(alphas)
ax.set_ylim(15, 85)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

figures_dir = os.path.join(PROJECT_ROOT, "進捗報告用/20261125/figures")
paper_fig_dir = os.path.join(PROJECT_ROOT, "docs/paper/figures")

for d in [figures_dir, paper_fig_dir]:
    plt.savefig(os.path.join(d, "exp3e_bayesian_composite_transfer.png"), dpi=300)
    plt.savefig(os.path.join(d, "exp3e_bayesian_composite_transfer.pdf"))
plt.close()
print("Figure saved successfully.")
