# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 日本語フォント設定
for font_name in ['Noto Sans CJK JP', 'TakaoPGothic', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'DejaVu Sans']:
    try:
        matplotlib.rcParams['font.family'] = font_name
        break
    except Exception:
        pass

sys.path.insert(0, '/home/shota/work/year1')
from src.satisfaction_analysis.constants import FEATURE_NAMES_12, OUTLIER_SHORTS_IDS
from src.satisfaction_analysis.data_validator import load_and_validate_datasets

df_survey, df_shorts = load_and_validate_datasets()
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(OUTLIER_SHORTS_IDS)].copy().reset_index(drop=True)

y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
y_shorts  = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values
n_survey  = len(df_survey)

FEATURE_COLS_11 = [f for f in FEATURE_NAMES_12 if f != 'pause_ratio']

sc_73 = StandardScaler()
X_73_z = sc_73.fit_transform(df_shorts_73[FEATURE_COLS_11].values)

sc_23 = StandardScaler()
X_23_z = sc_23.fit_transform(df_survey[FEATURE_COLS_11].values)

COMBINATIONS = [
    ('疑問比率 単体\n(question_ratio)', ['question_ratio'], '1変数'),
    ('動画尺 単体\n(segment_length_sec)', ['segment_length_sec'], '1変数'),
    ('コメント意味分散 単体\n(semantic_var)', ['comment_semantic_variance'], '1変数'),
    ('疑問比率 ＋ 動画尺\n(question + seg_len)', ['question_ratio', 'segment_length_sec'], '2変数'),
    ('動画尺 ＋ コメント意味分散\n(seg_len + sem_var)', ['segment_length_sec', 'comment_semantic_variance'], '2変数'),
    ('疑問比率 ＋ コメント意味分散\n(question + sem_var)', ['question_ratio', 'comment_semantic_variance'], '2変数'),
    ('上位3変数 統合\n(question + seg_len + sem_var)', ['question_ratio', 'segment_length_sec', 'comment_semantic_variance'], '3変数'),
]

rows = []
for label, feats, group in COMBINATIONS:
    idx = [FEATURE_COLS_11.index(f) for f in feats]
    svm = SVC(kernel='linear', C=1.0, random_state=42).fit(X_73_z[:, idx], y_shorts)
    p_svm = svm.predict(X_23_z[:, idx])
    acc_s = int(np.sum(p_svm == y_survey))
    
    lr = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit(X_73_z[:, idx], y_shorts)
    p_lr = lr.predict(X_23_z[:, idx])
    acc_l = int(np.sum(p_lr == y_survey))
    
    clean_label = label.split('\n')[0]
    rows.append({
        'group': group,
        'label': clean_label,
        'features': ' + '.join(feats),
        'svm_correct': f'{acc_s} / {n_survey}',
        'svm_acc': acc_s / n_survey * 100,
        'lr_correct': f'{acc_l} / {n_survey}',
        'lr_acc': acc_l / n_survey * 100,
    })

df_table = pd.DataFrame(rows)

csv_p1 = '/home/shota/work/year1/進捗報告用/20261125/tables/exp3c_new_core_features_results_11.csv'
csv_p2 = '/home/shota/work/year1/docs/paper/tables/exp3c_new_core_features_results_11.csv'
df_table.to_csv(csv_p1, index=False, encoding='utf-8-sig')
df_table.to_csv(csv_p2, index=False, encoding='utf-8-sig')
print('✅ CSV saved successfully')

# グラフ描画（ラベルは英語・短縮表記併記で豆腐文字を完全回避）
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(COMBINATIONS))
width = 0.35

svm_accs = [r['svm_acc'] for r in rows]
lr_accs = [r['lr_acc'] for r in rows]
plot_labels = [
    "Question Ratio\n(1-feat)",
    "Segment Length\n(1-feat)",
    "Semantic Var\n(1-feat)",
    "Question + Length\n(2-feats)",
    "Length + SemVar\n(2-feats)",
    "Question + SemVar\n(2-feats)",
    "Top 3 Integrated\n(3-feats)"
]

bars1 = ax.bar(x - width/2, svm_accs, width, label='Linear SVM', color='#2b5c8f')
bars2 = ax.bar(x + width/2, lr_accs, width, label='Logistic Regression', color='#e26d5c')

ax.axhline(73.9, color='#2e7d32', linestyle=':', alpha=0.8, linewidth=1.5, label='Max Accuracy: 73.9% (17/23)')
ax.axhline(65.2, color='gray', linestyle='--', alpha=0.6, label='All 11 Feats SVM: 65.2% (15/23)')

ax.set_title('Transfer Accuracy (73 -> 23) Across All Combinations of Top 3 Core Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Transfer Accuracy (%)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(plot_labels, fontsize=9)
ax.set_ylim(25, 85)
ax.legend(loc='lower right', fontsize=9)

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8)

ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

png_p1 = '/home/shota/work/year1/進捗報告用/20261125/figures/exp3c_new_core_features_transfer_11.png'
pdf_p1 = '/home/shota/work/year1/進捗報告用/20261125/figures/exp3c_new_core_features_transfer_11.pdf'
png_p2 = '/home/shota/work/year1/docs/paper/figures/exp3c_new_core_features_transfer_11.png'
pdf_p2 = '/home/shota/work/year1/docs/paper/figures/exp3c_new_core_features_transfer_11.pdf'

for p in [png_p1, png_p2]:
    plt.savefig(p, dpi=300)
for p in [pdf_p1, pdf_p2]:
    plt.savefig(p)
plt.close()
print('✅ Figures saved successfully')
