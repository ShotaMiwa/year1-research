# coding: utf-8
"""
run_exhaustive_feature_search.py
全13特徴量に対する網羅的探索（Exhaustive Search: 1特徴量13通り + 2特徴量78通り + 3特徴量286通り = 計377通り）
- 厳密な Leave-One-Out Cross-Validation (LOO-CV, 23分割)
- 主要ハイパーパラメータ設定（max_depth: 1~3, min_samples_leaf: 1~2, class_weight: None/balanced）
- 完全ランキング表 CSV、トップランキング棒グラフ、新特徴量セグメント長の出現率集計を出力
"""

import os, sys, warnings, time
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'feature_matrix.csv')
OUT_DIR = os.path.join(BASE_DIR, '進捗報告用', '20261021')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

PAPER_DIR = os.path.join(BASE_DIR, 'docs', 'paper')
PAPER_FIG_DIR = os.path.join(PAPER_DIR, 'figures')
PAPER_TAB_DIR = os.path.join(PAPER_DIR, 'tables')
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(PAPER_TAB_DIR, exist_ok=True)

# 1. データ読み込み
df = pd.read_csv(DATA_PATH)
TARGET = 'Target_Satisfaction'
median_target = df[TARGET].median()
y_all = (df[TARGET] >= median_target).astype(int).values
N = len(df)

FEATURE_COLS = [
    'comment_rate', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity',
    'speech_rate', 'pause_ratio', 'grass_ratio',
    'hm_mean', 'hm_max',
    'comment_semantic_variance', 'sub_comment_similarity',
    'segment_length_sec'
]

print(f"Loaded {N} samples, {len(FEATURE_COLS)} features.")

# 2. 探索空間の構築（1〜3特徴量: 計377通り）
comb_dict = {}
for k in [1, 2, 3]:
    comb_dict[k] = list(combinations(range(len(FEATURE_COLS)), k))
    print(f"  Level {k} ({k} features): {len(comb_dict[k])} combinations")

total_combs = sum(len(c) for c in comb_dict.values())
print(f"Total feature combinations: {total_combs}")

# 3. ハイパーパラメータグリッド
PARAM_SETTINGS = [
    {'name': 'd1_leaf1', 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'd2_leaf1', 'max_depth': 2, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'd2_leaf2', 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'd3_leaf1', 'max_depth': 3, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'd3_leaf2', 'max_depth': 3, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'd2_leaf1_bal', 'max_depth': 2, 'min_samples_leaf': 1, 'class_weight': 'balanced'},
]

X_matrix = df[FEATURE_COLS].values
results = []

start_time = time.time()
print("\nStarting exhaustive LOO-CV search across all combinations...")

for k, combs in comb_dict.items():
    # レベル1（単体特徴量）は深さ1のみ
    # レベル2は深さ1〜3
    # レベル3は深さ2〜3
    applicable_params = PARAM_SETTINGS
    if k == 1:
        applicable_params = [p for p in PARAM_SETTINGS if p['max_depth'] == 1]
    
    for param in applicable_params:
        p_name = param['name']
        d_val = param['max_depth']
        l_val = param['min_samples_leaf']
        w_val = param['class_weight']
        
        for comb in combs:
            feats = list(comb)
            feat_names = [FEATURE_COLS[idx] for idx in feats]
            feat_names_str = ", ".join(feat_names)
            has_segment_length = ('segment_length_sec' in feat_names)
            
            # LOO-CV 23分割
            preds = []
            for i in range(N):
                X_tr, y_tr = np.delete(X_matrix[:, feats], i, axis=0), np.delete(y_all, i)
                X_te = X_matrix[i:i+1, feats]
                clf = DecisionTreeClassifier(max_depth=d_val, min_samples_leaf=l_val, class_weight=w_val, random_state=42)
                clf.fit(X_tr, y_tr)
                preds.append(clf.predict(X_te)[0])
            
            preds = np.array(preds)
            acc = accuracy_score(y_all, preds)
            bal_acc = balanced_accuracy_score(y_all, preds)
            f1 = f1_score(y_all, preds, average='macro')
            correct_cnt = int(sum(preds == y_all))
            
            # 誤分類セグメントIDリスト
            misclassified = df[preds != y_all]['セグメント'].tolist()
            
            results.append({
                'Feature_Count': k,
                'Features': feat_names_str,
                'Has_Segment_Length': has_segment_length,
                'Max_Depth': d_val,
                'Min_Samples_Leaf': l_val,
                'Class_Weight': str(w_val),
                'Hyperparameters': f"depth={d_val}, leaf={l_val}, weight={w_val}",
                'Balanced_Accuracy': bal_acc,
                'Accuracy': acc,
                'F1_Macro': f1,
                'Correct_Count': f"{correct_cnt}/{N}",
                'Correct_Num': correct_cnt,
                'Misclassified_Segments': str(misclassified)
            })

elapsed = time.time() - start_time
print(f"Exhaustive search finished in {elapsed:.2f} seconds. Evaluated {len(results)} total configurations.")

# 4. データフレーム化とソート
df_res = pd.DataFrame(results)
# 優先順位: Balanced_Accuracy (降順) -> Accuracy (降順) -> Feature_Count (昇順, 少ない特徴量優先) -> Max_Depth (昇順)
df_res = df_res.sort_values(by=['Balanced_Accuracy', 'Accuracy', 'Feature_Count', 'Max_Depth'], ascending=[False, False, True, True]).reset_index(drop=True)
df_res['Rank'] = range(1, len(df_res) + 1)

# CSV出力
cols_order = ['Rank', 'Feature_Count', 'Features', 'Has_Segment_Length', 'Hyperparameters', 'Balanced_Accuracy', 'Accuracy', 'Correct_Count', 'F1_Macro', 'Misclassified_Segments']
df_out = df_res[cols_order]

csv_path = os.path.join(TAB_DIR, 'exp_exhaustive_feature_ranking.csv')
df_out.to_csv(csv_path, index=False)
df_out.to_csv(os.path.join(PAPER_TAB_DIR, 'exp_exhaustive_feature_ranking.csv'), index=False)
print(f"Saved ranking CSV to: {csv_path}")

# 5. 上位結果のサマリ表示
print("\n=== TOP 10 MODELS (Exhaustive Search) ===")
top10 = df_out.head(10)
for _, r in top10.iterrows():
    print(f"Rank {r['Rank']:2d} | BalAcc: {r['Balanced_Accuracy']*100:.1f}% ({r['Correct_Count']}) | k={r['Feature_Count']} | {r['Hyperparameters']} | {r['Features']} | Length?: {r['Has_Segment_Length']}")

# 6. 新特徴量「セグメント長」のプレゼンス分析
top20 = df_out.head(20)
top30 = df_out.head(30)
top50 = df_out.head(50)

len_ratio_top20 = (top20['Has_Segment_Length'].sum() / 20) * 100
len_ratio_top30 = (top30['Has_Segment_Length'].sum() / 30) * 100
len_ratio_top50 = (top50['Has_Segment_Length'].sum() / 50) * 100

print(f"\n--- Segment Length Presence in Top Rankings ---")
print(f"  Top 20 Models: {top20['Has_Segment_Length'].sum()}/20 ({len_ratio_top20:.1f}%) contain segment_length_sec")
print(f"  Top 30 Models: {top30['Has_Segment_Length'].sum()}/30 ({len_ratio_top30:.1f}%) contain segment_length_sec")
print(f"  Top 50 Models: {top50['Has_Segment_Length'].sum()}/50 ({len_ratio_top50:.1f}%) contain segment_length_sec")

# 7. 可視化図表①: トップ12モデルの比較棒グラフ
plt.figure(figsize=(12, 7))
top12 = df_out.head(12).copy()
y_indices = np.arange(len(top12))

# セグメント長を含むものは鮮やかな青色、含まないものはグレー
bar_colors = ['#2B6CB0' if has_l else '#A0AEC0' for has_l in top12['Has_Segment_Length']]

# 1位のバー（セグメント長＋疑問符率）を赤/ゴールドで強調
if top12.iloc[0]['Has_Segment_Length']:
    bar_colors[0] = '#E53E3E'

bars = plt.barh(y_indices, top12['Balanced_Accuracy'] * 100, color=bar_colors, height=0.6, alpha=0.9)

labels = [f"#{r['Rank']} [{r['Hyperparameters']}] {r['Features']}" for _, r in top12.iterrows()]
plt.yticks(y_indices, labels, fontsize=9)
plt.gca().invert_yaxis()  # 1位を一番上に表示

for idx, bar in enumerate(bars):
    w = bar.get_width()
    corr = top12.iloc[idx]['Correct_Count']
    plt.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}% ({corr})", va='center', fontsize=9, fontweight='bold', color='#1A202C')

plt.xlabel('Balanced Accuracy (%) - LOO-CV', fontsize=11)
plt.title('Top 12 Models from Exhaustive Feature Search (Decision Tree)\nRed/Blue: Contains new feature "segment_length_sec" | Gray: Baseline features only', fontsize=12, fontweight='bold')
plt.xlim(0, 100)
plt.axvline(x=50.0, color='gray', linestyle='--', alpha=0.6, label='Chance Level (50%)')
plt.axvline(x=63.1, color='orange', linestyle=':', alpha=0.7, label='Previous Baseline (63.1%)')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()

fig_top_path = os.path.join(FIG_DIR, 'exp_exhaustive_top_ranking_barplot.png')
plt.savefig(fig_top_path, dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, 'exp_exhaustive_top_ranking_barplot.png'), dpi=300)
plt.close()
print(f"Saved: {fig_top_path}")

# 8. 可視化図表②: 上位30モデルにおける特徴量別出現回数ランキング
feature_counts = []
for feats_str in top30['Features']:
    for f in feats_str.split(', '):
        feature_counts.append(f)

feat_counter = Counter(feature_counts)
feat_df = pd.DataFrame(feat_counter.most_common(), columns=['Feature', 'Count'])

plt.figure(figsize=(10, 6))
bar_colors_f = ['#E53E3E' if f == 'segment_length_sec' else '#3182CE' for f in feat_df['Feature']]
bars_f = plt.barh(range(len(feat_df)), feat_df['Count'], color=bar_colors_f, height=0.65)
plt.yticks(range(len(feat_df)), feat_df['Feature'], fontsize=10)
plt.gca().invert_yaxis()

for bar in bars_f:
    w = bar.get_width()
    plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{int(w)}回 ({w/30*100:.1f}%)", va='center', fontsize=9, color='#2D3748')

plt.xlabel('Occurrence Count in Top 30 Models', fontsize=11)
plt.title('Feature Appearance Frequency in Top 30 Models (Out of 30 Configurations)\nRed: New Feature "segment_length_sec"', fontsize=12, fontweight='bold')
plt.xlim(0, max(feat_df['Count']) + 3)
plt.grid(True, linestyle='--', alpha=0.4, axis='x')
plt.tight_layout()

fig_freq_path = os.path.join(FIG_DIR, 'exp_exhaustive_feature_frequency.png')
plt.savefig(fig_freq_path, dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, 'exp_exhaustive_feature_frequency.png'), dpi=300)
plt.close()
print(f"Saved: {fig_freq_path}")

print("\nALL EXHAUSTIVE EXPERIMENTS AND VISUALIZATIONS COMPLETE!")
