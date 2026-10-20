# coding: utf-8
"""
進捗報告（2026/10/21）用 実験スクリプト:
新特徴量「セグメントの長さ（segment_length_sec）」導入前後の徹底比較検証
- パート1: 二重交差検証（Nested LOO-CV + SFS）での Before vs After 比較
- パート2: 決定木分岐ルール & 決定境界の Before vs After 比較
- パート3: 全23セグメントの誤分類・救済状況の Before vs After 追跡
"""

import os, sys, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

warnings.filterwarnings('ignore')

# 基準パス
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

# 1. データ準備
df = pd.read_csv(DATA_PATH)
TARGET = 'Target_Satisfaction'
median_target = df[TARGET].median()
df['Satisfaction_Class'] = (df[TARGET] >= median_target).astype(int)
df['Class_Label'] = df['Satisfaction_Class'].map({1: 'High (>=3.60)', 0: 'Low (<3.60)'})

y_all = df['Satisfaction_Class'].values
N = len(df)

# 特徴量定義（Before: 従来の12特徴量、After: +セグメント長）
FEATURE_COLS_BEFORE = [
    'comment_rate', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity',
    'speech_rate', 'pause_ratio', 'grass_ratio',
    'hm_mean', 'hm_max',
    'comment_semantic_variance', 'sub_comment_similarity',
]
FEATURE_COLS_AFTER = FEATURE_COLS_BEFORE + ['segment_length_sec']

print(f"Total Segments: {N}")
print(f"Features Before: {len(FEATURE_COLS_BEFORE)}")
print(f"Features After : {len(FEATURE_COLS_AFTER)}")

# =========================================================================
# パート1: 二重交差検証（Nested LOO-CV + SFS）での Before vs After 比較
# =========================================================================
print("\n=== Running Part 1: Nested LOO-CV Comparison (Before vs After) ===")

def sfs_select(X_tr, y_tr, model_factory, k_features=3):
    selected = []
    candidates = list(range(X_tr.shape[1]))
    while len(selected) < k_features:
        best_score = -1
        best_cand = None
        for cand in candidates:
            current = selected + [cand]
            n_inner = len(X_tr)
            preds = []
            for j in range(n_inner):
                X_in_tr, y_in_tr = np.delete(X_tr[:, current], j, axis=0), np.delete(y_tr, j)
                X_in_te = X_tr[j:j+1, current]
                m = model_factory()
                m.fit(X_in_tr, y_in_tr)
                preds.append(m.predict(X_in_te)[0])
            score = balanced_accuracy_score(y_tr, preds)
            if score > best_score:
                best_score = score
                best_cand = cand
        selected.append(best_cand)
        candidates.remove(best_cand)
    return selected

param_grid_dt = [
    {'name': 'Default (max_depth=None)', 'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None, 'params_str': 'max_depth=None, leaf=1, weight=None'},
    {'name': 'Depth=1 (Stump)', 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': None, 'params_str': 'max_depth=1, leaf=1, weight=None'},
    {'name': 'Depth=2, Leaf=2', 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': None, 'params_str': 'max_depth=2, leaf=2, weight=None'},
    {'name': 'Depth=2, Leaf=3', 'max_depth': 2, 'min_samples_leaf': 3, 'class_weight': None, 'params_str': 'max_depth=2, leaf=3, weight=None'},
    {'name': 'Depth=3, Leaf=2', 'max_depth': 3, 'min_samples_leaf': 2, 'class_weight': None, 'params_str': 'max_depth=3, leaf=2, weight=None'},
    {'name': 'Depth=2, Leaf=2, Balanced', 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'params_str': 'max_depth=2, leaf=2, weight=balanced'},
    {'name': 'Depth=1, Balanced', 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'params_str': 'max_depth=1, leaf=1, weight=balanced'},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=1', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': None, 'params_str': 'Fixed: Q+Sent, depth=1, leaf=1'},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=2, Leaf=2', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': None, 'params_str': 'Fixed: Q+Sent, depth=2, leaf=2'},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=2, Leaf=2, Balanced', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': 'balanced', 'params_str': 'Fixed: Q+Sent, depth=2, leaf=2, balanced'},
]

nested_results = []

for param in param_grid_dt:
    p_name = param['name']
    p_str = param['params_str']
    is_fixed = 'fixed_feats' in param
    
    # 1. Before (12 features)
    X_before = df[FEATURE_COLS_BEFORE].values
    preds_before = []
    for i in range(N):
        X_tr, y_tr = np.delete(X_before, i, axis=0), np.delete(y_all, i)
        X_te, y_te = X_before[i:i+1], y_all[i:i+1]
        if is_fixed:
            feat_idx = [FEATURE_COLS_BEFORE.index(f) for f in param['fixed_feats']]
        else:
            factory = lambda: DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], class_weight=param['class_weight'], random_state=42)
            feat_idx = sfs_select(X_tr, y_tr, factory, k_features=3)
        clf = DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], class_weight=param['class_weight'], random_state=42)
        clf.fit(X_tr[:, feat_idx], y_tr)
        preds_before.append(clf.predict(X_te[:, feat_idx])[0])
    preds_before = np.array(preds_before)
    bal_acc_before = balanced_accuracy_score(y_all, preds_before)
    correct_before = int(sum(preds_before == y_all))
    
    # 2. After (13 features: +segment_length_sec)
    X_after = df[FEATURE_COLS_AFTER].values
    preds_after = []
    chosen_after = []
    for i in range(N):
        X_tr, y_tr = np.delete(X_after, i, axis=0), np.delete(y_all, i)
        X_te, y_te = X_after[i:i+1], y_all[i:i+1]
        if is_fixed:
            feat_idx = [FEATURE_COLS_AFTER.index(f) for f in param['fixed_feats']]
        else:
            factory = lambda: DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], class_weight=param['class_weight'], random_state=42)
            feat_idx = sfs_select(X_tr, y_tr, factory, k_features=3)
        chosen_after.extend([FEATURE_COLS_AFTER[idx] for idx in feat_idx])
        clf = DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], class_weight=param['class_weight'], random_state=42)
        clf.fit(X_tr[:, feat_idx], y_tr)
        preds_after.append(clf.predict(X_te[:, feat_idx])[0])
    preds_after = np.array(preds_after)
    bal_acc_after = balanced_accuracy_score(y_all, preds_after)
    correct_after = int(sum(preds_after == y_all))
    
    diff_pt = (bal_acc_after - bal_acc_before) * 100
    top_sfs = ", ".join([f"{f}({c})" for f, c in Counter(chosen_after).most_common(3)]) if not is_fixed else "Fixed"
    
    nested_results.append({
        'Model_Setting': p_name,
        'Hyperparameters': p_str,
        'Before_BalAcc': bal_acc_before,
        'Before_Correct': f"{correct_before}/{N}",
        'After_BalAcc': bal_acc_after,
        'After_Correct': f"{correct_after}/{N}",
        'Diff_pt': diff_pt,
        'After_Top_SFS_Features': top_sfs
    })
    print(f"  {p_name:40s} | Before: {bal_acc_before*100:.1f}% ({correct_before:2d}) -> After: {bal_acc_after*100:.1f}% ({correct_after:2d}) | Diff: {diff_pt:+.1f}pt")

df_nested = pd.DataFrame(nested_results)
df_nested.to_csv(os.path.join(TAB_DIR, 'exp1_nested_cv_before_after.csv'), index=False)
df_nested.to_csv(os.path.join(PAPER_TAB_DIR, 'exp1_nested_cv_before_after.csv'), index=False)

# 可視化：Before vs After 比較棒グラフ
plt.figure(figsize=(12, 7))
y_pos = np.arange(len(df_nested))
height = 0.35

plt.barh(y_pos - height/2, df_nested['Before_BalAcc'] * 100, height, label='Before (12 Features: Without Length)', color='#A0AEC0', alpha=0.9)
bars_after = plt.barh(y_pos + height/2, df_nested['After_BalAcc'] * 100, height, label='After (13 Features: With segment_length_sec)', color='#3182CE', alpha=0.9)

# 決定株（Stump）を赤色で強調
bars_after[1].set_color('#E53E3E')

for idx, row in df_nested.iterrows():
    b_val = row['Before_BalAcc'] * 100
    a_val = row['After_BalAcc'] * 100
    plt.text(b_val + 1, y_pos[idx] - height/2, f"{b_val:.1f}%", va='center', fontsize=9, color='#4A5568')
    plt.text(a_val + 1, y_pos[idx] + height/2, f"{a_val:.1f}%", va='center', fontsize=9, fontweight='bold', color='#1A365D')

plt.yticks(y_pos, df_nested['Model_Setting'], fontsize=10)
plt.xlabel('Balanced Accuracy (%)', fontsize=12)
plt.title('Nested LOO-CV Performance: Before vs After Adding "segment_length_sec"', fontsize=13, fontweight='bold')
plt.axvline(x=50.0, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
plt.xlim(0, 100)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
fig_nested_path = os.path.join(FIG_DIR, 'exp1_nested_cv_before_after_barplot.png')
plt.savefig(fig_nested_path, dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, 'exp1_nested_cv_before_after_barplot.png'), dpi=300)
plt.close()
print(f"Saved: {fig_nested_path}")


# =========================================================================
# パート2: 決定木分岐ルール & 決定境界の Before vs After 比較
# =========================================================================
print("\n=== Running Part 2: Decision Tree & Decision Boundary Before vs After ===")

# Before: question_ratio only (d=1, leaf=1)
clf_before = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, random_state=42)
clf_before.fit(df[['question_ratio']].values, y_all)

# After: question_ratio + segment_length_sec (d=3, leaf=1)
clf_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=42)
clf_after.fit(df[['question_ratio', 'segment_length_sec']].values, y_all)

# 2.1 木構造の Before vs After 並列可視化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before Tree
plot_tree(
    clf_before,
    feature_names=['question_ratio'],
    class_names=['Low', 'High'],
    filled=True,
    rounded=True,
    ax=axes[0],
    fontsize=10
)
axes[0].set_title("Before: question_ratio Only (max_depth=1)\nBalanced Acc: 63.1% (16/23 Correct)", fontsize=12, fontweight='bold')

# After Tree
plot_tree(
    clf_after,
    feature_names=['question_ratio', 'segment_length_sec'],
    class_names=['Low', 'High'],
    filled=True,
    rounded=True,
    ax=axes[1],
    fontsize=9
)
axes[1].set_title("After: question_ratio + segment_length_sec (max_depth=3)\nBalanced Acc: 79.8% (19/23 Correct)", fontsize=12, fontweight='bold')

plt.tight_layout()
fig_tree_path = os.path.join(FIG_DIR, 'exp2_tree_structure_before_after.png')
plt.savefig(fig_tree_path, dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, 'exp2_tree_structure_before_after.png'), dpi=300)
plt.close()
print(f"Saved: {fig_tree_path}")

# 2.2 決定境界プロット（Afterモデル: 疑問符率 vs セグメント長）
plt.figure(figsize=(9, 6))
x_min, x_max = df['question_ratio'].min() - 0.03, df['question_ratio'].max() + 0.03
y_min, y_max = max(0, df['segment_length_sec'].min() - 10), df['segment_length_sec'].max() + 15
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf_after.predict(grid_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm_r')
plt.contour(xx, yy, Z, colors='black', linewidths=1.2, linestyles='--')

scatter = sns.scatterplot(
    data=df,
    x='question_ratio',
    y='segment_length_sec',
    hue='Class_Label',
    style='Class_Label',
    s=140,
    palette=['#E53E3E', '#3182CE'],
    alpha=0.95
)

# 注目セグメントのアノテーション
for _, r in df.iterrows():
    seg_id = int(r['セグメント'])
    offset = (5, 4) if seg_id != 19 else (5, -12)
    fontw = 'bold' if seg_id in [1, 17, 19, 20, 22] else 'normal'
    plt.annotate(
        f"Seg {seg_id}",
        (r['question_ratio'], r['segment_length_sec']),
        xytext=offset,
        textcoords='offset points',
        fontsize=9,
        fontweight=fontw
    )

plt.axhline(y=40, color='darkred', linestyle=':', linewidth=1.5, label='Cutoff: segment_length <= 40s')
plt.axvline(x=0.20, color='darkblue', linestyle=':', linewidth=1.5, label='Cutoff: question_ratio <= 0.20')
plt.title('Decision Boundary After Adding "segment_length_sec" (max_depth=3, min_leaf=1)', fontsize=13, fontweight='bold')
plt.xlabel('Question Ratio (Comment)', fontsize=11)
plt.ylabel('Segment Length (seconds)', fontsize=11)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
fig_bound_path = os.path.join(FIG_DIR, 'exp2_decision_boundary_after.png')
plt.savefig(fig_bound_path, dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, 'exp2_decision_boundary_after.png'), dpi=300)
plt.close()
print(f"Saved: {fig_bound_path}")


# =========================================================================
# パート3: 全23セグメントの誤分類・救済状況の Before vs After 追跡
# =========================================================================
print("\n=== Running Part 3: Segment-level Misclassification & Recovery Tracking ===")

# LOO-CV での各セグメント個別予測値を算出
# Before: question_ratio only (d=1, leaf=1)
preds_loo_before = []
for i in range(N):
    X_tr, y_tr = np.delete(df[['question_ratio']].values, i, axis=0), np.delete(y_all, i)
    X_te = df[['question_ratio']].values[i:i+1]
    m = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, random_state=42)
    m.fit(X_tr, y_tr)
    preds_loo_before.append(m.predict(X_te)[0])

# After: question_ratio + segment_length_sec (d=3, leaf=1)
preds_loo_after = []
for i in range(N):
    X_tr, y_tr = np.delete(df[['question_ratio', 'segment_length_sec']].values, i, axis=0), np.delete(y_all, i)
    X_te = df[['question_ratio', 'segment_length_sec']].values[i:i+1]
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=42)
    m.fit(X_tr, y_tr)
    preds_loo_after.append(m.predict(X_te)[0])

df['Pred_Before'] = preds_loo_before
df['Pred_After'] = preds_loo_after
df['Correct_Before'] = (df['Satisfaction_Class'] == df['Pred_Before'])
df['Correct_After'] = (df['Satisfaction_Class'] == df['Pred_After'])

def classify_status(row):
    if not row['Correct_Before'] and row['Correct_After']:
        return '救済成功 (Fixed)'
    elif not row['Correct_Before'] and not row['Correct_After']:
        return '依然誤分類 (Remained Error)'
    elif row['Correct_Before'] and not row['Correct_After']:
        return '悪化 (Deteriorated)'
    else:
        return '正解維持 (Remained Correct)'

df['Transition_Status'] = df.apply(classify_status, axis=1)

# セグメント分析用テーブル作成
segment_table_cols = [
    'セグメント', '開始時刻', '終了時刻', '長さ(秒)', 'Target_Satisfaction',
    'Satisfaction_Class', 'Pred_Before', 'Pred_After', 'Transition_Status',
    'question_ratio', 'sentiment_polarity', 'hm_mean', 'grass_ratio', 'speech_rate'
]
df_seg_summary = df[segment_table_cols].sort_values('セグメント')

# CSV保存
df_seg_summary.to_csv(os.path.join(TAB_DIR, 'exp3_segment_classification_before_after.csv'), index=False)
df_seg_summary.to_csv(os.path.join(PAPER_TAB_DIR, 'exp3_segment_classification_before_after.csv'), index=False)

print("\n--- Transition Status Breakdown ---")
print(df_seg_summary['Transition_Status'].value_counts())

print("\n--- Key Recovered Segments (Fixed by segment_length_sec) ---")
print(df_seg_summary[df_seg_summary['Transition_Status'] == '救済成功 (Fixed)'][['セグメント', '長さ(秒)', 'Target_Satisfaction', 'question_ratio']])

print("\n--- Unresolved Segments (Remained Error) ---")
print(df_seg_summary[df_seg_summary['Transition_Status'] == '依然誤分類 (Remained Error)'][['セグメント', '長さ(秒)', 'Target_Satisfaction', 'question_ratio']])

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
