# coding: utf-8
"""
決定木（Decision Tree）における Nested LOO-CV（二重交差検証 + SFS）評価スクリプト
10/14のRandomForestと同じ条件（Nested LOO-CV, SFSで3特徴量選択, パラメータグリッド10種）にて、
新特徴量 segment_length_sec の追加前 (12特徴量) と追加後 (13特徴量) を厳密に比較・評価する。
"""

import os, sys, warnings
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

df = pd.read_csv(DATA_PATH)
TARGET = 'Target_Satisfaction'
median_target = df[TARGET].median()
y_all = (df[TARGET] >= median_target).astype(int).values
N = len(df)

FEATURE_COLS_BEFORE = [
    'comment_rate', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity',
    'speech_rate', 'pause_ratio', 'grass_ratio',
    'hm_mean', 'hm_max',
    'comment_semantic_variance', 'sub_comment_similarity',
]
FEATURE_COLS_AFTER = FEATURE_COLS_BEFORE + ['segment_length_sec']

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
    {'name': 'Default (max_depth=None)', 'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'Depth=1 (Stump)', 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'Depth=2, Leaf=2', 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'Depth=2, Leaf=3', 'max_depth': 2, 'min_samples_leaf': 3, 'class_weight': None},
    {'name': 'Depth=3, Leaf=2', 'max_depth': 3, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'Depth=2, Leaf=2, Balanced', 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': 'balanced'},
    {'name': 'Depth=1, Balanced', 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': 'balanced'},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=1', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=2, Leaf=2', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=2, Leaf=2, Balanced', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': 'balanced'},
]

results = []

for condition_label, feat_cols in [('Before (12 Features)', FEATURE_COLS_BEFORE), ('After (+segment_length, 13 Features)', FEATURE_COLS_AFTER)]:
    X_all = df[feat_cols].values
    for param in param_grid_dt:
        p_name = param['name']
        is_fixed = 'fixed_feats' in param
        outer_preds = []
        chosen_features_list = []
        for i in range(N):
            X_outer_tr, y_outer_tr = np.delete(X_all, i, axis=0), np.delete(y_all, i)
            X_outer_te, y_outer_te = X_all[i:i+1], y_all[i:i+1]
            if is_fixed:
                feat_indices = [feat_cols.index(f) for f in param['fixed_feats']]
            else:
                factory = lambda: DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], class_weight=param['class_weight'], random_state=42)
                feat_indices = sfs_select(X_outer_tr, y_outer_tr, factory, k_features=3)
            chosen_features_list.extend([feat_cols[idx] for idx in feat_indices])
            clf = DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], class_weight=param['class_weight'], random_state=42)
            clf.fit(X_outer_tr[:, feat_indices], y_outer_tr)
            pred = clf.predict(X_outer_te[:, feat_indices])[0]
            outer_preds.append(pred)
        outer_preds = np.array(outer_preds)
        acc = accuracy_score(y_all, outer_preds)
        bal_acc = balanced_accuracy_score(y_all, outer_preds)
        f1 = f1_score(y_all, outer_preds, average='macro')
        top_feats = ", ".join([f"{f}({c})" for f, c in Counter(chosen_features_list).most_common(3)]) if not is_fixed else "Fixed: " + ", ".join(param['fixed_feats'])
        results.append({
            'Condition': condition_label,
            'Model_Setting': p_name,
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'F1_Macro': f1,
            'Correct_Count': int(sum(outer_preds == y_all)),
            'Total': N,
            'Top_Selected_Features': top_feats
        })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(TAB_DIR, 'dt_nested_cv_comparison.csv'), index=False)
print("Saved table to:", os.path.join(TAB_DIR, 'dt_nested_cv_comparison.csv'))

# 可視化：Before vs After の比較プロット
plt.figure(figsize=(12, 7))
settings = [p['name'] for p in param_grid_dt]
df_before = res_df[res_df['Condition'] == 'Before (12 Features)'].set_index('Model_Setting').loc[settings]
df_after = res_df[res_df['Condition'] == 'After (+segment_length, 13 Features)'].set_index('Model_Setting').loc[settings]

y_pos = np.arange(len(settings))
height = 0.35

plt.barh(y_pos - height/2, df_before['Balanced_Accuracy'] * 100, height, label='Before (12 Features: No Length)', color='#A0AEC0', alpha=0.9)
bars_after = plt.barh(y_pos + height/2, df_after['Balanced_Accuracy'] * 100, height, label='After (13 Features: +segment_length)', color='#3182CE', alpha=0.9)

# 強調表示（Depth=1 Stump）
bars_after[1].set_color('#E53E3E')

for idx, val in enumerate(df_before['Balanced_Accuracy'] * 100):
    plt.text(val + 1, y_pos[idx] - height/2, f"{val:.1f}%", va='center', fontsize=9, color='#4A5568')
for idx, val in enumerate(df_after['Balanced_Accuracy'] * 100):
    plt.text(val + 1, y_pos[idx] + height/2, f"{val:.1f}%", va='center', fontsize=9, fontweight='bold', color='#1A365D')

plt.yticks(y_pos, settings, fontsize=10)
plt.xlabel('Balanced Accuracy (%)', fontsize=12)
plt.title('Decision Tree: Nested LOO-CV Performance Comparison (Before vs After New Feature)', fontsize=13, fontweight='bold')
plt.axvline(x=50.0, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
plt.xlim(0, 100)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'exp_dt_nested_cv_comparison.png'), dpi=300)
plt.close()
print("Saved figure to:", os.path.join(FIG_DIR, 'exp_dt_nested_cv_comparison.png'))
