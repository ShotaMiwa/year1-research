# coding: utf-8
"""
run_dt_nested_exhaustive_cv.py
決定木における「内側網羅探索（Inner Exhaustive Search）＋ 外側評価（Outer LOO-CV）」による完全二重交差検証 (Nested LOO-CV)
- サンプルサイズ: N=23 (LOO-CV 23分割)
- Inner Loop: 各分割の22件に対して 1~3特徴量（377通り）× パラメータ設定 (計2,197通り) を網羅探索
- 最適モデルを選定し、Outer Test (1件) に対する予測を実施
- 特徴量選抜安定性 (Feature Stability) および 真の汎化性能を出力
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

comb_dict = {}
for k in [1, 2, 3]:
    comb_dict[k] = list(combinations(range(len(FEATURE_COLS)), k))

PARAM_SETTINGS = [
    {'name': 'd1_leaf1', 'max_depth': 1, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'd2_leaf1', 'max_depth': 2, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'd2_leaf2', 'max_depth': 2, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'd3_leaf1', 'max_depth': 3, 'min_samples_leaf': 1, 'class_weight': None},
    {'name': 'd3_leaf2', 'max_depth': 3, 'min_samples_leaf': 2, 'class_weight': None},
    {'name': 'd2_leaf1_bal', 'max_depth': 2, 'min_samples_leaf': 1, 'class_weight': 'balanced'},
]

X_matrix = df[FEATURE_COLS].values

print(f"Starting Nested LOO-CV with Inner Exhaustive Search (N={N})...")
start_total = time.time()

outer_predictions = []
outer_details = []
all_selected_features = []

for outer_idx in range(N):
    t_fold = time.time()
    
    # 1. Outer split
    X_outer_tr = np.delete(X_matrix, outer_idx, axis=0)
    y_outer_tr = np.delete(y_all, outer_idx)
    X_outer_te = X_matrix[outer_idx:outer_idx+1]
    y_outer_te = y_all[outer_idx]
    
    n_inner = len(X_outer_tr) # 22
    
    # 2. Inner Exhaustive Search across 377 combinations * params
    best_inner_score = -1.0
    best_inner_acc = -1.0
    best_inner_config = None
    
    for k, combs in comb_dict.items():
        app_params = PARAM_SETTINGS if k > 1 else [p for p in PARAM_SETTINGS if p['max_depth'] == 1]
        for param in app_params:
            d = param['max_depth']
            l = param['min_samples_leaf']
            w = param['class_weight']
            
            for comb in combs:
                feats = list(comb)
                inner_preds = []
                for inner_idx in range(n_inner):
                    X_in_tr = np.delete(X_outer_tr[:, feats], inner_idx, axis=0)
                    y_in_tr = np.delete(y_outer_tr, inner_idx)
                    X_in_te = X_outer_tr[inner_idx:inner_idx+1, feats]
                    
                    clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, class_weight=w, random_state=42)
                    clf.fit(X_in_tr, y_in_tr)
                    inner_preds.append(clf.predict(X_in_te)[0])
                
                inner_preds = np.array(inner_preds)
                inner_bal = balanced_accuracy_score(y_outer_tr, inner_preds)
                inner_acc = accuracy_score(y_outer_tr, inner_preds)
                
                update = False
                if inner_bal > best_inner_score:
                    update = True
                elif inner_bal == best_inner_score:
                    if inner_acc > best_inner_acc:
                        update = True
                    elif inner_acc == best_inner_acc and best_inner_config is not None:
                        prev_feats, prev_param = best_inner_config
                        if len(feats) < len(prev_feats):
                            update = True
                        elif len(feats) == len(prev_feats) and d < prev_param['max_depth']:
                            update = True
                
                if update:
                    best_inner_score = inner_bal
                    best_inner_acc = inner_acc
                    best_inner_config = (feats, param)
                    
    # 3. Outer Evaluation with the best selected model
    best_feats, best_param = best_inner_config
    best_feat_names = [FEATURE_COLS[idx] for idx in best_feats]
    all_selected_features.extend(best_feat_names)
    
    final_clf = DecisionTreeClassifier(
        max_depth=best_param['max_depth'],
        min_samples_leaf=best_param['min_samples_leaf'],
        class_weight=best_param['class_weight'],
        random_state=42
    )
    final_clf.fit(X_outer_tr[:, best_feats], y_outer_tr)
    pred_outer = final_clf.predict(X_outer_te[:, best_feats])[0]
    outer_predictions.append(pred_outer)
    
    seg_id = df.iloc[outer_idx]['セグメント']
    is_correct = bool(pred_outer == y_outer_te)
    elapsed_fold = time.time() - t_fold
    
    outer_details.append({
        'Outer_Fold': outer_idx + 1,
        'Segment_ID': seg_id,
        'True_Class': int(y_outer_te),
        'Predicted_Class': int(pred_outer),
        'Is_Correct': is_correct,
        'Inner_Best_BalAcc': best_inner_score,
        'Inner_Best_Acc': best_inner_acc,
        'Selected_Feature_Count': len(best_feats),
        'Selected_Features': ', '.join(best_feat_names),
        'Selected_Hyperparameters': f"depth={best_param['max_depth']}, leaf={best_param['min_samples_leaf']}, weight={best_param['class_weight']}",
        'Has_Segment_Length': bool('segment_length_sec' in best_feat_names)
    })
    
    status_str = "CORRECT" if is_correct else "WRONG"
    print(f"Fold {outer_idx+1:2d}/{N} [{elapsed_fold:.1f}s]: Seg {seg_id} -> {status_str} (True:{y_outer_te}, Pred:{pred_outer}) | Inner BalAcc: {best_inner_score*100:.1f}% | Feats: {best_feat_names}")

outer_predictions = np.array(outer_predictions)
final_bal_acc = balanced_accuracy_score(y_all, outer_predictions)
final_acc = accuracy_score(y_all, outer_predictions)
final_f1 = f1_score(y_all, outer_predictions, average='macro')
correct_cnt = int(sum(outer_predictions == y_all))
total_elapsed = time.time() - start_total

print("\n=============================================")
print("=== NESTED LOO-CV FINAL RESULTS (UNBIASED) ===")
print("=============================================")
print(f"Total Samples       : {N}")
print(f"Correct Predictions : {correct_cnt} / {N} ({final_acc*100:.2f}%)")
print(f"Balanced Accuracy   : {final_bal_acc*100:.2f}%")
print(f"Macro F1 Score      : {final_f1*100:.2f}%")
print(f"Total Elapsed Time  : {total_elapsed/60:.2f} minutes")

df_details = pd.DataFrame(outer_details)
csv_detail_path = os.path.join(TAB_DIR, 'exp_dt_nested_exhaustive_details.csv')
df_details.to_csv(csv_detail_path, index=False)
df_details.to_csv(os.path.join(PAPER_TAB_DIR, 'exp_dt_nested_exhaustive_details.csv'), index=False)
print(f"Saved fold details to: {csv_detail_path}")

feat_counts = Counter(all_selected_features)
df_feat_stability = pd.DataFrame([
    {'Feature': f, 'Selected_Count': feat_counts.get(f, 0), 'Selection_Rate': feat_counts.get(f, 0) / N * 100}
    for f in FEATURE_COLS
]).sort_values(by='Selected_Count', ascending=False).reset_index(drop=True)

csv_feat_path = os.path.join(TAB_DIR, 'exp_dt_nested_feature_stability.csv')
df_feat_stability.to_csv(csv_feat_path, index=False)
df_feat_stability.to_csv(os.path.join(PAPER_TAB_DIR, 'exp_dt_nested_feature_stability.csv'), index=False)
print(f"Saved feature stability to: {csv_feat_path}")

print("\n=== Feature Selection Stability across 23 Folds ===")
print(df_feat_stability.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if f == 'segment_length_sec' else '#3498db' for f in df_feat_stability['Feature']]
sns.barplot(data=df_feat_stability, x='Selection_Rate', y='Feature', palette=colors)
plt.title('Feature Selection Stability in Nested LOO-CV (23 Folds, Inner Exhaustive Search)', fontsize=13, fontweight='bold')
plt.xlabel('Selection Rate (%) [Percentage of outer folds selecting feature]')
plt.ylabel('Feature Name')
plt.xlim(0, 105)
for idx, r in df_feat_stability.iterrows():
    cnt = int(r['Selected_Count'])
    rate = r['Selection_Rate']
    plt.text(rate + 1, idx, f"{cnt}/23 ({rate:.1f}%)", va='center', fontsize=9)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'exp_dt_nested_feature_stability.png')
plt.savefig(fig_path, dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, 'exp_dt_nested_feature_stability.png'), dpi=300)
plt.close()
print(f"Saved stability plot to: {fig_path}")
