# coding: utf-8
import os, sys, warnings, time
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_matrix.csv")
OUT_DIR = os.path.join(BASE_DIR, "進捗報告用", "20261021")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

PAPER_DIR = os.path.join(BASE_DIR, "docs", "paper")
PAPER_FIG_DIR = os.path.join(PAPER_DIR, "figures")
PAPER_TAB_DIR = os.path.join(PAPER_DIR, "tables")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(PAPER_TAB_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "Target_Satisfaction"
median_target = df[TARGET].median()
y_all = (df[TARGET] >= median_target).astype(int).values
N = len(df)

FEATURE_COLS = [
    "comment_rate", "question_ratio", "exclamation_ratio",
    "sentiment_polarity", "subtitle_sentiment_polarity",
    "speech_rate", "pause_ratio", "grass_ratio",
    "hm_mean", "hm_max",
    "comment_semantic_variance", "sub_comment_similarity",
    "segment_length_sec"
]

all_combs = []
for k in [1, 2, 3]:
    all_combs.extend(list(combinations(range(len(FEATURE_COLS)), k)))

X_matrix = df[FEATURE_COLS].values

print(f"Starting RF Nested LOO-CV with Inner Exhaustive Search (N={N})...")
print(f"Total feature combinations per fold: {len(all_combs)} (RF: n_est=50, max_depth=2, min_leaf=2)")
start_total = time.time()

def eval_single_comb(comb, X_tr_22, y_tr_22):
    feats = list(comb)
    n_in = len(X_tr_22)
    preds = []
    for i in range(n_in):
        X_sub_tr = np.delete(X_tr_22[:, feats], i, axis=0)
        y_sub_tr = np.delete(y_tr_22, i)
        X_sub_te = X_tr_22[i:i+1, feats]
        clf = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=2, random_state=42, n_jobs=1)
        clf.fit(X_sub_tr, y_sub_tr)
        preds.append(clf.predict(X_sub_te)[0])
    preds = np.array(preds)
    bal = balanced_accuracy_score(y_tr_22, preds)
    acc = accuracy_score(y_tr_22, preds)
    return (bal, acc, comb)

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
    
    # 2. Parallel evaluation of all 377 combinations
    results = Parallel(n_jobs=14)(
        delayed(eval_single_comb)(comb, X_outer_tr, y_outer_tr) for comb in all_combs
    )
    
    # Find best config
    best_bal = -1.0
    best_acc = -1.0
    best_comb = None
    
    for bal, acc, comb in results:
        update = False
        if bal > best_bal:
            update = True
        elif bal == best_bal:
            if acc > best_acc:
                update = True
            elif acc == best_acc and best_comb is not None:
                if len(comb) < len(best_comb):
                    update = True
        if update:
            best_bal = bal
            best_acc = acc
            best_comb = comb
            
    best_feats = list(best_comb)
    best_feat_names = [FEATURE_COLS[idx] for idx in best_feats]
    all_selected_features.extend(best_feat_names)
    
    # 3. Outer fit & predict
    final_clf = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=2, random_state=42, n_jobs=-1)
    final_clf.fit(X_outer_tr[:, best_feats], y_outer_tr)
    pred_outer = final_clf.predict(X_outer_te[:, best_feats])[0]
    outer_predictions.append(pred_outer)
    
    seg_id = outer_idx + 1
    is_correct = bool(pred_outer == y_outer_te)
    elapsed_fold = time.time() - t_fold
    
    outer_details.append({
        "Outer_Fold": outer_idx + 1,
        "Segment_ID": seg_id,
        "True_Class": int(y_outer_te),
        "Predicted_Class": int(pred_outer),
        "Is_Correct": is_correct,
        "Inner_Best_BalAcc": best_bal,
        "Inner_Best_Acc": best_acc,
        "Selected_Feature_Count": len(best_feats),
        "Selected_Features": ", ".join(best_feat_names),
        "Has_Segment_Length": bool("segment_length_sec" in best_feat_names)
    })
    
    status_str = "CORRECT" if is_correct else "WRONG"
    print(f"Fold {outer_idx+1:2d}/{N} [{elapsed_fold:.1f}s]: Seg {seg_id} -> {status_str} (True:{y_outer_te}, Pred:{pred_outer}) | Inner BalAcc: {best_bal*100:.1f}% | Feats: {best_feat_names}", flush=True)

outer_predictions = np.array(outer_predictions)
final_bal_acc = balanced_accuracy_score(y_all, outer_predictions)
final_acc = accuracy_score(y_all, outer_predictions)
final_f1 = f1_score(y_all, outer_predictions, average="macro")
correct_cnt = int(sum(outer_predictions == y_all))
total_elapsed = time.time() - start_total

print("\n=============================================")
print("=== RF NESTED LOO-CV FINAL RESULTS (UNBIASED) ===")
print("=============================================")
print(f"Total Samples       : {N}")
print(f"Correct Predictions : {correct_cnt} / {N} ({final_acc*100:.2f}%)")
print(f"Balanced Accuracy   : {final_bal_acc*100:.2f}%")
print(f"Macro F1 Score      : {final_f1*100:.2f}%")
print(f"Total Elapsed Time  : {total_elapsed/60:.2f} minutes")

df_details = pd.DataFrame(outer_details)
csv_detail_path = os.path.join(TAB_DIR, "exp_rf_nested_exhaustive_details.csv")
df_details.to_csv(csv_detail_path, index=False)
df_details.to_csv(os.path.join(PAPER_TAB_DIR, "exp_rf_nested_exhaustive_details.csv"), index=False)
print(f"Saved fold details to: {csv_detail_path}")

feat_counts = Counter(all_selected_features)
df_feat_stability = pd.DataFrame([
    {"Feature": f, "Selected_Count": feat_counts.get(f, 0), "Selection_Rate": feat_counts.get(f, 0) / N * 100}
    for f in FEATURE_COLS
]).sort_values(by="Selected_Count", ascending=False).reset_index(drop=True)

csv_feat_path = os.path.join(TAB_DIR, "exp_rf_nested_feature_stability.csv")
df_feat_stability.to_csv(csv_feat_path, index=False)
df_feat_stability.to_csv(os.path.join(PAPER_TAB_DIR, "exp_rf_nested_feature_stability.csv"), index=False)
print(f"Saved feature stability to: {csv_feat_path}")

print("\n=== RF Feature Selection Stability across 23 Folds ===")
print(df_feat_stability.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
colors = ["#e74c3c" if f == "segment_length_sec" else "#2ecc71" for f in df_feat_stability["Feature"]]
sns.barplot(data=df_feat_stability, x="Selection_Rate", y="Feature", palette=colors)
plt.title("RF Feature Selection Stability in Nested LOO-CV (23 Folds, Inner Exhaustive Search)", fontsize=13, fontweight="bold")
plt.xlabel("Selection Rate (%)")
plt.ylabel("Feature Name")
plt.xlim(0, 105)
for i, v in enumerate(df_feat_stability["Selection_Rate"]):
    cnt = df_feat_stability.loc[i, "Selected_Count"]
    plt.text(v + 1, i, f"{v:.1f}% ({cnt}/{N})", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "exp_rf_nested_feature_stability.png"), dpi=300)
plt.savefig(os.path.join(PAPER_FIG_DIR, "exp_rf_nested_feature_stability.png"), dpi=300)
plt.close()
print("Saved figure: exp_rf_nested_feature_stability.png")
