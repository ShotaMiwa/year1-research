# coding: utf-8
import os, sys, warnings, time
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_matrix.csv")
OUT_DIR = os.path.join(BASE_DIR, "進捗報告用", "20261021")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "Target_Satisfaction"
median_target = df[TARGET].median()
y_all = (df[TARGET] >= median_target).astype(int).values
N = len(df)

# pause_ratio進捗報告用11???
BASE_FEATURES = [
    "comment_rate", "question_ratio", "exclamation_ratio",
    "sentiment_polarity", "subtitle_sentiment_polarity",
    "speech_rate", "grass_ratio",
    "hm_mean", "hm_max",
    "comment_semantic_variance", "sub_comment_similarity"
]

ALL_FEATURES = BASE_FEATURES + ["segment_length_sec"]

EXPERIMENTS = [
    {
        "id": "A",
        "name": "A進捗報告用????2",
        "features": BASE_FEATURES,
        "max_depth": 2,
        "max_k": 2
    },
    {
        "id": "B",
        "name": "B進捗報告用???2",
        "features": ALL_FEATURES,
        "max_depth": 2,
        "max_k": 2
    },
    {
        "id": "C",
        "name": "C進捗報告用????3",
        "features": BASE_FEATURES,
        "max_depth": 3,
        "max_k": 3
    },
    {
        "id": "D",
        "name": "D進捗報告用???3",
        "features": ALL_FEATURES,
        "max_depth": 3,
        "max_k": 3
    }
]

summary_rows = []
stability_dict = {}

for exp in EXPERIMENTS:
    exp_id = exp["id"]
    exp_name = exp["name"]
    feats = exp["features"]
    d = exp["max_depth"]
    max_k = exp["max_k"]
    
    # 進捗報告用?
    comb_list = []
    for k in range(1, max_k + 1):
        comb_list.extend(list(combinations(range(len(feats)), k)))
        
    X_mat = df[feats].values
    
    print(f"\n==================================================")
    print(f"Running Experiment {exp_name} (Combinations: {len(comb_list)})...")
    t0 = time.time()
    
    outer_preds = []
    outer_details = []
    selected_features_all = []
    
    for out_i in range(N):
        X_tr = np.delete(X_mat, out_i, axis=0)
        y_tr = np.delete(y_all, out_i)
        X_te = X_mat[out_i:out_i+1]
        y_te = y_all[out_i]
        
        n_in = len(X_tr)
        best_bal = -1.0
        best_acc = -1.0
        best_comb = None
        
        for comb in comb_list:
            sub_feats = list(comb)
            in_preds = []
            for in_i in range(n_in):
                in_X_tr = np.delete(X_tr[:, sub_feats], in_i, axis=0)
                in_y_tr = np.delete(y_tr, in_i)
                in_X_te = X_tr[in_i:in_i+1, sub_feats]
                
                clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=2, random_state=42)
                clf.fit(in_X_tr, in_y_tr)
                in_preds.append(clf.predict(in_X_te)[0])
                
            in_preds = np.array(in_preds)
            bal = balanced_accuracy_score(y_tr, in_preds)
            acc = accuracy_score(y_tr, in_preds)
            
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
                
        # Outer predict
        best_f_names = [feats[idx] for idx in best_comb]
        selected_features_all.extend(best_f_names)
        
        final_clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=2, random_state=42)
        final_clf.fit(X_tr[:, list(best_comb)], y_tr)
        pred = final_clf.predict(X_te[:, list(best_comb)])[0]
        outer_preds.append(pred)
        
        outer_details.append({
            "Outer_Fold": out_i + 1,
            "True_Class": y_te,
            "Pred_Class": pred,
            "Is_Correct": pred == y_te,
            "Inner_Best_BalAcc": best_bal,
            "Inner_Best_Acc": best_acc,
            "Selected_Features": ", ".join(best_f_names)
        })
        
    outer_preds = np.array(outer_preds)
    final_acc = accuracy_score(y_all, outer_preds)
    final_bal = balanced_accuracy_score(y_all, outer_preds)
    final_f1 = f1_score(y_all, outer_preds, average="macro")
    cm = confusion_matrix(y_all, outer_preds)
    correct_cnt = int(sum(outer_preds == y_all))
    
    elapsed = time.time() - t0
    print(f"Finished {exp_name} in {elapsed:.2f}s | BalAcc: {final_bal*100:.2f}% | Acc: {correct_cnt}/{N} ({final_acc*100:.2f}%) | Macro F1: {final_f1*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save details
    pd.DataFrame(outer_details).to_csv(os.path.join(TAB_DIR, f"exp_dt_nested_{exp_id}_details.csv"), index=False)
    
    # Feature stability
    cnts = Counter(selected_features_all)
    df_st = pd.DataFrame([
        {"Feature": f, "Selected_Count": cnts.get(f, 0), "Selection_Rate": cnts.get(f, 0) / N * 100}
        for f in feats
    ]).sort_values("Selected_Count", ascending=False).reset_index(drop=True)
    df_st.to_csv(os.path.join(TAB_DIR, f"exp_dt_nested_{exp_id}_stability.csv"), index=False)
    stability_dict[exp_id] = df_st
    
    summary_rows.append({
        "??": exp_name,
        "Accuracy": f"{correct_cnt}/{N} ({final_acc*100:.1f}%)",
        "Balanced Accuracy": f"{final_bal*100:.1f}%",
        "Macro F1": f"{final_f1*100:.1f}%",
        "TN (Low??)": cm[0, 0],
        "FP (Low??)": cm[0, 1],
        "FN (High??)": cm[1, 0],
        "TP (High??)": cm[1, 1],
        "Top1_Feature": f"{df_st.iloc[0]['Feature']} ({df_st.iloc[0]['Selection_Rate']:.1f}%)",
        "Top2_Feature": f"{df_st.iloc[1]['Feature']} ({df_st.iloc[1]['Selection_Rate']:.1f}%)",
        "Top3_Feature": f"{df_st.iloc[2]['Feature']} ({df_st.iloc[2]['Selection_Rate']:.1f}%)" if len(df_st) > 2 else "-"
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TAB_DIR, "exp_dt_nested_4cond_summary.csv"), index=False)
print("\n=== FINAL 4-CONDITION SUMMARY TABLE ===")
print(summary_df[["??", "Accuracy", "Balanced Accuracy", "Macro F1"]].to_string(index=False))
