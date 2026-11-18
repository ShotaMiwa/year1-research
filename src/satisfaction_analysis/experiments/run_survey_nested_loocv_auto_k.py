# -*- coding: utf-8 -*-
"""
run_survey_nested_loocv_auto_k.py

アンケート23件内部に対しても、
「特徴量個数の制限なし・内部LOO-CVで最高精度となったステップ k を自動選抜する」
厳密な Nested LOO-CV を実行し、完全自動ピーク決定の真の汎化精度を算出する。
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

PROJECT_ROOT = "/home/shota/work/year1"
df_survey = pd.read_csv(f"{PROJECT_ROOT}/data/processed/feature_matrix.csv")

FEATURE_COLS = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_vocab_diversity',
    'comment_semantic_variance', 'segment_length_sec'
]

y = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
n = len(df_survey)

def run_nested_auto_k(clf_fn, name):
    outer_preds = []
    selected_k_log = []
    
    for i in range(n):
        tr_idx = [j for j in range(n) if j != i]
        te_idx = [i]
        
        df_tr = df_survey.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        df_te = df_survey.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]
        
        n_in = len(df_tr)
        
        # 内側22件を標準化
        sc_in = StandardScaler()
        X_in_all = sc_in.fit_transform(df_tr[FEATURE_COLS].values)
        X_te_val = sc_in.transform(df_te[FEATURE_COLS].values)
        
        # 内側SFS: step 1 から 12 まで進め、内側LOO-CVで最良ステップを記録
        selected = []
        rem = list(range(len(FEATURE_COLS)))
        step_scores = []
        step_features = []
        
        for step in range(len(FEATURE_COLS)):
            best_cand = None
            best_cv = -1.0
            for cand in rem:
                curr = selected + [cand]
                # 内側LOO-CV (22分割)
                in_preds = []
                for k_in in range(n_in):
                    k_tr = [m for m in range(n_in) if m != k_in]
                    k_va = [k_in]
                    c = clf_fn()
                    c.fit(X_in_all[k_tr][:, curr], y_tr[k_tr])
                    in_preds.append(c.predict(X_in_all[k_va][:, curr])[0])
                acc = np.mean(in_preds == y_tr)
                if acc > best_cv:
                    best_cv = acc
                    best_cand = cand
            selected.append(best_cand)
            rem.remove(best_cand)
            step_scores.append(best_cv)
            step_features.append(list(selected))
            
        # 内側で最高CVスコアとなったステップ（個数 k）を採用
        best_step = np.argmax(step_scores)
        best_idxs = step_features[best_step]
        selected_k_log.append(len(best_idxs))
        
        # 内側全体で学習 -> 未知の1件をテスト
        final_clf = clf_fn()
        final_clf.fit(X_in_all[:, best_idxs], y_tr)
        out_pred = final_clf.predict(X_te_val[:, best_idxs])[0]
        outer_preds.append(out_pred)
        
    outer_preds = np.array(outer_preds)
    acc = accuracy_score(y, outer_preds) * 100
    cor = sum(outer_preds == y)
    print(f"[{name}] 完全自動ピーク決定 (Nested LOO-CV):")
    print(f"  的中数: {cor} / 23 ({acc:.1f}%)")
    print(f"  各Foldで選ばれた個数 k: {selected_k_log}")
    return acc, cor, selected_k_log

print("=== アンケート23件側：完全自動ピーク決定のNested LOO-CV計算開始 ===")
svm_acc, svm_cor, _ = run_nested_auto_k(lambda: SVC(kernel="linear", C=1.0, random_state=42), "Linear SVM")
lr_acc, lr_cor, _ = run_nested_auto_k(lambda: LogisticRegression(C=1.0, random_state=42), "ロジスティック回帰")
