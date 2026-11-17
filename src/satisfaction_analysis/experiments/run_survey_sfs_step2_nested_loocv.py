# -*- coding: utf-8 -*-
"""
run_survey_sfs_step2_nested_loocv.py

アンケートデータ（N=23）に対して、
今回の73件と全く同一の手続き（制限なしSFSのStep 2）を
データ漏洩のないNested LOO-CVで実行し、
Linear SVM および ロジスティック回帰（L2）の真の未知汎化精度を算出する。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

PROJECT_ROOT = Path("/home/shota/work/year1")
SURVEY_CSV = PROJECT_ROOT / "data/processed/feature_matrix.csv"

# 共通全11特徴量（ヒートマップなし）
FEATS_COMMON11 = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_semantic_variance',
    'segment_length_sec'
]

def run_sfs_step2_nested_cv(model_name: str, model_fn):
    df = pd.read_csv(SURVEY_CSV)
    y = (df['Target_Satisfaction'] >= df['Target_Satisfaction'].median()).astype(int).values
    n = len(df)
    
    outer_preds = []
    selected_pairs_log = []
    
    for i in range(n):
        # 外側分割: 訓練22件, テスト1件
        tr_idx = [j for j in range(n) if j != i]
        te_idx = [i]
        
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        df_te = df.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]
        
        # 内側22件の中で SFS (Step 1 -> Step 2) を実行
        # 内側の評価は内側LOO-CV (22分割)
        n_in = len(df_tr)
        
        # --- Step 1: 最良の1特徴量を探索 ---
        best_f1 = None
        best_score1 = -1.0
        
        for cand in FEATS_COMMON11:
            scores = []
            for k in range(n_in):
                in_tr = [m for m in range(n_in) if m != k]
                in_va = [k]
                
                scaler = StandardScaler()
                X_in_tr = scaler.fit_transform(df_tr.iloc[in_tr][[cand]].values)
                X_in_va = scaler.transform(df_tr.iloc[in_va][[cand]].values)
                
                clf = model_fn()
                clf.fit(X_in_tr, y_tr[in_tr])
                pred = clf.predict(X_in_va)[0]
                scores.append(pred == y_tr[k])
            
            acc = np.mean(scores)
            if acc > best_score1:
                best_score1 = acc
                best_f1 = cand
                
        # --- Step 2: 2つ目の特徴量を追加 ---
        best_f2 = None
        best_score2 = -1.0
        
        rem_cands = [c for c in FEATS_COMMON11 if c != best_f1]
        for cand in rem_cands:
            pair = [best_f1, cand]
            scores = []
            for k in range(n_in):
                in_tr = [m for m in range(n_in) if m != k]
                in_va = [k]
                
                scaler = StandardScaler()
                X_in_tr = scaler.fit_transform(df_tr.iloc[in_tr][pair].values)
                X_in_va = scaler.transform(df_tr.iloc[in_va][pair].values)
                
                clf = model_fn()
                clf.fit(X_in_tr, y_tr[in_tr])
                pred = clf.predict(X_in_va)[0]
                scores.append(pred == y_tr[k])
                
            acc = np.mean(scores)
            if acc > best_score2:
                best_score2 = acc
                best_f2 = cand
                
        final_pair = [best_f1, best_f2]
        selected_pairs_log.append(tuple(final_pair))
        
        # 内側22件全体で scaler fit してモデル学習
        scaler_outer = StandardScaler()
        X_outer_tr = scaler_outer.fit_transform(df_tr[final_pair].values)
        X_outer_te = scaler_outer.transform(df_te[final_pair].values)
        
        clf_outer = model_fn()
        clf_outer.fit(X_outer_tr, y_tr)
        out_pred = clf_outer.predict(X_outer_te)[0]
        outer_preds.append(out_pred)
        
    outer_preds = np.array(outer_preds)
    bacc = balanced_accuracy_score(y, outer_preds)
    acc = accuracy_score(y, outer_preds)
    f1 = f1_score(y, outer_preds, average='macro')
    correct = sum(y == outer_preds)
    
    print(f"\n[{model_name}] 制限なしSFS Step 2 結果 (Nested LOO-CV):")
    print(f"  Balanced Accuracy: {bacc*100:.2f}%")
    print(f"  Accuracy         : {acc*100:.2f}% ({correct}/{n} 正解)")
    print(f"  Macro F1         : {f1:.4f}")
    
    # 頻出ペアの表示
    from collections import Counter
    pair_counts = Counter(selected_pairs_log)
    print("  選ばれたペアTop3:", pair_counts.most_common(3))
    
    return bacc, acc, correct

def main():
    print("=" * 70)
    print("アンケート23件：制限なしSFS Step 2 の厳密Nested LOO-CV検証")
    print("=" * 70)
    
    svm_bacc, svm_acc, svm_corr = run_sfs_step2_nested_cv(
        "Linear SVM (C=1.0)",
        lambda: SVC(kernel='linear', C=1.0, random_state=42)
    )
    
    lr_bacc, lr_acc, lr_corr = run_sfs_step2_nested_cv(
        "Logistic Regression (L2, C=1.0)",
        lambda: LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)
    )

if __name__ == "__main__":
    main()
