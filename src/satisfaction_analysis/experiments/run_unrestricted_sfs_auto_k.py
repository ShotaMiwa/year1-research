# -*- coding: utf-8 -*-
"""
run_unrestricted_sfs_auto_k.py

特徴量個数の上限（2個などの制限）を一切設けず、
アルゴリズムに1個から全次元まで前進選択させ、
交差検証スコアが最大となった「最適な特徴量数 k」と「その特徴量セット」を自律決定させる。
① アンケート23件内部 (Nested LOO-CV)
② YouTube実測73件内部 (5-Fold CV) -> 23件へ外部転移
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# 1. データ読み込み
PROJECT_ROOT = "/home/shota/work/year1"
df_survey = pd.read_csv(f"{PROJECT_ROOT}/data/processed/feature_matrix.csv")
df_shorts = pd.read_csv(f"{PROJECT_ROOT}/data/processed/extended_feature_matrix_77.csv")
outliers = {'LD06GCUS6cI', 'Msd39_BhMqU', '4KMwiyfGxA0', 'UEvV6gO5l88'}
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(outliers)].copy()

FEATURE_COLS = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_vocab_diversity',
    'comment_semantic_variance', 'segment_length_sec'
]

y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
y_shorts = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values


# ==============================================================================
# 実験②：73件側で「個数制限なしSFS」を実行（5-Fold CVで最良ステップを自動判定）
# ==============================================================================
def run_sfs_unrestricted_73(clf_fn):
    sc_tr = StandardScaler()
    X_tr_z = sc_tr.fit_transform(df_shorts_73[FEATURE_COLS].values)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    selected = []
    rem = list(range(len(FEATURE_COLS)))
    
    step_scores = []
    step_features = []
    
    for step in range(len(FEATURE_COLS)):
        best_cand = None
        best_cv = -1.0
        
        for cand in rem:
            curr_feats = selected + [cand]
            fold_accs = []
            for tr_idx, va_idx in cv.split(X_tr_z):
                clf = clf_fn()
                clf.fit(X_tr_z[tr_idx][:, curr_feats], y_shorts[tr_idx])
                preds = clf.predict(X_tr_z[va_idx][:, curr_feats])
                fold_accs.append(accuracy_score(y_shorts[va_idx], preds))
            mean_acc = np.mean(fold_accs)
            if mean_acc > best_cv:
                best_cv = mean_acc
                best_cand = cand
                
        selected.append(best_cand)
        rem.remove(best_cand)
        step_scores.append(best_cv)
        step_features.append(list(selected))
        
    best_step = np.argmax(step_scores)
    best_k = best_step + 1
    best_feat_idxs = step_features[best_step]
    best_feat_names = [FEATURE_COLS[i] for i in best_feat_idxs]
    
    # 最良特徴量セットで全量学習 -> アンケート23件テスト
    sc_te = StandardScaler()
    X_te_z = sc_te.fit_transform(df_survey[FEATURE_COLS].values)
    
    final_clf = clf_fn()
    final_clf.fit(X_tr_z[:, best_feat_idxs], y_shorts)
    test_preds = final_clf.predict(X_te_z[:, best_feat_idxs])
    test_acc = accuracy_score(y_survey, test_preds) * 100
    test_cor = sum(test_preds == y_survey)
    
    return best_k, step_scores[best_step]*100, best_feat_names, test_acc, test_cor, step_scores


print("=== 【実験②: 73件側】個数制限なしSFS（自動ピーク決定） ===")
# Linear SVM
k_svm, cv_svm, feats_svm, te_acc_svm, te_cor_svm, scores_svm = run_sfs_unrestricted_73(
    lambda: SVC(kernel="linear", C=1.0, random_state=42)
)
print(f"[Linear SVM]")
print(f"  各ステップ(k=1〜12)の73件内部CV精度: {[round(s*100, 1) for s in scores_svm]}")
print(f"  -> 最良個数 k = {k_svm} 個 (73件内部CV: {cv_svm:.1f}%)")
print(f"  -> 選ばれた特徴量: {feats_svm}")
print(f"  -> 未知アンケート的中数: {te_cor_svm} / 23 ({te_acc_svm:.1f}%)\n")

# LogReg
k_lr, cv_lr, feats_lr, te_acc_lr, te_cor_lr, scores_lr = run_sfs_unrestricted_73(
    lambda: LogisticRegression(C=1.0, random_state=42)
)
print(f"[ロジスティック回帰]")
print(f"  各ステップ(k=1〜12)の73件内部CV精度: {[round(s*100, 1) for s in scores_lr]}")
print(f"  -> 最良個数 k = {k_lr} 個 (73件内部CV: {cv_lr:.1f}%)")
print(f"  -> 選ばれた特徴量: {feats_lr}")
print(f"  -> 未知アンケート的中数: {te_cor_lr} / 23 ({te_acc_lr:.1f}%)")
