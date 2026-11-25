# -*- coding: utf-8 -*-
"""
evaluate_no_pause_ratio_pipeline.py

pause_ratio (ポーズ比率) を除外した「全11特徴量」環境における、
11/25進捗報告の全実験の完全再計算スクリプト。

【対象実験】
1. 73件モデルの初期特徴量重みランキング (|w|)
2. SFS (前進貪欲選択) の自動選抜 (アンケート23件 Nested LOO-CV & 73件 5-Fold CV -> 23件転移)
3. RFE (再帰的特徴消去) の自動選抜 (アンケート23件 Nested LOO-CV & 73件 5-Fold CV -> 23件転移)
4. アンケート23件 Nested LOO-CV の全23フォールドにおける特徴量選抜頻度 (新コア特徴量の特定)
5. 固定特徴量セットによる転移的中率 (新Core3, 新Core4など)
6. 全11特徴量そのまま投入時の転移的中率
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

PROJECT_ROOT = "/home/shota/work/year1"
sys.path.insert(0, PROJECT_ROOT)

from src.satisfaction_analysis.constants import FEATURE_NAMES_12, OUTLIER_SHORTS_IDS
from src.satisfaction_analysis.data_validator import load_and_validate_datasets

df_survey, df_shorts = load_and_validate_datasets()
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(OUTLIER_SHORTS_IDS)].copy().reset_index(drop=True)

y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
y_shorts = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values

n_survey = len(df_survey)
n_shorts = len(df_shorts_73)

# ★ pause_ratio を除外した「全11特徴量」
FEATURE_COLS_11 = [f for f in FEATURE_NAMES_12 if f != 'pause_ratio']
print(f"特徴量数: {len(FEATURE_COLS_11)} 個 (pause_ratio 除外)")
print(f"特徴量一覧: {FEATURE_COLS_11}")

def get_weights(clf):
    return np.abs(clf.coef_[0])

# ==============================================================================
# 実験1: 73件全量における初期11特徴量の重み絶対値 |w|
# ==============================================================================
print("\n" + "="*80)
print("【実験1】73件全量における初期11特徴量の重み絶対値 |w| (重要度)")
print("="*80)

sc_73 = StandardScaler()
X_73_z = sc_73.fit_transform(df_shorts_73[FEATURE_COLS_11].values)

svm_init = SVC(kernel='linear', C=1.0, random_state=42).fit(X_73_z, y_shorts)
lr_init  = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000).fit(X_73_z, y_shorts)

w_svm = get_weights(svm_init)
w_lr  = get_weights(lr_init)

df_weights = pd.DataFrame({
    'feature': FEATURE_COLS_11,
    'svm_weight': w_svm,
    'lr_weight': w_lr
}).sort_values('svm_weight', ascending=False).reset_index(drop=True)
df_weights['rank'] = range(1, len(df_weights) + 1)
print(df_weights[['rank', 'feature', 'svm_weight', 'lr_weight']].to_string(index=False))


# ==============================================================================
# 実験2: 全11特徴量をそのまま投入した時の精度
# ==============================================================================
print("\n" + "="*80)
print("【実験2】全11特徴量をそのまま投入した時の精度")
print("="*80)

sc_23 = StandardScaler()
X_23_z = sc_23.fit_transform(df_survey[FEATURE_COLS_11].values)

# 73件全量学習 -> 23件テスト
svm_all = SVC(kernel='linear', C=1.0, random_state=42).fit(X_73_z, y_shorts)
pred_svm_all = svm_all.predict(X_23_z)
acc_svm_all = np.sum(pred_svm_all == y_survey)

lr_all = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000).fit(X_73_z, y_shorts)
pred_lr_all = lr_all.predict(X_23_z)
acc_lr_all = np.sum(pred_lr_all == y_survey)

print(f"全11特徴量投入 Linear SVM 転移的中率        : {acc_svm_all:2d}/{n_survey} ({acc_svm_all/n_survey*100:.1f}%)")
print(f"全11特徴量投入 Logistic Regression 転移的中率 : {acc_lr_all:2d}/{n_survey} ({acc_lr_all/n_survey*100:.1f}%)")


# ==============================================================================
# 実験3: SFS (前進探索) 完全自動選抜
# ==============================================================================
print("\n" + "="*80)
print("【実験3】SFS (前進探索) 完全自動選抜 (pause_ratio 除外)")
print("="*80)

def run_sfs_survey_nested(clf_fn):
    outer_preds = []
    selected_k_log = []
    for i in range(n_survey):
        tr_idx = [j for j in range(n_survey) if j != i]
        te_idx = [i]
        df_tr = df_survey.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_survey[tr_idx]
        df_te = df_survey.iloc[te_idx].reset_index(drop=True)
        y_te = y_survey[te_idx]
        
        sc_in = StandardScaler()
        X_in_all = sc_in.fit_transform(df_tr[FEATURE_COLS_11].values)
        X_te_val = sc_in.transform(df_te[FEATURE_COLS_11].values)
        
        selected = []
        rem = list(range(len(FEATURE_COLS_11)))
        step_scores = []
        step_feats = []
        
        for step in range(len(FEATURE_COLS_11)):
            best_c = None
            best_cv = -1.0
            for cand in rem:
                curr = selected + [cand]
                in_preds = []
                for k_in in range(len(df_tr)):
                    k_tr = [m for m in range(len(df_tr)) if m != k_in]
                    c = clf_fn()
                    c.fit(X_in_all[k_tr][:, curr], y_tr[k_tr])
                    in_preds.append(c.predict(X_in_all[[k_in]][:, curr])[0])
                acc = np.mean(in_preds == y_tr)
                if acc > best_cv:
                    best_cv = acc
                    best_c = cand
            selected.append(best_c)
            rem.remove(best_c)
            step_scores.append(best_cv)
            step_feats.append(list(selected))
            
        best_step = np.argmax(step_scores)
        best_k = best_step + 1
        best_feat_idxs = step_feats[best_step]
        selected_k_log.append(best_k)
        
        clf_out = clf_fn()
        clf_out.fit(X_in_all[:, best_feat_idxs], y_tr)
        outer_preds.append(clf_out.predict(X_te_val[:, best_feat_idxs])[0])
        
    correct = np.sum(outer_preds == y_survey)
    return correct, correct / n_survey, np.mean(selected_k_log)

def run_sfs_transfer(clf_fn):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    selected = []
    rem = list(range(len(FEATURE_COLS_11)))
    step_scores = []
    step_feats = []
    
    for step in range(len(FEATURE_COLS_11)):
        best_c = None
        best_cv = -1.0
        for cand in rem:
            curr = selected + [cand]
            fold_accs = []
            for tr_idx, va_idx in cv.split(X_73_z):
                c = clf_fn()
                c.fit(X_73_z[tr_idx][:, curr], y_shorts[tr_idx])
                preds = c.predict(X_73_z[va_idx][:, curr])
                fold_accs.append(accuracy_score(y_shorts[va_idx], preds))
            mean_acc = np.mean(fold_accs)
            if mean_acc > best_cv:
                best_cv = mean_acc
                best_c = cand
        selected.append(best_c)
        rem.remove(best_c)
        step_scores.append(best_cv)
        step_feats.append(list(selected))
        
    best_step = np.argmax(step_scores)
    best_k = best_step + 1
    best_feat_idxs = step_feats[best_step]
    best_feat_names = [FEATURE_COLS_11[i] for i in best_feat_idxs]
    
    clf_final = clf_fn()
    clf_final.fit(X_73_z[:, best_feat_idxs], y_shorts)
    test_preds = clf_final.predict(X_23_z[:, best_feat_idxs])
    correct = np.sum(test_preds == y_survey)
    return best_k, step_scores[best_step], best_feat_names, correct, correct / n_survey

# SFS実行
svm_fn = lambda: SVC(kernel='linear', C=1.0, random_state=42)
lr_fn  = lambda: LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)

sfs_svm_in_corr, sfs_svm_in_acc, sfs_svm_in_k = run_sfs_survey_nested(svm_fn)
sfs_svm_tr_k, sfs_svm_tr_cv, sfs_svm_tr_feats, sfs_svm_tr_corr, sfs_svm_tr_acc = run_sfs_transfer(svm_fn)

sfs_lr_in_corr, sfs_lr_in_acc, sfs_lr_in_k = run_sfs_survey_nested(lr_fn)
sfs_lr_tr_k, sfs_lr_tr_cv, sfs_lr_tr_feats, sfs_lr_tr_corr, sfs_lr_tr_acc = run_sfs_transfer(lr_fn)

print(f"SFS Linear SVM: 内部Nested {sfs_svm_in_corr}/{n_survey} ({sfs_svm_in_acc*100:.1f}%, 平均k={sfs_svm_in_k:.1f}) | 転移 {sfs_svm_tr_corr}/{n_survey} ({sfs_svm_tr_acc*100:.1f}%, k={sfs_svm_tr_k}: {sfs_svm_tr_feats})")
print(f"SFS Logistic:   内部Nested {sfs_lr_in_corr}/{n_survey} ({sfs_lr_in_acc*100:.1f}%, 平均k={sfs_lr_in_k:.1f}) | 転移 {sfs_lr_tr_corr}/{n_survey} ({sfs_lr_tr_acc*100:.1f}%, k={sfs_lr_tr_k}: {sfs_lr_tr_feats})")


# ==============================================================================
# 実験4: RFE (再帰的消去) 完全自動選抜 & 各フォールド詳細
# ==============================================================================
print("\n" + "="*80)
print("【実験4】RFE (再帰的消去) 完全自動選抜 (pause_ratio 除外)")
print("="*80)

def run_rfe_survey_nested(clf_fn, name):
    outer_preds = []
    selected_k_log = []
    feat_freq = {f: 0 for f in FEATURE_COLS_11}
    
    for i in range(n_survey):
        tr_idx = [j for j in range(n_survey) if j != i]
        te_idx = [i]
        df_tr = df_survey.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_survey[tr_idx]
        df_te = df_survey.iloc[te_idx].reset_index(drop=True)
        y_te = y_survey[te_idx]
        
        sc_in = StandardScaler()
        X_in_all = sc_in.fit_transform(df_tr[FEATURE_COLS_11].values)
        X_te_val = sc_in.transform(df_te[FEATURE_COLS_11].values)
        
        current = list(range(len(FEATURE_COLS_11)))
        rfe_history = []
        
        while len(current) >= 1:
            k = len(current)
            in_preds = []
            for k_in in range(len(df_tr)):
                k_tr = [m for m in range(len(df_tr)) if m != k_in]
                c = clf_fn()
                c.fit(X_in_all[k_tr][:, current], y_tr[k_tr])
                in_preds.append(c.predict(X_in_all[[k_in]][:, current])[0])
            inner_acc = np.mean(in_preds == y_tr)
            rfe_history.append((k, list(current), inner_acc))
            
            if k == 1:
                break
            clf_full = clf_fn().fit(X_in_all[:, current], y_tr)
            worst = np.argmin(get_weights(clf_full))
            current.pop(worst)
            
        best_entry = max(rfe_history, key=lambda x: (x[2], -x[0]))
        best_k, best_feats, _ = best_entry
        selected_k_log.append(best_k)
        for f_idx in best_feats:
            feat_freq[FEATURE_COLS_11[f_idx]] += 1
            
        clf_out = clf_fn().fit(X_in_all[:, best_feats], y_tr)
        outer_preds.append(clf_out.predict(X_te_val[:, best_feats])[0])
        
    correct = np.sum(outer_preds == y_survey)
    return correct, correct / n_survey, np.mean(selected_k_log), feat_freq

def run_rfe_transfer(clf_fn):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    current = list(range(len(FEATURE_COLS_11)))
    rfe_history = []
    
    while len(current) >= 1:
        k = len(current)
        fold_accs = []
        for tr_idx, va_idx in cv.split(X_73_z):
            c = clf_fn()
            c.fit(X_73_z[tr_idx][:, current], y_shorts[tr_idx])
            preds = c.predict(X_73_z[va_idx][:, current])
            fold_accs.append(accuracy_score(y_shorts[va_idx], preds))
        mean_acc = np.mean(fold_accs)
        rfe_history.append((k, list(current), mean_acc))
        if k == 1:
            break
        clf_full = clf_fn().fit(X_73_z[:, current], y_shorts)
        worst = np.argmin(get_weights(clf_full))
        current.pop(worst)
        
    best_entry = max(rfe_history, key=lambda x: (x[2], -x[0]))
    best_k, best_feats, best_cv = best_entry
    best_feat_names = [FEATURE_COLS_11[i] for i in best_feats]
    
    clf_final = clf_fn().fit(X_73_z[:, best_feats], y_shorts)
    test_preds = clf_final.predict(X_23_z[:, best_feats])
    correct = np.sum(test_preds == y_survey)
    return best_k, best_cv, best_feat_names, correct, correct / n_survey

rfe_svm_in_corr, rfe_svm_in_acc, rfe_svm_in_k, rfe_svm_freq = run_rfe_survey_nested(svm_fn, "Linear SVM")
rfe_svm_tr_k, rfe_svm_tr_cv, rfe_svm_tr_feats, rfe_svm_tr_corr, rfe_svm_tr_acc = run_rfe_transfer(svm_fn)

rfe_lr_in_corr, rfe_lr_in_acc, rfe_lr_in_k, rfe_lr_freq = run_rfe_survey_nested(lr_fn, "Logistic Regression")
rfe_lr_tr_k, rfe_lr_tr_cv, rfe_lr_tr_feats, rfe_lr_tr_corr, rfe_lr_tr_acc = run_rfe_transfer(lr_fn)

print(f"RFE Linear SVM: 内部Nested {rfe_svm_in_corr}/{n_survey} ({rfe_svm_in_acc*100:.1f}%, 平均k={rfe_svm_in_k:.1f}) | 転移 {rfe_svm_tr_corr}/{n_survey} ({rfe_svm_tr_acc*100:.1f}%, k={rfe_svm_tr_k}: {rfe_svm_tr_feats})")
print(f"RFE Logistic:   内部Nested {rfe_lr_in_corr}/{n_survey} ({rfe_lr_in_acc*100:.1f}%, 平均k={rfe_lr_in_k:.1f}) | 転移 {rfe_lr_tr_corr}/{n_survey} ({rfe_lr_tr_acc*100:.1f}%, k={rfe_lr_tr_k}: {rfe_lr_tr_feats})")

print("\n--- アンケート内部Nested CVにおける特徴量選抜頻度 (全23フォールド中) ---")
for f in FEATURE_COLS_11:
    print(f"  {f:<28}: SVM={rfe_svm_freq[f]:2d}/23, LR={rfe_lr_freq[f]:2d}/23")


# ==============================================================================
# 実験5: 新コア特徴量セット固定転移 (Core3: question + seg_len + sem_var など)
# ==============================================================================
print("\n" + "="*80)
print("【実験5】新コア特徴量セット固定転移 (73件学習 -> 23件テスト)")
print("="*80)

# 新Core3: pause_ratio が抜けたため、残る3大コアは question_ratio, segment_length_sec, comment_semantic_variance
new_core_sets = {
    "New Core2 (question + seg_len)": ['question_ratio', 'segment_length_sec'],
    "New Core3 (question + seg_len + sem_var)": ['question_ratio', 'segment_length_sec', 'comment_semantic_variance'],
    "New Core4 (New Core3 + grass_ratio)": ['question_ratio', 'segment_length_sec', 'comment_semantic_variance', 'grass_ratio'],
    "New Core4b (New Core3 + sentiment)": ['question_ratio', 'segment_length_sec', 'comment_semantic_variance', 'sentiment_polarity'],
    "New Core5 (New Core3 + grass + sentiment)": ['question_ratio', 'segment_length_sec', 'comment_semantic_variance', 'grass_ratio', 'sentiment_polarity']
}

for set_name, feats in new_core_sets.items():
    idx = [FEATURE_COLS_11.index(f) for f in feats]
    c_svm = svm_fn().fit(X_73_z[:, idx], y_shorts)
    p_svm = c_svm.predict(X_23_z[:, idx])
    acc_s = np.sum(p_svm == y_survey)
    
    c_lr = lr_fn().fit(X_73_z[:, idx], y_shorts)
    p_lr = c_lr.predict(X_23_z[:, idx])
    acc_l = np.sum(p_lr == y_survey)
    
    print(f"[{set_name}]")
    print(f"  Linear SVM          : {acc_s:2d}/{n_survey} ({acc_s/n_survey*100:.1f}%)")
    print(f"  Logistic Regression : {acc_l:2d}/{n_survey} ({acc_l/n_survey*100:.1f}%)")
