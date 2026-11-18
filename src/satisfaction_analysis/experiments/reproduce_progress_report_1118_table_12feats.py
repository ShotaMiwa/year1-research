# -*- coding: utf-8 -*-
"""
reproduce_progress_report_1118_table_12feats.py

「全12特徴量」（語彙多様性TTR ＋ BERT意味多様性を含む）を完全投入し、
進捗報告の比較表を単純正解率（Accuracy: 的中数/23と%が完全一致）で算出する。
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import LeaveOneOut

PROJECT_ROOT = "/home/shota/work/year1"
SURVEY_CSV = f"{PROJECT_ROOT}/data/processed/feature_matrix.csv"
SHORTS_CSV = f"{PROJECT_ROOT}/data/processed/extended_feature_matrix_77.csv"
OUTLIER_IDS = {"LD06GCUS6cI", "Msd39_BhMqU", "4KMwiyfGxA0", "UEvV6gO5l88"}

df_survey = pd.read_csv(SURVEY_CSV)
df_shorts = pd.read_csv(SHORTS_CSV)
df_shorts_73 = df_shorts[~df_shorts["shorts_id"].isin(OUTLIER_IDS)].copy()

# ★真の全12特徴量（TTR + BERT意味分散の両方を完全網羅）
FEATURE_COLS = [
    "comment_rate", "grass_ratio", "question_ratio", "exclamation_ratio",
    "sentiment_polarity", "subtitle_sentiment_polarity", "speech_rate",
    "pause_ratio", "sub_comment_similarity",
    "comment_vocab_diversity",   # 語彙多様性 (TTR)
    "comment_semantic_variance", # 意味多様性 (BERT分散)
    "segment_length_sec"        # 動画尺
]

print(f"投入特徴量数: {len(FEATURE_COLS)} 個")
print(FEATURE_COLS)

y_survey = (df_survey["Target_Satisfaction"] >= df_survey["Target_Satisfaction"].median()).astype(int).values
y_shorts_73 = (df_shorts_73["like_rate"] >= df_shorts_73["like_rate"].median()).astype(int).values

X_survey_raw = df_survey[FEATURE_COLS].values
X_shorts_raw = df_shorts_73[FEATURE_COLS].values


# ① アンケート23件 Nested LOO-CV
def run_nested_loocv(model_name="svm", use_sfs=False):
    loo = LeaveOneOut()
    y_preds = []
    selected_features_log = []

    for train_idx, test_idx in loo.split(X_survey_raw):
        X_tr, X_te = X_survey_raw[train_idx], X_survey_raw[test_idx]
        y_tr, y_te = y_survey[train_idx], y_survey[test_idx]

        scaler = StandardScaler()
        X_tr_z = scaler.fit_transform(X_tr)
        X_te_z = scaler.transform(X_te)

        base_clf = (
            SVC(kernel="linear", C=1.0, random_state=42)
            if model_name == "svm"
            else LogisticRegression(C=1.0, random_state=42)
        )

        if use_sfs:
            sfs = SequentialFeatureSelector(
                base_clf,
                n_features_to_select=2,
                direction="forward",
                cv=LeaveOneOut(),
                scoring="accuracy",
                n_jobs=-1
            )
            sfs.fit(X_tr_z, y_tr)
            feat_idx = sfs.get_support(indices=True)
            selected_features_log.append(tuple(feat_idx))
            X_tr_sub = X_tr_z[:, feat_idx]
            X_te_sub = X_te_z[:, feat_idx]
        else:
            X_tr_sub = X_tr_z
            X_te_sub = X_te_z

        clf = (
            SVC(kernel="linear", C=1.0, random_state=42)
            if model_name == "svm"
            else LogisticRegression(C=1.0, random_state=42)
        )
        clf.fit(X_tr_sub, y_tr)
        pred = clf.predict(X_te_sub)[0]
        y_preds.append(pred)

    acc = accuracy_score(y_survey, y_preds) * 100
    n_correct = sum(y_survey == y_preds)
    return acc, n_correct, selected_features_log


# ② 健全73件学習 -> アンケート23件テスト
def run_transfer(model_name="svm", use_sfs=False):
    scaler_tr = StandardScaler()
    X_tr_z = scaler_tr.fit_transform(X_shorts_raw)
    scaler_te = StandardScaler()
    X_te_z = scaler_te.fit_transform(X_survey_raw)

    base_clf = (
        SVC(kernel="linear", C=1.0, random_state=42)
        if model_name == "svm"
        else LogisticRegression(C=1.0, random_state=42)
    )

    if use_sfs:
        sfs = SequentialFeatureSelector(
            base_clf,
            n_features_to_select=2,
            direction="forward",
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        sfs.fit(X_tr_z, y_shorts_73)
        feat_idx = sfs.get_support(indices=True)
        selected_names = [FEATURE_COLS[i] for i in feat_idx]
        X_tr_sub = X_tr_z[:, feat_idx]
        X_te_sub = X_te_z[:, feat_idx]
    else:
        X_tr_sub = X_tr_z
        X_te_sub = X_te_z
        selected_names = FEATURE_COLS

    clf = (
        SVC(kernel="linear", C=1.0, random_state=42)
        if model_name == "svm"
        else LogisticRegression(C=1.0, random_state=42)
    )
    clf.fit(X_tr_sub, y_shorts_73)
    y_preds = clf.predict(X_te_sub)

    acc = accuracy_score(y_survey, y_preds) * 100
    n_correct = sum(y_survey == y_preds)
    return acc, n_correct, selected_names


print("=== 全12特徴量による正確な再計算開始 ===")
svm_all_23_acc, svm_all_23_cor, _ = run_nested_loocv("svm", False)
svm_all_73_acc, svm_all_73_cor, _ = run_transfer("svm", False)

svm_sfs_23_acc, svm_sfs_23_cor, _ = run_nested_loocv("svm", True)
svm_sfs_73_acc, svm_sfs_73_cor, svm_sfs_names = run_transfer("svm", True)

lr_all_23_acc, lr_all_23_cor, _ = run_nested_loocv("lr", False)
lr_all_73_acc, lr_all_73_cor, _ = run_transfer("lr", False)

lr_sfs_23_acc, lr_sfs_23_cor, _ = run_nested_loocv("lr", True)
lr_sfs_73_acc, lr_sfs_73_cor, lr_sfs_names = run_transfer("lr", True)

print("\n=== 単純正解率 (Accuracy: 的中数/23と%が一致) ===")
print(f"Linear SVM (全12特徴量) : 23件学習 = {svm_all_23_acc:.1f}% ({svm_all_23_cor}/23) | 73件転移 = {svm_all_73_acc:.1f}% ({svm_all_73_cor}/23)")
print(f"Linear SVM (自律探索)   : 23件学習 = {svm_sfs_23_acc:.1f}% ({svm_sfs_23_cor}/23) | 73件転移 = {svm_sfs_73_acc:.1f}% ({svm_sfs_73_cor}/23) | 選択: {svm_sfs_names}")
print(f"LogReg (全12特徴量)     : 23件学習 = {lr_all_23_acc:.1f}% ({lr_all_23_cor}/23) | 73件転移 = {lr_all_73_acc:.1f}% ({lr_all_73_cor}/23)")
print(f"LogReg (自律探索)       : 23件学習 = {lr_sfs_23_acc:.1f}% ({lr_sfs_23_cor}/23) | 73件転移 = {lr_sfs_73_acc:.1f}% ({lr_sfs_73_cor}/23) | 選択: {lr_sfs_names}")
