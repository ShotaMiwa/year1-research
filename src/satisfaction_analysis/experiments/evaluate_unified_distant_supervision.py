# -*- coding: utf-8 -*-
"""
evaluate_unified_distant_supervision.py

完全に同一モジュール・同一定義で統一されたデータセットを用いて、
Shorts 73件（外れ値除外）学習 -> アンケート23件テストの
転移精度（線形SVM / ロジスティック回帰）を再検証する。
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = "/home/shota/work/year1"
SURVEY_CSV = f"{PROJECT_ROOT}/data/processed/feature_matrix.csv"
SHORTS_CSV = f"{PROJECT_ROOT}/data/processed/extended_feature_matrix_77.csv"

# 外れ値4本のShorts ID
OUTLIER_IDS = {"LD06GCUS6cI", "Msd39_BhMqU", "4KMwiyfGxA0", "UEvV6gO5l88"}

df_survey = pd.read_csv(SURVEY_CSV)
df_shorts = pd.read_csv(SHORTS_CSV)

# 73件に絞り込み
df_shorts_73 = df_shorts[~df_shorts["shorts_id"].isin(OUTLIER_IDS)].copy()

# ターゲットラベル作成（中央値二値化）
y_train = (df_shorts_73["like_rate"] >= df_shorts_73["like_rate"].median()).astype(int).values
y_test  = (df_survey["Target_Satisfaction"] >= df_survey["Target_Satisfaction"].median()).astype(int).values

print(f"訓練データ (Shorts健全): {len(y_train)}件 (Positive: {sum(y_train)}, Negative: {len(y_train)-sum(y_train)})")
print(f"テストデータ (アンケート): {len(y_test)}件 (Positive: {sum(y_test)}, Negative: {len(y_test)-sum(y_test)})")

# 検証する特徴量セット
experiments = {
    "① 語彙多様性(TTR) + 動画尺": ["comment_vocab_diversity", "segment_length_sec"],
    "② BERT意味多様性 + 動画尺": ["comment_semantic_variance", "segment_length_sec"],
    "③ 語彙多様性(TTR) + BERT意味多様性 + 動画尺": ["comment_vocab_diversity", "comment_semantic_variance", "segment_length_sec"],
    "④ 全10特徴量 (TTR版)": [
        "comment_rate", "grass_ratio", "question_ratio", "exclamation_ratio",
        "sentiment_polarity", "subtitle_sentiment_polarity", "speech_rate",
        "pause_ratio", "sub_comment_similarity", "comment_vocab_diversity"
    ],
    "⑤ 全10特徴量 (BERT意味多様性版)": [
        "comment_rate", "grass_ratio", "question_ratio", "exclamation_ratio",
        "sentiment_polarity", "subtitle_sentiment_polarity", "speech_rate",
        "pause_ratio", "sub_comment_similarity", "comment_semantic_variance"
    ],
}

results = []
for exp_name, feat_cols in experiments.items():
    X_train = df_shorts_73[feat_cols].values
    X_test  = df_survey[feat_cols].values

    # 各データセットで標準化
    scaler_tr = StandardScaler()
    X_train_z = scaler_tr.fit_transform(X_train)
    scaler_te = StandardScaler()
    X_test_z  = scaler_te.fit_transform(X_test)

    # 1. Linear SVM
    svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm.fit(X_train_z, y_train)
    y_pred_svm = svm.predict(X_test_z)
    y_prob_svm = svm.predict_proba(X_test_z)[:, 1]

    svm_bacc = balanced_accuracy_score(y_test, y_pred_svm) * 100
    svm_acc  = accuracy_score(y_test, y_pred_svm) * 100
    svm_f1   = f1_score(y_test, y_pred_svm) * 100
    svm_auc  = roc_auc_score(y_test, y_prob_svm) * 100

    # 2. Logistic Regression
    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(X_train_z, y_train)
    y_pred_lr = lr.predict(X_test_z)
    y_prob_lr = lr.predict_proba(X_test_z)[:, 1]

    lr_bacc = balanced_accuracy_score(y_test, y_pred_lr) * 100
    lr_acc  = accuracy_score(y_test, y_pred_lr) * 100
    lr_f1   = f1_score(y_test, y_pred_lr) * 100
    lr_auc  = roc_auc_score(y_test, y_prob_lr) * 100

    results.append({
        "実験条件": exp_name,
        "特徴量数": len(feat_cols),
        "SVM Balanced Acc": f"{svm_bacc:.2f}%",
        "SVM Acc": f"{svm_acc:.2f}%",
        "SVM AUC": f"{svm_auc:.2f}%",
        "LogReg Balanced Acc": f"{lr_bacc:.2f}%",
        "LogReg Acc": f"{lr_acc:.2f}%",
        "LogReg AUC": f"{lr_auc:.2f}%",
    })

    if exp_name in ["① 語彙多様性(TTR) + 動画尺", "② BERT意味多様性 + 動画尺"]:
        print(f"\n[{exp_name}] SVM 係数:")
        for feat, coef in zip(feat_cols, svm.coef_[0]):
            print(f"  {feat}: {coef:.4f}")
        print(f"  bias: {svm.intercept_[0]:.4f}")

df_res = pd.DataFrame(results)
print("\n=== 完全統一パイプラインにおける転移検証結果 ===")
print(df_res.to_string(index=False))

# 保存
output_table = f"{PROJECT_ROOT}/docs/paper/tables/exp_unified_distant_supervision_results.csv"
df_res.to_csv(output_table, index=False)
print(f"\n✅ 結果を保存: {output_table}")
