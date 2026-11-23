# -*- coding: utf-8 -*-
"""
evaluate_bayesian_composite_label_transfer.py

案1：信頼度付き加重合成スコア（ベイズ的アプローチ）の実装とアンケート23件転移検証

【計算式】
各動画 i に対し、コメント件数 N_i に応じた信頼度重み w_i を計算:
  w_i = N_i / (N_i + K)   (デフォルト K=3)
  Composite_Score_i = Z(like_rate_i) + alpha * w_i * Z(sentiment_polarity_i)

alpha を 0.0（現状ベースライン: いいね率のみ）から 0.3, 0.5, 0.8, 1.0, 1.5 まで変化させ、
合成スコアの中央値で二値化した疑似教師ラベルを作成。

【評価条件】
- 訓練データ: 73件 (外れ値4本除外)
- テストデータ: 未見のアンケート23件
- 特徴量:
  1. Core4特徴量 ('question_ratio', 'pause_ratio', 'segment_length_sec', 'comment_semantic_variance')
  2. 全12特徴量
- モデル: Linear SVM & Logistic Regression
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROJECT_ROOT = "/home/shota/work/year1"
sys.path.insert(0, PROJECT_ROOT)

from src.satisfaction_analysis.constants import FEATURE_NAMES_12, OUTLIER_SHORTS_IDS
from src.satisfaction_analysis.data_validator import load_and_validate_datasets

# 1. データセットの読み込み
df_survey, df_shorts = load_and_validate_datasets()
df_sent = pd.read_csv(f"{PROJECT_ROOT}/data/processed/shorts_comments_sentiment_summary.csv")

# 感情スコアを結合
df_shorts = df_shorts.merge(
    df_sent[['shorts_id', 'sentiment_polarity', 'n_comments']],
    on='shorts_id',
    how='left',
    suffixes=('', '_shorts_comment')
)

# 外れ値4本を除外 (73本)
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(OUTLIER_SHORTS_IDS)].copy().reset_index(drop=True)

y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
n_survey = len(df_survey)

CORE4_FEATURES = ['question_ratio', 'pause_ratio', 'segment_length_sec', 'comment_semantic_variance']

# 2. 各シグナルの標準化
z_like = zscore(df_shorts_73['like_rate'].values)
z_sent = zscore(df_shorts_73['sentiment_polarity_shorts_comment'].fillna(0.0).values)
n_comments = df_shorts_73['n_comments_shorts_comment'].fillna(0).values

# 信頼度重み w_i = N / (N + K)
K = 3.0
weights = n_comments / (n_comments + K)

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]

models = [
    ("Linear SVM", lambda: SVC(kernel='linear', C=1.0, random_state=42)),
    ("Logistic Regression", lambda: LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)),
]

print("=" * 80)
print("【案1: 信頼度付き加重合成スコア（ベイズ的アプローチ）】転移精度検証")
print("式: Composite_Score = Z(いいね率) + alpha * (N / (N + 3)) * Z(Shortsコメント感情)")
print("=" * 80)

rows = []

for alpha in alphas:
    # 加重合成スコア
    comp_score = z_like + alpha * (weights * z_sent)
    y_train = (comp_score >= np.median(comp_score)).astype(int)
    
    # alpha=0 の時は現状ベースライン
    label_name = f"alpha={alpha:.1f}" + (" (現状: いいね率のみ)" if alpha == 0.0 else "")
    print(f"\n■ 条件: {label_name}")
    
    for feat_name, feats in [("Core4特徴量", CORE4_FEATURES), ("全12特徴量", FEATURE_NAMES_12)]:
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_shorts_73[feats].values)
        X_te = sc.transform(df_survey[feats].values)
        
        for model_name, clf_fn in models:
            clf = clf_fn()
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            
            correct = int(np.sum(preds == y_survey))
            acc = correct / n_survey
            print(f"  [{feat_name:<8}] {model_name:20s}: {correct:2d}/{n_survey} ({acc*100:.1f}%)")
            
            rows.append({
                "alpha": alpha,
                "label_name": label_name,
                "feature_condition": feat_name,
                "model": model_name,
                "correct": correct,
                "total": n_survey,
                "accuracy": f"{acc*100:.1f}%",
                "acc_val": acc
            })

# CSV保存（論文tablesに一時記録用として保存、進捗報告mdには触らない）
out_csv = f"{PROJECT_ROOT}/docs/paper/tables/exp3e_bayesian_composite_results.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ 検証ログを保存しました: {out_csv}")
