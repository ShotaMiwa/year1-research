# -*- coding: utf-8 -*-
"""
evaluate_label_refinement_transfer.py

73件の疑似教師ラベル品質改善実験
単一の「いいね率中央値」だけでなく、よりノイズの少ない多面的なラベル生成戦略を比較検証する。

【検証するラベル生成戦略】
1. Baseline:
   - like_rate >= median(like_rate) (現状)

2. Composite Engagement Score (いいね率 + コメント率の標準化合成):
   - z_like = zscore(like_rate)
   - z_comment = zscore(comment_rate)
   - composite_eng = z_like + z_comment
   - label = composite_eng >= median

3. Sentiment-weighted Engagement (感情極性を加味したエンゲージメント):
   - z_sentiment = zscore(sentiment_polarity)
   - composite_sent = z_like + 0.5 * z_comment + 0.5 * z_sentiment
   - label = composite_sent >= median

4. Margin Threshold / Ambiguity Filter (中央値付近の境界曖昧サンプルを除外または極端値重視):
   - 上位40% (ラベル1) vs 下位40% (ラベル0) で学習（中央20%のノイズグレーゾーンを除外して強いシグナルのみで学習）

【評価対象】
- Core4特徴量 ('question_ratio', 'pause_ratio', 'segment_length_sec', 'comment_semantic_variance')
- 全12特徴量
- Linear SVM & Logistic Regression
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

df_survey, df_shorts = load_and_validate_datasets()
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(OUTLIER_SHORTS_IDS)].copy().reset_index(drop=True)

y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
n_survey = len(df_survey)

CORE4_FEATURES = ['question_ratio', 'pause_ratio', 'segment_length_sec', 'comment_semantic_variance']

# 疑似ラベル生成戦略の定義
z_like = zscore(df_shorts_73['like_rate'].values)
z_comm = zscore(df_shorts_73['comment_rate'].values)
z_sent = zscore(df_shorts_73['sentiment_polarity'].values)

# 戦略1: 現状ベースライン（いいね率）
y_label_baseline = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values

# 戦略2: いいね率 + コメント率 (複合エンゲージメント)
comp_eng = z_like + z_comm
y_label_comp_eng = (comp_eng >= np.median(comp_eng)).astype(int)

# 戦略3: エンゲージメント + コメント感情 (ポジティブ関与度)
comp_sent = z_like + 0.5 * z_comm + 0.5 * z_sent
y_label_comp_sent = (comp_sent >= np.median(comp_sent)).astype(int)

# 戦略4: 曖昧ゾーン除去 (上位35% vs 下位35%、中央30%を除外してクリアな差のみ学習)
q_low = np.percentile(df_shorts_73['like_rate'], 35)
q_high = np.percentile(df_shorts_73['like_rate'], 65)
mask_clear = (df_shorts_73['like_rate'] <= q_low) | (df_shorts_73['like_rate'] >= q_high)
y_label_clear = (df_shorts_73.loc[mask_clear, 'like_rate'] >= q_high).astype(int).values

# 戦略5: 複合エンゲージメントの曖昧ゾーン除去 (上位35% vs 下位35%)
comp_q_low = np.percentile(comp_eng, 35)
comp_q_high = np.percentile(comp_eng, 65)
mask_comp_clear = (comp_eng <= comp_q_low) | (comp_eng >= comp_q_high)
y_label_comp_clear = (comp_eng[mask_comp_clear] >= comp_q_high).astype(int)

label_strategies = [
    ("1. 現状ベースライン (いいね率中央値)", y_label_baseline, np.ones(len(df_shorts_73), dtype=bool)),
    ("2. 複合エンゲージメント (いいね率+コメント率)", y_label_comp_eng, np.ones(len(df_shorts_73), dtype=bool)),
    ("3. 感情加味エンゲージメント (いいね+0.5コメント+0.5感情)", y_label_comp_sent, np.ones(len(df_shorts_73), dtype=bool)),
    ("4. いいね率クリア層抽出 (上位35% vs 下位35%, N=52)", y_label_clear, mask_clear),
    ("5. 複合エンゲージメントクリア層 (上位35% vs 下位35%, N=52)", y_label_comp_clear, mask_comp_clear),
]

models = [
    ("Linear SVM", lambda: SVC(kernel='linear', C=1.0, random_state=42)),
    ("Logistic Regression", lambda: LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)),
]

feature_conditions = [
    ("Core4特徴量", CORE4_FEATURES),
    ("全12特徴量", FEATURE_NAMES_12)
]

print("=" * 80)
print("疑似教師ラベル品質改善によるアンケート23件逆転移精度検証")
print("=" * 80)

results = []

for strat_name, y_train_subset, mask in label_strategies:
    print(f"\n【戦略】{strat_name} (訓練N={np.sum(mask)}件)")
    df_train_sub = df_shorts_73.loc[mask].reset_index(drop=True)
    
    for feat_cond_name, feats in feature_conditions:
        for model_name, clf_fn in models:
            # スケーリング
            sc = StandardScaler()
            X_tr = sc.fit_transform(df_train_sub[feats].values)
            X_te = sc.transform(df_survey[feats].values)
            
            clf = clf_fn()
            clf.fit(X_tr, y_train_subset)
            preds = clf.predict(X_te)
            
            correct = int(np.sum(preds == y_survey))
            acc = correct / n_survey
            
            print(f"  [{feat_cond_name}] {model_name:20s}: {correct}/{n_survey} ({acc*100:.1f}%)")
            
            results.append({
                "strategy": strat_name,
                "train_n": int(np.sum(mask)),
                "feature_condition": feat_cond_name,
                "model": model_name,
                "correct": correct,
                "total": n_survey,
                "accuracy": f"{acc*100:.1f}%",
                "acc_val": acc
            })

# CSV保存
df_res = pd.DataFrame(results)
out_csv = os.path.join(PROJECT_ROOT, "docs/paper/tables/exp3d_label_refinement_results.csv")
df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
# 11/25進捗報告用フォルダにも同期
out_report_csv = os.path.join(PROJECT_ROOT, "進捗報告用/20261125/tables/exp3d_label_refinement_results.csv")
df_res.to_csv(out_report_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ 結果を保存しました: {out_csv}")
