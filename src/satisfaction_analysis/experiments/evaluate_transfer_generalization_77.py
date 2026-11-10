# -*- coding: utf-8 -*-
"""
evaluate_transfer_generalization_77.py

【モデル転移・外部汎化性能評価実験】
被験者アンケート（N=23）で学習した機械学習モデル（SVM / ロジスティック回帰 / 決定木）を用いて、
完全に未知の外部テストセット（N=77: YouTube Shorts + 元配信実測チャット）における
「いいね率の高低（二値分類）」を予測し、外部汎化性能を評価する。

実験条件:
- 訓練データ (N=23): feature_matrix.csv (Target_Satisfaction >= median=3.6 を正例=1)
- 外部テストデータ (N=77): extended_feature_matrix_77.csv (like_rate >= median=0.4541 を高評価=1)
- 共通特徴量セット:
  1) 全10音響・言語・チャット特徴量 + segment_length_sec (計11特徴量)
  2) 前回の最良2特徴量セット: question_ratio + sentiment_polarity
  3) 前回の決定株最良2特徴量セット: comment_rate + segment_length_sec
- 評価モデル:
  - Linear SVM (C=1.0)
  - Logistic Regression (L2正則化, C=1.0)
  - Logistic Regression (L1正則化 / Lasso)
  - Decision Tree (Depth=1, Stump: 決定株)
  - Decision Tree (Depth=2, min_samples_leaf=2)
  - Random Forest (n_estimators=100, max_depth=3)

出力:
- docs/paper/tables/exp_transfer_generalization_77.csv
- docs/paper/figures/exp_transfer_generalization_77_barplots.png
- docs/paper/figures/exp_transfer_confusion_matrices.png
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path("/home/shota/work/year1")
TRAIN_DATA_PATH = PROJECT_ROOT / "data/processed/feature_matrix.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
FIG_DIR = PROJECT_ROOT / "docs/paper/figures"
TAB_DIR = PROJECT_ROOT / "docs/paper/tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# 共通特徴量11種
FEATS_ALL11 = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_semantic_variance',
    'segment_length_sec'
]

# 代表的少数特徴量サブセット（先行検証で特定された重要セット）
FEATS_SUBSET_QS = ['question_ratio', 'sentiment_polarity']
FEATS_SUBSET_STUMP = ['comment_rate', 'segment_length_sec']
FEATS_SUBSET_Q_LEN = ['question_ratio', 'segment_length_sec']

FEATURE_SETS = {
    "全11特徴量 (All 11 Feats)": FEATS_ALL11,
    "先行最良2特徴量 [疑問文+感情極性]": FEATS_SUBSET_QS,
    "決定株最良2特徴量 [コメント率+動画尺]": FEATS_SUBSET_STUMP,
    "重要2特徴量 [疑問文+動画尺]": FEATS_SUBSET_Q_LEN
}

def main():
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    print(f"訓練データ (被験者アンケート): {len(df_train)} 件")
    print(f"テストデータ (外部Shorts実測): {len(df_test)} 件")

    # 目的変数の2値化（中央値分割）
    train_median = df_train['Target_Satisfaction'].median()
    y_train = (df_train['Target_Satisfaction'] >= train_median).astype(int).values
    
    test_median = df_test['like_rate'].median()
    y_test = (df_test['like_rate'] >= test_median).astype(int).values

    print(f"訓練ラベル分布: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"テストラベル分布: 0={sum(y_test==0)}, 1={sum(y_test==1)}")

    model_configs = [
        ("Linear SVM (C=1.0)", lambda: SVC(kernel='linear', C=1.0, random_state=42)),
        ("Logistic Regression (L2)", lambda: LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)),
        ("Logistic Regression (L1/Lasso)", lambda: LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)),
        ("Decision Tree (Depth=1, Stump)", lambda: DecisionTreeClassifier(max_depth=1, random_state=42)),
        ("Decision Tree (Depth=2, Leaf=2)", lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=42)),
        ("Random Forest (100 Trees, Depth=3)", lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
    ]

    results = []
    confusion_dict = {}

    for fset_name, feats in FEATURE_SETS.items():
        X_train_raw = df_train[feats].values
        X_test_raw = df_test[feats].values

        # 訓練データの平均・分散で標準化（データ漏洩防止）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        for model_name, model_fn in model_configs:
            model = model_fn()
            
            # 木構造モデルはスケーリング有無問わないが統一してscaledを使用
            model.fit(X_train_scaled, y_train)

            # 訓練内LOOスコアではなく「完全未知テストセット(N=77)」での評価
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # ROC-AUC (予測確率がある場合)
            try:
                if hasattr(model, "predict_proba"):
                    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
                elif hasattr(model, "decision_function"):
                    auc = roc_auc_score(y_test, model.decision_function(X_test_scaled))
                else:
                    auc = np.nan
            except:
                auc = np.nan

            key = f"{model_name} | {fset_name}"
            confusion_dict[key] = cm

            results.append({
                "feature_set": fset_name,
                "n_features": len(feats),
                "model": model_name,
                "balanced_acc": round(bacc * 100, 2),
                "accuracy": round(acc * 100, 2),
                "f1_macro": round(f1, 4),
                "roc_auc": round(auc, 4) if not np.isnan(auc) else "-",
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "correct_total": tp + tn
            })

    df_results = pd.DataFrame(results)
    out_table_path = TAB_DIR / "exp_transfer_generalization_77.csv"
    df_results.to_csv(out_table_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved transfer evaluation table to: {out_table_path}")
    print("\n=== モデル転移・外部汎化性能サマリー (上位順) ===")
    print(df_results.sort_values(by="balanced_acc", ascending=False)[["feature_set", "model", "balanced_acc", "accuracy", "f1_macro", "correct_total"]].head(15).to_string())

    # 可視化1: 特徴量セット別・モデル別のBalanced Accuracy棒グラフ
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=df_results,
        x="model",
        y="balanced_acc",
        hue="feature_set",
        palette="viridis"
    )
    plt.axhline(50.0, color="#d93025", linestyle="--", linewidth=1.5, label="チャンスレベル (50%)")
    plt.title("アンケート(N=23)学習モデルの外部テストセット(N=77)に対する転移汎化性能 (Balanced Accuracy)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("機械学習モデル", fontsize=12)
    plt.ylabel("外部テスト Balanced Accuracy (%)", fontsize=12)
    plt.ylim(30, 80)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.legend(title="特徴量セット", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_fig_bar = FIG_DIR / "exp_transfer_generalization_77_barplots.png"
    plt.savefig(out_fig_bar, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved barplot to: {out_fig_bar}")

    # 可視化2: 性能上位4モデルの混同行列 (Confusion Matrix)
    top_models = df_results.sort_values(by="balanced_acc", ascending=False).head(4)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, (_, row) in enumerate(top_models.iterrows()):
        ax = axes[i]
        key = f"{row['model']} | {row['feature_set']}"
        cm = confusion_dict[key]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        ax.set_title(f"Rank {i+1}: {row['model']}\n({row['feature_set']})\nB-Acc: {row['balanced_acc']}% ({row['correct_total']}/77)", fontsize=10, fontweight='bold')
        ax.set_xlabel("予測 (0: 低いいね率, 1: 高いいね率)", fontsize=10)
        ax.set_ylabel("実測 (0: 低, 1: 高)", fontsize=10)
        ax.set_xticklabels(['Low', 'High'])
        ax.set_yticklabels(['Low', 'High'])

    plt.suptitle("外部テストセット(N=77)に対する予測混同行列 (Top 4モデル)", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    out_fig_cm = FIG_DIR / "exp_transfer_confusion_matrices.png"
    plt.savefig(out_fig_cm, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix figure to: {out_fig_cm}")

if __name__ == "__main__":
    main()
