# -*- coding: utf-8 -*-
"""
run_svm_rfe_pipeline_v3.py

対策③: 実験①（アンケート23件 Nested LOO-CV）で安定して選ばれた特徴量を
固定セットとして実験②（73件学習→23件転移）に使用する。

73件側のノイジーないいね率ラベルに引きずられず、
主観満足度との関連が確認された特徴量セットで転移精度を評価する。

固定セット候補:
  - Core3: question_ratio + pause_ratio + segment_length_sec  (両モデル安定Top3)
  - Core4: Core3 + comment_semantic_variance                  (SVM 19/23, LR 13/23)
  - Core5: Core4 + sentiment_polarity                         (SVM 18/23)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
y_shorts  = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values
n_survey  = len(df_survey)

# 実験①の安定特徴量セット（固定）
FIXED_FEATURE_SETS = {
    "Core3 (question + pause + seg_len)":
        ['question_ratio', 'pause_ratio', 'segment_length_sec'],
    "Core4 (Core3 + sem_var)":
        ['question_ratio', 'pause_ratio', 'segment_length_sec', 'comment_semantic_variance'],
    "Core5 (Core4 + sentiment)":
        ['question_ratio', 'pause_ratio', 'segment_length_sec',
         'comment_semantic_variance', 'sentiment_polarity'],
}

def run_fixed_transfer(clf_fn, model_name, feat_names):
    """固定特徴量セットで73件学習→23件転移テスト"""
    feat_idx = [FEATURE_NAMES_12.index(f) for f in feat_names]

    sc = StandardScaler()
    X_tr = sc.fit_transform(df_shorts_73[FEATURE_NAMES_12].values)[:, feat_idx]
    X_te = sc.transform(df_survey[FEATURE_NAMES_12].values)[:, feat_idx]

    # 再スケーリング（テスト側は訓練側のスケーラーで変換済み）
    sc2 = StandardScaler()
    X_tr2 = sc2.fit_transform(df_shorts_73[feat_names].values)
    X_te2 = sc2.transform(df_survey[feat_names].values)

    clf = clf_fn()
    clf.fit(X_tr2, y_shorts)
    preds = clf.predict(X_te2)

    correct = int(np.sum(preds == y_survey))
    acc = correct / n_survey
    return correct, acc


def main():
    models = [
        ("Linear SVM",         lambda: SVC(kernel='linear', C=1.0, random_state=42)),
        ("Logistic Regression", lambda: LogisticRegression(penalty='l2', C=1.0,
                                         random_state=42, max_iter=1000)),
    ]

    print("\n" + "="*70)
    print("対策③: 実験①安定特徴量を固定して実験②（73件→23件転移）を実行")
    print("="*70)

    rows = []
    for set_name, feat_names in FIXED_FEATURE_SETS.items():
        print(f"\n【特徴量セット】{set_name}")
        print(f"  特徴量: {feat_names}")
        for model_name, clf_fn in models:
            correct, acc = run_fixed_transfer(clf_fn, model_name, feat_names)
            print(f"  {model_name:22s}: {correct}/{n_survey} ({acc*100:.1f}%)")
            rows.append({
                "feature_set": set_name,
                "model": model_name,
                "features": " + ".join(feat_names),
                "transfer_correct": correct,
                "transfer_total": n_survey,
                "transfer_accuracy": f"{acc*100:.1f}%",
            })

    # --- 参考: 前回条件との比較表示 ---
    print("\n" + "-"*70)
    print("【参考: 比較ベースライン】")
    print("  SFS (前回)   SVM: 15/23 (65.2%) | LR: 12/23 (52.2%)")
    print("  RFE v1       SVM: 16/23 (69.6%) | LR: 13/23 (56.5%)  ← k=1(seg_len)のみ")
    print("  RFE v2       SVM: 16/23 (69.6%) | LR: 16/23 (69.6%)  ← k=1(seg_len)のみ")

    # --- 集計表保存 ---
    tables_dir = os.path.join(PROJECT_ROOT, "docs/paper/tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "exp3c_rfe_fixed_features_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 集計表保存: {csv_path}")

    # --- 棒グラフ ---
    figures_dir = os.path.join(PROJECT_ROOT, "docs/paper/figures")
    os.makedirs(figures_dir, exist_ok=True)

    set_labels = list(FIXED_FEATURE_SETS.keys())
    svm_accs = []
    lr_accs  = []
    for set_name in set_labels:
        for r in rows:
            if r["feature_set"] == set_name and r["model"] == "Linear SVM":
                svm_accs.append(float(r["transfer_accuracy"].strip('%')))
            if r["feature_set"] == set_name and r["model"] == "Logistic Regression":
                lr_accs.append(float(r["transfer_accuracy"].strip('%')))

    x = np.arange(len(set_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, svm_accs, width, label='Linear SVM', color='steelblue')
    bars2 = ax.bar(x + width/2, lr_accs,  width, label='Logistic Regression', color='coral')

    # ベースライン
    ax.axhline(65.2, color='steelblue', linestyle='--', alpha=0.5, label='SFS SVM baseline (65.2%)')
    ax.axhline(52.2, color='coral',     linestyle='--', alpha=0.5, label='SFS LR baseline (52.2%)')

    ax.set_title("Transfer Accuracy (73→23) with Fixed Feature Sets from Exp.①", fontsize=13)
    ax.set_ylabel("Transfer Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels, fontsize=9)
    ax.set_ylim(40, 85)
    ax.legend(fontsize=9)
    ax.bar_label(bars1, fmt='%.1f%%', padding=3)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    png_path = os.path.join(figures_dir, "exp3c_fixed_features_transfer.png")
    pdf_path = os.path.join(figures_dir, "exp3c_fixed_features_transfer.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"✅ 図保存: {png_path}")


if __name__ == "__main__":
    main()
