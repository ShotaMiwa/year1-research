# -*- coding: utf-8 -*-
"""
evaluate_distant_supervision_73_to_23.py

【再生数外れ値除去（N=73）弱教師学習によるアンケート満足度（N=23）ブラインド逆転移検証実験】
完全版:
1. 全11特徴量投入（ドメイン内標準化 / 生値標準化）による線形モデル評価
2. 73本の訓練データ内部のみ（Stratified 5-Fold CV）による自律的特徴量選択（Sequential Forward Selection: SFS）
3. 先行研究・アブレーション特徴量セットとの厳密な対照比較
4. 全77本（外れ値あり） vs 健全73本（外れ値除去）のデノイジング効果検証

運用ルール準拠:
- PDF標準出力 (+同名PNG)
- CSVテーブル出力
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data/processed"
FIG_DIR = PROJECT_ROOT / "docs/paper/figures"
TAB_DIR = PROJECT_ROOT / "docs/paper/tables"
PROG_FIG_DIR = PROJECT_ROOT / "進捗報告用/20261118/figures"
PROG_TAB_DIR = PROJECT_ROOT / "進捗報告用/20261118/tables"

for d in [FIG_DIR, TAB_DIR, PROG_FIG_DIR, PROG_TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

OUTLIER_SHORTS_IDS = {
    'jwJZMRaxafU',  # Views: 374, Like rate: 1.60% (低再生インフレ)
    'cAQ0YcMl_14',  # Views: 534, Like rate: 1.50% (低再生インフレ)
    'Msd39_BhMqU',  # Views: 112,681, Like rate: 0.26% (超高再生希釈)
    'LD06GCUS6cI'   # Views: 361,790, Like rate: 0.30% (超高再生希釈)
}

FEATS_ALL11 = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_semantic_variance',
    'segment_length_sec'
]

FEATS_Z11 = [
    'comment_rate_zscore', 'grass_ratio_zscore', 'question_ratio_zscore',
    'exclamation_ratio_zscore', 'sentiment_polarity_zscore',
    'subtitle_sentiment_polarity_zscore', 'speech_rate_zscore',
    'pause_ratio_zscore', 'sub_comment_similarity_zscore',
    'comment_semantic_variance_zscore', 'segment_length_zscore'
]

def run_sfs_cv(df_train, y_train, feat_candidates, cv):
    """訓練データ内部CVのみで特徴量を前進選択する"""
    selected = []
    rem = list(feat_candidates)
    step_records = []
    
    for step in range(1, len(feat_candidates) + 1):
        best_cand = None
        best_cv_score = -1.0
        for cand in rem:
            cur = selected + [cand]
            X = df_train[cur].values
            scores = []
            for tr_idx, va_idx in cv.split(X, y_train):
                clf = LogisticRegression(C=0.5, random_state=42)
                clf.fit(X[tr_idx], y_train[tr_idx])
                pred = clf.predict(X[va_idx])
                scores.append(balanced_accuracy_score(y_train[va_idx], pred))
            m = np.mean(scores)
            if m > best_cv_score:
                best_cv_score = m
                best_cand = cand
        selected.append(best_cand)
        rem.remove(best_cand)
        step_records.append({
            "step": step,
            "added_feature": best_cand,
            "cv_bacc": best_cv_score,
            "current_set": list(selected)
        })
    return step_records

def main():
    print("=" * 80)
    print("再生数外れ値除去（N=73）弱教師学習によるアンケート満足度（N=23）完全検証")
    print("=" * 80)

    df_train_77 = pd.read_csv(DATA_DIR / "extended_feature_matrix_77.csv")
    df_test_23 = pd.read_csv(DATA_DIR / "feature_matrix.csv")

    # segment_length_zscoreの追加（データセット内標準化）
    df_train_77["segment_length_zscore"] = (df_train_77["segment_length_sec"] - df_train_77["segment_length_sec"].mean()) / df_train_77["segment_length_sec"].std()
    df_test_23["segment_length_zscore"] = (df_test_23["segment_length_sec"] - df_test_23["segment_length_sec"].mean()) / df_test_23["segment_length_sec"].std()

    df_train_73 = df_train_77[~df_train_77['shorts_id'].isin(OUTLIER_SHORTS_IDS)].copy()
    # 73本基準で再度zscore計算
    df_train_73["segment_length_zscore"] = (df_train_73["segment_length_sec"] - df_train_73["segment_length_sec"].mean()) / df_train_73["segment_length_sec"].std()

    y_train_77 = (df_train_77['like_rate'] >= df_train_77['like_rate'].median()).astype(int).values
    y_train_73 = (df_train_73['like_rate'] >= df_train_73['like_rate'].median()).astype(int).values
    y_test_23 = (df_test_23['Target_Satisfaction'] >= df_test_23['Target_Satisfaction'].median()).astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. 訓練データ73本内部のみでの自律的特徴量選択（SFS）
    print(">>> 73本訓練データ内部CVによる自律的特徴量選択 (SFS) を実行中...")
    sfs_records = run_sfs_cv(df_train_73, y_train_73, FEATS_Z11, cv)
    
    # 73本CVで最良だった上位セットを抽出
    best_sfs_step = max(sfs_records, key=lambda x: x["cv_bacc"])
    sfs_top2_feats = sfs_records[1]["current_set"]   # Step 2
    sfs_peak_feats = best_sfs_step["current_set"]    # CV最高スコアセット
    print(f"73本内部SFS 最良Step: Step {best_sfs_step['step']} (CV B-Acc: {best_sfs_step['cv_bacc']*100:.2f}%)")
    print(f"最良特徴量セット ({len(sfs_peak_feats)}次元): {sfs_peak_feats}")

    # 特徴量セットの定義
    feature_sets = {
        "全11特徴量 [ドメイン内正規化Z-Score]": FEATS_Z11,
        f"73本内部SFS自律選択 [Top-2: 語彙多様性+動画尺]": sfs_top2_feats,
        f"73本内部SFS自律選択 [Peak-{len(sfs_peak_feats)}次元: CV最良]": sfs_peak_feats,
        "重要2特徴量 [疑問文+動画尺] (先行最良)": ['question_ratio_zscore', 'segment_length_zscore'],
        "先行2特徴量 [疑問文+極性] (動画尺なし)": ['question_ratio_zscore', 'sentiment_polarity_zscore'],
        "全11特徴量 [生値投入]": FEATS_ALL11
    }

    models = [
        ("Linear SVM", lambda c: SVC(kernel='linear', C=c, random_state=42), [0.05, 0.1, 0.5, 1.0, 5.0]),
        ("Logistic Regression (L2)", lambda c: LogisticRegression(penalty='l2', C=c, solver='lbfgs', random_state=42), [0.05, 0.1, 0.5, 1.0, 5.0]),
        ("Logistic Regression (L1)", lambda c: LogisticRegression(penalty='l1', C=c, solver='liblinear', random_state=42), [0.05, 0.1, 0.5, 1.0, 5.0]),
        ("Decision Tree", lambda d: DecisionTreeClassifier(max_depth=d, random_state=42), [1, 2, 3]),
        ("Random Forest", lambda d: RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42), [2, 3])
    ]

    results = []
    confusion_dict = {}

    for train_name, df_tr, y_tr in [
        ("全77本学習 (外れ値あり)", df_train_77, y_train_77),
        ("健全73本学習 (外れ値4本除去・提案)", df_train_73, y_train_73)
    ]:
        for fset_name, feats in feature_sets.items():
            # 生値投入の場合はStandardScaler、Z-scoreの場合はそのまま
            is_raw = "生値" in fset_name
            X_tr_raw = df_tr[feats].values
            X_te_raw = df_test_23[feats].values

            for m_name, m_fn, params in models:
                best_p = None
                best_cv = -1.0
                
                # 訓練内部5-Fold CVでハイパーパラメータ探索
                for p in params:
                    f_scores = []
                    for tr_i, va_i in cv.split(X_tr_raw, y_tr):
                        if is_raw:
                            sc = StandardScaler()
                            x_tr = sc.fit_transform(X_tr_raw[tr_i])
                            x_va = sc.transform(X_tr_raw[va_i])
                        else:
                            x_tr = X_tr_raw[tr_i]
                            x_va = X_tr_raw[va_i]
                        clf = m_fn(p)
                        clf.fit(x_tr, y_tr[tr_i])
                        f_scores.append(balanced_accuracy_score(y_tr[va_i], clf.predict(x_va)))
                    m_score = np.mean(f_scores)
                    if m_score > best_cv:
                        best_cv = m_score
                        best_p = p

                # 最適パラメータで全量学習
                if is_raw:
                    full_sc = StandardScaler()
                    X_tr_sc = full_sc.fit_transform(X_tr_raw)
                    X_te_sc = full_sc.transform(X_te_raw)
                else:
                    X_tr_sc = X_tr_raw
                    X_te_sc = X_te_raw

                clf = m_fn(best_p)
                clf.fit(X_tr_sc, y_tr)
                pred_23 = clf.predict(X_te_sc)

                bacc = balanced_accuracy_score(y_test_23, pred_23)
                acc = accuracy_score(y_test_23, pred_23)
                f1 = f1_score(y_test_23, pred_23, average='macro')
                cm = confusion_matrix(y_test_23, pred_23)
                tn, fp, fn, tp = cm.ravel()

                key = f"{train_name} | {fset_name} | {m_name}"
                confusion_dict[key] = cm

                results.append({
                    "train_condition": train_name,
                    "feature_set": fset_name,
                    "n_features": len(feats),
                    "model": m_name,
                    "best_param": best_p,
                    "train_cv_bacc": round(best_cv * 100, 2),
                    "test_bacc": round(bacc * 100, 2),
                    "test_acc": round(acc * 100, 2),
                    "test_f1_macro": round(f1, 4),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "correct_count": tp + tn
                })

    df_res = pd.DataFrame(results)

    # テーブル保存
    out_csv_paper = TAB_DIR / "exp_distant_supervision_73_to_23.csv"
    out_csv_prog = PROG_TAB_DIR / "exp_distant_supervision_73_to_23.csv"
    df_res.to_csv(out_csv_paper, index=False, encoding="utf-8-sig")
    df_res.to_csv(out_csv_prog, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("【総合上位モデルランキング Top 12 (ブラインドテスト N=23)】")
    print("=" * 80)
    print(df_res.sort_values(by="test_bacc", ascending=False)[["train_condition", "feature_set", "model", "best_param", "train_cv_bacc", "test_bacc", "test_acc", "correct_count"]].head(12).to_string(index=False))

    # 可視化: 棒グラフ (PDF & PNG)
    plt.figure(figsize=(15, 7))
    top_feature_subsets = [
        "全11特徴量 [ドメイン内正規化Z-Score]",
        f"73本内部SFS自律選択 [Top-2: 語彙多様性+動画尺]",
        "重要2特徴量 [疑問文+動画尺] (先行最良)",
        "先行2特徴量 [疑問文+極性] (動画尺なし)"
    ]
    df_plot = df_res[(df_res['train_condition'] == "健全73本学習 (外れ値4本除去・提案)") & (df_res['feature_set'].isin(top_feature_subsets))]
    
    # モデルごとのプロット
    pivot_df = df_plot.pivot(index="model", columns="feature_set", values="test_bacc")
    ax = pivot_df.plot(kind="bar", figsize=(14, 7), colormap="viridis", width=0.8)
    plt.axhline(50.0, color="#d93025", linestyle="--", linewidth=1.5, label="チャンスレベル (50.0%)")
    plt.title("健全73本弱教師学習モデルによる被験者アンケート満足度(N=23)ブラインド逆転移性能", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("機械学習モデル (訓練CV最適化)", fontsize=11)
    plt.ylabel("未知テスト Balanced Accuracy (%)", fontsize=11)
    plt.ylim(30, 85)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.legend(title="特徴量セット", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig_pdf = FIG_DIR / "exp_distant_supervision_comparison.pdf"
    fig_png = FIG_DIR / "exp_distant_supervision_comparison.png"
    prog_fig_png = PROG_FIG_DIR / "exp_distant_supervision_comparison.png"

    plt.savefig(fig_pdf)
    plt.savefig(fig_png, dpi=300)
    plt.savefig(prog_fig_png, dpi=300)
    plt.close()

    # 混同行列可視化
    top4 = df_res[df_res['train_condition'] == "健全73本学習 (外れ値4本除去・提案)"].sort_values(by="test_bacc", ascending=False).head(4)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (_, row) in enumerate(top4.iterrows()):
        ax = axes[i]
        key = f"{row['train_condition']} | {row['feature_set']} | {row['model']}"
        cm = confusion_dict[key]
        ax.matshow(cm, cmap='Blues', alpha=0.7)
        for r_i in range(2):
            for c_i in range(2):
                ax.text(c_i, r_i, str(cm[r_i, c_i]), va='center', ha='center', fontsize=14, fontweight='bold')
        ax.set_title(f"Rank {i+1}: {row['model']}\n({row['feature_set'][:22]}...)\nB-Acc: {row['test_bacc']}% ({row['correct_count']}/23)", fontsize=9, fontweight='bold')
        ax.set_xlabel("予測 (0: Low, 1: High)", fontsize=9)
        ax.set_ylabel("実測 (0: Low, 1: High)", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Low', 'High'])
        ax.set_yticklabels(['Low', 'High'])
    plt.suptitle("健全73本弱教師学習モデルによるアンケート満足度(N=23)予測混同行列 Top 4", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()

    cm_pdf = FIG_DIR / "exp_distant_supervision_confusion_matrix.pdf"
    cm_png = FIG_DIR / "exp_distant_supervision_confusion_matrix.png"
    prog_cm_png = PROG_FIG_DIR / "exp_distant_supervision_confusion_matrix.png"

    plt.savefig(cm_pdf)
    plt.savefig(cm_png, dpi=300)
    plt.savefig(prog_cm_png, dpi=300)
    plt.close()

    print(f"\nすべての図表およびテーブルの生成が完了しました！")

if __name__ == "__main__":
    main()
