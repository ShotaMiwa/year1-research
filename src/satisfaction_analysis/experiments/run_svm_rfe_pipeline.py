# -*- coding: utf-8 -*-
"""
run_svm_rfe_pipeline.py

SVM-RFE（再帰的特徴消去 / Recursive Feature Elimination）パイプライン
全12特徴量から、モデルの重み |w_i|（係数の絶対値）が最も小さい特徴量を
1つずつ削ぎ落とし、各ステップでの交差検証スコアを評価して最適サブセットを自律選抜する。

【実行内容】
① アンケート23件内部 Nested LOO-CV:
   - 外側LOO-CV（23分割）でテスト1件を完全分離。
   - 内側22件で標準化＆RFE（12個から1個まで消去）。
   - 内側LOO-CV（22分割）の正解率が最大となるステップ k* を自動決定。
   - 最適特徴量セットで外側テスト1件を予測 $\to$ リーク一切なしの真の汎化正解率を算出。

② Shorts 73件外部弱教師学習 $\to$ アンケート23件逆転移:
   - 73件データで5-Fold CVを用いたRFEを実行し、CV正解率が最大となる特徴量セットを選抜。
   - その特徴量セットで73件全量を学習し、未見のアンケート23件に直接適用して的中率を評価。

【比較モデル】
- Linear SVM (`SVC(kernel='linear', C=1.0)`)
- ロジスティック回帰 (`LogisticRegression(penalty='l2', C=1.0)`)

【エビデンス出力】
- 集計表: `docs/paper/tables/exp3_rfe_comparison_results.csv`
- 図表: `docs/paper/figures/exp3_rfe_cv_curves.png`, `docs/paper/figures/exp3_rfe_cv_curves.pdf`
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from src.satisfaction_analysis.constants import FEATURE_NAMES_12, OUTLIER_SHORTS_IDS
from src.satisfaction_analysis.data_validator import load_and_validate_datasets

PROJECT_ROOT = "/home/shota/work/year1"

# 1. データセットの読み込み＆自動バリデーション
df_survey, df_shorts = load_and_validate_datasets()
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(OUTLIER_SHORTS_IDS)].copy().reset_index(drop=True)

y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
y_shorts = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values

n_survey = len(df_survey)
n_shorts = len(df_shorts_73)

def get_weights(clf):
    """線形モデルから特徴量重要度（係数の絶対値 |w|）を取得"""
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    raise ValueError("Classifier must have coef_ attribute.")


# ==============================================================================
# 実験①: アンケート23件内部 Nested LOO-CV による RFE
# ==============================================================================
def run_nested_loocv_rfe(clf_fn, model_name):
    print(f"\n--- [実験①] アンケート23件内部 Nested LOO-CV (RFE): {model_name} ---")
    outer_preds = []
    selected_k_list = []
    selected_features_frequency = {feat: 0 for feat in FEATURE_NAMES_12}

    for i in range(n_survey):
        tr_idx = [j for j in range(n_survey) if j != i]
        te_idx = [i]

        df_tr = df_survey.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_survey[tr_idx]
        df_te = df_survey.iloc[te_idx].reset_index(drop=True)
        y_te = y_survey[te_idx]

        # 内側22件の標準化
        sc_in = StandardScaler()
        X_in_all = sc_in.fit_transform(df_tr[FEATURE_NAMES_12].values)
        X_te_all = sc_in.transform(df_te[FEATURE_NAMES_12].values)

        n_in = len(df_tr)

        # RFEのシミュレーション: 12個から1個まで消去
        # 各特徴量数 k (12, 11, ..., 1) における内側LOO-CVスコアを測定
        current_features = list(range(len(FEATURE_NAMES_12)))
        rfe_history = []  # [(k, [feat_indices], inner_cv_acc)]

        while len(current_features) >= 1:
            k = len(current_features)
            # 現在の特徴量セットで内側LOO-CVを評価
            in_preds = []
            for k_in in range(n_in):
                k_tr = [m for m in range(n_in) if m != k_in]
                k_va = [k_in]
                clf = clf_fn()
                clf.fit(X_in_all[k_tr][:, current_features], y_tr[k_tr])
                in_preds.append(clf.predict(X_in_all[k_va][:, current_features])[0])
            inner_acc = np.mean(in_preds == y_tr)
            rfe_history.append((k, list(current_features), inner_acc))

            if k == 1:
                break

            # 内側22件全量で学習して重み最小の特徴量を1つ削除
            clf_full = clf_fn()
            clf_full.fit(X_in_all[:, current_features], y_tr)
            weights = get_weights(clf_full)
            worst_local_idx = np.argmin(weights)
            current_features.pop(worst_local_idx)

        # 最高スコアの k を選択（複数同点の場合は特徴量数が少ない＝より簡潔な方を優先）
        # rfe_history は k=12 から k=1 の順
        best_entry = max(rfe_history, key=lambda x: (x[2], -x[0]))
        best_k, best_feats, best_inner_score = best_entry

        selected_k_list.append(best_k)
        for f_idx in best_feats:
            selected_features_frequency[FEATURE_NAMES_12[f_idx]] += 1

        # 最適特徴量で外側テスト1件を予測
        clf_outer = clf_fn()
        clf_outer.fit(X_in_all[:, best_feats], y_tr)
        pred = clf_outer.predict(X_te_all[:, best_feats])[0]
        outer_preds.append(pred)

    outer_acc = np.mean(outer_preds == y_survey)
    correct_count = np.sum(outer_preds == y_survey)
    print(f"結果: {correct_count}/{n_survey} 的中 ({outer_acc*100:.1f}%)")
    print(f"選択された特徴量数 k の分布: 平均 {np.mean(selected_k_list):.1f} 個 (中央値 {np.median(selected_k_list):.0f})")

    # 選択頻度上位
    sorted_freq = sorted(selected_features_frequency.items(), key=lambda x: x[1], reverse=True)
    top_feats = [f"{k}({v}/{n_survey})" for k, v in sorted_freq[:5]]
    print(f"高頻度選抜特徴量 Top 5: {', '.join(top_feats)}")

    return {
        "model": model_name,
        "correct": correct_count,
        "total": n_survey,
        "accuracy": outer_acc,
        "avg_k": np.mean(selected_k_list),
        "selected_k_list": selected_k_list,
        "feature_freq": selected_features_frequency
    }


# ==============================================================================
# 実験②: Shorts 73件外部学習 $\to$ アンケート23件逆転移 RFE
# ==============================================================================
def run_transfer_rfe(clf_fn, model_name):
    print(f"\n--- [実験②] Shorts 73件 5-Fold CV (RFE) -> 23件逆転移: {model_name} ---")
    sc_tr = StandardScaler()
    X_tr_all = sc_tr.fit_transform(df_shorts_73[FEATURE_NAMES_12].values)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    current_features = list(range(len(FEATURE_NAMES_12)))
    rfe_history = []  # [(k, [feat_indices], cv_acc, cv_std)]

    while len(current_features) >= 1:
        k = len(current_features)
        # 5-Fold CV で正解率を評価
        fold_accs = []
        for tr_idx, va_idx in cv.split(X_tr_all):
            clf = clf_fn()
            clf.fit(X_tr_all[tr_idx][:, current_features], y_shorts[tr_idx])
            preds = clf.predict(X_tr_all[va_idx][:, current_features])
            fold_accs.append(accuracy_score(y_shorts[va_idx], preds))
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        rfe_history.append((k, list(current_features), mean_acc, std_acc))

        if k == 1:
            break

        # 73件全量で学習して重み最小の特徴量を削除
        clf_full = clf_fn()
        clf_full.fit(X_tr_all[:, current_features], y_shorts)
        weights = get_weights(clf_full)
        worst_local_idx = np.argmin(weights)
        current_features.pop(worst_local_idx)

    # 5-Fold CVスコアが最大の k を自動決定
    best_entry = max(rfe_history, key=lambda x: (x[2], -x[0]))
    best_k, best_feats, best_cv_acc, best_cv_std = best_entry
    best_feat_names = [FEATURE_NAMES_12[i] for i in best_feats]

    print(f"73件 5-Fold CV 最良ステップ: k={best_k} (CV正解率: {best_cv_acc*100:.1f}% ± {best_cv_std*100:.1f}%)")
    print(f"選抜された特徴量 ({best_k}個): {best_feat_names}")

    # 選ばれた特徴量セットで73件全量を学習 $\to$ アンケート23件をテスト
    sc_te = StandardScaler()
    X_te_all = sc_te.fit_transform(df_survey[FEATURE_NAMES_12].values)

    clf_final = clf_fn()
    clf_final.fit(X_tr_all[:, best_feats], y_shorts)
    test_preds = clf_final.predict(X_te_all[:, best_feats])

    transfer_acc = accuracy_score(y_survey, test_preds)
    transfer_correct = np.sum(test_preds == y_survey)

    print(f"23件逆転移テスト的中数: {transfer_correct}/{n_survey} ({transfer_acc*100:.1f}%)")

    return {
        "model": model_name,
        "best_k": best_k,
        "best_features": best_feat_names,
        "cv_accuracy": best_cv_acc,
        "transfer_correct": transfer_correct,
        "transfer_total": n_survey,
        "transfer_accuracy": transfer_acc,
        "rfe_history": rfe_history
    }


# ==============================================================================
# メイン実行ルーチン
# ==============================================================================
def main():
    models = [
        ("Linear SVM", lambda: SVC(kernel='linear', C=1.0, random_state=42)),
        ("Logistic Regression", lambda: LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000))
    ]

    nested_results = []
    transfer_results = []

    for name, clf_fn in models:
        res_nested = run_nested_loocv_rfe(clf_fn, name)
        nested_results.append(res_nested)

        res_transfer = run_transfer_rfe(clf_fn, name)
        transfer_results.append(res_transfer)

    # --------------------------------------------------------------------------
    # 結果の集約とCSV保存 (AGENTS.md ルール: docs/paper/tables/)
    # --------------------------------------------------------------------------
    tables_dir = os.path.join(PROJECT_ROOT, "docs/paper/tables")
    figures_dir = os.path.join(PROJECT_ROOT, "docs/paper/figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    summary_rows = []
    for n_res, t_res in zip(nested_results, transfer_results):
        summary_rows.append({
            "model": n_res["model"],
            "nested_loocv_acc": f"{n_res['correct']}/{n_res['total']} ({n_res['accuracy']*100:.1f}%)",
            "nested_avg_k": f"{n_res['avg_k']:.1f}",
            "transfer_73_cv_best_k": t_res["best_k"],
            "transfer_73_cv_acc": f"{t_res['cv_accuracy']*100:.1f}%",
            "transfer_test_acc": f"{t_res['transfer_correct']}/{t_res['transfer_total']} ({t_res['transfer_accuracy']*100:.1f}%)",
            "transfer_selected_features": " + ".join(t_res["best_features"])
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(tables_dir, "exp3_rfe_comparison_results.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 集計結果表を保存しました: {csv_path}")

    # --------------------------------------------------------------------------
    # グラフ描画（73件 5-Fold CV における RFE 削減曲線）
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for t_res in transfer_results:
        # rfe_history: [(k, feats, acc, std), ...] (k=12 down to 1)
        # グラフは見やすく k=1 から 12 へソート
        history_sorted = sorted(t_res["rfe_history"], key=lambda x: x[0])
        ks = [h[0] for h in history_sorted]
        accs = [h[2] * 100 for h in history_sorted]
        stds = [h[3] * 100 for h in history_sorted]

        line = plt.plot(ks, accs, marker='o', label=f"{t_res['model']} (Peak k={t_res['best_k']})")
        plt.fill_between(ks, np.array(accs) - np.array(stds), np.array(accs) + np.array(stds), alpha=0.15)
        
        # ピークにマーカー
        peak_idx = ks.index(t_res['best_k'])
        plt.scatter([t_res['best_k']], [accs[peak_idx]], s=120, zorder=5, edgecolors='black')

    plt.title("SVM-RFE Feature Elimination Curves on 73 YouTube Shorts (5-Fold CV)", fontsize=13)
    plt.xlabel("Number of Selected Features (k)", fontsize=12)
    plt.ylabel("5-Fold CV Accuracy (%)", fontsize=12)
    plt.xticks(range(1, 13))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    png_path = os.path.join(figures_dir, "exp3_rfe_cv_curves.png")
    pdf_path = os.path.join(figures_dir, "exp3_rfe_cv_curves.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"✅ RFE削減曲線の図を保存しました: {png_path} / {pdf_path}")


if __name__ == "__main__":
    main()
