# -*- coding: utf-8 -*-
"""
run_svm_rfe_pipeline_v2.py

実験②改善版: Repeated Stratified K-Fold (5分割×10回=50回平均) でCVを安定化し、
k=1トラップを回避する。k選択には「ピーク±1%以内なら特徴量数が多い方を優先」の許容幅も追加。

実験①（アンケート23件内部 Nested LOO-CV）は変更なし。
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
from sklearn.model_selection import RepeatedStratifiedKFold
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

# 許容幅: ピークから何%pt以内なら「より多い特徴量を優先」するか
TOLERANCE_PT = 1.0  # 1%pt


def get_weights(clf):
    return np.abs(clf.coef_[0])


# ==============================================================================
# 実験①: アンケート23件内部 Nested LOO-CV（変更なし）
# ==============================================================================
def run_nested_loocv_rfe(clf_fn, model_name):
    print(f"\n--- [実験①] アンケート23件内部 Nested LOO-CV (RFE): {model_name} ---")
    outer_preds = []
    selected_k_list = []

    for i in range(n_survey):
        tr_idx = [j for j in range(n_survey) if j != i]
        df_tr = df_survey.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_survey[tr_idx]
        df_te = df_survey.iloc[[i]].reset_index(drop=True)
        y_te = y_survey[[i]]

        n_in = len(df_tr)
        sc_in = StandardScaler()
        X_in_all = sc_in.fit_transform(df_tr[FEATURE_NAMES_12].values)
        X_te_all = sc_in.transform(df_te[FEATURE_NAMES_12].values)

        current_features = list(range(len(FEATURE_NAMES_12)))
        rfe_history = []

        while len(current_features) >= 1:
            k = len(current_features)
            in_preds = []
            for k_in in range(n_in):
                k_tr = [m for m in range(n_in) if m != k_in]
                clf = clf_fn()
                clf.fit(X_in_all[k_tr][:, current_features], y_tr[k_tr])
                in_preds.append(clf.predict(X_in_all[[k_in]][:, current_features])[0])
            inner_acc = np.mean(in_preds == y_tr)
            rfe_history.append((k, list(current_features), inner_acc))
            if k == 1:
                break
            clf_full = clf_fn()
            clf_full.fit(X_in_all[:, current_features], y_tr)
            worst = np.argmin(get_weights(clf_full))
            current_features.pop(worst)

        best_entry = max(rfe_history, key=lambda x: (x[2], -x[0]))
        best_k, best_feats, _ = best_entry
        selected_k_list.append(best_k)

        clf_outer = clf_fn()
        clf_outer.fit(X_in_all[:, best_feats], y_tr)
        outer_preds.append(clf_outer.predict(X_te_all[:, best_feats])[0])

    correct = int(np.sum(outer_preds == y_survey))
    acc = correct / n_survey
    print(f"結果: {correct}/{n_survey} ({acc*100:.1f}%) | 平均k={np.mean(selected_k_list):.1f}")
    return {"model": model_name, "correct": correct, "total": n_survey, "accuracy": acc,
            "avg_k": np.mean(selected_k_list)}


# ==============================================================================
# 実験②改善版: Repeated Stratified K-Fold (5×10) + 許容幅付きk選択
# ==============================================================================
def run_transfer_rfe_v2(clf_fn, model_name):
    print(f"\n--- [実験②改善] 73件 Repeated Stratified 5-Fold×10 (RFE) → 23件: {model_name} ---")

    sc_tr = StandardScaler()
    X_tr_all = sc_tr.fit_transform(df_shorts_73[FEATURE_NAMES_12].values)

    # Repeated Stratified K-Fold: 5分割×10回 = 50回の平均
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    current_features = list(range(len(FEATURE_NAMES_12)))
    rfe_history = []  # (k, [feat_idx], mean_acc, std_acc)

    while len(current_features) >= 1:
        k = len(current_features)
        fold_accs = []
        for tr_idx, va_idx in cv.split(X_tr_all, y_shorts):
            clf = clf_fn()
            clf.fit(X_tr_all[tr_idx][:, current_features], y_shorts[tr_idx])
            preds = clf.predict(X_tr_all[va_idx][:, current_features])
            fold_accs.append(accuracy_score(y_shorts[va_idx], preds))
        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        rfe_history.append((k, list(current_features), mean_acc, std_acc))

        if k == 1:
            break
        clf_full = clf_fn()
        clf_full.fit(X_tr_all[:, current_features], y_shorts)
        worst = np.argmin(get_weights(clf_full))
        current_features.pop(worst)

    # ピーク選択: 「ピーク精度 - TOLERANCE_PT%」以上のkの中で最大kを採用
    best_acc = max(h[2] for h in rfe_history)
    threshold = (best_acc * 100 - TOLERANCE_PT) / 100
    candidates = [(h[0], h[1], h[2], h[3]) for h in rfe_history if h[2] >= threshold]
    best_entry = max(candidates, key=lambda x: x[0])  # 最大k
    best_k, best_feats, best_cv_acc, best_cv_std = best_entry
    best_feat_names = [FEATURE_NAMES_12[i] for i in best_feats]

    print(f"CV ピーク精度: {best_acc*100:.1f}%")
    print(f"許容幅 ±{TOLERANCE_PT}%pt 内で最大kを採用 → k={best_k} (CV: {best_cv_acc*100:.1f}% ± {best_cv_std*100:.1f}%)")
    print(f"選抜特徴量 ({best_k}個): {best_feat_names}")

    # 73件全量で学習 → アンケート23件テスト
    sc_te = StandardScaler()
    X_te_all = sc_te.fit_transform(df_survey[FEATURE_NAMES_12].values)
    clf_final = clf_fn()
    clf_final.fit(X_tr_all[:, best_feats], y_shorts)
    test_preds = clf_final.predict(X_te_all[:, best_feats])

    transfer_acc = accuracy_score(y_survey, test_preds)
    transfer_correct = int(np.sum(test_preds == y_survey))
    print(f"23件転移テスト: {transfer_correct}/{n_survey} ({transfer_acc*100:.1f}%)")

    # スコア推移曲線データ
    history_sorted = sorted(rfe_history, key=lambda x: x[0])

    return {
        "model": model_name,
        "best_k": best_k,
        "best_features": best_feat_names,
        "cv_accuracy": best_cv_acc,
        "cv_std": best_cv_std,
        "transfer_correct": transfer_correct,
        "transfer_total": n_survey,
        "transfer_accuracy": transfer_acc,
        "rfe_history": history_sorted
    }


# ==============================================================================
# メイン
# ==============================================================================
def main():
    models = [
        ("Linear SVM",          lambda: SVC(kernel='linear', C=1.0, random_state=42)),
        ("Logistic Regression",  lambda: LogisticRegression(penalty='l2', C=1.0,
                                          random_state=42, max_iter=1000)),
    ]

    nested_results   = []
    transfer_results = []

    for name, clf_fn in models:
        nested_results.append(run_nested_loocv_rfe(clf_fn, name))
        transfer_results.append(run_transfer_rfe_v2(clf_fn, name))

    # --- 集計表 ---
    tables_dir  = os.path.join(PROJECT_ROOT, "docs/paper/tables")
    figures_dir = os.path.join(PROJECT_ROOT, "docs/paper/figures")
    os.makedirs(tables_dir,  exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    rows = []
    for n_res, t_res in zip(nested_results, transfer_results):
        rows.append({
            "model": n_res["model"],
            "nested_loocv_acc":      f"{n_res['correct']}/{n_res['total']} ({n_res['accuracy']*100:.1f}%)",
            "nested_avg_k":          f"{n_res['avg_k']:.1f}",
            "transfer_cv_best_k":    t_res["best_k"],
            "transfer_cv_acc":       f"{t_res['cv_accuracy']*100:.1f}% ± {t_res['cv_std']*100:.1f}%",
            "transfer_test_acc":     f"{t_res['transfer_correct']}/{t_res['transfer_total']} ({t_res['transfer_accuracy']*100:.1f}%)",
            "transfer_selected_features": " + ".join(t_res["best_features"]),
        })
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "exp3b_rfe_v2_results.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 集計表保存: {csv_path}")

    # --- RFE削減曲線 ---
    plt.figure(figsize=(10, 6))
    for t_res in transfer_results:
        ks   = [h[0] for h in t_res["rfe_history"]]
        accs = [h[2] * 100 for h in t_res["rfe_history"]]
        stds = [h[3] * 100 for h in t_res["rfe_history"]]
        plt.plot(ks, accs, marker='o', label=f"{t_res['model']} (Peak k={t_res['best_k']})")
        plt.fill_between(ks, np.array(accs)-np.array(stds),
                             np.array(accs)+np.array(stds), alpha=0.15)
        pk_idx = ks.index(t_res['best_k'])
        plt.scatter([t_res['best_k']], [accs[pk_idx]], s=120, zorder=5, edgecolors='black')

    plt.title("SVM-RFE v2: Repeated Stratified 5-Fold×10 CV Curves (73 Shorts)", fontsize=13)
    plt.xlabel("Number of Selected Features (k)", fontsize=12)
    plt.ylabel("Repeated CV Accuracy (%)", fontsize=12)
    plt.xticks(range(1, 13))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    png_path = os.path.join(figures_dir, "exp3b_rfe_v2_cv_curves.png")
    pdf_path = os.path.join(figures_dir, "exp3b_rfe_v2_cv_curves.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"✅ 図保存: {png_path}")


if __name__ == "__main__":
    main()
