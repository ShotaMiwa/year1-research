# -*- coding: utf-8 -*-
"""
run_svm_rfe_detail.py
Nested LOO-CVの各フォールドで「何個の特徴量が選ばれたか」「何が選ばれたか」「内側CVのスコア推移」を詳細出力する診断スクリプト。
"""
import os
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = "/home/shota/work/year1"
sys.path.insert(0, PROJECT_ROOT)

from src.satisfaction_analysis.constants import FEATURE_NAMES_12, OUTLIER_SHORTS_IDS
from src.satisfaction_analysis.data_validator import load_and_validate_datasets

df_survey, df_shorts = load_and_validate_datasets()
y = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values
n = len(df_survey)

def get_weights(clf):
    return np.abs(clf.coef_[0])

def run_nested_rfe_detail(clf_fn, model_name):
    print(f"\n{'='*70}")
    print(f"[Nested LOO-CV RFE 詳細ログ] モデル: {model_name}")
    print(f"{'='*70}")

    outer_preds = []
    outer_corrects = []
    fold_details = []  # 各フォールドの詳細

    for i in range(n):
        tr_idx = [j for j in range(n) if j != i]
        te_idx = [i]

        df_tr = df_survey.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        df_te = df_survey.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]

        n_in = len(df_tr)  # 22

        sc_in = StandardScaler()
        X_in_all = sc_in.fit_transform(df_tr[FEATURE_NAMES_12].values)
        X_te_all = sc_in.transform(df_te[FEATURE_NAMES_12].values)

        # RFE: k=12 から k=1 まで消去しながら内側LOO-CVスコアを記録
        current_features = list(range(len(FEATURE_NAMES_12)))
        rfe_history = []  # (k, [feat_idx], inner_acc)

        while len(current_features) >= 1:
            k = len(current_features)
            # 内側22件 LOO-CV
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

            clf_full = clf_fn()
            clf_full.fit(X_in_all[:, current_features], y_tr)
            weights = get_weights(clf_full)
            worst_local_idx = np.argmin(weights)
            current_features.pop(worst_local_idx)

        # ピーク: 同点は k が小さい方優先
        best_entry = max(rfe_history, key=lambda x: (x[2], -x[0]))
        best_k, best_feats, best_inner_acc = best_entry
        best_feat_names = [FEATURE_NAMES_12[f] for f in best_feats]

        # 外側テスト
        clf_outer = clf_fn()
        clf_outer.fit(X_in_all[:, best_feats], y_tr)
        pred = clf_outer.predict(X_te_all[:, best_feats])[0]
        correct = int(pred == y_te[0])
        outer_preds.append(pred)
        outer_corrects.append(correct)

        # 内側CVスコア推移（表示用：k=1→12順に並び替え）
        history_sorted = sorted(rfe_history, key=lambda x: x[0])
        cv_curve = {h[0]: f"{h[2]*100:.1f}%" for h in history_sorted}

        fold_details.append({
            "fold": i + 1,
            "true_label": y_te[0],
            "pred_label": pred,
            "correct": "✅" if correct else "❌",
            "best_k": best_k,
            "best_inner_cv": f"{best_inner_acc*100:.1f}%",
            "selected_features": best_feat_names,
            "inner_cv_curve": cv_curve
        })

    # --- フォールド詳細出力 ---
    print(f"\n{'フォールド':<4} {'真ラベル':<6} {'予測':<6} {'判定':<4} {'k*':<4} {'内側CVピーク':<12} {'選ばれた特徴量'}")
    print("-" * 100)
    for d in fold_details:
        feat_str = " + ".join(d["selected_features"])
        print(f"  {d['fold']:<4} {d['true_label']:<6} {d['pred_label']:<6} {d['correct']:<4} {d['best_k']:<4} {d['best_inner_cv']:<12} {feat_str}")

    # --- 内側CVスコア推移（全フォールド平均） ---
    print(f"\n--- 内側 LOO-CV スコア推移（k=1〜12 の平均正解率）---")
    all_curves = {k: [] for k in range(1, 13)}
    for d in fold_details:
        for k, acc_str in d["inner_cv_curve"].items():
            all_curves[k].append(float(acc_str.strip('%')))
    print(f"{'k':<4} {'平均内側CVスコア':>16}")
    for k in range(1, 13):
        mean_acc = np.mean(all_curves[k]) if all_curves[k] else float('nan')
        bar = "█" * int(mean_acc / 5)
        print(f"  {k:<4} {mean_acc:>6.1f}%  {bar}")

    # --- k* の分布 ---
    k_list = [d["best_k"] for d in fold_details]
    print(f"\n--- 選ばれた k* の分布 ---")
    for k_val in sorted(set(k_list)):
        count = k_list.count(k_val)
        print(f"  k={k_val:2d}: {'▮' * count} ({count}回)")
    print(f"  平均 k = {np.mean(k_list):.1f}、中央値 k = {np.median(k_list):.0f}")

    # --- 特徴量選択頻度 ---
    freq = {f: 0 for f in FEATURE_NAMES_12}
    for d in fold_details:
        for f in d["selected_features"]:
            freq[f] += 1
    print(f"\n--- 特徴量選択頻度（全{n}フォールド中）---")
    for feat, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"  {feat:<35} {bar} ({cnt}/{n})")

    # --- 最終精度 ---
    total_correct = sum(outer_corrects)
    total_acc = total_correct / n
    print(f"\n>>> 最終的中率: {total_correct}/{n} ({total_acc*100:.1f}%)")
    return fold_details

if __name__ == "__main__":
    run_nested_rfe_detail(
        lambda: SVC(kernel='linear', C=1.0, random_state=42),
        "Linear SVM"
    )
    run_nested_rfe_detail(
        lambda: LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000),
        "Logistic Regression"
    )
