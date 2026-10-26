# coding: utf-8
"""
マルチモデル Nested LOO-CV 最適特徴量数 (k=1〜3) 探索スクリプト (実験C)
========================================================================
目的:
  切り抜き動画の満足度二値分類（High: 14件 vs Low: 9件, N=23）において、
  特徴量数 k の制約を 1〜2 個から 1〜3 個（全298通り）へと拡張。
  「各モデルにおいて、何個の特徴量を組み合わせるのが最も汎化精度（Balanced Accuracy）が高いか」
  「3個以上組み合わせた場合に過学習が発生するか」を定量解明する。

探索空間:
  - 特徴量プール: 12特徴量
  - k = 1: 12通り
  - k = 2: 66通り
  - k = 3: 220通り
  - 合計: 298通りの組み合わせを内側LOO-CVで総当たり探索

対象モデル:
  1. LogisticRegression (L2線形)
  2. SVC_Linear (線形SVM)
  3. SVC_RBF (RBFカーネルSVM)
  4. DecisionTree (決定木 max_depth=3, min_samples_leaf=2)
  5. RandomForest (ランダムフォレスト max_depth=3, n_est=50)
"""

import os, sys, warnings, time
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

# パス設定
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_matrix.csv")
OUT_DIR = os.path.join(BASE_DIR, "進捗報告用", "20261028")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# データ読み込み
df = pd.read_csv(DATA_PATH)
TARGET = "Target_Satisfaction"
median_target = df[TARGET].median()
y_all = (df[TARGET] >= median_target).astype(int).values
N = len(df)

# 全12特徴量
FEATURES = [
    "comment_rate", "question_ratio", "exclamation_ratio",
    "sentiment_polarity", "subtitle_sentiment_polarity",
    "speech_rate", "grass_ratio",
    "hm_mean", "hm_max",
    "comment_semantic_variance", "sub_comment_similarity",
    "segment_length_sec"
]

# 特徴量組み合わせ (k = 1〜3: 12 + 66 + 220 = 298通り)
MAX_K = 3
comb_list = []
for k in range(1, MAX_K + 1):
    comb_list.extend(list(combinations(range(len(FEATURES)), k)))

def get_model(model_type):
    if model_type == "LogisticRegression":
        return LogisticRegression(penalty="l2", C=1.0, solver="liblinear", random_state=42)
    elif model_type == "SVC_Linear":
        return SVC(kernel="linear", C=1.0, random_state=42)
    elif model_type == "SVC_RBF":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    elif model_type == "DecisionTree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
    elif model_type == "RandomForest":
        return RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def eval_single_comb(comb, X_tr, y_tr, model_type, need_scale):
    sub_X = X_tr[:, list(comb)]
    n_in = len(sub_X)
    in_preds = []
    
    for in_i in range(n_in):
        in_X_tr = np.delete(sub_X, in_i, axis=0)
        in_y_tr = np.delete(y_tr, in_i)
        in_X_te = sub_X[in_i:in_i+1]
        
        if need_scale:
            scaler = StandardScaler()
            in_X_tr = scaler.fit_transform(in_X_tr)
            in_X_te = scaler.transform(in_X_te)
            
        clf = get_model(model_type)
        clf.fit(in_X_tr, in_y_tr)
        in_preds.append(clf.predict(in_X_te)[0])
        
    in_preds = np.array(in_preds)
    bal = balanced_accuracy_score(y_tr, in_preds)
    acc = accuracy_score(y_tr, in_preds)
    return comb, bal, acc

MODEL_EXPERIMENTS = [
    {
        "id": "LogReg_k3",
        "name": "ロジスティック回帰 (k=1〜3)",
        "model_type": "LogisticRegression",
        "need_scale": True,
        "category": "線形モデル"
    },
    {
        "id": "SVC_Lin_k3",
        "name": "線形SVM (k=1〜3)",
        "model_type": "SVC_Linear",
        "need_scale": True,
        "category": "線形モデル"
    },
    {
        "id": "SVC_RBF_k3",
        "name": "非線形SVM (k=1〜3)",
        "model_type": "SVC_RBF",
        "need_scale": True,
        "category": "カーネル非線形"
    },
    {
        "id": "DT_k3",
        "name": "決定木 深さ3 (k=1〜3)",
        "model_type": "DecisionTree",
        "need_scale": False,
        "category": "非線形・解釈性"
    },
    {
        "id": "RF_k3",
        "name": "ランダムフォレスト 深さ3 (k=1〜3)",
        "model_type": "RandomForest",
        "need_scale": False,
        "category": "非線形・アンサンブル"
    }
]

def main():
    print(f"=== 実験C: 最適特徴量数 (k=1〜3, 298通り) 探索 Nested LOO-CV 開始 (N={N}) ===")
    X_mat = df[FEATURES].values
    
    summary_rows = []
    stability_dict = {}
    k_distribution_dict = {}
    
    for exp in MODEL_EXPERIMENTS:
        exp_id = exp["id"]
        exp_name = exp["name"]
        model_type = exp["model_type"]
        need_scale = exp["need_scale"]
        category = exp["category"]
        
        print(f"\n" + "="*70)
        print(f"[{category}] {exp_name} の Nested LOO-CV を実行中 (内側探索空間: {len(comb_list)}通り)...")
        t0 = time.time()
        
        outer_preds = []
        outer_details = []
        selected_features_all = []
        selected_k_all = []
        
        for out_i in range(N):
            f_t0 = time.time()
            X_tr = np.delete(X_mat, out_i, axis=0)
            y_tr = np.delete(y_all, out_i)
            X_te = X_mat[out_i:out_i+1]
            y_te = y_all[out_i]
            
            # 内側LOO-CVを並列評価
            results = Parallel(n_jobs=-1)(
                delayed(eval_single_comb)(comb, X_tr, y_tr, model_type, need_scale)
                for comb in comb_list
            )
            
            best_bal = -1.0
            best_acc = -1.0
            best_comb = None
            
            for comb, bal, acc in results:
                update = False
                if bal > best_bal:
                    update = True
                elif bal == best_bal:
                    if acc > best_acc:
                        update = True
                    elif acc == best_acc and best_comb is not None:
                        # 同スコアならより少ない特徴量（節約性）を優先
                        if len(comb) < len(best_comb):
                            update = True
                            
                if update:
                    best_bal = bal
                    best_acc = acc
                    best_comb = comb
                    
            best_f_names = [FEATURES[idx] for idx in best_comb]
            k_val = len(best_comb)
            selected_features_all.extend(best_f_names)
            selected_k_all.append(k_val)
            
            final_sub_X_tr = X_tr[:, list(best_comb)]
            final_sub_X_te = X_te[:, list(best_comb)]
            
            if need_scale:
                scaler = StandardScaler()
                final_sub_X_tr = scaler.fit_transform(final_sub_X_tr)
                final_sub_X_te = scaler.transform(final_sub_X_te)
                
            final_clf = get_model(model_type)
            final_clf.fit(final_sub_X_tr, y_tr)
            pred = final_clf.predict(final_sub_X_te)[0]
            outer_preds.append(pred)
            
            outer_details.append({
                "Outer_Fold": out_i + 1,
                "True_Class": y_te,
                "Pred_Class": pred,
                "Is_Correct": bool(pred == y_te),
                "Inner_Best_BalAcc": best_bal,
                "Inner_Best_Acc": best_acc,
                "Selected_k": k_val,
                "Selected_Features": ", ".join(best_f_names)
            })
            print(f"  Fold {out_i+1:02d}/{N} 完了 ({time.time()-f_t0:.1f}s) | k={k_val} | 選出: {', '.join(best_f_names)} | 正解: {pred == y_te}")
            
        outer_preds = np.array(outer_preds)
        final_acc = accuracy_score(y_all, outer_preds)
        final_bal = balanced_accuracy_score(y_all, outer_preds)
        final_f1 = f1_score(y_all, outer_preds, average="macro")
        cm = confusion_matrix(y_all, outer_preds)
        correct_cnt = int(sum(outer_preds == y_all))
        
        elapsed = time.time() - t0
        print(f"--> {exp_name} 完了 ({elapsed:.1f}s) | Balanced Acc: {final_bal*100:.2f}% | 正解: {correct_cnt}/{N} ({final_acc*100:.2f}%)")
        
        # 詳細保存
        pd.DataFrame(outer_details).to_csv(os.path.join(TAB_DIR, f"exp_multimodel_k3_{exp_id}_details.csv"), index=False)
        
        # 特徴量安定性集計
        cnts = Counter(selected_features_all)
        df_st = pd.DataFrame([
            {"Feature": f, "Selected_Count": cnts.get(f, 0), "Selection_Rate": cnts.get(f, 0) / N * 100}
            for f in FEATURES
        ]).sort_values("Selected_Count", ascending=False).reset_index(drop=True)
        df_st.to_csv(os.path.join(TAB_DIR, f"exp_multimodel_k3_{exp_id}_stability.csv"), index=False)
        stability_dict[exp_id] = df_st
        
        # k値の内訳集計
        k_cnts = Counter(selected_k_all)
        k_dist_str = f"k=1: {k_cnts.get(1, 0)}回, k=2: {k_cnts.get(2, 0)}回, k=3: {k_cnts.get(3, 0)}回"
        k_distribution_dict[exp_id] = k_cnts
        
        summary_rows.append({
            "モデルID": exp_id,
            "モデル名": exp_name,
            "カテゴリ": category,
            "正解率 (Accuracy)": f"{correct_cnt}/{N} ({final_acc*100:.1f}%)",
            "Balanced Accuracy": final_bal * 100,
            "Macro F1": final_f1 * 100,
            "Low正解 (TN/9)": cm[0, 0],
            "Low誤判定 (FP/9)": cm[0, 1],
            "High誤判定 (FN/14)": cm[1, 0],
            "High正解 (TP/14)": cm[1, 1],
            "選出kの内訳 (23回中)": k_dist_str,
            "Top1特徴量": f"{df_st.iloc[0]['Feature']} ({df_st.iloc[0]['Selection_Rate']:.1f}%)",
            "Top2特徴量": f"{df_st.iloc[1]['Feature']} ({df_st.iloc[1]['Selection_Rate']:.1f}%)" if len(df_st) > 1 else "-"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TAB_DIR, "exp_multimodel_k_search_summary.csv"), index=False)
    
    print("\n" + "="*80)
    print("=== 実験C: 最適特徴量数 (k=1〜3) 探索 評価サマリー ===")
    print(summary_df[["モデル名", "正解率 (Accuracy)", "Balanced Accuracy", "Macro F1", "選出kの内訳 (23回中)", "Top1特徴量"]].to_string(index=False))
    
    # プロット作成
    create_k3_plots(summary_df, k_distribution_dict)
    print(f"\n実験Cの全処理が完了しました。")

def create_k3_plots(summary_df, k_distribution_dict):
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Balanced Accuracy
    ax1 = axes[0]
    colors = ["#4C72B0" if cat == "線形モデル" else "#55A868" if "カーネル" in cat else "#C44E52" for cat in summary_df["カテゴリ"]]
    bars = ax1.bar(summary_df["モデルID"], summary_df["Balanced Accuracy"], color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.axhline(50.0, color="gray", linestyle="--", linewidth=1.5, label="チャンスレベル (50.0%)")
    
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
        
    ax1.set_title("各モデルの汎化性能 [特徴量数 k=1〜3探索]\n(最大298通りから最適kを選抜)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 80)
    ax1.set_xticklabels(summary_df["モデルID"], fontsize=10, fontweight="bold", rotation=15)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    
    # 2. 選出されたkの積み上げ棒グラフ
    ax2 = axes[1]
    m_ids = summary_df["モデルID"].tolist()
    k1_vals = [k_distribution_dict[m].get(1, 0) for m in m_ids]
    k2_vals = [k_distribution_dict[m].get(2, 0) for m in m_ids]
    k3_vals = [k_distribution_dict[m].get(3, 0) for m in m_ids]
    
    ax2.bar(m_ids, k1_vals, label="k=1 (1特徴量)", color="#4C72B0", width=0.55, edgecolor="black")
    ax2.bar(m_ids, k2_vals, bottom=k1_vals, label="k=2 (2特徴量)", color="#55A868", width=0.55, edgecolor="black")
    bottom_k3 = np.array(k1_vals) + np.array(k2_vals)
    ax2.bar(m_ids, k3_vals, bottom=bottom_k3, label="k=3 (3特徴量)", color="#C44E52", width=0.55, edgecolor="black")
    
    ax2.set_title("各モデルで選抜された特徴量数 k の内訳\n(外側23回中)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("選出回数 (合計23回)", fontsize=12)
    ax2.set_ylim(0, 26)
    ax2.set_xticklabels(m_ids, fontsize=10, fontweight="bold", rotation=15)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plot_path = os.path.join(FIG_DIR, "exp_multimodel_k3_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"実験Cの比較グラフを保存しました: {plot_path}")

if __name__ == "__main__":
    main()
