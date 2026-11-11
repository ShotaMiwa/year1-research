# -*- coding: utf-8 -*-
"""
evaluate_stratified_transfer_77.py

YouTube Shortsの再生回数バイアス（即座スワイプによるいいね率希釈）を緩和するため、
再生回数（view_count）に基づき3分位（Tiers: Low, Mid, High）にビン分割し、
ビン内中央値分割による層化ラベル（Stratified Like Rate Label）を作成。
被験者アンケート（N=23）学習モデルの外部転移汎化性能を再評価・比較する。
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

PROJECT_ROOT = Path("/home/shota/work/year1")
ALIGNMENT_JSON = PROJECT_ROOT / "data/processed/verified_alignments.json"
FEATURE_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data/processed/feature_matrix.csv"

FIG_DIR = PROJECT_ROOT / "docs/paper/figures"
TAB_DIR = PROJECT_ROOT / "docs/paper/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    # 77件の特徴量
    df_test = pd.read_csv(FEATURE_CSV)
    
    # alignment json から view_count をマージ
    with open(ALIGNMENT_JSON, encoding="utf-8") as f:
        align_dict = json.load(f)
    
    view_counts = {}
    like_counts = {}
    for cid_str, info in align_dict.items():
        if info.get("status") == "OK":
            sid = info.get("shorts_id")
            view_counts[sid] = info.get("view_count", np.nan)
            like_counts[sid] = info.get("like_count", np.nan)
            
    df_test["view_count"] = df_test["shorts_id"].map(view_counts)
    df_test["like_count"] = df_test["shorts_id"].map(like_counts)
    
    # 訓練データ
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    
    return df_train, df_test


def main():
    df_train, df_test = load_data()
    print(f"Loaded train (N={len(df_train)}) and test (N={len(df_test)})")
    print("view_count null count:", df_test["view_count"].isnull().sum())
    
    # 1. 再生回数といいね率の相関確認
    r_pearson = df_test["view_count"].corr(df_test["like_rate"], method="pearson")
    r_spearman = df_test["view_count"].corr(df_test["like_rate"], method="spearman")
    print(f"\n[再生回数 vs いいね率 相関]")
    print(f"  Pearson r  = {r_pearson:.4f}")
    print(f"  Spearman ρ = {r_spearman:.4f}")

    # 散布図プロット
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_test["view_count"], df_test["like_rate"], color="#2b5c8f", alpha=0.8, edgecolors="k", s=50)
    ax.set_xscale("log")
    ax.set_xlabel("再生回数 (Views, 対数スケール)", fontsize=11)
    ax.set_ylabel("いいね率 (%)", fontsize=11)
    ax.set_title(f"YouTube Shorts 再生回数 vs いいね率 (N=77)\nSpearman ρ = {r_spearman:.3f}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_view_count_vs_like_rate.png", dpi=200)
    plt.close()

    # 2. 再生回数による3分位（Tiers）分割
    # qcut で 3等分
    df_test["view_tier"] = pd.qcut(df_test["view_count"], q=3, labels=["Tier1_LowViews", "Tier2_MidViews", "Tier3_HighViews"])
    
    print("\n[3分位ごとの再生回数範囲といいね率統計]")
    for tier in ["Tier1_LowViews", "Tier2_MidViews", "Tier3_HighViews"]:
        sub = df_test[df_test["view_tier"] == tier]
        print(f"  {tier} (N={len(sub)}):")
        print(f"    Views range: {sub['view_count'].min():,.0f} 〜 {sub['view_count'].max():,.0f} (Mean: {sub['view_count'].mean():,.0f})")
        print(f"    Like rate  : Mean={sub['like_rate'].mean():.4f}%, Median={sub['like_rate'].median():.4f}%, Min={sub['like_rate'].min():.4f}%, Max={sub['like_rate'].max():.4f}%")

    # 3. 単純中央値ラベル vs ビン内層化ラベルの作成
    # 単純全体中央値
    overall_median = df_test["like_rate"].median()
    df_test["y_simple"] = (df_test["like_rate"] >= overall_median).astype(int)
    
    # ビン内中央値分割 (層化ラベル)
    df_test["y_stratified"] = df_test.groupby("view_tier")["like_rate"].transform(
        lambda s: (s >= s.median()).astype(int)
    )
    
    # 差分の確認
    diff_count = (df_test["y_simple"] != df_test["y_stratified"]).sum()
    print(f"\n単純ラベルと層化ラベルで反転した動画数: {diff_count} / {len(df_test)} ({diff_count/len(df_test)*100:.1f}%)")

    # 4. モデル転移評価の実行と比較
    FEATS_ALL11 = [
        'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
        'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
        'pause_ratio', 'sub_comment_similarity', 'comment_semantic_variance',
        'segment_length_sec'
    ]
    FEATS_SUBSET_QS = ['question_ratio', 'sentiment_polarity']
    FEATS_SUBSET_STUMP = ['comment_rate', 'segment_length_sec']
    FEATS_SUBSET_Q_LEN = ['question_ratio', 'segment_length_sec']

    FEATURE_SETS = {
        "全11特徴量": FEATS_ALL11,
        "重要2特徴量[疑問文+動画尺]": FEATS_SUBSET_Q_LEN,
        "決定株最良2特徴量[コメント率+動画尺]": FEATS_SUBSET_STUMP,
        "先行最良2特徴量[疑問文+感情極性]": FEATS_SUBSET_QS,
    }

    models = [
        ("Linear SVM (C=1.0)", lambda: SVC(kernel='linear', C=1.0, random_state=42)),
        ("Logistic Regression (L2)", lambda: LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)),
        ("Logistic Regression (L1)", lambda: LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)),
        ("Decision Tree (Depth=1)", lambda: DecisionTreeClassifier(max_depth=1, random_state=42)),
        ("Random Forest (Depth=3)", lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]

    # 訓練正解ラベル (N=23 Target_Satisfaction 中央値分割)
    train_median = df_train['Target_Satisfaction'].median()
    y_train = (df_train['Target_Satisfaction'] >= train_median).astype(int).values

    results = []

    for fset_name, fcols in FEATURE_SETS.items():
        X_train_raw = df_train[fcols].values
        X_test_raw = df_test[fcols].values
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        for mname, mfn in models:
            clf = mfn()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # 単純ラベルでの評価
            bacc_simple = balanced_accuracy_score(df_test["y_simple"], y_pred)
            acc_simple = accuracy_score(df_test["y_simple"], y_pred)
            f1_simple = f1_score(df_test["y_simple"], y_pred)
            
            # 層化ラベルでの評価
            bacc_strat = balanced_accuracy_score(df_test["y_stratified"], y_pred)
            acc_strat = accuracy_score(df_test["y_stratified"], y_pred)
            f1_strat = f1_score(df_test["y_stratified"], y_pred)
            
            results.append({
                "feature_set": fset_name,
                "model": mname,
                "simple_bacc": bacc_simple,
                "simple_acc": acc_simple,
                "simple_f1": f1_simple,
                "strat_bacc": bacc_strat,
                "strat_acc": acc_strat,
                "strat_f1": f1_strat,
                "bacc_diff": bacc_strat - bacc_simple
            })

    res_df = pd.DataFrame(results)
    out_csv = TAB_DIR / "exp_stratified_transfer_comparison.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved comparison to {out_csv}")

    # 上位比較の表示
    print("\n=== 層化（再生回数ビン分割）による性能変化（上位抜粋） ===")
    disp_cols = ["feature_set", "model", "simple_bacc", "strat_bacc", "bacc_diff"]
    top_df = res_df.sort_values(by="strat_bacc", ascending=False).head(10)
    for idx, row in top_df.iterrows():
        diff_str = f"+{row['bacc_diff']*100:.2f}%" if row['bacc_diff'] >= 0 else f"{row['bacc_diff']*100:.2f}%"
        print(f"[{row['feature_set']}] {row['model']}:")
        print(f"   単純B-Acc: {row['simple_bacc']*100:.2f}% -> 層化B-Acc: {row['strat_bacc']*100:.2f}% ({diff_str})")

    # 比較グラフの作成
    fig, ax = plt.subplots(figsize=(10, 6))
    top_models = res_df[res_df["feature_set"].isin(["全11特徴量", "重要2特徴量[疑問文+動画尺]"])].copy()
    top_models["label"] = top_models["feature_set"] + "\n" + top_models["model"]
    
    x = np.arange(len(top_models))
    width = 0.35
    
    ax.bar(x - width/2, top_models["simple_bacc"] * 100, width, label="単純ラベル (全体中央値)", color="#8faadc")
    ax.bar(x + width/2, top_models["strat_bacc"] * 100, width, label="層化ラベル (再生回数ビン内中央値)", color="#203864")
    
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1, label="ベースライン (Random 50%)")
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_title("再生回数層化ラベルによる外部転移性能の改善効果 (N=77)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(top_models["label"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(40, 75)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_stratified_transfer_comparison.png", dpi=200)
    plt.close()
    print(f"Saved figure to {FIG_DIR / 'exp_stratified_transfer_comparison.png'}")

if __name__ == "__main__":
    main()
