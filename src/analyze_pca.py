#!/usr/bin/env python3
"""
アンケート質問項目の主成分分析 (Principal Component Analysis: PCA) スクリプト

目的:
  アンケートの評価項目（8項目）の情報を少数の合成変数（主成分得点）に集約し、
  統計的なマッピングおよび合併スコアを作成するための基準値を出力・可視化する。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import japanize_matplotlib

from src.utils.config import (
    init_directories,
    PROJECT_ROOT,
    DATA_PROCESSED_DIR,
    OUT_CORRELATION_DIR
)

# ディレクトリ初期化
init_directories()

MERGED_CSV = DATA_PROCESSED_DIR / "merged_survey_heatmap.csv"
OUT_DIR = OUT_CORRELATION_DIR

def run_pca():
    print("📐 主成分分析 (Principal Component Analysis) を開始します...")
    
    # データ読み込み
    df = pd.read_csv(MERGED_CSV)
    
    items = [
        "面白かった",
        "話に引き込まれた",
        "有益な内容だった",
        "誰かに共有したいと思った"
    ]
    
    X = df[items].dropna()
    print(f"分析対象セグメント数: {len(X)}")
    
    # データを標準化 (平均0, 分散1に揃える - PCAでは必須ステップ)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCAの実行
    pca = PCA()
    pca.fit(X_scaled)
    
    # ── 1. 主成分決定のための指標算出 ──
    # 固有値 (sklearnでは explained_variance_ に相当)
    eigenvalues = pca.explained_variance_
    # 寄与率 (Proportion of Variance)
    explained_variance_ratio = pca.explained_variance_ratio_
    # 累積寄与率 (Cumulative Variance)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    pca_stats = pd.DataFrame({
        "固有値 (Eigenvalue)": eigenvalues,
        "寄与率 (Proportion Var)": explained_variance_ratio,
        "累積寄与率 (Cumulative Var)": cumulative_variance_ratio
    }, index=[f"第{i}主成分" for i in range(1, len(items) + 1)])
    
    print("\n[ 主成分分析 統計量 ]")
    print(pca_stats.round(4))
    
    # カイザー基準 (固有値 > 1.0) の数
    kaiser_n = sum(eigenvalues > 1.0)
    print(f"\n-> カイザー基準 (固有値 > 1.0) による推奨主成分数: {kaiser_n}")
    
    # ── 2. 主成分負荷量 (Factor Loadings) の計算 ──
    # sklearnの components_ に 固有値の平方根を掛けることで負荷量に変換
    # 負荷量は各変数と主成分の相関係数に相当する
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    loadings_df = pd.DataFrame(
        loadings,
        index=items,
        columns=[f"PC{i}" for i in range(1, len(items) + 1)]
    )
    
    # 上位3つの主成分を表示
    print("\n[ 主成分負荷量 (変数の合成の重み) - 上位3主成分 ]")
    print(loadings_df.iloc[:, :3].round(4))
    
    # CSVに保存
    loadings_df.to_csv(OUT_DIR / "pca_loadings.csv", encoding="utf-8-sig")
    
    # ── 3. 主成分得点 (PC Scores) の算出と保存 ──
    # 各セグメントの合成スコアを計算
    pca_scores = pca.transform(X_scaled)
    for i in range(kaiser_n):
        df[f"主成分得点_PC{i+1}"] = pca_scores[:, i]
        
    df.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 主成分得点を結合データに保存しました: {MERGED_CSV}")
    
    # ── 4. スクリープロットの作成・保存 ──
    fig, ax1 = plt.subplots(figsize=(8, 5))
    pc_indices = np.arange(1, len(items) + 1)
    
    # 固有値プロット（折れ線グラフ）
    color = "#2C3E50"
    ax1.set_xlabel("主成分番号", fontsize=12)
    ax1.set_ylabel("固有値 (Eigenvalue)", color=color, fontsize=12)
    line1 = ax1.plot(pc_indices, eigenvalues, marker="o", color=color, linewidth=2, label="固有値")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axhline(1.0, color="#E74C3C", linestyle="--", alpha=0.7, label="カイザー基準 (固有値=1.0)")
    
    # 累積寄与率プロット（棒グラフ）
    ax2 = ax1.twinx()
    color = "#2ECC71"
    ax2.set_ylabel("累積寄与率 (Cumulative Proportion)", color=color, fontsize=12)
    bar1 = ax2.bar(pc_indices, cumulative_variance_ratio, alpha=0.3, color=color, width=0.4, label="累積寄与率")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.1)
    
    # 凡例の追加
    lines = line1 + [bar1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")
    
    plt.title("主成分分析 スクリープロット & 累積寄与率", fontsize=14, pad=15)
    ax1.set_xticks(pc_indices)
    ax1.grid(True, alpha=0.3)
    
    scree_path = OUT_DIR / "pca_scree_plot.png"
    plt.tight_layout()
    fig.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"✅ スクリープロットを保存しました: {scree_path}")
    
    # ── 5. 主成分負荷量のヒートマップ作成・保存 ──
    fig, ax = plt.subplots(figsize=(8, 8))
    # 上位3主成分に絞ってヒートマップ描画
    sns.heatmap(
        loadings_df.iloc[:, :3],
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "主成分負荷量 (相関係数)"},
        linewidths=0.5,
        ax=ax
    )
    plt.title("主成分負荷量ヒートマップ (上位3成分)", fontsize=13, pad=15)
    plt.tight_layout()
    heatmap_path = OUT_DIR / "pca_loadings_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"✅ 負荷量ヒートマップを保存しました: {heatmap_path}")

if __name__ == "__main__":
    run_pca()
