#!/usr/bin/env python3
"""
アンケート質問項目（8項目版）の主成分分析と可視化スクリプト
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

def run_pca_8vars():
    print("📐 8項目版主成分分析 (PCA) を開始します...")
    df = pd.read_csv(MERGED_CSV)
    
    items = [
        "面白かった",
        "話に引き込まれた",
        "テンポが良かった",
        "新しい情報を得られた",
        "有益な内容だった",
        "誰かに共有したいと思った",
        "気軽に視聴できた",
        "気分転換になった"
    ]
    
    X = df[items].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 負荷量の計算
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        index=items,
        columns=[f"PC{i}" for i in range(1, len(items) + 1)]
    )
    
    # CSVに保存
    loadings_df.to_csv(OUT_DIR / "pca_loadings_8vars.csv", encoding="utf-8-sig")
    
    # スクリープロットの作成・保存
    fig, ax1 = plt.subplots(figsize=(8, 5))
    pc_indices = np.arange(1, len(items) + 1)
    
    # 固有値プロット
    color = "#2C3E50"
    ax1.set_xlabel("主成分番号", fontsize=12)
    ax1.set_ylabel("固有値 (Eigenvalue)", color=color, fontsize=12)
    line1 = ax1.plot(pc_indices, eigenvalues, marker="o", color=color, linewidth=2, label="固有値")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axhline(1.0, color="#E74C3C", linestyle="--", alpha=0.7, label="カイザー基準 (固有値=1.0)")
    
    # 累積寄与率プロット
    ax2 = ax1.twinx()
    color = "#2ECC71"
    ax2.set_ylabel("累積寄与率 (Cumulative Proportion)", color=color, fontsize=12)
    bar1 = ax2.bar(pc_indices, cumulative_variance_ratio, alpha=0.3, color=color, width=0.4, label="累積寄与率")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.1)
    
    lines = line1 + [bar1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")
    
    plt.title("主成分分析 スクリープロット & 累積寄与率 (8項目版)", fontsize=14, pad=15)
    ax1.set_xticks(pc_indices)
    ax1.grid(True, alpha=0.3)
    
    scree_path = OUT_DIR / "pca_scree_plot_8vars.png"
    plt.tight_layout()
    fig.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"✅ スクリープロット (8項目) を保存しました: {scree_path}")
    
    # 負荷量のヒートマップ作成・保存
    fig, ax = plt.subplots(figsize=(8, 8))
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
    plt.title("主成分負荷量ヒートマップ (8項目版, 上位3成分)", fontsize=13, pad=15)
    plt.tight_layout()
    heatmap_path = OUT_DIR / "pca_loadings_heatmap_8vars.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"✅ 負荷量ヒートマップ (8項目) を保存しました: {heatmap_path}")

if __name__ == "__main__":
    run_pca_8vars()
