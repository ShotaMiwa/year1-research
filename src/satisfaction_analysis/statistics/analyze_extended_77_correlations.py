# -*- coding: utf-8 -*-
"""
analyze_extended_77_correlations.py

確定実測データセット（N=77: extended_feature_matrix_77.csv）における
全11特徴量（10音響言語チャット特徴量 + segment_length_sec）と「いいね率（like_rate）」との
単変量相関分析（Pearson相関係数 / Spearman順位相関係数、p値、95%信頼区間）を算出し、
散布図グリッド画像（docs/paper/figures/exp_extended_77_scatter_grid.png）および
相関結果テーブル（docs/paper/tables/exp_extended_77_correlations.csv）を出力する。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("/home/shota/work/year1")
DATA_PATH = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
FIG_DIR = PROJECT_ROOT / "docs/paper/figures"
TAB_DIR = PROJECT_ROOT / "docs/paper/tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

def calc_pearson_ci(r, n, alpha=0.05):
    """Fisher's z-transformation for Pearson correlation CI"""
    if np.isnan(r) or abs(r) >= 1.0 or n <= 3:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return lo, hi

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    feature_cols = [
        ("question_ratio", "疑問文比率 (question_ratio)"),
        ("comment_semantic_variance", "コメント意味多様性 (comment_semantic_variance)"),
        ("segment_length_sec", "動画尺(秒) (segment_length_sec)"),
        ("comment_rate", "コメント密度 (comment_rate)"),
        ("grass_ratio", "草・笑い率 (grass_ratio)"),
        ("exclamation_ratio", "感嘆符比率 (exclamation_ratio)"),
        ("sentiment_polarity", "コメント感情極性 (sentiment_polarity)"),
        ("subtitle_sentiment_polarity", "字幕感情極性 (subtitle_sentiment_polarity)"),
        ("speech_rate", "発話速度 (speech_rate)"),
        ("pause_ratio", "ポーズ率 (pause_ratio)"),
        ("sub_comment_similarity", "字幕-コメント類似度 (sub_comment_similarity)"),
    ]

    target_col = "like_rate"
    n = len(df)
    results = []

    for col, label in feature_cols:
        x = df[col].values
        y = df[target_col].values
        
        # Pearson
        pr, p_pval = stats.pearsonr(x, y)
        p_ci_lo, p_ci_hi = calc_pearson_ci(pr, n)
        
        # Spearman
        sr, s_pval = stats.spearmanr(x, y)
        
        results.append({
            "feature": col,
            "feature_label": label,
            "pearson_r": round(pr, 4),
            "pearson_p": round(p_pval, 4),
            "pearson_ci_95": f"[{p_ci_lo:.3f}, {p_ci_hi:.3f}]",
            "spearman_rho": round(sr, 4),
            "spearman_p": round(s_pval, 4),
            "significant_p05": "Yes" if (p_pval < 0.05 or s_pval < 0.05) else "No",
            "significant_p10": "Yes" if (p_pval < 0.10 or s_pval < 0.10) else "No"
        })

    df_res = pd.DataFrame(results)
    out_table_path = TAB_DIR / "exp_extended_77_correlations.csv"
    df_res.to_csv(out_table_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved correlation table to: {out_table_path}")
    print(df_res[["feature", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]].to_string())

    # 散布図グリッドの描画 (3行4列)
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()

    for i, (col, label) in enumerate(feature_cols):
        ax = axes[i]
        x = df[col]
        y = df[target_col]
        
        row_stat = df_res[df_res["feature"] == col].iloc[0]
        pr = row_stat["pearson_r"]
        pp = row_stat["pearson_p"]
        sr = row_stat["spearman_rho"]
        sp = row_stat["spearman_p"]

        # 回帰直線付き散布図
        sns.regplot(
            data=df, x=col, y=target_col, ax=ax,
            scatter_kws={"alpha": 0.65, "color": "#1a73e8", "s": 35},
            line_kws={"color": "#d93025" if (pp < 0.05 or sp < 0.05) else "#5f6368", "linewidth": 2}
        )
        
        sig_mark = "★" if (pp < 0.05 or sp < 0.05) else ("†" if (pp < 0.10 or sp < 0.10) else "")
        ax.set_title(f"{label} {sig_mark}\nPearson r={pr:.3f} (p={pp:.3f}) | Spearman ρ={sr:.3f} (p={sp:.3f})", fontsize=11, fontweight="bold" if sig_mark else "normal")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("いいね率 (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    # 12個目の空きサブプロットをサマリー表示用に活用
    ax_last = axes[-1]
    ax_last.axis('off')
    summary_text = (
        "【相関分析サマリー (N=77)】\n"
        "------------------------------------\n"
        "★ p < 0.05 有意\n"
        "† p < 0.10 傾向あり\n\n"
        "目的変数: YouTubeいいね率 (Like Rate, %)\n"
        "対象: 完全実測Shorts動画 77本\n"
        "(生チャット + 字幕BERT極性)\n\n"
        "アンケート知見との対比:\n"
        "・疑問文比率: 負の重み\n"
        "・意味多様性: 正の重み\n"
        "・動画尺: 正の重み\n"
        "------------------------------------"
    )
    ax_last.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dadce0'))

    plt.suptitle("完全実測データセット(N=77)における各特徴量とYouTubeいいね率(Like Rate)の単変量相関・散布図", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_fig_path = FIG_DIR / "exp_extended_77_scatter_grid.png"
    plt.savefig(out_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved scatter grid figure to: {out_fig_path}")

if __name__ == "__main__":
    main()

