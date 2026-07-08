#!/usr/bin/env python3
"""
古い4グループ分類の境界線や色分けをすべて取り除き、
素のデータとしての「ヒートマップ値（平均・最大） × 総合満足度」の散布図を描画する。
ラベルの重なりを手動オフセットで解消。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

plt.rcParams["axes.unicode_minus"] = False

# ── パス設定（相対パスでプロジェクトルートを解決） ──────────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "analysis_questionnaire"
OUT_DIR = DATA_DIR / "explore_satisfaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV = DATA_DIR / "merged_survey_heatmap.csv"

# ── データ読み込み ──────────────────────────────────────────
df = pd.read_csv(MERGED_CSV)

SAT_COL = "この動画に満足した"
y = df[SAT_COL].values
segs = df["セグメント"].values

# ════════════════════════════════════════════════════════════
# 散布図プロット（平均ヒートマップ vs 総合満足度）
# ════════════════════════════════════════════════════════════
x_mean = df["平均ヒートマップ値"].values

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_mean, y, color="#3498db", s=100, edgecolors="#2c3e50", alpha=0.8, linewidths=1.2, zorder=3)

# 重なり回避のための手動オフセット定義
# (x方向オフセット, y方向オフセット) のテキスト相対位置
manual_offsets_mean = {
    5:  (-12, 6),
    12: (6, -10),
    23: (-12, 6),
    8:  (6, -10),
    21: (6, 6),
    15: (-15, -12),
}

for i, txt in enumerate(segs):
    seg_num = int(txt)
    offset = manual_offsets_mean.get(seg_num, (6, 6))  # デフォルトは右上
    ax.annotate(
        str(seg_num), 
        (x_mean[i], y[i]), 
        textcoords="offset points", 
        xytext=offset, 
        fontsize=10, 
        color="#2c3e50", 
        fontweight="bold"
    )

ax.set_xlabel("平均ヒートマップ値", fontsize=12)
ax.set_ylabel("総合満足度（平均スコア）", fontsize=12)
ax.set_title("セグメント別：平均ヒートマップ値 × 総合満足度 散布図", fontsize=13, pad=15)
ax.grid(True, alpha=0.3)

out_mean = OUT_DIR / "raw_scatter_mean_heatmap_satisfaction.png"
fig.savefig(out_mean, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ 平均ヒートマップ散布図を保存: {out_mean}")

# ════════════════════════════════════════════════════════════
# 散布図プロット（最大ヒートマップ vs 総合満足度）
# ════════════════════════════════════════════════════════════
x_max = df["最大ヒートマップ値"].values

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_max, y, color="#e74c3c", s=100, edgecolors="#2c3e50", alpha=0.8, linewidths=1.2, zorder=3)

# 最大ヒートマップ用の手動オフセット
manual_offsets_max = {
    11: (-12, 6),
    17: (6, -10),
    15: (6, 6),
    21: (-15, -12),
}

for i, txt in enumerate(segs):
    seg_num = int(txt)
    offset = manual_offsets_max.get(seg_num, (6, 6))
    ax.annotate(
        str(seg_num), 
        (x_max[i], y[i]), 
        textcoords="offset points", 
        xytext=offset, 
        fontsize=10, 
        color="#2c3e50", 
        fontweight="bold"
    )

ax.set_xlabel("最大ヒートマップ値", fontsize=12)
ax.set_ylabel("総合満足度（平均スコア）", fontsize=12)
ax.set_title("セグメント別：最大ヒートマップ値 × 総合満足度 散布図", fontsize=13, pad=15)
ax.grid(True, alpha=0.3)

out_max = OUT_DIR / "raw_scatter_max_heatmap_satisfaction.png"
fig.savefig(out_max, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ 最大ヒートマップ散布図を保存: {out_max}")
