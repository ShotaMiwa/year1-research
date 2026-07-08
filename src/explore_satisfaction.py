#!/usr/bin/env python3
"""
満足度の分布・特徴量との関係を探索的に可視化するスクリプト
Step 1: 満足度スコアの分布（ヒストグラム + KDE）
Step 2: 満足度 × 各定量変数の散布図
Step 3: 満足度のみによる階層的クラスタリング（デンドログラム）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import seaborn as sns
from pathlib import Path

plt.rcParams["axes.unicode_minus"] = False

# ── パス設定（相対パスでプロジェクトルートを解決） ──────────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "analysis_questionnaire"
OUT_DIR = DATA_DIR / "explore_satisfaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV = DATA_DIR / "merged_survey_heatmap.csv"
GROUP_CSV  = DATA_DIR / "group_quantitative_comparison.csv"

# ── データ読み込み ──────────────────────────────────────────
df = pd.read_csv(MERGED_CSV)
df_group = pd.read_csv(GROUP_CSV)

SAT_COL = "この動画に満足した"
sat = df[SAT_COL].values
segs = df["セグメント"].values

print(f"=== データ概要 ===")
print(f"セグメント数: {len(df)}")
print(f"満足度 min={sat.min():.2f}, max={sat.max():.2f}, "
      f"mean={sat.mean():.2f}, median={np.median(sat):.2f}, std={sat.std():.2f}")
print()

# ════════════════════════════════════════════════════════════
# Step 1: 満足度の分布（ヒストグラム + KDE + 統計量）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

# ビンの幅を0.1刻みに固定（境界値3.5や3.8がビンの端に揃うように調整）
custom_bins = np.arange(2.75, 4.35, 0.1)

# ヒストグラム
n, bins, patches = ax.hist(sat, bins=custom_bins, color="#4A90D9", alpha=0.55,
                            edgecolor="white", linewidth=1.2, label="頻度")

# KDE（密度推定カーブ）
sat_lin = np.linspace(sat.min() - 0.1, sat.max() + 0.1, 300)
kde = stats.gaussian_kde(sat, bw_method=0.5)
ax2 = ax.twinx()
ax2.plot(sat_lin, kde(sat_lin), color="#E74C3C", linewidth=2.5, label="KDE（密度）")
ax2.set_ylabel("確率密度", fontsize=11, color="#E74C3C")
ax2.tick_params(axis="y", colors="#E74C3C")

# グループ境界の縦線（低満足: 3.5未満, 高満足: 3.8以上）
ax.axvline(3.5, color="#E74C3C", linestyle="-", linewidth=2, label="低満足境界 (< 3.5)")
ax.axvline(3.8, color="#2ECC71", linestyle="-", linewidth=2, label="高満足境界 (>= 3.8)")

# グループの背景色塗り分け
ax.axvspan(2.7, 3.5, color="#E74C3C", alpha=0.08)  # 低満足エリア
ax.axvspan(3.5, 3.8, color="#95A5A6", alpha=0.08)  # 中間エリア
ax.axvspan(3.8, 4.3, color="#2ECC71", alpha=0.08)  # 高満足エリア

# 平均値・中央値
med = np.median(sat)
mean = sat.mean()
ax.axvline(med,  color="#2D3E50", linestyle="--", linewidth=1.5, label=f"中央値={med:.2f}")
ax.axvline(mean, color="#F39C12", linestyle=":",  linewidth=1.5, label=f"平均値={mean:.2f}")

# 全セグメントのスコアをrug plotとして表示
for s in sat:
    ax.axvline(s, color="#4A90D9", alpha=0.25, linewidth=0.8)

ax.set_xlim(2.7, 4.3)
ax.set_xticks(np.arange(2.7, 4.4, 0.1))
ax.set_yticks(np.arange(0, 6, 1))  # 頻度を整数に固定
ax.set_xlabel("総合満足度（平均スコア）", fontsize=13)
ax.set_ylabel("頻度（セグメント数）", fontsize=11)
ax.set_title("満足度スコアの分布と分析グループ定義 (N=23)", fontsize=15, pad=15)

# 凡例をまとめる
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

ax.grid(True, alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "1_satisfaction_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Step1: 満足度分布 → {out}")

# ════════════════════════════════════════════════════════════
# Step 2: 満足度 × 各定量変数の散布図（一覧）
# ════════════════════════════════════════════════════════════
quantitative_vars = [
    ("平均ヒートマップ値", "HM平均値（視聴維持率）"),
    ("最大ヒートマップ値", "HM最大値"),
    ("面白かった",       "面白かった（娯楽性）"),
    ("話に引き込まれた", "話に引き込まれた（娯楽性）"),
    ("テンポが良かった", "テンポが良かった（娯楽性）"),
    ("新しい情報を得られた", "新しい情報を得られた（情報性）"),
    ("有益な内容だった", "有益な内容だった（情報性）"),
    ("誰かに共有したいと思った", "誰かに共有したいと思った（社会性）"),
    ("気軽に視聴できた", "気軽に視聴できた（リラックス）"),
    ("気分転換になった", "気分転換になった（リラックス）"),
]

nrows, ncols = 4, 3  # 10変数 → 4行×3列
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 18))
axes_flat = axes.flatten()

for idx, (col, label) in enumerate(quantitative_vars):
    ax = axes_flat[idx]
    x = df[col].values

    # 散布図
    ax.scatter(x, sat, s=70, color="#4A90D9", alpha=0.75, edgecolors="white",
               linewidth=1, zorder=5)

    # セグメント番号ラベル
    for i, (xi, yi) in enumerate(zip(x, sat)):
        ax.annotate(str(int(segs[i])), (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="#2C3E50")

    # ピアソン相関係数
    r, p = stats.pearsonr(x, sat)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.set_title(f"{label}\nr={r:+.3f}{sig} (p={p:.3f})", fontsize=9, pad=6)

    # 回帰直線
    m, b = np.polyfit(x, sat, 1)
    x_lin = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_lin, m * x_lin + b, color="#E74C3C", linewidth=1.5, alpha=0.7)

    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("総合満足度", fontsize=8)
    ax.grid(True, alpha=0.25)

# 余った軸を非表示
for idx in range(len(quantitative_vars), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("満足度 × 各変数の散布図（回帰直線・相関係数付き）", fontsize=16, y=1.01)
plt.tight_layout()
out = OUT_DIR / "2_scatter_satisfaction_vs_all.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Step2: 散布図一覧 → {out}")

# ════════════════════════════════════════════════════════════
# Step 3: 満足度スコアのみに基づくデンドログラム
# ════════════════════════════════════════════════════════════
sat_2d = sat.reshape(-1, 1)
Z = linkage(sat_2d, method="ward")

fig, ax = plt.subplots(figsize=(14, 7))
dendrogram(
    Z,
    labels=[f"Seg{int(s)}\n({v:.2f})" for s, v in zip(segs, sat)],
    color_threshold=None,
    leaf_rotation=0,
    ax=ax
)
ax.set_title("満足度スコアのみによる階層的クラスタリング（Ward法）\n"
             "→ データが自然に何グループに分かれるか？", fontsize=14, pad=15)
ax.set_xlabel("セグメント番号（満足度スコア）", fontsize=12)
ax.set_ylabel("結合距離（Ward法 / 分散）", fontsize=12)

# 水平線で「どこで切るとN分割か」を視覚化
heights = sorted([c[2] for c in Z], reverse=True)
colors_cut = ["#2ECC71", "#F39C12", "#E74C3C"]
n_cuts = [2, 3, 4]  # 2分割・3分割・4分割の境界
for ncut, color in zip(n_cuts, colors_cut):
    threshold = (heights[ncut-2] + heights[ncut-1]) / 2
    ax.axvline(0, visible=False)  # 警告防止
    ax.axhline(threshold, linestyle="--", color=color, linewidth=1.5, alpha=0.8,
               label=f"{ncut}分割の境界 (h≈{threshold:.3f})")

ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
out = OUT_DIR / "3_dendrogram_satisfaction_only.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Step3: デンドログラム（満足度のみ） → {out}")

# ════════════════════════════════════════════════════════════
# Step 4: 満足度の「区切りの候補」を数値で出力
# ════════════════════════════════════════════════════════════
print("\n=== 満足度の分布サマリー ===")
percentiles = [10, 25, 33, 50, 67, 75, 90]
for p in percentiles:
    print(f"  第{p:2d}パーセンタイル: {np.percentile(sat, p):.4f}")

print("\n=== 各スコアとセグメント番号（昇順） ===")
sorted_idx = np.argsort(sat)
for i in sorted_idx:
    print(f"  Seg{int(segs[i]):2d}: {sat[i]:.2f}")

print("\n=== デンドログラムの結合高さ（最後の5ステップ） ===")
for i, h in enumerate(heights[:6], 1):
    print(f"  {i}個→{i-1}個に統合する時の距離: {h:.4f}")

print(f"\n✅ すべての図を {OUT_DIR} に保存しました")
