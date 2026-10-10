import os
import shutil
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/home/shota/work/year1"
DATA_PATH = os.path.join(BASE_DIR, "data/processed/feature_matrix.csv")
PAPER_DIR = os.path.join(BASE_DIR, "docs/paper")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TAB_DIR = os.path.join(PAPER_DIR, "tables")
SRC_DIR = os.path.join(BASE_DIR, "src")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# 1. 過去のLOO-CV図表を figures/ にコピー
old_fig = os.path.join(BASE_DIR, "data/processed/model_comparison_loocv_barplots.png")
if os.path.exists(old_fig):
    shutil.copyfile(old_fig, os.path.join(FIG_DIR, "exp1_loocv_barplots.png"))

# 2. データ読み込み
df = pd.read_csv(DATA_PATH)
print("Data loaded. Shape:", df.shape)

# 正しい目的変数（5因子）
target_map = {
    "Target_Satisfaction": "総合満足度",
    "Factor_Entertainment": "娯楽性・没頭感",
    "Factor_Information": "情報性・学習価値",
    "Factor_Relaxation": "リラックス・気軽さ",
    "Factor_SocialShare": "社会的共有性"
}
target_cols = list(target_map.keys())

# 説明変数（13の定量的特徴量）
feat_map = {
    "comment_rate": "コメント密度(件/秒)",
    "grass_ratio": "草・笑い率",
    "question_ratio": "質問率",
    "exclamation_ratio": "感嘆符率(!/?)",
    "sentiment_polarity": "チャット感情極性",
    "sub_comment_similarity": "字幕-コメント類似度",
    "comment_semantic_variance": "コメント意味多様性",
    "high_satisfaction_sim": "高満足表現類似度",
    "speech_rate": "発話速度(文字/秒)",
    "pause_ratio": "無音(ポーズ)割合",
    "hm_mean": "平均ヒートマップ値",
    "hm_max": "最大ヒートマップ値",
    "hm_volatility": "ヒートマップ変動性"
}
feat_cols = list(feat_map.keys())

print(f"Features ({len(feat_cols)}):", feat_cols)
print(f"Targets ({len(target_cols)}):", target_cols)

# 3. 相関係数・95%信頼区間の計算
def pearson_ci(r, n, alpha=0.05):
    if np.isnan(r) or abs(r) >= 1.0 or n <= 3:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return lo, hi

def spearman_bootstrap_ci(x, y, n_boot=2000, alpha=0.05):
    np.random.seed(42)
    rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rhos.append(r)
    if len(rhos) == 0:
        return np.nan, np.nan
    lo = np.percentile(rhos, 100 * (alpha / 2))
    hi = np.percentile(rhos, 100 * (1 - alpha / 2))
    return lo, hi

corr_records = []
n_samples = len(df)

for t_col in target_cols:
    t_name = target_map[t_col]
    for f_col in feat_cols:
        f_name = feat_map[f_col]
        x = df[f_col].values
        y = df[t_col].values
        
        # Pearson
        pr, pp = stats.pearsonr(x, y)
        pr_lo, pr_hi = pearson_ci(pr, n_samples)
        
        # Spearman
        sr, sp = stats.spearmanr(x, y)
        sr_lo, sr_hi = spearman_bootstrap_ci(x, y)
        
        corr_records.append({
            "target_key": t_col,
            "target_name": t_name,
            "feature_key": f_col,
            "feature_name": f_name,
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 4),
            "pearson_ci_95": f"[{pr_lo:.3f}, {pr_hi:.3f}]",
            "spearman_rho": round(sr, 4),
            "spearman_p": round(sp, 4),
            "spearman_ci_95": f"[{sr_lo:.3f}, {sr_hi:.3f}]",
            "abs_pearson_r": abs(pr)
        })

corr_df = pd.DataFrame(corr_records)
corr_csv_path = os.path.join(TAB_DIR, "exp2_bivariate_correlations_ci.csv")
corr_df.to_csv(corr_csv_path, index=False, encoding="utf-8-sig")
print(f"Saved correlation table to {corr_csv_path}")

# 4. 可視化プロット生成 (95% CI付き散布図グリッド & 相関ヒートマップ)
plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'TakaoPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 各因子に対する上位2特徴量 (計10プロット) を 5行2列 で描画
fig, axes = plt.subplots(len(target_cols), 2, figsize=(14, 22))

for i, t_col in enumerate(target_cols):
    t_name = target_map[t_col]
    sub_df = corr_df[corr_df["target_key"] == t_col].sort_values("abs_pearson_r", ascending=False)
    top2_rows = sub_df.head(2).to_dict('records')
    
    for j, row in enumerate(top2_rows):
        ax = axes[i, j]
        f_col = row["feature_key"]
        f_name = row["feature_name"]
        pr = row["pearson_r"]
        sr = row["spearman_rho"]
        pr_ci = row["pearson_ci_95"]
        
        sns.regplot(data=df, x=f_col, y=t_col, ax=ax, ci=95,
                    scatter_kws={"color": "#1f77b4", "alpha": 0.8, "s": 50},
                    line_kws={"color": "#ff7f0e", "linewidth": 2})
        
        ax.set_title(f"{t_name} vs {f_name}\n(Pearson r = {pr:+.3f} {pr_ci}, Spearman ρ = {sr:+.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"{f_name} ({f_col})", fontsize=10)
        ax.set_ylabel(f"{t_name}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
scatter_path = os.path.join(FIG_DIR, "exp2_scatter_grid_ci.png")
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved scatter plot grid to {scatter_path}")

# 相関ヒートマップ (Pearson vs Spearman)
p_matrix = corr_df.pivot(index="feature_name", columns="target_name", values="pearson_r")
s_matrix = corr_df.pivot(index="feature_name", columns="target_name", values="spearman_rho")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
sns.heatmap(p_matrix, annot=True, fmt="+.2f", cmap="coolwarm", center=0, ax=ax1, cbar_kws={'label': 'Pearson r'}, vmin=-1, vmax=1)
ax1.set_title("Pearson r (線形相関)", fontsize=13, fontweight='bold')

sns.heatmap(s_matrix, annot=True, fmt="+.2f", cmap="coolwarm", center=0, ax=ax2, cbar_kws={'label': 'Spearman ρ'}, vmin=-1, vmax=1)
ax2.set_title("Spearman ρ (順位相関 / 単調性)", fontsize=13, fontweight='bold')

plt.tight_layout()
heatmap_path = os.path.join(FIG_DIR, "exp2_correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Saved correlation heatmap to {heatmap_path}")

# 5. 再現スクリプトを src/ に保存
src_script_path = os.path.join(SRC_DIR, "analyze_feature_factor_correlations_ci.py")
shutil.copyfile(__file__, src_script_path)
print(f"Saved script to {src_script_path}")
