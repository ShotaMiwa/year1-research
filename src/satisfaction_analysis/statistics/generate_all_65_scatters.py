import os
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

# 1. データ読み込み
df = pd.read_csv(DATA_PATH)

target_map = {
    "Target_Satisfaction": "総合満足度",
    "Factor_Entertainment": "娯楽性・没頭感",
    "Factor_Information": "情報性・学習価値",
    "Factor_Relaxation": "リラックス・気軽さ",
    "Factor_SocialShare": "社会的共有性"
}
target_cols = list(target_map.keys())

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

# 2. フォント設定
plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'TakaoPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 3. 65個（13行 × 5列）のフル散布図マトリクス作成
fig, axes = plt.subplots(len(feat_cols), len(target_cols), figsize=(20, 42))

# 乖離（非線形や外れ値の疑い）チェック用リスト
discrepancy_list = []

for r, f_col in enumerate(feat_cols):
    f_name = feat_map[f_col]
    for c, t_col in enumerate(target_cols):
        t_name = target_map[t_col]
        ax = axes[r, c]
        
        x = df[f_col].values
        y = df[t_col].values
        
        # Pearson & Spearman
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        
        # |Pearson - Spearman| が大きい（0.15以上）場合は非線形・外れ値の可能性
        diff = abs(pr - sr)
        if diff >= 0.15 or (abs(pr) < 0.2 and abs(sr) > 0.35) or (abs(pr) > 0.35 and abs(sr) < 0.2):
            discrepancy_list.append({
                "feature": f_name,
                "feature_key": f_col,
                "target": t_name,
                "target_key": t_col,
                "pearson_r": round(pr, 3),
                "spearman_rho": round(sr, 3),
                "diff": round(diff, 3),
                "note": "非線形性または外れ値影響の疑いあり"
            })
            bg_color = "#fff3cd" # ハイライト（黄色）
        else:
            bg_color = "#ffffff"
            
        ax.set_facecolor(bg_color)
        
        # 散布図 + 95%信頼区間バンド
        sns.regplot(data=df, x=f_col, y=t_col, ax=ax, ci=95,
                    scatter_kws={"color": "#1f77b4", "alpha": 0.7, "s": 35},
                    line_kws={"color": "#d62728", "linewidth": 1.5})
        
        ax.set_title(f"{f_name}\n× {t_name}\n(r={pr:+.2f}, ρ={sr:+.2f})", fontsize=8.5, fontweight='bold')
        ax.set_xlabel(f_col, fontsize=7.5)
        ax.set_ylabel(t_col, fontsize=7.5)
        ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
full_scatter_path = os.path.join(FIG_DIR, "exp2_all_65_scatter_matrix.png")
plt.savefig(full_scatter_path, dpi=200)
plt.close()
print(f"Saved full 65-scatter matrix to {full_scatter_path}")

# 4. 外れ値・非線形の疑いがある組み合わせをCSV出力
disc_df = pd.DataFrame(discrepancy_list).sort_values("diff", ascending=False)
disc_csv_path = os.path.join(TAB_DIR, "exp2_non_linear_discrepancies.csv")
disc_df.to_csv(disc_csv_path, index=False, encoding="utf-8-sig")
print(f"Saved discrepancy analysis to {disc_csv_path}")

# 5. スクリプト保存
shutil.copyfile(__file__, os.path.join(SRC_DIR, "generate_all_65_scatters.py"))
print("Saved script to src/generate_all_65_scatters.py")
