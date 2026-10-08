#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実測特徴量で相関分析を再実行し、all_analysis_results_master.md と
correlation_analysis_presentation_draft.md を更新する
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path("/home/shota/work/year1")
MATRIX_CSV = PROJECT_ROOT / "data/processed/feature_matrix.csv"
TABLE_DIR  = PROJECT_ROOT / "docs/paper/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MATRIX_CSV)
print("Loaded feature_matrix:", df.shape)

FEATURES = [
    ('comment_rate',               'コメント密度(件/秒)'),
    ('grass_ratio',                '草・笑い率'),
    ('question_ratio',             '質問率'),
    ('exclamation_ratio',          '感嘆符率(!/?)'),
    ('sentiment_polarity',         'チャット感情極性'),
    ('subtitle_sentiment_polarity','字幕テキスト感情極性'),
    ('speech_rate',                '発話速度(字幕数/秒)'),
    ('pause_ratio',                '無音(ポーズ)割合'),
    ('sub_comment_similarity',     '字幕-コメント類似度'),
    ('comment_semantic_variance',  'コメント意味多様性'),
    ('hm_mean',                    '平均ヒートマップ値'),
    ('hm_max',                     '最大ヒートマップ値'),
    ('hm_volatility',              'ヒートマップ変動性'),
]

TARGETS = [
    ('Target_Satisfaction',   '総合満足度'),
    ('Factor_Entertainment',  '娯楽性・没頭感'),
    ('Factor_Relaxation',     'リラックス・気軽さ'),
    ('Factor_Information',    '情報性・学習価値'),
    ('Factor_SocialShare',    '社会的共有性'),
]

def bootstrap_ci(x, y, n_boot=2000, ci=0.95):
    """Bootstrap 95% CI for Pearson r"""
    n = len(x)
    rs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if xi.std() < 1e-9 or yi.std() < 1e-9:
            continue
        rs.append(np.corrcoef(xi, yi)[0,1])
    rs = np.array(rs)
    lo = np.percentile(rs, (1-ci)/2*100)
    hi = np.percentile(rs, (1+ci)/2*100)
    return lo, hi

def bootstrap_ci_spearman(x, y, n_boot=2000, ci=0.95):
    """Bootstrap 95% CI for Spearman rho"""
    n = len(x)
    rs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        r, _ = stats.spearmanr(xi, yi)
        if not np.isnan(r):
            rs.append(r)
    rs = np.array(rs)
    lo = np.percentile(rs, (1-ci)/2*100)
    hi = np.percentile(rs, (1+ci)/2*100)
    return lo, hi

# 全ペアの相関を計算
all_results = []
for (f_key, f_label) in FEATURES:
    for (t_key, t_label) in TARGETS:
        x = df[f_key].values.astype(float)
        y = df[t_key].values.astype(float)
        
        # Pearson
        r, p_r = stats.pearsonr(x, y)
        ci_r_lo, ci_r_hi = bootstrap_ci(x, y)
        
        # Spearman
        rho, p_rho = stats.spearmanr(x, y)
        ci_rho_lo, ci_rho_hi = bootstrap_ci_spearman(x, y)
        
        all_results.append({
            'feature_key': f_key,
            'feature_label': f_label,
            'target_key': t_key,
            'target_label': t_label,
            'pearson_r': round(r, 4),
            'pearson_p': round(p_r, 4),
            'pearson_ci_lo': round(ci_r_lo, 3),
            'pearson_ci_hi': round(ci_r_hi, 3),
            'spearman_rho': round(rho, 4),
            'spearman_p': round(p_rho, 4),
            'spearman_ci_lo': round(ci_rho_lo, 3),
            'spearman_ci_hi': round(ci_rho_hi, 3),
            'delta_abs': round(abs(r - rho), 4),
        })

df_results = pd.DataFrame(all_results)
df_results.to_csv(TABLE_DIR / "exp2_bivariate_correlations_ci_real.csv", index=False, encoding='utf-8-sig')

# ============================================================
# 因子別サマリーを出力
# ============================================================
print("\n" + "="*80)
print("REAL DATA CORRELATION RESULTS")
print("="*80)

for t_key, t_label in TARGETS:
    sub = df_results[df_results['target_key'] == t_key].sort_values('pearson_r', key=abs, ascending=False)
    print(f"\n### 【{t_label}】との相関（|r| 降順）")
    print(f"{'特徴量':<24} {'Pearson r':>10} {'95%CI':>20} {'p(r)':>7} {'Spearman ρ':>11} {'p(ρ)':>7}")
    print("-"*85)
    for _, row in sub.iterrows():
        sig_r   = "*" if row['pearson_p'] < 0.05 else ""
        sig_rho = "*" if row['spearman_p'] < 0.05 else ""
        print(f"{row['feature_label']:<24} "
              f"{row['pearson_r']:>+9.3f}{sig_r:<1} "
              f"[{row['pearson_ci_lo']:>+6.3f},{row['pearson_ci_hi']:>+6.3f}] "
              f"{row['pearson_p']:>7.4f} "
              f"{row['spearman_rho']:>+10.3f}{sig_rho:<1} "
              f"{row['spearman_p']:>7.4f}")

# 非線形乖離スクリーニング
print("\n\n### Pearson–Spearman 乖離 TOP（非線形/外れ値の疑い）")
df_nonlinear = df_results.nlargest(10, 'delta_abs')[
    ['feature_label','target_label','pearson_r','spearman_rho','delta_abs']
]
print(df_nonlinear.to_string(index=False))

# 有意なペアのみ
df_sig = df_results[(df_results['pearson_p'] < 0.05) | (df_results['spearman_p'] < 0.05)].sort_values('pearson_r', key=abs, ascending=False)
print(f"\n\n### 有意ペア（p<0.05 いずれか）: {len(df_sig)} 件")
print(df_sig[['feature_label','target_label','pearson_r','pearson_p','spearman_rho','spearman_p']].to_string(index=False))
