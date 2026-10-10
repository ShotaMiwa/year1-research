import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path('/home/shota/work/year1')
SURVEY_CSV = PROJECT_ROOT / 'ライブ配信動画に関する予備調査アンケート/ライブ配信動画に関する予備調査アンケート.csv'
FEATURE_CSV = PROJECT_ROOT / 'data/processed/feature_matrix.csv'
TABLE_DIR = PROJECT_ROOT / 'docs/paper/tables'
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = TABLE_DIR / 'exp2_lmm_results.csv'

LIKERT_MAP = {
    '非常にそう思う': 5,
    'ややそう思う': 4,
    'どちらでもない': 3,
    'あまりそう思わない': 2,
    '全くそう思わない': 1,
}

ITEM_NAMES = [
    '面白かった', '話に引き込まれた', 'テンポが良かった',
    '新しい情報を得られた', '有益な内容だった', '誰かに共有したいと思った',
    '気軽に視聴できた', '気分転換になった', 'この動画に満足した'
]

FEATURES = [
    ('comment_rate',               'コメント密度(件/秒)'),
    ('grass_ratio',                '草・笑い率'),
    ('question_ratio',             '質問率'),
    ('exclamation_ratio',          '感嘆符率'),
    ('sentiment_polarity',         'チャット感情極性'),
    ('subtitle_sentiment_polarity','字幕テキスト感情極性'),
    ('speech_rate',                '発話速度(字幕数/秒)'),
    ('pause_ratio',                '無音割合'),
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

def load_long_survey_data():
    df_raw = pd.read_csv(SURVEY_CSV)
    records = []
    for sub_idx, row in df_raw.iterrows():
        sub_id = f'sub_{sub_idx+1:02d}'
        for seg_num in range(1, 24):
            seg_label = f'セグメント{seg_num}'
            scores = {}
            for item in ITEM_NAMES:
                match_cols = [c for c in df_raw.columns if seg_label in c and item in c]
                if match_cols:
                    val = LIKERT_MAP.get(row[match_cols[0]], np.nan)
                    scores[item] = val
            ent = np.mean([scores['面白かった'], scores['話に引き込まれた'], scores['テンポが良かった']])
            info = np.mean([scores['新しい情報を得られた'], scores['有益な内容だった']])
            rel = np.mean([scores['気軽に視聴できた'], scores['気分転換になった']])
            soc = scores['誰かに共有したいと思った']
            sat = scores['この動画に満足した']
            records.append({
                'subject_id': sub_id,
                'セグメント': seg_num,
                'Factor_Entertainment': ent,
                'Factor_Information': info,
                'Factor_Relaxation': rel,
                'Factor_SocialShare': soc,
                'Target_Satisfaction': sat,
            })
    return pd.DataFrame(records)

print('Reading data...')
df_survey_long = load_long_survey_data()
df_feats = pd.read_csv(FEATURE_CSV)
feat_cols = ['セグメント'] + [f[0] for f in FEATURES]
df_merged = pd.merge(df_survey_long, df_feats[feat_cols], on='セグメント', how='inner')

for f_key, _ in FEATURES:
    df_merged[f'{f_key}_std'] = (df_merged[f_key] - df_merged[f_key].mean()) / df_merged[f_key].std()
for t_key, _ in TARGETS:
    df_merged[f'{t_key}_std'] = (df_merged[t_key] - df_merged[t_key].mean()) / df_merged[t_key].std()

CORR_CSV = TABLE_DIR / 'exp2_bivariate_correlations_ci_real.csv'
df_corr = pd.read_csv(CORR_CSV) if CORR_CSV.exists() else None

results = []
print('Running LMM fits for 65 pairs...')
for f_key, f_label in FEATURES:
    for t_key, t_label in TARGETS:
        formula = f'{t_key}_std ~ {f_key}_std'
        model = smf.mixedlm(formula, df_merged, groups=df_merged['subject_id'])
        try:
            fit = model.fit(reml=True, method='lbfgs')
            beta = fit.params[f'{f_key}_std']
            se = fit.bse[f'{f_key}_std']
            z_val = fit.tvalues[f'{f_key}_std']
            p_val = fit.pvalues[f'{f_key}_std']
            ci_lo = fit.conf_int().loc[f'{f_key}_std', 0]
            ci_hi = fit.conf_int().loc[f'{f_key}_std', 1]
            var_subject = fit.cov_re.iloc[0, 0] if hasattr(fit, 'cov_re') else np.nan
            var_resid = fit.scale
            icc = var_subject / (var_subject + var_resid) if (var_subject + var_resid) > 0 else np.nan
        except Exception as e:
            beta, se, z_val, p_val, ci_lo, ci_hi, icc = [np.nan]*7

        r_val, p_r = np.nan, np.nan
        if df_corr is not None:
            m = df_corr[(df_corr['feature_key'] == f_key) & (df_corr['target_key'] == t_key)]
            if not m.empty:
                r_val = m['pearson_r'].values[0]
                p_r = m['pearson_p'].values[0]

        results.append({
            'feature_key': f_key,
            'feature_label': f_label,
            'target_key': t_key,
            'target_label': t_label,
            'lmm_beta': round(beta, 4),
            'lmm_se': round(se, 4),
            'lmm_z': round(z_val, 4),
            'lmm_p': round(p_val, 4),
            'lmm_ci_lo': round(ci_lo, 3),
            'lmm_ci_hi': round(ci_hi, 3),
            'subject_icc': round(icc, 4),
            'pearson_r_n23': round(r_val, 4) if not np.isnan(r_val) else np.nan,
            'pearson_p_n23': round(p_r, 4) if not np.isnan(p_r) else np.nan,
        })

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print('SUCCESS_SAVED:', OUTPUT_CSV)

df_sig = df_res[df_res['lmm_p'] < 0.05].sort_values('lmm_p')
print(f'Sig pairs count: {len(df_sig)}')
print(df_sig[['feature_label', 'target_label', 'lmm_beta', 'lmm_p', 'subject_icc', 'pearson_r_n23']].to_string(index=False))