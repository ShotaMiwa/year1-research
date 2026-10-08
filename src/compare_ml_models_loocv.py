# -*- coding: utf-8 -*-
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/shota/work/year1")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.base import clone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

FEATURE_MATRIX_CSV = PROJECT_ROOT / 'data' / 'processed' / 'feature_matrix.csv'
REPORT_MD = PROJECT_ROOT / 'data' / 'processed' / 'model_comparison_loocv_report.md'
BARPLOT_PNG = PROJECT_ROOT / 'data' / 'processed' / 'model_comparison_loocv_barplots.png'

def calculate_adjusted_r2(r2, n, p):
    if n - p - 1 <= 0:
        return 0.0
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def loo_eval(pipe, X, y):
    """
    LOO-CVで1サンプルテストではR²が定義不能 (UndefinedMetricWarning)。
    全LOO予測値を収集後、全体SS_res/SS_totから LOO-R²・RMSE・MAE を算出する。
    """
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        p_clone = clone(pipe)
        p_clone.fit(X[train_idx], y[train_idx])
        y_pred_loo[test_idx] = p_clone.predict(X[test_idx])
    ss_res = np.sum((y - y_pred_loo) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    loo_r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    loo_rmse = float(np.sqrt(np.mean((y - y_pred_loo) ** 2)))
    loo_mae  = float(np.mean(np.abs(y - y_pred_loo)))
    return loo_r2, loo_rmse, loo_mae

def run_loocv_comparison():
    print('Starting LOO-CV Evaluation (Lasso vs ElasticNet vs SVR vs RandomForest)...')
    df = pd.read_csv(FEATURE_MATRIX_CSV)
    n_samples = len(df)

    target_factors = [
        'Factor_Entertainment', 'Factor_Information',
        'Factor_Relaxation', 'Factor_SocialShare', 'Target_Satisfaction'
    ]
    factor_names_ja = {
        'Factor_Entertainment': '娯楽性・没頭感',
        'Factor_Information': '情報性・学習価値',
        'Factor_Relaxation': 'リラックス・気軽さ',
        'Factor_SocialShare': '社会的共有性',
        'Target_Satisfaction': '総合満足度'
    }
    quant_features = [
        'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
        'sentiment_polarity', 'subtitle_sentiment_polarity', 'pause_ratio', 'speech_rate',
        'sub_comment_similarity', 'comment_semantic_variance',
        'hm_mean', 'hm_max', 'hm_volatility'
    ]
    p_feat = len(quant_features)
    results = {}
    model_names = ['Lasso', 'ElasticNet', 'SVR', 'RandomForest']

    for factor in target_factors:
        print(f'  Processing factor: {factor_names_ja[factor]}...')
        X = df[quant_features].values
        y = df[factor].values
        results[factor] = {}

        # 1. Lasso
        lasso_pipe = Pipeline([('scaler', StandardScaler()),
                                ('model', LassoCV(cv=5, random_state=42, max_iter=10000))])
        lasso_pipe.fit(X, y)
        r2_lasso = r2_score(y, lasso_pipe.predict(X))
        selected_l = int(np.sum(np.abs(lasso_pipe.named_steps['model'].coef_) > 1e-4))
        loo_r2, loo_rmse, loo_mae = loo_eval(lasso_pipe, X, y)
        results[factor]['Lasso'] = {
            'r2': float(r2_lasso),
            'adj_r2': float(calculate_adjusted_r2(r2_lasso, n_samples, max(1, selected_l))),
            'loo_r2': loo_r2, 'loo_rmse': loo_rmse, 'loo_mae': loo_mae,
            'params': f"alpha={lasso_pipe.named_steps['model'].alpha_:.4f}, n_features={selected_l}"
        }

        # 2. ElasticNet
        enet_pipe = Pipeline([('scaler', StandardScaler()),
                               ('model', ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,0.95,0.99],
                                                      cv=5, random_state=42, max_iter=10000))])
        enet_pipe.fit(X, y)
        r2_enet = r2_score(y, enet_pipe.predict(X))
        selected_e = int(np.sum(np.abs(enet_pipe.named_steps['model'].coef_) > 1e-4))
        loo_r2, loo_rmse, loo_mae = loo_eval(enet_pipe, X, y)
        results[factor]['ElasticNet'] = {
            'r2': float(r2_enet),
            'adj_r2': float(calculate_adjusted_r2(r2_enet, n_samples, max(1, selected_e))),
            'loo_r2': loo_r2, 'loo_rmse': loo_rmse, 'loo_mae': loo_mae,
            'params': f"alpha={enet_pipe.named_steps['model'].alpha_:.4f}, l1_ratio={enet_pipe.named_steps['model'].l1_ratio_:.2f}"
        }

        # 3. SVR
        svr_grid = GridSearchCV(
            Pipeline([('scaler', StandardScaler()), ('svr', SVR())]),
            param_grid={'svr__kernel': ['rbf', 'linear'], 'svr__C': [0.1, 1.0, 10.0],
                        'svr__epsilon': [0.01, 0.1, 0.2], 'svr__gamma': ['scale', 'auto', 0.01, 0.1]},
            cv=5, scoring='neg_mean_squared_error')
        svr_grid.fit(X, y)
        best_svr = svr_grid.best_estimator_
        r2_svr = r2_score(y, best_svr.predict(X))
        loo_r2, loo_rmse, loo_mae = loo_eval(best_svr, X, y)
        bp_svr = svr_grid.best_params_
        results[factor]['SVR'] = {
            'r2': float(r2_svr),
            'adj_r2': float(calculate_adjusted_r2(r2_svr, n_samples, p_feat)),
            'loo_r2': loo_r2, 'loo_rmse': loo_rmse, 'loo_mae': loo_mae,
            'params': f"kernel={bp_svr['svr__kernel']}, C={bp_svr['svr__C']}, eps={bp_svr['svr__epsilon']}"
        }

        # 4. Random Forest
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid={'n_estimators': [50, 100], 'max_depth': [2, 3, 4, None],
                        'min_samples_leaf': [1, 2, 3], 'max_features': ['sqrt', 1.0]},
            cv=5, scoring='neg_mean_squared_error')
        rf_grid.fit(X, y)
        best_rf = rf_grid.best_estimator_
        r2_rf = r2_score(y, best_rf.predict(X))
        loo_r2, loo_rmse, loo_mae = loo_eval(best_rf, X, y)
        bp_rf = rf_grid.best_params_
        results[factor]['RandomForest'] = {
            'r2': float(r2_rf),
            'adj_r2': float(calculate_adjusted_r2(r2_rf, n_samples, p_feat)),
            'loo_r2': loo_r2, 'loo_rmse': loo_rmse, 'loo_mae': loo_mae,
            'params': f"depth={bp_rf['max_depth']}, leaf={bp_rf['min_samples_leaf']}, feat={bp_rf['max_features']}"
        }

    # プロット
    palette = sns.color_palette('Set2', 4)
    fig, axes = plt.subplots(len(target_factors), 2, figsize=(14, 18))
    fig.suptitle('機械学習モデル比較 (LOO-CV評価)', fontsize=16, y=0.99)

    for i, factor in enumerate(target_factors):
        fname = factor_names_ja[factor]
        loo_r2_vals = [results[factor][m]['loo_r2'] for m in model_names]
        r2_vals     = [results[factor][m]['r2'] for m in model_names]
        adj_r2_vals = [results[factor][m]['adj_r2'] for m in model_names]

        ax_loo = axes[i, 0]
        bars = ax_loo.bar(model_names, loo_r2_vals, color=palette, alpha=0.85, edgecolor='black')
        ax_loo.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax_loo.set_title(f'【{fname}】 LOO-CV $R^2$ (汎化性能)', fontsize=12, fontweight='bold')
        ax_loo.set_ylabel('LOO $R^2$')
        ax_loo.grid(axis='y', linestyle=':', alpha=0.6)
        for bar, val in zip(bars, loo_r2_vals):
            offset = 0.02 if val >= 0 else -0.08
            ax_loo.text(bar.get_x() + bar.get_width()/2, val + offset, f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')

        x = np.arange(len(model_names))
        width = 0.35
        ax_fit = axes[i, 1]
        ax_fit.bar(x - width/2, r2_vals, width, label='$R^2$', color='#4A90E2', alpha=0.85, edgecolor='black')
        ax_fit.bar(x + width/2, adj_r2_vals, width, label='調整済み $R^2$', color='#50E3C2', alpha=0.85, edgecolor='black')
        ax_fit.set_title(f'【{fname}】 全データ適合度', fontsize=12, fontweight='bold')
        ax_fit.set_xticks(x)
        ax_fit.set_xticklabels(model_names)
        ax_fit.set_ylabel('$R^2$')
        ax_fit.set_ylim(-0.1, 1.05)
        ax_fit.legend(loc='lower right', fontsize=9)
        ax_fit.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(BARPLOT_PNG, dpi=300)
    plt.close()
    print('Saved LOO-CV barplots.')

    # レポート
    lines = [
        '# 視聴満足度因子 機械学習モデル比較検証レポート（LOO-CV版）', '',
        'N=24 の小サンプル環境において、5-Fold CV よりも統計的に適切な **Leave-One-Out CV（LOO-CV）** を採用し、4種類のモデルの汎化性能を再評価しました。', '',
        '> [!NOTE]',
        '> **LOO-CVにおけるR²の算出方法**: 1サンプルでのR²は数学的に定義不能なため、全24回のLOO予測値を一括収集し、$LOO\\text{-}R^2 = 1 - SS_{res}^{LOO} / SS_{tot}$ として算出しています。これは「LOO予測で残差を最小化できたか」を表す汎化指標です。', '',
        '## 1. LOO-CV スコア比較サマリー', '',
        '| 満足度因子 | モデル | 全データ $R^2$ | 調整済み $R^2$ | LOO-CV $R^2$ (汎化) | LOO RMSE | LOO MAE | 最適パラメータ |',
        '|---|---|---|---|---|---|---|---|'
    ]
    for factor in target_factors:
        fname = factor_names_ja[factor]
        for m in model_names:
            res = results[factor][m]
            lines.append(
                f"| **{fname}** | **{m}** | {res['r2']:.3f} | {res['adj_r2']:.3f} | **{res['loo_r2']:.3f}** | {res['loo_rmse']:.3f} | {res['loo_mae']:.3f} | `{res['params']}` |"
            )
    lines.extend(['',
        '## 2. 5-Fold CV vs LOO-CV 比較考察', '',
        '- **LOO-R²** は全LOO予測値を一括評価するため、5-Fold CVより安定した汎化性能の推定値となります。',
        '- 5-Fold CVでは「外れ値セグメントが1つのFoldに集中する」リスクがあり、LOO-CVではこのリスクが解消されます。',
        '- LOO RMSEは「1セグメントを抜いた場合の予測精度」を直接示す指標として論文掲載に適しています。', '',
        '![LOO-CV モデル比較プロット](model_comparison_loocv_barplots.png)', '',
        '## 3. 参考文献', '',
        '```text',
        '[1] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. JRSS-B. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x',
        '[2] Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS-B. https://doi.org/10.1111/j.1467-9868.2005.00503.x',
        '[3] Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. Stat. Comput. https://doi.org/10.1023/B:STCO.0000035301.49549.88',
        '[4] Breiman, L. (2001). Random forests. Mach. Learn. https://doi.org/10.1023/A:1010933404324',
        '```'
    ])
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('Saved LOO-CV report.')

    print('\n=== LOO-CV R² Summary ===')
    for factor in target_factors:
        fname = factor_names_ja[factor]
        print(f'\n{fname}:')
        for m in model_names:
            res = results[factor][m]
            print(f'  {m:15s}: LOO-R²={res["loo_r2"]:+.3f}, LOO-RMSE={res["loo_rmse"]:.3f}')

    return results

if __name__ == '__main__':
    run_loocv_comparison()
