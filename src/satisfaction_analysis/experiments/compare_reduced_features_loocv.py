# -*- coding: utf-8 -*-
"""
compare_reduced_features_loocv.py
アプローチ4（特徴量厳選・PCA次元削減による過学習抑制）のLOO-CV検証スクリプト。
・Baseline: 全13特徴量
・Selected: 相関/LMM/Permutationに基づく厳選2〜3特徴量
・PCA: 訓練データ内PCA（PC1〜PC2）による2次元合成特徴量
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.base import clone

PROJECT_ROOT = Path("/home/shota/work/year1")
FEATURE_MATRIX_CSV = PROJECT_ROOT / 'data' / 'processed' / 'feature_matrix.csv'

def loo_eval(pipe, X, y):
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        p_clone = clone(pipe)
        p_clone.fit(X[train_idx], y[train_idx])
        y_pred_loo[test_idx] = p_clone.predict(X[test_idx])
    ss_res = np.sum((y - y_pred_loo) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    loo_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    loo_rmse = float(np.sqrt(np.mean((y - y_pred_loo) ** 2)))
    loo_mae = float(np.mean(np.abs(y - y_pred_loo)))
    return loo_r2, loo_rmse, loo_mae

def get_best_model_for_factor(factor_name, X, y):
    models = {}
    
    # 1. SVR
    svr_grid = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('svr', SVR())]),
        param_grid={'svr__kernel': ['rbf', 'linear'], 'svr__C': [0.1, 1.0, 10.0],
                    'svr__epsilon': [0.01, 0.1, 0.2], 'svr__gamma': ['scale', 'auto', 0.01, 0.1]},
        cv=5, scoring='neg_mean_squared_error'
    )
    svr_grid.fit(X, y)
    best_svr = svr_grid.best_estimator_
    r2_svr, rmse_svr, mae_svr = loo_eval(best_svr, X, y)
    models['SVR'] = {'pipe': best_svr, 'loo_r2': r2_svr, 'loo_rmse': rmse_svr, 'params': svr_grid.best_params_}

    # 2. RandomForest
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={'n_estimators': [50, 100], 'max_depth': [2, 3, 4, None],
                    'min_samples_leaf': [1, 2, 3], 'max_features': ['sqrt', 1.0] if X.shape[1] > 1 else [1.0]},
        cv=5, scoring='neg_mean_squared_error'
    )
    rf_grid.fit(X, y)
    best_rf = rf_grid.best_estimator_
    r2_rf, rmse_rf, mae_rf = loo_eval(best_rf, X, y)
    models['RandomForest'] = {'pipe': best_rf, 'loo_r2': r2_rf, 'loo_rmse': rmse_rf, 'params': rf_grid.best_params_}

    # 3. Lasso
    lasso_pipe = Pipeline([('scaler', StandardScaler()), ('model', LassoCV(cv=5, random_state=42, max_iter=10000))])
    lasso_pipe.fit(X, y)
    r2_lasso, rmse_lasso, mae_lasso = loo_eval(lasso_pipe, X, y)
    models['Lasso'] = {'pipe': lasso_pipe, 'loo_r2': r2_lasso, 'loo_rmse': rmse_lasso, 'params': f"alpha={lasso_pipe.named_steps['model'].alpha_:.4f}"}

    # 4. ElasticNet
    enet_pipe = Pipeline([('scaler', StandardScaler()),
                          ('model', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=42, max_iter=10000))])
    enet_pipe.fit(X, y)
    r2_enet, rmse_enet, mae_enet = loo_eval(enet_pipe, X, y)
    models['ElasticNet'] = {'pipe': enet_pipe, 'loo_r2': r2_enet, 'loo_rmse': rmse_enet, 'params': f"alpha={enet_pipe.named_steps['model'].alpha_:.4f}"}

    return models

def run_experiment():
    print("=== 特徴量次元削減 LOO-CV 比較実験開始 ===")
    df = pd.read_csv(FEATURE_MATRIX_CSV)

    all_13_features = [
        'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
        'sentiment_polarity', 'subtitle_sentiment_polarity', 'pause_ratio', 'speech_rate',
        'sub_comment_similarity', 'comment_semantic_variance',
        'hm_mean', 'hm_max', 'hm_volatility'
    ]

    selected_features_map = {
        'Factor_Information': ['subtitle_sentiment_polarity', 'hm_mean', 'question_ratio'],
        'Factor_Relaxation': ['hm_volatility', 'comment_rate', 'speech_rate'],
        'Factor_Entertainment': ['hm_volatility', 'hm_max', 'comment_semantic_variance'],
        'Factor_SocialShare': ['grass_ratio', 'comment_semantic_variance'],
        'Target_Satisfaction': ['question_ratio', 'pause_ratio']
    }

    factor_names_ja = {
        'Factor_Information': '情報性・学習価値',
        'Factor_Relaxation': 'リラックス・気軽さ',
        'Factor_Entertainment': '娯楽性・没頭感',
        'Factor_SocialShare': '社会的共有性',
        'Target_Satisfaction': '総合満足度'
    }

    results = {}

    for factor_col, factor_name in factor_names_ja.items():
        print(f"\n▶ 因子: {factor_name}")
        y = df[factor_col].values
        results[factor_col] = {}

        # 1. Baseline (全13特徴量)
        print("  - パターン1: Baseline (全13特徴量)")
        X_base = df[all_13_features].values
        results[factor_col]['Baseline'] = get_best_model_for_factor(factor_col, X_base, y)

        # 2. Selected (厳選2〜3特徴量)
        sel_cols = selected_features_map[factor_col]
        print(f"  - パターン2: Selected (厳選{len(sel_cols)}特徴量: {sel_cols})")
        X_sel = df[sel_cols].values
        results[factor_col]['Selected'] = get_best_model_for_factor(factor_col, X_sel, y)

        # 3. PCA (訓練内PCA 2次元)
        print("  - パターン3: PCA (PC1〜PC2 2次元)")
        pca_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2, random_state=42))])
        X_pca = pca_pipe.fit_transform(X_base)
        results[factor_col]['PCA'] = get_best_model_for_factor(factor_col, X_pca, y)

    # 結果まとめ
    print("\n=== 全因子の比較結果サマリー ===")
    summary_rows = []
    for factor_col, factor_name in factor_names_ja.items():
        sel_cols = selected_features_map[factor_col]
        for model in ['SVR', 'RandomForest', 'Lasso', 'ElasticNet']:
            base_r2 = results[factor_col]['Baseline'][model]['loo_r2']
            base_rmse = results[factor_col]['Baseline'][model]['loo_rmse']
            sel_r2 = results[factor_col]['Selected'][model]['loo_r2']
            sel_rmse = results[factor_col]['Selected'][model]['loo_rmse']
            pca_r2 = results[factor_col]['PCA'][model]['loo_r2']
            pca_rmse = results[factor_col]['PCA'][model]['loo_rmse']
            diff_r2 = sel_r2 - base_r2
            
            summary_rows.append({
                'factor_col': factor_col,
                'factor_name': factor_name,
                'model': model,
                'selected_features': ", ".join(sel_cols),
                'base_r2': base_r2,
                'base_rmse': base_rmse,
                'sel_r2': sel_r2,
                'sel_rmse': sel_rmse,
                'pca_r2': pca_r2,
                'pca_rmse': pca_rmse,
                'diff_r2': diff_r2
            })
            print(f"[{factor_name}] {model:12s}: Base R²={base_r2:+.3f} -> Sel R²={sel_r2:+.3f} (Δ={diff_r2:+.3f}), PCA R²={pca_r2:+.3f}")

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(PROJECT_ROOT / 'data/processed/feature_reduction_comparison.csv', index=False)
    print("\nCSV saved to data/processed/feature_reduction_comparison.csv")

if __name__ == '__main__':
    run_experiment()
