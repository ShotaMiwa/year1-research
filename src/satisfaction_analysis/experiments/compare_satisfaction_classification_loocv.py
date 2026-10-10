# -*- coding: utf-8 -*-
"""
compare_satisfaction_classification_loocv.py
総合満足度の二値分類（High: 14件 vs Low: 9件, 閾値=3.60）LOO-CV検証スクリプト
・4つの特徴量パターン:
   1. All13 (全13特徴量)
   2. SelectedA (質問率, 無音割合: 2次元)
   3. SelectedB (字幕極性, HM変動性, 平均HM, コメント密度: 4次元)
   4. PCA (訓練内PCA 2次元)
・4つの分類モデル:
   1. LogisticRegression (L2)
   2. SVC (RBFカーネル)
   3. RandomForestClassifier
   4. Dummy (多数派予測: Chance Level)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.base import clone

PROJECT_ROOT = Path("/home/shota/work/year1")
FEATURE_MATRIX_CSV = PROJECT_ROOT / 'data' / 'processed' / 'feature_matrix.csv'

def loo_classify(pipe, X, y, has_proba=True):
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    y_proba = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        p = clone(pipe)
        p.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = p.predict(X[test_idx])
        if has_proba:
            if hasattr(p, "predict_proba"):
                y_proba[test_idx] = p.predict_proba(X[test_idx])[:, 1]
            elif hasattr(p, "decision_function"):
                df_val = p.decision_function(X[test_idx])
                # sigmoid
                y_proba[test_idx] = 1.0 / (1.0 + np.exp(-df_val))
            else:
                y_proba[test_idx] = y_pred[test_idx]
        else:
            y_proba[test_idx] = y_pred[test_idx]

    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    # Recall for Class 0 (Low Satisfaction)
    recall_c0 = recall_score(y, y_pred, pos_label=0, zero_division=0)
    # Recall for Class 1 (High Satisfaction)
    recall_c1 = recall_score(y, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y, y_proba)
    except:
        auc = 0.5
    cm = confusion_matrix(y, y_pred)
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1,
        'recall_low_c0': recall_c0,
        'recall_high_c1': recall_c1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def run_classification_experiment():
    print("=== 総合満足度 二値分類 LOO-CV 比較実験開始 ===")
    df = pd.read_csv(FEATURE_MATRIX_CSV)

    sat_med = df['Target_Satisfaction'].median()
    y = (df['Target_Satisfaction'] >= sat_med).astype(int).values
    n_high = int(np.sum(y == 1))
    n_low = int(np.sum(y == 0))
    print(f"目的変数二値化完了: 閾値={sat_med:.2f}, High(1)={n_high}件, Low(0)={n_low}件 (合計={len(y)}件)")

    all_13_features = [
        'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
        'sentiment_polarity', 'subtitle_sentiment_polarity', 'pause_ratio', 'speech_rate',
        'sub_comment_similarity', 'comment_semantic_variance',
        'hm_mean', 'hm_max', 'hm_volatility'
    ]

    feature_patterns = {
        'Pattern1_All13': all_13_features,
        'Pattern2_SelectedA_2d': ['question_ratio', 'pause_ratio'],
        'Pattern3_SelectedB_4d': ['subtitle_sentiment_polarity', 'hm_volatility', 'hm_mean', 'comment_rate'],
        'Pattern4_PCA_2d': 'PCA'
    }

    results = []
    confusion_matrices = {}
    preds_dict = {}

    for p_name, feats in feature_patterns.items():
        print(f"\n▶ 特徴量パターン: {p_name}")
        if feats == 'PCA':
            X_raw = df[all_13_features].values
            pca_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2, random_state=42))])
            X = pca_pipe.fit_transform(X_raw)
            feat_desc = "PCA (PC1, PC2 from 13 feats)"
        else:
            X = df[feats].values
            feat_desc = ", ".join(feats)

        models = {
            'Dummy_Majority': DummyClassifier(strategy='most_frequent'),
            'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, random_state=42))]),
            'SVC_RBF': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))]),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42)
        }

        for m_name, model in models.items():
            eval_res = loo_classify(model, X, y, has_proba=True)
            results.append({
                'feature_pattern': p_name,
                'feature_desc': feat_desc,
                'num_features': X.shape[1],
                'model_name': m_name,
                'accuracy': eval_res['accuracy'],
                'balanced_accuracy': eval_res['balanced_accuracy'],
                'f1_macro': eval_res['f1_macro'],
                'recall_low_c0': eval_res['recall_low_c0'],
                'recall_high_c1': eval_res['recall_high_c1'],
                'roc_auc': eval_res['roc_auc']
            })
            key = f"{p_name}_{m_name}"
            confusion_matrices[key] = eval_res['confusion_matrix']
            preds_dict[key] = eval_res['y_pred']

            print(f"  [{m_name:20s}] Acc={eval_res['accuracy']:.3f} | BalAcc={eval_res['balanced_accuracy']:.3f} | F1={eval_res['f1_macro']:.3f} | Recall_Low={eval_res['recall_low_c0']:.3f} | AUC={eval_res['roc_auc']:.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(PROJECT_ROOT / 'data/processed/satisfaction_classification_comparison.csv', index=False)
    print("\n✅ CSV saved to data/processed/satisfaction_classification_comparison.csv")

    # 最良モデルにおけるロジスティック回帰の係数
    print("\n=== Pattern 3 (SelectedB) ロジスティック回帰 係数・オッズ比分析 ===")
    sel_b_cols = feature_patterns['Pattern3_SelectedB_4d']
    X_b = df[sel_b_cols].values
    lr_b = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, random_state=42))])
    lr_b.fit(X_b, y)
    coefs = lr_b.named_steps['clf'].coef_[0]
    odds = np.exp(coefs)
    coef_df = pd.DataFrame({
        'feature': sel_b_cols,
        'coef': coefs,
        'odds_ratio': odds
    }).sort_values(by='coef', ascending=False)
    print(coef_df)
    coef_df.to_csv(PROJECT_ROOT / 'data/processed/satisfaction_lr_odds_ratio.csv', index=False)

    return df_res, confusion_matrices, coef_df, df['Target_Satisfaction'].values, y, preds_dict

if __name__ == '__main__':
    run_classification_experiment()
