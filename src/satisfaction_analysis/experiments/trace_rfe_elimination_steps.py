# -*- coding: utf-8 -*-
"""
trace_rfe_elimination_steps.py

YouTube実測73件データにおいて、全12特徴量からRFE（再帰的特徴消去）によって
「重みの小さい特徴量から1つずつ削ぎ落としていった完全な過程（k=12 -> 1）」と、
各ステップで残った特徴量セットによるアンケート23件への転移的中率を追跡・出力する。
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df_shorts = pd.read_csv('/home/shota/work/year1/data/processed/extended_feature_matrix_77.csv')
outliers = {'LD06GCUS6cI', 'Msd39_BhMqU', '4KMwiyfGxA0', 'UEvV6gO5l88'}
df_73 = df_shorts[~df_shorts['shorts_id'].isin(outliers)].copy().reset_index(drop=True)
y_73 = (df_73['like_rate'] >= df_73['like_rate'].median()).astype(int).values

df_survey = pd.read_csv('/home/shota/work/year1/data/processed/feature_matrix.csv')
y_survey = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values

FEATURE_NAMES_12 = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_vocab_diversity',
    'comment_semantic_variance', 'segment_length_sec'
]

def trace_rfe(clf_fn, model_name):
    sc_73 = StandardScaler()
    X_73 = sc_73.fit_transform(df_73[FEATURE_NAMES_12].values)
    
    sc_23 = StandardScaler()
    X_23 = sc_23.fit_transform(df_survey[FEATURE_NAMES_12].values)
    
    current_feats = list(range(12))
    
    print("\n" + "=" * 90)
    print(f"【{model_name}】RFEによる特徴量重要度ベース削減ステップ (k=12 -> 1)")
    print("=" * 90)
    
    records = []
    
    while len(current_feats) >= 1:
        k = len(current_feats)
        clf = clf_fn()
        clf.fit(X_73[:, current_feats], y_73)
        preds = clf.predict(X_23[:, current_feats])
        acc = accuracy_score(y_survey, preds)
        correct = np.sum(preds == y_survey)
        
        weights = np.abs(clf.coef_[0])
        feat_names = [FEATURE_NAMES_12[i] for i in current_feats]
        
        worst_local_idx = np.argmin(weights)
        elim_feat = feat_names[worst_local_idx]
        worst_weight = weights[worst_local_idx]
        
        records.append({
            "k": k,
            "correct": correct,
            "acc": acc,
            "features": list(feat_names),
            "eliminated": elim_feat if k > 1 else "(最後残存)",
            "min_weight": worst_weight
        })
        
        if k == 1:
            break
        current_feats.pop(worst_local_idx)
        
    for r in records:
        feat_str = " + ".join(r['features'])
        print(f"k={r['k']:2d} | 23件的中: {r['correct']:2d}/23 ({r['acc']*100:4.1f}%) | 消去: {r['eliminated']:<28} (重み={r['min_weight']:.4f})")
        print(f"     残存特徴量 ({r['k']}個): {feat_str}")
        print("-" * 90)

trace_rfe(lambda: SVC(kernel='linear', C=1.0, random_state=42), "Linear SVM")
trace_rfe(lambda: LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000), "Logistic Regression")
