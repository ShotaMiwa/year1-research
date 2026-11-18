# -*- coding: utf-8 -*-
import itertools
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df_survey = pd.read_csv('/home/shota/work/year1/data/processed/feature_matrix.csv')
df_shorts = pd.read_csv('/home/shota/work/year1/data/processed/extended_feature_matrix_77.csv')
outliers = {'LD06GCUS6cI', 'Msd39_BhMqU', '4KMwiyfGxA0', 'UEvV6gO5l88'}
df_shorts_73 = df_shorts[~df_shorts['shorts_id'].isin(outliers)].copy()

FEATURE_COLS = [
    'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity', 'speech_rate',
    'pause_ratio', 'sub_comment_similarity', 'comment_vocab_diversity',
    'comment_semantic_variance', 'segment_length_sec'
]

y_train = (df_shorts_73['like_rate'] >= df_shorts_73['like_rate'].median()).astype(int).values
y_test  = (df_survey['Target_Satisfaction'] >= df_survey['Target_Satisfaction'].median()).astype(int).values

pairs = list(itertools.combinations(FEATURE_COLS, 2))
print(f'全ペア数: {len(pairs)} 通り')

results = []
for f1, f2 in pairs:
    feats = [f1, f2]
    sc_tr = StandardScaler().fit(df_shorts_73[feats].values)
    sc_te = StandardScaler().fit(df_survey[feats].values)
    X_tr = sc_tr.transform(df_shorts_73[feats].values)
    X_te = sc_te.transform(df_survey[feats].values)

    # SVM
    clf_svm = SVC(kernel='linear', C=1.0, random_state=42).fit(X_tr, y_train)
    pred_svm = clf_svm.predict(X_te)
    acc_svm = accuracy_score(y_test, pred_svm) * 100
    cor_svm = sum(pred_svm == y_test)

    # LogReg
    clf_lr = LogisticRegression(C=1.0, random_state=42).fit(X_tr, y_train)
    pred_lr = clf_lr.predict(X_te)
    acc_lr = accuracy_score(y_test, pred_lr) * 100
    cor_lr = sum(pred_lr == y_test)

    results.append({
        'feature_1': f1,
        'feature_2': f2,
        'pair_name': f'{f1} + {f2}',
        'svm_acc': acc_svm,
        'svm_correct': cor_svm,
        'lr_acc': acc_lr,
        'lr_correct': cor_lr,
    })

df_res = pd.DataFrame(results)

print('\n=== 【全66通り総当たり】Linear SVM 転移精度 Top 10 ===')
top_svm = df_res.sort_values(by='svm_acc', ascending=False).head(10)
for _, r in top_svm.iterrows():
    print(f"{r['pair_name']:50s}: {r['svm_acc']:.1f}% ({r['svm_correct']}/23)")

print('\n=== 【全66通り総当たり】ロジスティック回帰 転移精度 Top 10 ===')
top_lr = df_res.sort_values(by='lr_acc', ascending=False).head(10)
for _, r in top_lr.iterrows():
    print(f"{r['pair_name']:50s}: {r['lr_acc']:.1f}% ({r['lr_correct']}/23)")

df_res.to_csv('/home/shota/work/year1/docs/paper/tables/exp_all_66_pairs_transfer_gridsearch.csv', index=False)
