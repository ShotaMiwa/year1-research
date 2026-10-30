# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

BASE_DIR = '/home/shota/work/year1'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'feature_matrix.csv')
OUT_DIR = os.path.join(BASE_DIR, '進捗報告用', '20261111')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = 'Target_Satisfaction'
median_target = df[TARGET].median()
y = (df[TARGET] >= median_target).astype(int).values
N = len(df)

FEATURES = [
    'comment_rate', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity',
    'speech_rate', 'grass_ratio',
    'hm_mean', 'hm_max',
    'comment_semantic_variance', 'sub_comment_similarity',
    'segment_length_sec'
]

FEATURE_LABELS_EN = {
    'comment_rate': 'Comment Rate',
    'question_ratio': 'Question Ratio',
    'exclamation_ratio': 'Exclamation Ratio',
    'sentiment_polarity': 'Comment Sentiment',
    'subtitle_sentiment_polarity': 'Subtitle Sentiment',
    'speech_rate': 'Speech Rate',
    'grass_ratio': 'Laughter (w) Ratio',
    'hm_mean': 'Heatmap Mean',
    'hm_max': 'Heatmap Max',
    'comment_semantic_variance': 'Comment Semantic Variance',
    'sub_comment_similarity': 'Sub-Comment Similarity',
    'segment_length_sec': 'Segment Length (sec)'
}

X_all = df[FEATURES].values

conditions = [
    {
        'id': 'svc_all12_l2',
        'name': 'Linear SVM (All 12, L2)',
        'model_func': lambda: SVC(kernel='linear', C=1.0, random_state=42),
        'extract_coef': True
    },
    {
        'id': 'logreg_all12_l2',
        'name': 'Logistic Reg (All 12, L2)',
        'model_func': lambda: LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42),
        'extract_coef': True
    },
    {
        'id': 'logreg_all12_l1',
        'name': 'Logistic Reg (All 12, L1/Lasso)',
        'model_func': lambda: LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42),
        'extract_coef': True
    }
]

eval_summary = []
eval_summary.append({
    'Condition': 'Linear SVM (Exp A: k<=2 selection)',
    'Feature_Count': 'k<=2 (Selected)',
    'Regularization': 'L2 (C=1.0)',
    'Accuracy': 16/23,
    'Accuracy_Ratio': '16/23 (69.6%)',
    'Balanced_Accuracy': 0.6508,
    'Macro_F1': 0.6538,
    'TN_FP_FN_TP': '[Low: 5/9, High: 11/14]'
})
eval_summary.append({
    'Condition': 'Logistic Reg (Exp A: k<=2 selection)',
    'Feature_Count': 'k<=2 (Selected)',
    'Regularization': 'L2 (C=1.0)',
    'Accuracy': 13/23,
    'Accuracy_Ratio': '13/23 (56.5%)',
    'Balanced_Accuracy': 0.5437,
    'Macro_F1': 0.5437,
    'TN_FP_FN_TP': '[Low: 4/9, High: 9/14]'
})

coef_records = {cond['id']: [] for cond in conditions}
intercept_records = {cond['id']: [] for cond in conditions}

for cond in conditions:
    preds = []
    for i in range(N):
        tr_idx = [j for j in range(N) if j != i]
        te_idx = [i]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_idx])
        X_te = scaler.transform(X_all[te_idx])
        
        clf = cond['model_func']()
        clf.fit(X_tr, y[tr_idx])
        
        p = clf.predict(X_te)[0]
        preds.append(p)
        
        if cond['extract_coef']:
            coef_records[cond['id']].append(clf.coef_[0].copy())
            intercept_records[cond['id']].append(clf.intercept_[0])
            
    preds = np.array(preds)
    acc = accuracy_score(y, preds)
    bal = balanced_accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    eval_summary.append({
        'Condition': cond['name'],
        'Feature_Count': 'All 12 Features',
        'Regularization': 'L2' if 'L2' in cond['name'] else 'L1 (Lasso)',
        'Accuracy': acc,
        'Accuracy_Ratio': f'{int(acc*N)}/{N} ({acc*100:.1f}%)',
        'Balanced_Accuracy': bal,
        'Macro_F1': f1,
        'TN_FP_FN_TP': f'[Low: {tn}/9, High: {tp}/14]'
    })

sum_df = pd.DataFrame(eval_summary)
sum_csv_path = os.path.join(TAB_DIR, 'feature_dimension_comparison_summary.csv')
sum_df.to_csv(sum_csv_path, index=False, encoding='utf-8-sig')
print(f'Saved summary to {sum_csv_path}')
print(sum_df[['Condition', 'Feature_Count', 'Accuracy_Ratio', 'Balanced_Accuracy', 'Macro_F1']])

coef_summary_list = []
for cond in conditions:
    cid = cond['id']
    arr = np.array(coef_records[cid])
    mean_coef = arr.mean(axis=0)
    std_coef = arr.std(axis=0)
    
    for f_idx, feat in enumerate(FEATURES):
        coef_summary_list.append({
            'Condition_ID': cid,
            'Condition_Name': cond['name'],
            'Feature': feat,
            'Feature_Label': FEATURE_LABELS_EN[feat],
            'Mean_Weight': mean_coef[f_idx],
            'Std_Weight': std_coef[f_idx]
        })

coef_df = pd.DataFrame(coef_summary_list)
coef_csv_path = os.path.join(TAB_DIR, 'all12_feature_weights_summary.csv')
coef_df.to_csv(coef_csv_path, index=False, encoding='utf-8-sig')
print(f'Saved weights summary to {coef_csv_path}')

# Visualization 1: Compare Balanced Accuracy
plt.figure(figsize=(10, 5.5), dpi=300)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

labels = [
    'Linear SVM\n(Exp A: k<=2)',
    'Linear SVM\n(All 12, L2)',
    'Logistic Reg\n(Exp A: k<=2)',
    'Logistic Reg\n(All 12, L2)',
    'Logistic Reg\n(All 12, L1)'
]
bals = sum_df['Balanced_Accuracy'].values * 100
colors = ['#2ecc71', '#e74c3c', '#95a5a6', '#3498db', '#f39c12']

bars = plt.bar(labels, bals, color=colors, width=0.55, edgecolor='#333333', linewidth=1.0)
plt.axhline(50.0, color='#888888', linestyle='--', linewidth=1.2, label='Chance Level (50.0%)')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Performance Comparison: 2 Features (Exp A) vs All 12 Features', fontsize=13, fontweight='bold', pad=15)
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()

fig1_path = os.path.join(FIG_DIR, 'compare_2features_vs_all12.png')
plt.savefig(fig1_path)
plt.close()
print(f'Saved Figure 1 to {fig1_path}')

# Visualization 2: Feature weights
fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300, sharey=True)

svm_weights = coef_df[coef_df['Condition_ID'] == 'svc_all12_l2'].sort_values('Mean_Weight', ascending=True)
colors_svm = ['#e74c3c' if w < 0 else '#2ecc71' for w in svm_weights['Mean_Weight']]
y_pos = np.arange(len(svm_weights))

axes[0].barh(y_pos, svm_weights['Mean_Weight'], xerr=svm_weights['Std_Weight'], color=colors_svm, edgecolor='#333333', alpha=0.85, capsize=3)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(svm_weights['Feature_Label'], fontsize=10)
axes[0].axvline(0, color='#333333', linewidth=1.0)
axes[0].set_title('Linear SVM (All 12 Features, L2)\nMean Weights across 23 Folds', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Weight (w)', fontsize=11)
axes[0].grid(axis='x', linestyle=':', alpha=0.6)

log_weights = coef_df[coef_df['Condition_ID'] == 'logreg_all12_l2'].set_index('Feature').loc[svm_weights['Feature']]
colors_log = ['#e74c3c' if w < 0 else '#2ecc71' for w in log_weights['Mean_Weight']]

axes[1].barh(y_pos, log_weights['Mean_Weight'], xerr=log_weights['Std_Weight'], color=colors_log, edgecolor='#333333', alpha=0.85, capsize=3)
axes[1].axvline(0, color='#333333', linewidth=1.0)
axes[1].set_title('Logistic Regression (All 12 Features, L2)\nMean Weights across 23 Folds', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Weight (w)', fontsize=11)
axes[1].grid(axis='x', linestyle=':', alpha=0.6)

plt.suptitle('Learned Weights (w) for All 12 Features\n[Green = Contributes to High Satisfaction, Red = Low Satisfaction]', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()

fig2_path = os.path.join(FIG_DIR, 'all12_feature_weights_comparison.png')
plt.savefig(fig2_path)
plt.close()
print(f'Saved Figure 2 to {fig2_path}')
