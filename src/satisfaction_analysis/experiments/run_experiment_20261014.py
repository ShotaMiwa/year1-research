# coding: utf-8
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

BASE_DIR = '/home/shota/work/year1'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'feature_matrix.csv')
OUT_DIR = os.path.join(BASE_DIR, '進捗報告用', '20261014')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
FEATURE_COLS = [
    'comment_rate', 'question_ratio', 'exclamation_ratio',
    'sentiment_polarity', 'subtitle_sentiment_polarity',
    'speech_rate', 'pause_ratio', 'grass_ratio',
    'hm_mean', 'hm_max',
    'comment_semantic_variance', 'sub_comment_similarity',
]
available = [c for c in FEATURE_COLS if c in df.columns]
TARGET = 'Target_Satisfaction'

df_clean = df.copy()
median_target = df_clean[TARGET].median()
df_clean['Satisfaction_Class'] = (df_clean[TARGET] >= median_target).astype(int)
df_clean['Class_Label'] = df_clean['Satisfaction_Class'].map({1: 'High (>=3.60)', 0: 'Low (<3.60)'})

print('Loaded data:', len(df_clean), 'segments')
print('Target counts:', df_clean['Satisfaction_Class'].value_counts().to_dict())

# 課題2: 分布比較
target_vars = ['question_ratio', 'sentiment_polarity']
stats_rows = []
for var in target_vars:
    high_vals = df_clean[df_clean['Satisfaction_Class'] == 1][var]
    low_vals = df_clean[df_clean['Satisfaction_Class'] == 0][var]
    u_stat, p_val = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')
    t_stat, p_val_ttest = stats.ttest_ind(high_vals, low_vals, equal_var=False)
    stats_rows.append({
        'Variable': var,
        'High_Mean': high_vals.mean(),
        'High_Std': high_vals.std(),
        'High_Median': high_vals.median(),
        'Low_Mean': low_vals.mean(),
        'Low_Std': low_vals.std(),
        'Low_Median': low_vals.median(),
        'MW_U_stat': u_stat,
        'MW_p_value': p_val,
        'Ttest_p_value': p_val_ttest
    })
pd.DataFrame(stats_rows).to_csv(os.path.join(TAB_DIR, 'feature_stats_comparison.csv'), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titles = {'question_ratio': 'Question Ratio (Comment)', 'sentiment_polarity': 'Sentiment Polarity (Comment)'}
for idx, var in enumerate(target_vars):
    ax = axes[idx]
    sns.boxplot(x='Class_Label', y=var, data=df_clean, palette=['#4A90E2', '#E94E77'], ax=ax, width=0.4, boxprops=dict(alpha=0.6))
    sns.stripplot(x='Class_Label', y=var, data=df_clean, color='black', size=7, jitter=0.2, ax=ax)
    ax.set_title(titles[var], fontsize=13, fontweight='bold')
    ax.set_xlabel('Satisfaction Group', fontsize=11)
    ax.set_ylabel(var, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_distribution_boxplots.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='sentiment_polarity', y='question_ratio',
    hue='Class_Label', style='Class_Label',
    data=df_clean, s=120, palette=['#4A90E2', '#E94E77'], alpha=0.9
)
for _, r in df_clean.iterrows():
    plt.annotate(str(int(r['セグメント'])), (r['sentiment_polarity'], r['question_ratio']), fontsize=9, xytext=(4, 4), textcoords='offset points')
plt.title('Question Ratio vs Sentiment Polarity', fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_scatter_high_low.png'), dpi=300)
plt.close()

# 課題3: 疑問符率単体分類
X_q = df_clean[['question_ratio']].values
y = df_clean['Satisfaction_Class'].values
N = len(df_clean)
preds_dt1 = []
for i in range(N):
    X_train, y_train = np.delete(X_q, i, axis=0), np.delete(y, i)
    X_test, y_test = X_q[i:i+1], y[i:i+1]
    clf_stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf_stump.fit(X_train, y_train)
    preds_dt1.append(clf_stump.predict(X_test)[0])

preds_dt1 = np.array(preds_dt1)
df_clean['Pred_QuestionOnly'] = preds_dt1
df_clean['Is_Correct'] = (df_clean['Satisfaction_Class'] == df_clean['Pred_QuestionOnly'])
df_clean['Error_Type'] = 'Correct'
df_clean.loc[(df_clean['Satisfaction_Class'] == 0) & (df_clean['Pred_QuestionOnly'] == 1), 'Error_Type'] = 'False Positive (High予測だが実際Low)'
df_clean.loc[(df_clean['Satisfaction_Class'] == 1) & (df_clean['Pred_QuestionOnly'] == 0), 'Error_Type'] = 'False Negative (Low予測だが実際High)'

mis_cols = ['セグメント', '開始時刻', '終了時刻', 'Target_Satisfaction', 'Satisfaction_Class', 'Pred_QuestionOnly', 'Error_Type', 'question_ratio', 'sentiment_polarity', 'grass_ratio', 'comment_rate', 'speech_rate', 'pause_ratio', 'hm_mean']
mis_df = df_clean[df_clean['Is_Correct'] == False][mis_cols].sort_values('セグメント')
mis_df.to_csv(os.path.join(TAB_DIR, 'misclassified_segments_question_only.csv'), index=False)

cm_q = confusion_matrix(y, preds_dt1)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_q, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Pred Low', 'Pred High'], yticklabels=['True Low', 'True High'])
plt.title('Confusion Matrix (question_ratio Only)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'question_only_confusion_matrix.png'), dpi=300)
plt.close()

# 課題1: RF HP調整
X_all = df_clean[available].values
y_all = df_clean['Satisfaction_Class'].values

def sfs_select(X_tr, y_tr, model_factory, k_features=3):
    selected = []
    candidates = list(range(X_tr.shape[1]))
    while len(selected) < k_features:
        best_score = -1
        best_cand = None
        for cand in candidates:
            current = selected + [cand]
            n_inner = len(X_tr)
            preds = []
            for j in range(n_inner):
                X_in_tr, y_in_tr = np.delete(X_tr[:, current], j, axis=0), np.delete(y_tr, j)
                X_in_te = X_tr[j:j+1, current]
                m = model_factory()
                m.fit(X_in_tr, y_in_tr)
                preds.append(m.predict(X_in_te)[0])
            score = balanced_accuracy_score(y_tr, preds)
            if score > best_score:
                best_score = score
                best_cand = cand
        selected.append(best_cand)
        candidates.remove(best_cand)
    return selected

param_grid = [
    {'name': 'Default (Baseline)', 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Depth=1 (Stump Ensemble)', 'max_depth': 1, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Depth=2, Leaf=2', 'max_depth': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Depth=2, Leaf=3', 'max_depth': 2, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Depth=3, Leaf=2', 'max_depth': 3, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Depth=2, Leaf=2, Balanced', 'max_depth': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'name': 'Depth=1, Balanced', 'max_depth': 1, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=1', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 1, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=2, Leaf=2', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None},
    {'name': 'Fixed 2 Feats (Q+Sent) Depth=2, Leaf=2, Balanced', 'fixed_feats': ['question_ratio', 'sentiment_polarity'], 'max_depth': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
]

rf_results = []
for param in param_grid:
    p_name = param['name']
    is_fixed = 'fixed_feats' in param
    outer_preds = []
    for i in range(N):
        X_outer_tr, y_outer_tr = np.delete(X_all, i, axis=0), np.delete(y_all, i)
        X_outer_te, y_outer_te = X_all[i:i+1], y_all[i:i+1]
        if is_fixed:
            feat_indices = [available.index(f) for f in param['fixed_feats']]
        else:
            factory = lambda: RandomForestClassifier(n_estimators=100, max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], max_features=param['max_features'], class_weight=param['class_weight'], random_state=42)
            feat_indices = sfs_select(X_outer_tr, y_outer_tr, factory, k_features=3)
        rf = RandomForestClassifier(n_estimators=100, max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], max_features=param['max_features'], class_weight=param['class_weight'], random_state=42)
        rf.fit(X_outer_tr[:, feat_indices], y_outer_tr)
        pred = rf.predict(X_outer_te[:, feat_indices])[0]
        outer_preds.append(pred)
    outer_preds = np.array(outer_preds)
    acc = accuracy_score(y_all, outer_preds)
    bal_acc = balanced_accuracy_score(y_all, outer_preds)
    f1 = f1_score(y_all, outer_preds, average='macro')
    print(p_name, '-> Acc:', round(acc, 4), 'BalAcc:', round(bal_acc, 4), 'F1:', round(f1, 4))
    rf_results.append({
        'Model_Setting': p_name,
        'Accuracy': acc,
        'Correct_Count': int(sum(outer_preds == y_all)),
        'Total': N,
        'Balanced_Accuracy': bal_acc,
        'F1_Macro': f1
    })

rf_res_df = pd.DataFrame(rf_results)
rf_res_df.to_csv(os.path.join(TAB_DIR, 'rf_tuning_results.csv'), index=False)

plt.figure(figsize=(10, 6))
y_pos = range(len(rf_res_df))
bars = plt.barh(y_pos, rf_res_df['Balanced_Accuracy'] * 100, color='#4A90E2', alpha=0.8, height=0.6)
bars[0].set_color('#E94E77')
plt.yticks(y_pos, rf_res_df['Model_Setting'], fontsize=10)
plt.xlabel('Balanced Accuracy (%)', fontsize=11)
plt.title('RandomForest Hyperparameter Tuning Comparison (Nested LOO-CV)', fontsize=13, fontweight='bold')
plt.axvline(x=50.0, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
plt.axvline(x=38.1, color='red', linestyle=':', alpha=0.7, label='Default Baseline (38.1%)')
for bar in bars:
    w = bar.get_width()
    plt.text(w + 1, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=9)
plt.xlim(0, 100)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_param_comparison.png'), dpi=300)
plt.close()
print('DONE!')
