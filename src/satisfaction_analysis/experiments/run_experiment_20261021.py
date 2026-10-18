# coding: utf-8
import os, sys, warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# ワークスペース基準パス
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'feature_matrix.csv')
OUT_DIR = os.path.join(BASE_DIR, '進捗報告用', '20261021')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

print(f"Data Path: {DATA_PATH}")
print(f"Output Fig: {FIG_DIR}")
print(f"Output Tab: {TAB_DIR}")

df = pd.read_csv(DATA_PATH)
TARGET = 'Target_Satisfaction'
median_target = df[TARGET].median()
df['Satisfaction_Class'] = (df[TARGET] >= median_target).astype(int)
df['Class_Label'] = df['Satisfaction_Class'].map({1: 'High (>=3.60)', 0: 'Low (<3.60)'})

y = df['Satisfaction_Class'].values
N = len(df)

# =========================================================================
# 1. 決定木（Decision Tree）におけるパラメータ & 特徴量探索 (Nested LOO-CV & Fixed)
# =========================================================================

models_to_evaluate = [
    {
        'category': 'Single Feature (Stump d=1)',
        'name': 'DT: question_ratio only (d=1)',
        'feats': ['question_ratio'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=1, random_state=42)
    },
    {
        'category': 'Single Feature (Stump d=1)',
        'name': 'DT: hm_mean only (d=1)',
        'feats': ['hm_mean'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=1, random_state=42)
    },
    {
        'category': 'Single Feature (Stump d=1)',
        'name': 'DT: segment_length only (d=1)',
        'feats': ['segment_length_sec'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=1, random_state=42)
    },
    {
        'category': 'Multiple Features (DT)',
        'name': 'DT: question_ratio + hm_mean (d=2, l=1)',
        'feats': ['question_ratio', 'hm_mean'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, random_state=42)
    },
    {
        'category': 'Multiple Features (DT)',
        'name': 'DT: question_ratio + segment_length (d=3, l=1)',
        'feats': ['question_ratio', 'segment_length_sec'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=42)
    },
    {
        'category': 'Multiple Features (DT)',
        'name': 'DT: question_ratio + sentiment_polarity (d=3, l=1)',
        'feats': ['question_ratio', 'sentiment_polarity'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=42)
    },
    {
        'category': 'Multiple Features (DT)',
        'name': 'DT: question_ratio + hm_mean + segment_length (d=3, l=1)',
        'feats': ['question_ratio', 'hm_mean', 'segment_length_sec'],
        'model_fn': lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=42)
    },
    {
        'category': 'Random Forest Comparison',
        'name': 'RF: question_ratio + sentiment (d=2, l=2)',
        'feats': ['question_ratio', 'sentiment_polarity'],
        'model_fn': lambda: RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=2, random_state=42)
    },
    {
        'category': 'Random Forest Comparison',
        'name': 'RF: question_ratio + hm_mean (d=2, l=2)',
        'feats': ['question_ratio', 'hm_mean'],
        'model_fn': lambda: RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=2, random_state=42)
    },
    {
        'category': 'Random Forest Comparison',
        'name': 'RF: question_ratio + segment_length (d=2, l=2)',
        'feats': ['question_ratio', 'segment_length_sec'],
        'model_fn': lambda: RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=2, random_state=42)
    },
]

eval_results = []
pred_dict = {}

for m_info in models_to_evaluate:
    m_name = m_info['name']
    fts = m_info['feats']
    X_mat = df[fts].values
    preds = []
    for i in range(N):
        X_tr, y_tr = np.delete(X_mat, i, axis=0), np.delete(y, i)
        X_te = X_mat[i:i+1]
        clf = m_info['model_fn']()
        clf.fit(X_tr, y_tr)
        preds.append(clf.predict(X_te)[0])
    preds = np.array(preds)
    pred_dict[m_name] = preds

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    cm = confusion_matrix(y, preds)
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    mis_segs = df.iloc[np.where(preds != y)[0]]['セグメント'].tolist()

    eval_results.append({
        'Category': m_info['category'],
        'Model_Name': m_name,
        'Features': ', '.join(fts),
        'Accuracy': acc,
        'Balanced_Accuracy': bal_acc,
        'F1_Macro': f1,
        'Correct_Count': int(np.sum(preds == y)),
        'Total': N,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Misclassified_Segments': str(mis_segs)
    })

eval_df = pd.DataFrame(eval_results)
eval_df.to_csv(os.path.join(TAB_DIR, 'model_performance_comparison.csv'), index=False)
print("Saved: model_performance_comparison.csv")

# =========================================================================
# 2. 誤分類セグメント（特にSeg 19, 22）の詳細分析表
# =========================================================================
mis_analysis_rows = []
for idx, row in df.iterrows():
    seg_id = int(row['セグメント'])
    true_cls = row['Satisfaction_Class']
    true_score = row['Target_Satisfaction']
    
    # 3つの主要モデルの予測
    p_q = pred_dict['DT: question_ratio only (d=1)'][idx]
    p_hm = pred_dict['DT: question_ratio + hm_mean (d=2, l=1)'][idx]
    p_len = pred_dict['DT: question_ratio + segment_length (d=3, l=1)'][idx]
    
    # どこかで誤分類されたセグメント、あるいはSeg 19, 22
    if (p_q != true_cls) or (p_hm != true_cls) or (p_len != true_cls) or seg_id in [19, 22]:
        mis_analysis_rows.append({
            'セグメント': seg_id,
            '開始時刻': row['開始時刻'],
            '終了時刻': row['終了時刻'],
            '長さ(秒)': row['segment_length_sec'],
            'Target_Satisfaction': true_score,
            'True_Class': 'High' if true_cls == 1 else 'Low',
            'Pred_Q_Only': 'High' if p_q == 1 else 'Low',
            'Pred_Q_HM': 'High' if p_hm == 1 else 'Low',
            'Pred_Q_Len': 'High' if p_len == 1 else 'Low',
            'question_ratio': round(row['question_ratio'], 4),
            'hm_mean': round(row['hm_mean'], 4),
            'sentiment_polarity': round(row['sentiment_polarity'], 4),
            'grass_ratio': round(row['grass_ratio'], 4),
            'speech_rate': round(row['speech_rate'], 2),
            'comment_rate': round(row['comment_rate'], 2),
        })

mis_df = pd.DataFrame(mis_analysis_rows).sort_values('セグメント')
mis_df.to_csv(os.path.join(TAB_DIR, 'misclassified_segment_analysis.csv'), index=False)
print("Saved: misclassified_segment_analysis.csv")

# =========================================================================
# 3. 図表生成 (Visualizations)
# =========================================================================

# (A) モデル精度比較バープロット
plt.figure(figsize=(10, 6))
plot_df = eval_df.sort_values('Balanced_Accuracy', ascending=True).reset_index(drop=True)
colors = ['#5DADE2' if 'DT' in name else '#E67E22' for name in plot_df['Model_Name']]
bars = plt.barh(range(len(plot_df)), plot_df['Balanced_Accuracy'] * 100, color=colors, alpha=0.85, height=0.65)

# 強調
for i, name in enumerate(plot_df['Model_Name']):
    if 'question_ratio + hm_mean' in name or 'question_ratio + segment_length' in name:
        bars[i].set_edgecolor('black')
        bars[i].set_linewidth(1.5)

plt.yticks(range(len(plot_df)), plot_df['Model_Name'], fontsize=10)
plt.xlabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Performance Comparison: Decision Tree vs Random Forest (LOO-CV)', fontsize=13, fontweight='bold')
plt.axvline(x=50.0, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
plt.axvline(x=63.1, color='red', linestyle=':', alpha=0.8, label='Q-Ratio Baseline (63.1%)')

for bar in bars:
    w = bar.get_width()
    plt.text(w + 1.0, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=9.5, fontweight='bold')

plt.xlim(0, 100)
plt.legend(loc='lower right', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'exp_model_performance_comparison.png'), dpi=300)
plt.close()

# (B) 決定木の構造可視化 (Tree 1: question_ratio + hm_mean)
clf_tree1 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, random_state=42)
clf_tree1.fit(df[['question_ratio', 'hm_mean']].values, y)

fig, ax = plt.subplots(figsize=(9, 5))
plot_tree(
    clf_tree1,
    feature_names=['Question Ratio', 'Heatmap Mean'],
    class_names=['Low Satisfaction', 'High Satisfaction'],
    filled=True,
    rounded=True,
    fontsize=11,
    ax=ax
)
plt.title('Decision Tree Structure: Question Ratio + Heatmap Mean (Depth=2)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'exp_dt_tree_structure_hm.png'), dpi=300)
plt.close()

# (C) 決定木の構造可視化 (Tree 2: question_ratio + segment_length_sec)
clf_tree2 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=42)
clf_tree2.fit(df[['question_ratio', 'segment_length_sec']].values, y)

fig, ax = plt.subplots(figsize=(11, 6))
plot_tree(
    clf_tree2,
    feature_names=['Question Ratio', 'Segment Length (s)'],
    class_names=['Low Satisfaction', 'High Satisfaction'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title('Decision Tree Structure: Question Ratio + Segment Length (Depth=3)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'exp_dt_tree_structure_length.png'), dpi=300)
plt.close()

# (D) 決定境界散布図 (Question Ratio vs Heatmap Mean)
plt.figure(figsize=(9, 6.5))
x_min, x_max = df['question_ratio'].min() - 0.02, df['question_ratio'].max() + 0.02
y_min, y_max = df['hm_mean'].min() - 0.02, df['hm_mean'].max() + 0.03
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf_tree1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
scatter = sns.scatterplot(
    x='question_ratio', y='hm_mean', hue='Class_Label', style='Class_Label',
    data=df, s=130, palette=['#E94E77', '#4A90E2'], alpha=0.9
)
for _, r in df.iterrows():
    seg_idx = int(r['セグメント'])
    color = 'red' if seg_idx in [19, 22] else 'black'
    fontweight = 'bold' if seg_idx in [19, 22] else 'normal'
    plt.annotate(
        f"Seg {seg_idx}",
        (r['question_ratio'], r['hm_mean']),
        fontsize=9.5, fontweight=fontweight, color=color,
        xytext=(5, 4), textcoords='offset points'
    )

plt.title('Decision Boundary: Question Ratio vs Heatmap Mean (DT)', fontsize=13, fontweight='bold')
plt.xlabel('Question Ratio (Comment)', fontsize=11)
plt.ylabel('Heatmap Mean (User Rewatch Peak)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'exp_dt_decision_boundary_hm.png'), dpi=300)
plt.close()

# (E) 決定境界散布図 (Question Ratio vs Segment Length)
plt.figure(figsize=(9, 6.5))
x_min, x_max = df['question_ratio'].min() - 0.02, df['question_ratio'].max() + 0.02
y_min, y_max = df['segment_length_sec'].min() - 10, df['segment_length_sec'].max() + 15
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf_tree2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
sns.scatterplot(
    x='question_ratio', y='segment_length_sec', hue='Class_Label', style='Class_Label',
    data=df, s=130, palette=['#E94E77', '#4A90E2'], alpha=0.9
)
for _, r in df.iterrows():
    seg_idx = int(r['セグメント'])
    color = 'red' if seg_idx in [19, 22] else 'black'
    fontweight = 'bold' if seg_idx in [19, 22] else 'normal'
    plt.annotate(
        f"Seg {seg_idx}",
        (r['question_ratio'], r['segment_length_sec']),
        fontsize=9.5, fontweight=fontweight, color=color,
        xytext=(5, 4), textcoords='offset points'
    )

plt.title('Decision Boundary: Question Ratio vs Segment Length (DT)', fontsize=13, fontweight='bold')
plt.xlabel('Question Ratio (Comment)', fontsize=11)
plt.ylabel('Segment Length (seconds)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'exp_dt_decision_boundary_length.png'), dpi=300)
plt.close()

print("All figures and tables successfully generated!")
