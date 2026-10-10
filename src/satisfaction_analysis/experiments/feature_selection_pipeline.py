# -*- coding: utf-8 -*-
"""
特徴量選択パイプライン: RFECV → SFS
目的: 最適な特徴量の個数 n* と最適な組み合わせを客観的に決定する

設計書に基づく実装:
  Step 1: RFECV (Recursive Feature Elimination + LOO-CV) → n* を自動決定
  Step 2: SFS Forward/Backward → n* 個の最適組み合わせを決定
  Step 3: 最終モデル評価と既存結果との比較表作成

出力:
  data/processed/rfecv_cv_scores.csv
  data/processed/sfs_selected_features.csv
  docs/paper/feature_selection_pipeline_results.md
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import copy
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. データ読み込み
# ──────────────────────────────────────────────
BASE_DIR = "/home/shota/work/year1"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_matrix.csv")
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed")
DOC_DIR   = os.path.join(BASE_DIR, "docs", "paper")

df = pd.read_csv(DATA_PATH)
print(f"[INFO] データ読込: {df.shape[0]} サンプル, {df.shape[1]} 列")

# ──────────────────────────────────────────────
# 1. 特徴量・目的変数の定義
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "comment_rate", "question_ratio", "exclamation_ratio",
    "sentiment_polarity", "subtitle_sentiment_polarity",
    "speech_rate", "pause_ratio", "grass_ratio",
    "hm_mean", "hm_max",
    "comment_semantic_variance", "sub_comment_similarity",
    "avg_watching_ratio"
]
TARGET_COL = "Target_Satisfaction"

available = [c for c in FEATURE_COLS if c in df.columns]
missing   = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"[WARN] 不在の列を除外: {missing}")

df_clean = df[available + [TARGET_COL]].dropna()
print(f"[INFO] 有効サンプル: {len(df_clean)}, 特徴量数: {len(available)}")

X = df_clean[available].values
threshold = df_clean[TARGET_COL].median()
y = (df_clean[TARGET_COL] >= threshold).astype(int).values
print(f"[INFO] 閾値(中央値): {threshold:.4f}, High={y.sum()}, Low={(1-y).sum()}")

loo = LeaveOneOut()
scaler_global = StandardScaler()
X_scaled = scaler_global.fit_transform(X)

# ──────────────────────────────────────────────
# 2. Step 1: RFECV
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("Step 1: RFECV による最適次元数 n* の決定")
print("="*60)

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42),
}

rfecv_results = {}

for model_name, base_model in MODELS.items():
    print(f"\n  [{model_name}] RFECV 実行中... (少し時間がかかります)")
    rfecv = RFECV(
        estimator=copy.deepcopy(base_model),
        step=1,
        cv=loo,
        scoring="balanced_accuracy",
        min_features_to_select=1,
        n_jobs=-1,
    )
    rfecv.fit(X_scaled, y)

    n_optimal = rfecv.n_features_
    selected_mask = rfecv.support_
    selected_features = [available[i] for i in range(len(available)) if selected_mask[i]]
    cv_scores = rfecv.cv_results_["mean_test_score"]
    n_values = list(range(1, len(cv_scores) + 1))

    rfecv_results[model_name] = {
        "n_optimal":         n_optimal,
        "selected_features": selected_features,
        "cv_scores":         cv_scores,
        "n_values":          n_values,
        "best_score":        max(cv_scores),
    }

    print(f"    最適次元数 n* = {n_optimal}")
    print(f"    選定特徴量   = {selected_features}")
    print(f"    最高 Balanced Acc = {max(cv_scores):.4f}")
    for n, s in zip(n_values, cv_scores):
        marker = " ← n*" if n == n_optimal else ""
        print(f"      n={n:2d}: {s:.4f}{marker}")

# CSV保存
rows = []
for model_name, res in rfecv_results.items():
    for n, s in zip(res["n_values"], res["cv_scores"]):
        rows.append({"model": model_name, "n_features": n, "balanced_accuracy": round(s, 4)})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "rfecv_cv_scores.csv"), index=False)
print(f"\n[SAVED] rfecv_cv_scores.csv")

# ──────────────────────────────────────────────
# 3. Step 2: SFS
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("Step 2: SFS (Forward/Backward) による最適組み合わせの決定")
print("="*60)

sfs_results = []

for model_name, base_model in MODELS.items():
    n_opt = rfecv_results[model_name]["n_optimal"]
    for direction in ["forward", "backward"]:
        print(f"\n  [{model_name}] SFS-{direction.capitalize()}, n_features={n_opt} ...")
        sfs = SequentialFeatureSelector(
            estimator=copy.deepcopy(base_model),
            n_features_to_select=n_opt,
            direction=direction,
            cv=loo,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        sfs.fit(X_scaled, y)
        sel_feats = [available[i] for i in range(len(available)) if sfs.get_support()[i]]
        print(f"    選定特徴量: {sel_feats}")
        sfs_results.append({
            "model":      model_name,
            "direction":  direction,
            "n_optimal":  n_opt,
            "selected":   ", ".join(sel_feats),
        })

sfs_df = pd.DataFrame(sfs_results)
sfs_df.to_csv(os.path.join(OUT_DIR, "sfs_selected_features.csv"), index=False)
print(f"\n[SAVED] sfs_selected_features.csv")

# ──────────────────────────────────────────────
# 4. Step 3: 最終評価
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("Step 3: 最終モデル評価 (LOO-CV)")
print("="*60)

def loo_evaluate(X_sub, y_arr, model):
    preds, trues, probas = [], [], []
    for train_idx, test_idx in loo.split(X_sub):
        Xtr, Xte = X_sub[train_idx], X_sub[test_idx]
        ytr, yte = y_arr[train_idx], y_arr[test_idx]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        m = copy.deepcopy(model)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)[0]
        preds.append(pred); trues.append(yte[0])
        if hasattr(m, "predict_proba"):
            probas.append(m.predict_proba(Xte)[0][1])
        else:
            probas.append(float(pred))
    preds = np.array(preds); trues = np.array(trues); probas = np.array(probas)
    try:
        auc = roc_auc_score(trues, probas)
    except Exception:
        auc = float("nan")
    return {
        "acc":        (preds == trues).mean(),
        "bal_acc":    balanced_accuracy_score(trues, preds),
        "f1_macro":   f1_score(trues, preds, average="macro", zero_division=0),
        "recall_low": recall_score(trues, preds, pos_label=0, zero_division=0),
        "roc_auc":    auc,
    }

final_results = []

# RFECV 選定特徴量で評価
for base_name, res in rfecv_results.items():
    sel_feats = res["selected_features"]
    sel_idx = [available.index(f) for f in sel_feats]
    X_sub = X[:, sel_idx]
    for eval_name, eval_model in MODELS.items():
        metrics = loo_evaluate(X_sub, y, eval_model)
        final_results.append({
            "selection_method": f"RFECV({base_name})",
            "eval_model": eval_name,
            "n_features": res["n_optimal"],
            "features": ", ".join(sel_feats),
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        print(f"  RFECV({base_name})×{eval_name}: Acc={metrics['acc']:.4f}, BalAcc={metrics['bal_acc']:.4f}")

# SFS 選定特徴量で評価（重複スキップ）
seen = set()
for row in sfs_results:
    feat_set = frozenset(row["selected"].split(", "))
    if feat_set in seen:
        continue
    seen.add(feat_set)
    sel_feats = [f for f in row["selected"].split(", ") if f in available]
    sel_idx = [available.index(f) for f in sel_feats]
    X_sub = X[:, sel_idx]
    for eval_name, eval_model in MODELS.items():
        metrics = loo_evaluate(X_sub, y, eval_model)
        final_results.append({
            "selection_method": f"SFS-{row['direction'].capitalize()}({row['model']})",
            "eval_model": eval_name,
            "n_features": row["n_optimal"],
            "features": row["selected"],
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        print(f"  SFS-{row['direction'].capitalize()}({row['model']})×{eval_name}: Acc={metrics['acc']:.4f}")

# 既存の比較ベースライン
final_results.append({
    "selection_method": "事前仮説(相関上位2個)",
    "eval_model": "LogisticRegression",
    "n_features": 2,
    "features": "question_ratio, pause_ratio",
    "acc": 0.6957, "bal_acc": 0.6706, "f1_macro": 0.6731,
    "recall_low": 0.5556, "roc_auc": 0.6984,
})
final_results.append({
    "selection_method": "全78ペア最良",
    "eval_model": "LogisticRegression",
    "n_features": 2,
    "features": "question_ratio, speech_rate",
    "acc": 0.7826, "bal_acc": 0.7421, "f1_macro": 0.7531,
    "recall_low": 0.5556, "roc_auc": 0.7302,
})

final_df = pd.DataFrame(final_results)
final_df.to_csv(os.path.join(OUT_DIR, "feature_selection_final_results.csv"), index=False)
print(f"\n[SAVED] feature_selection_final_results.csv")

# ──────────────────────────────────────────────
# 5. Markdown レポート生成
# ──────────────────────────────────────────────
print("\nMarkdown レポート生成中...")

lr_res = rfecv_results.get("LogisticRegression", {})
rf_res = rfecv_results.get("RandomForest", {})

md = []
md.append("# 特徴量選択パイプライン 実験レポート")
md.append("")
md.append("## 1. 概要")
md.append("")
md.append("本レポートは RFECV（再帰的特徴量削減 + LOO-CV）および SFS（逐次特徴量選択）を用いて、")
md.append("「最適な特徴量の個数 $n^*$」と「最適な組み合わせ」を客観的に決定した結果をまとめる。")
md.append("")
md.append("| 項目 | 設定 |")
md.append("| :--- | :--- |")
md.append(f"| データ | feature_matrix.csv (N={len(df_clean)}, 特徴量={len(available)}個) |")
md.append(f"| 目的変数 | 総合満足度の高低 (閾値={threshold:.4f}、中央値) |")
md.append(f"| クラス分布 | High={y.sum()}, Low={(1-y).sum()} |")
md.append("| 交差検証 | Leave-One-Out CV (LOO-CV) |")
md.append("| 主評価指標 | Balanced Accuracy（クラス不均衡補正済み） |")
md.append("")
md.append("---")
md.append("")
md.append("## 2. Step 1: RFECV — 最適次元数 $n^*$ の決定")
md.append("")
md.append("全 13 特徴量から 1 個ずつ再帰的に削除し、各次元数 $n$ における LOO-CV Balanced Accuracy を算出。")
md.append("スコアが最大となる $n$ を最適次元数 $n^*$ として採用した。")
md.append("")

for model_name, res in rfecv_results.items():
    md.append(f"### {model_name}")
    md.append("")
    md.append(f"- **最適次元数 $n^* = {res['n_optimal']}$**")
    md.append(f"- 選定特徴量: `{'`, `'.join(res['selected_features'])}`")
    md.append(f"- 最高 Balanced Accuracy: **{res['best_score']:.4f}**")
    md.append("")
    md.append("| 次元数 $n$ | Balanced Acc (LOO-CV) | 備考 |")
    md.append("| :---: | :---: | :--- |")
    for n, s in zip(res["n_values"], res["cv_scores"]):
        marker = "← **$n^*$（採用）**" if n == res["n_optimal"] else ""
        md.append(f"| {n} | {s:.4f} | {marker} |")
    md.append("")

md.append("---")
md.append("")
md.append("## 3. Step 2: SFS — 最適組み合わせの決定")
md.append("")
md.append("Step 1 で確定した $n^*$ 個の特徴量を使い、前向き（Forward）・後向き（Backward）の両方向で")
md.append("最適な特徴量の組み合わせを探索した。")
md.append("")
md.append("| モデル | 方向 | 選定個数 | 選定特徴量 |")
md.append("| :--- | :--- | :---: | :--- |")
for row in sfs_results:
    md.append(f"| {row['model']} | {row['direction'].capitalize()} | {row['n_optimal']} | `{row['selected']}` |")
md.append("")

for model_name in MODELS.keys():
    model_sfs = [r for r in sfs_results if r["model"] == model_name]
    if len(model_sfs) == 2:
        fwd = set(model_sfs[0]["selected"].split(", "))
        bwd = set(model_sfs[1]["selected"].split(", "))
        if fwd == bwd:
            md.append(f"> ✅ **{model_name}**: Forward と Backward が**同一の特徴量セットを選定** → 選択の頑健性を確認")
        else:
            md.append(f"> ⚠️ **{model_name}**: Forward (`{model_sfs[0]['selected']}`) と Backward (`{model_sfs[1]['selected']}`) で結果が異なる")
md.append("")
md.append("---")
md.append("")
md.append("## 4. Step 3: 最終評価結果と手法比較")
md.append("")
md.append("| 選択手法 | 評価モデル | $n$ | 特徴量 | LOO Acc | Balanced Acc | F1-Macro | Recall(Low) | ROC-AUC |")
md.append("| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
best_acc = final_df["acc"].max()
for _, row in final_df.iterrows():
    flag = "🥇 " if row["acc"] == best_acc else ""
    md.append(
        f"| {flag}{row['selection_method']} | {row['eval_model']} | {row['n_features']} "
        f"| `{row['features']}` "
        f"| **{row['acc']:.4f}** | {row['bal_acc']:.4f} | {row['f1_macro']:.4f} "
        f"| {row['recall_low']:.4f} | {row['roc_auc']:.4f} |"
    )
md.append("")
md.append("---")
md.append("")
md.append("## 5. 考察・結論")
md.append("")
md.append(f"1. **最適次元数 $n^*$ の確定**:  ")
md.append(f"   RFECV により LogisticRegression では $n^* = {lr_res.get('n_optimal','?')}$、")
md.append(f"   RandomForest では $n^* = {rf_res.get('n_optimal','?')}$ が最適と決定された。")
md.append(f"   これにより「なぜ {lr_res.get('n_optimal','?')} 次元か」の根拠をモデルのCV性能から客観的に示すことができる。")
md.append("")
md.append("2. **SFS による最適組み合わせの確定**:  ")
md.append("   前向き・後向きの両方向探索が同一（または近い）特徴量セットを選んだ場合、")
md.append("   その選択が探索方向に依存しない頑健な選択であることを意味する。")
md.append("")
md.append("3. **既存手法との比較**:  ")
md.append("   RFECV+SFS（事前的なアルゴリズムによる選択）と「全78ペア探索」（事後的な全数探索）が")
md.append("   同一の特徴量を選定した場合、その特徴量の重要性が異なる方法論で二重に裏付けられることになる。")
md.append("")
md.append("4. **論文への記述（推奨文）**:  ")
md.append("")
md.append("   > 特徴量選択には再帰的特徴量削減（RFECV; Guyon et al., 2002）を採用し、")
md.append("   > LOO-CV の Balanced Accuracy を基準として最適次元数 $n^*$ を決定した。")
md.append("   > さらに逐次前向き・後向き特徴量選択（SFS）により最終的な特徴量の組み合わせを確定した。")
md.append("   > この手順は，特定の仮説に依存しない客観的な特徴量選択プロセスとして，")
md.append("   > 機械学習を用いた小サンプル研究において広く採用されている手法である。")

report_path = os.path.join(DOC_DIR, "feature_selection_pipeline_results.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"[SAVED] {report_path}")
print("\n✅ 全処理完了!")
