# 機械学習・統計モデルのパラメータ記録および再現性ルール

機械学習（決定木、ランダムフォレスト、SVM、回帰モデル等）や統計分析の実行にあたっては、実験結果の再現性と学術的透明性を担保するため、以下のルールを厳格に適用すること。

---

## 1. モデル実行パラメータのドキュメント記録義務（必須）

モデルを訓練・評価した際は、評価指標（Accuracy、Balanced Accuracy、F1、$R^2$、RMSE等）だけでなく、**実行時に使用した全ハイパーパラメータ設定を必ずドキュメント（進捗報告md、論文用実験記録md）に明記**すること。

### 記録すべき必須パラメータ項目（例）
- **決定木（DecisionTreeClassifier）**:
  - `max_depth`（最大深さ）
  - `min_samples_leaf`（葉ノードの最小サンプル数）
  - `min_samples_split`（分割に必要な最小サンプル数）
  - `class_weight`（不均衡補正: None / 'balanced'）
  - `criterion`（gini / entropy）
  - `random_state`（乱数シード）
- **ランダムフォレスト（RandomForestClassifier / Regressor）**:
  - `n_estimators`（決定木の数）
  - `max_depth`, `min_samples_leaf`, `max_features`（特徴量サンプリング数）
  - `class_weight`, `random_state`
- **特徴選択・交差検証（CV）設定**:
  - 分割手法（LOO-CV, K-Fold, Nested LOO-CV）
  - 逐次特徴選択（SFS）の採否、選択特徴量数（$k$）、選択評価指標（Balanced Accuracy等）

---

## 2. 図表・結果とパラメータの対応付け（一対一対応）

生成される図表（グラフ、決定境界図、木構造可視化、散布図等）および集計表（CSV、Markdown表）について、**「どの図表がどのパラメータ設定の結果なのか」**が一目で分かるようにすること。

1. **表形式データ（CSV・Markdown表）**:
   - 必ず表のカラムまたはモデル名にパラメータ設定（例: `DT (max_depth=2, min_samples_leaf=1)`）を明記すること。
2. **図表（PNGグラフ・プロット等）**:
   - グラフのタイトル、サブタイトル、または凡例に主要パラメータを必ず含めること。
   - キャプションやドキュメント掲載時の説明文にも対応するパラメータ設定を注記すること。
3. **ファイル命名規則**:
   - 図表やテーブルのファイル名にパラメータやモデル種別を反映させること（例: `exp_dt_tree_structure_hm_d2_leaf1.png`）。

---

## 3. パラメータ設定を保持・記録するファイル形式（3層連携標準）

今後実施するすべての機械学習・統計実験において、パラメータ設定は以下の3層ファイル形式を標準として保存・同期し、完全な再現性と検証性を保つこと。

| ファイル形式 | 格納先パス規約 | 記録・保持内容 |
| :--- | :--- | :--- |
| **① Markdown (`.md`)** | `進捗報告用/YYYYMMDD/*.md`<br>`docs/paper/01_experiment_results.md` | **人間可読・発表・論文用**<br>・スライドおよび論文本文の各比較表内に `max_depth`, `min_samples_leaf`, `class_weight` などのパラメータ設定列を明記。<br>・決定木構造図や境界図のキャプション・見出しに対応パラメータを付記。 |
| **② CSV (`.csv`)** | `進捗報告用/YYYYMMDD/tables/*.csv`<br>`docs/paper/tables/*.csv` | **データ集計・検証用**<br>・各行に `Model_Setting`（設定名）と併せて、使用パラメータ、評価指標、選出特徴量を明示して保存。 |
| **③ Python (`.py`)** | `src/satisfaction_analysis/experiments/*.py` | **完全再現用（スクリプトコード）**<br>・実行スクリプト内に `param_grid` などの辞書リストとして全探索パラメータを定義・保持し、ワンコマンドで同一実験が再実行できるようにする。 |

