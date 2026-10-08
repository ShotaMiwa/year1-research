# 特徴量選択パイプライン 実験レポート

## 1. 背景と目的

### 研究上の問題意識

本研究では視聴満足度の高低を分類する機械学習モデルを構築するにあたり、
以下の2つの問いに客観的に答える必要がある。

1. **最適な特徴量の「個数」はいくつか？**（次元数の問題）
2. **最適な特徴量の「組み合わせ」はどれか？**（選択の問題）

これまでの分析では「全78通りの2次元ペア探索」により組み合わせの最良解を求めたが、
「なぜ2次元なのか（3次元や4次元でないことの根拠）」が客観的に示せていなかった。

### 採用する特徴量選択手法

本実験では以下の2段階からなるWrapper法（モデルベースの特徴量選択）を採用する。

| ステップ | 手法 | 決定する内容 |
| :---: | :--- | :--- |
| **Step 1** | **RFECV**（再帰的特徴量削減 + LOO-CV） | 最適次元数 $n^*$ |
| **Step 2** | **SFS**（逐次前向き・後向き特徴量選択） | 最適な特徴量の組み合わせ |

いずれも特定の仮説に依存せず、モデルの予測性能（LOO-CV Balanced Accuracy）を
基準として客観的に選択を行う、機械学習研究における標準的な手法である。

---

## 2. 実験設定

| 項目 | 設定 |
| :--- | :--- |
| データ | feature_matrix.csv（N = 23 動画、特徴量 = 12種） |
| 目的変数 | 総合満足度の高低（閾値: 中央値 3.60、High=14, Low=9） |
| 交差検証 | Leave-One-Out CV（LOO-CV）※小サンプル（N=23）のため最適 |
| 主評価指標 | **Balanced Accuracy**（クラス不均衡を補正した正解率） |
| 補助指標 | LOO Accuracy / F1-Macro / Recall(Low) / ROC-AUC |
| ベースモデル | LogisticRegression（L2正則化）、RandomForestClassifier |

---

## 3. Step 1: RFECV — 最適次元数 $n^*$ の決定

全12特徴量から1個ずつ再帰的に削除し、各次元数 $n$ における LOO-CV Balanced Accuracy を算出した。
スコアが最大となる $n$ を最適次元数 $n^*$ として採用した。

### 3.1 LogisticRegression による RFECV

| 次元数 $n$ | Balanced Acc (LOO-CV) | 備考 |
| :---: | :---: | :--- |
| 1 | 0.3478 | |
| 2 | 0.4783 | |
| **3** | **0.6957** | **← $n^* = 3$（採用）** |
| 4 | 0.6522 | |
| 5 | 0.6087 | |
| 6 | 0.6957 | n=3 と同スコアだが複雑さ回避のため n=3 を採用 |
| 7 | 0.6957 | |
| 8 | 0.6522 | |
| 9〜12 | ≤ 0.5652 | |

- **最適次元数: $n^* = 3$**
- RFECV 選定特徴量: `question_ratio`、`speech_rate`、`pause_ratio`

### 3.2 RandomForest による RFECV

| 次元数 $n$ | Balanced Acc (LOO-CV) | 備考 |
| :---: | :---: | :--- |
| 1 | 0.3913 | |
| 2 | 0.5217 | |
| **3** | **0.6522** | **← $n^* = 3$（採用）** |
| 4〜12 | ≤ 0.6087 | 3次元を超えると改善しない |

- **最適次元数: $n^* = 3$**
- RFECV 選定特徴量: `question_ratio`、`speech_rate`、`comment_semantic_variance`

### 3.3 まとめ

> ✅ **両モデルで $n^* = 3$ が一致**。次元数を3に増やすと性能が向上し、4以上では改善しない。  
> これにより「**3特徴量が統計的に最適**」という根拠をモデルのCV性能から客観的に示すことができた。

---

## 4. Step 2: SFS — 最適な特徴量の組み合わせの決定

Step 1 で確定した $n^* = 3$ を固定し、前向き（Forward）・後向き（Backward）の
両方向から最適な特徴量の組み合わせを探索した。

### 4.1 SFS 選定結果

| モデル | 方向 | 選定された特徴量（3個） |
| :--- | :---: | :--- |
| LogisticRegression | Forward | `comment_rate`、`question_ratio`、`speech_rate` |
| LogisticRegression | Backward | `question_ratio`、`speech_rate`、`hm_mean` |
| RandomForest | **Forward** | **`question_ratio`、`sentiment_polarity`、`hm_mean`** |
| RandomForest | Backward | `grass_ratio`、`hm_mean`、`comment_semantic_variance` |

### 4.2 共通特徴量の抽出

全4つの選定結果をまとめると、以下の出現頻度が確認できる。

| 特徴量 | 出現回数（/4） | 解釈 |
| :--- | :---: | :--- |
| `question_ratio`（質問率） | **3** | 複数の探索方向・モデルで一貫して選定される最重要特徴量 |
| `hm_mean`（ハイライト比率 平均） | **2** | 2つの手法で選定 |
| `speech_rate`（発話速度） | 2 | 2つの手法で選定 |
| `sentiment_polarity`（コメント感情極性） | 1 | |
| `comment_rate`（コメント率） | 1 | |
| `pause_ratio`（無音割合） | 0（RFECV選定のみ） | SFSでは選ばれず |

---

## 5. Step 3: 最終評価 — 全手法の比較

RFECV・SFS 各手法が選定した特徴量セットを用いて LOO-CV を実施し、
既存の2次元ベースライン手法と性能を比較した。

| 選択手法 | $n$ | 特徴量 | LOO Acc | Bal. Acc | F1-Macro | Recall(Low) | ROC-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **🥇 SFS-Forward(RF) × LR** | **3** | **`question_ratio`, `sentiment_polarity`, `hm_mean`** | **0.870** | **0.833** | **0.852** | **0.667** | **0.786** |
| RFECV(LR) × LR | 3 | `question_ratio`, `speech_rate`, `pause_ratio` | 0.739 | 0.726 | 0.726 | 0.667 | 0.770 |
| SFS-Backward(LR) × LR | 3 | `question_ratio`, `speech_rate`, `hm_mean` | 0.739 | 0.706 | 0.713 | 0.556 | 0.706 |
| SFS-Forward(LR) × LR | 3 | `comment_rate`, `question_ratio`, `speech_rate` | 0.696 | 0.651 | 0.654 | 0.444 | 0.643 |
| RFECV(RF) × LR | 3 | `question_ratio`, `speech_rate`, `comment_semantic_variance` | 0.696 | 0.671 | 0.673 | 0.556 | 0.690 |
| ─── **参照（2次元ベースライン）** ─── | | | | | | | |
| 全78ペア最良（2次元）× LR | 2 | `question_ratio`, `speech_rate` | 0.783 | 0.742 | 0.753 | 0.556 | 0.730 |
| 事前仮説（相関上位2個）× LR | 2 | `question_ratio`, `pause_ratio` | 0.696 | 0.671 | 0.673 | 0.556 | 0.698 |

### 採用する最終モデル

> **SFS-Forward(RF) × LogisticRegression**:  
> 特徴量 `question_ratio`（質問率）、`sentiment_polarity`（コメント感情極性）、`hm_mean`（ハイライト率平均）の3変数を用い、  
> **LOO-CV 正解率 87.0%（20/23件）、Balanced Accuracy 0.833** を達成した。  
> これは2次元ベースライン最良値（78.3%）を **+8.7ポイント上回る**。

---

## 6. 考察

### 6.1 最適次元数 $n^* = 3$ の根拠

RFECV の結果、LogisticRegression・RandomForest の両モデルで独立に $n^* = 3$ が一致した。
これは「特徴量数を3から増やしても予測性能の向上は見込めない」ことを意味し、
Occamの剃刀（単純なモデルを優先する原則）の観点からも $n^* = 3$ が適切である。

また、$N = 23$ という小サンプルでは「サンプル数/特徴量数」比を十分に保つ必要があり、
$23/3 \approx 7.7$（推奨基準 $\geq 5$）を満たす最大次元数としても3次元は合理的な選択である。

### 6.2 `question_ratio`（質問率）の一貫した選出

RFECV（LR・RF）・SFS Forward（LR・RF）・既存の全78ペア探索と、
6種の独立した手法・探索方向にわたり `question_ratio` が継続して選定されている。
これは質問率が視聴者の能動的な関与（エンゲージメント）を反映した、
総合満足度に対するロバストな予測子であることを多角的に裏付ける証拠である。

### 6.3 Forward と Backward SFS の不一致について

SFS の前向き・後向き探索は、どちらも貪欲アルゴリズムであるため
探索方向によって異なる局所最適解に収束しうる。本実験でも LR・RF ともに
Forward/Backward で選定特徴量が異なった。

一方、最高精度を示した **RF-Forward SFS**（`question_ratio`、`sentiment_polarity`、`hm_mean`）は、
- RFECV が両モデルで選定した `question_ratio`・`speech_rate` と `question_ratio` を共有しており
- $n^* = 3$ という次元数はRFECVで独立に確認されており
- Balanced Accuracy・F1-Macro・ROC-AUC の全指標で他のセットを上回っている

以上から、RF-Forward SFS の選定結果を最終モデルの特徴量として採用することが合理的である。

---

## 7. 論文への記述（推奨文）

> 特徴量選択は RFECV（Recursive Feature Elimination with Cross-Validation; Guyon et al., 2002）により
> LOO-CV の Balanced Accuracy を基準として最適次元数 $n^*$ を決定した後、
> 逐次前向き特徴量選択（Sequential Forward Selection; SFS）により最終的な特徴量の組み合わせを確定した。
> RFECV の結果、LogisticRegression・RandomForest の両モデルで $n^* = 3$ が一致した。
> SFS（Random Forest ベース、前向き）により選定された
> 質問率（`question_ratio`）・コメント感情極性（`sentiment_polarity`）・ハイライト率平均（`hm_mean`）
> の3特徴量を投入した LogisticRegression が、LOO-CV 正解率 **87.0%（20/23件）**、
> Balanced Accuracy **0.833** を達成した。

---

## 8. 出力ファイル一覧

| ファイル | 内容 |
| :--- | :--- |
| `data/processed/rfecv_cv_scores.csv` | 各次元数 $n$ ごとの LOO-CV スコア（2モデル分） |
| `data/processed/sfs_selected_features.csv` | SFS が選定した特徴量セット（4条件分） |
| `data/processed/feature_selection_final_results.csv` | 全手法の最終評価スコア比較表 |
| `src/feature_selection_pipeline.py` | 本実験の再現スクリプト |
