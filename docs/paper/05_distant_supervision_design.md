# Distant Supervision 実装設計書
## YouTube公開シグナルを用いた疑似ラベル付き大規模データ拡張

---

## 0. 研究文脈の再確認と問題の正確な定義

### 元の研究目標
> **「ライブ配信アーカイブの各セグメントに対して、視聴満足度（5因子）が高いか低いかを、公開データ（チャット・字幕・ヒートマップ）から自動判別する」**

### Distant Supervision で解く問題
- **核心的な問い**: 「アンケートで測定した"主観的満足度"」と「YouTube公開シグナル」は本当に相関するか？
- **鍵となる仮定（Distant Supervision仮定）**: 
  > 多数の一般ユーザーが示す集合的な行動（再視聴・高評価・盛り上がったコメント）は、実験参加者10名の主観満足度スコアの**集合的な近似値**である

---

## 1. プロキシラベルの選定と正当性

### YouTube公開シグナル → 満足度因子のマッピング

| YouTube公開シグナル | 取得方法 | 対応する満足度因子 | 正当性・先行研究 |
|---|---|---|---|
| **再生維持率ヒートマップ** (Most Replayed) | yt-dlp / DOM解析 | 総合満足度・娯楽性・リラックス | **既に実験で有効性確認済み** (hm_mean, hm_max, hm_volatility の3特徴量) |
| **コメント感情極性** (BERT) | yt-dlp → 既存BERTパイプライン | 総合満足度・娯楽性 | **既に実験で有効性確認済み** (sentiment_polarity, $r=+0.431$) |
| **コメント密度** (件/秒) | yt-dlp | 総合満足度・娯楽性 | **既に特徴量として活用済み** |
| **高評価率** (likes/views) | YouTube Data API v3 | 総合満足度（動画全体） | Hofmann et al. (2014): 高評価はpositive afferent signal |
| **草率・質問率** | yt-dlp → 既存正規表現パイプライン | 娯楽性・情報性 | **既に実験で確認済み** |

> **重要な洞察**: 実測データ23サンプルで有効性が確認された特徴量は、まさにYouTube公開データから算出できるものと一致している。これがDistant Supervisionを採用する最大の理由。

---

## 2. プロキシラベルの設計（最重要）

### 2-1. セグメント単位のコンポジットスコア算出

各セグメント $s$ に対して、以下の式でプロキシ満足度スコア $\tilde{Y}(s)$ を算出する。

$$
\tilde{Y}(s) = w_1 \cdot \text{hm\_mean}(s) + w_2 \cdot \text{hm\_max}(s) + w_3 \cdot \text{sentiment\_polarity}(s) + w_4 \cdot \text{comment\_rate\_norm}(s)
$$

重み $(w_1, w_2, w_3, w_4)$ は**実測23サンプルのアンケートスコアとの相関から決定**する：

```python
# 実測データで各シグナルとアンケートY1(総合満足度)の相関を算出
corr_hm_mean    = pearsonr(X['hm_mean'], Y['Target_Satisfaction'])[0]
corr_hm_max     = pearsonr(X['hm_max'], Y['Target_Satisfaction'])[0]
corr_sentiment  = pearsonr(X['sentiment_polarity'], Y['Target_Satisfaction'])[0]
corr_comment    = pearsonr(X['comment_rate'], Y['Target_Satisfaction'])[0]

# 相関係数の絶対値を重みに使用（正規化）
weights = np.abs([corr_hm_mean, corr_hm_max, corr_sentiment, corr_comment])
weights /= weights.sum()
```

### 2-2. バイナリ化（高満足度 vs 低満足度）

実測データのアンケート中央値を閾値として使用:
```python
# 実測23サンプルの中央値を閾値として使用
threshold = np.median(Y_real['Target_Satisfaction'])
pseudo_label = (Y_proxy >= threshold_proxy).astype(int)
# 0: 低満足度セグメント, 1: 高満足度セグメント
```

---

## 3. データ収集パイプライン

### Step 1: 対象動画の選定方針

```
推奨: 元の実験で使用した動画と同一配信者・同ジャンルのアーカイブ
理由: ドメインシフト（特徴量分布のズレ）を最小化するため

収集規模:
- 最低目標: 20動画 × 平均10セグメント = 200セグメント（実測の約9倍）
- 理想目標: 50動画 × 平均10セグメント = 500セグメント
```

### Step 2: yt-dlp によるデータ取得

```bash
# チャットログ付きダウンロード（字幕・メタデータ込み）
yt-dlp \
  --write-auto-subs \        # 自動字幕（VTT形式）
  --write-subs \             # 手動字幕（存在する場合）
  --sub-lang ja \            # 日本語のみ
  --write-thumbnail \        # サムネイル
  --write-info-json \        # 動画メタデータ（高評価数・再生数等）
  --skip-download \          # 動画本体はDLしない（容量節約）
  "[VIDEO_URL]"

# チャットログ専用（リプレイチャット）
yt-dlp \
  --get-comments \           # コメント取得
  "[VIDEO_URL]"
```

**注意**: YouTubeの利用規約を遵守し、研究目的での利用であることを明記。

### Step 3: 再生維持率ヒートマップの取得

```python
# Most Replayed データは公開API非公開 → yt-dlp の heatmap 機能を使用
import yt_dlp

with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
    info = ydl.extract_info(url, download=False)
    heatmap = info.get('heatmap', [])
    # [{'start_time': 0.0, 'end_time': 5.0, 'value': 0.123}, ...]
```

### Step 4: 高評価率の取得（YouTube Data API v3）

```python
from googleapiclient.discovery import build

youtube = build('youtube', 'v3', developerKey=API_KEY)
response = youtube.videos().list(
    part='statistics',
    id=video_id
).execute()

likes = int(response['items'][0]['statistics']['likeCount'])
views = int(response['items'][0]['statistics']['viewCount'])
like_ratio = likes / views  # 高評価率（総合満足度の代理指標）
```

---

## 4. セグメント分割戦略

### 自動セグメント分割（トピック分割との統合）

```python
# 既存のsrc/topic_segmentation/ を活用
# → 字幕テキストのトピック境界でセグメント分割

# 推奨: 実験セグメントと同様の長さ分布に合わせる
# 実測データの長さ分布: 64秒〜334秒（平均 ≈ 145秒）
TARGET_SEGMENT_RANGE = (60, 360)  # 秒
```

---

## 5. モデル学習戦略（2段階アプローチ）

```
[Stage 1] 事前学習（Pre-training）
    疑似ラベルデータ (N≈200〜500セグメント)
    → 同じ13特徴量を使って ElasticNet / SVR を学習
    → 疑似データでのモデル重みを初期化

[Stage 2] 微調整（Fine-tuning / 重み付き学習）
    実測アンケートデータ (N=23) を高重みで再学習
    → 実測データの重みを疑似データの10〜20倍に設定
```

```python
from sklearn.linear_model import ElasticNet
import numpy as np

# 実測データに大きな重みをつける
sample_weights = np.concatenate([
    np.ones(len(X_pseudo)),          # 疑似データ: 重み=1
    np.full(len(X_real), 15.0)       # 実測データ: 重み=15 (= N_pseudo/N_real)
])

X_all = np.vstack([X_pseudo, X_real])
y_all = np.concatenate([y_pseudo, y_real])

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_all, y_all, sample_weight=sample_weights)
```

---

## 6. 評価設計（厳密なリーク防止）

```
[評価の原則]
実測23サンプルは「テストセット」として完全に隔離。
疑似データは学習のみに使用。

推奨評価プロトコル:
  - 外側ループ: LOO-CV on 実測23サンプル（これが最終評価）
  - 内側ループ: 疑似データ + 残り22実測サンプルで学習
  - テスト: ホールドアウトした1実測サンプルで評価

比較実験（アブレーションスタディ）:
  1. 実測23のみ (ベースライン)
  2. 疑似データのみ事前学習
  3. 疑似 + 実測（提案手法）
```

---

## 7. 論文記述における位置づけ

### 「弱教師あり学習」としての正式な定義

> **"We adopt a distant supervision approach (Mintz et al., 2009), using YouTube engagement signals (replay heatmaps, comment sentiment, like ratio) as proxy labels for viewer satisfaction. The proxy labels are calibrated using the 23 ground-truth survey segments to learn the mapping between public signals and subjective satisfaction scores."**

### 仮定の明示と限界の記述

- **明示すべき仮定**: 「集合的ユーザー行動 ≈ 実験参加者の主観満足度」の仮定は完全には成立しない
- **正当化**: 実測23サンプルで公開シグナルと満足度の相関を確認済み（$r \geq 0.40$）
- **限界**: 疑似ラベルのノイズが学習に影響する可能性 → 実測データへの高重みで軽減

---

## 8. 実装タスクリスト

### 着手順序

```
[Week 1]
□ 1. yt-dlp で同配信者アーカイブ動画10本のチャット・字幕・ヒートマップ取得
   → src/data_creaters/distant_supervision/01_crawl_youtube_data.py

□ 2. 既存の特徴量抽出パイプラインを任意動画に適用できるよう汎化
   → src/data_creaters/distant_supervision/02_extract_features.py

□ 3. プロキシスコア算出（実測データの相関から重み決定）
   → src/data_creaters/distant_supervision/03_compute_proxy_labels.py

[Week 2]
□ 4. 2段階学習（事前学習 + 微調整）の実装・LOO-CV評価
   → src/satisfaction_analysis/experiments/run_experiment_distant_supervision.py

□ 5. アブレーションスタディ（疑似のみ / 実測のみ / 両方）
   → 論文のTable用に比較結果を出力
```

---

## 9. リスクと対策

| リスク | 深刻度 | 対策 |
|---|---|---|
| yt-dlpが仕様変更でヒートマップ取得不可になる | 中 | 複数取得方法を用意（DOM直接解析のバックアップ） |
| 疑似ラベルのノイズが大きく精度が下がる | 高 | 実測データへの高重み付け / ノイズ耐性のある手法（Lasso）を優先 |
| 対象配信者以外の動画を使うとドメインシフト | 中 | 同配信者動画のみに限定（拡張は後続研究） |
| YouTube Data API の無料枠制限 | 低 | 1日10,000ユニット無料 → 50動画程度なら十分 |

---

## 10. 参考文献

- Mintz et al. (2009). *Distant supervision for relation extraction without labeled data.* ACL. → Distant Supervision の原典
- Davidson et al. (2020). *Measuring YouTube Audience Engagement.* CHI. → YouTube行動シグナルと満足度の関連性
- Zhang et al. (2023). *Survey on pseudo-labeling methods.* → 疑似ラベル手法のサーベイ
