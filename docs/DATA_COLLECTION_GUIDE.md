# Shortsデータ収集＆元配信アライメント 実行手順書

本ドキュメントは、別PC（持ち運び用PCなど）で `@hiroyukike` チャンネルのShorts動画から「元配信アーカイブの該当区間 [T_start, T_end]」および「高評価率（Like Rate）」を全件・無制限で自動収集・照合するための手順書です。

---

## 1. 概要・処理の流れ

1. **Shortsメタデータ**: `@hiroyukike/shorts` タブから抽出済みの全776件（`data/raw/chat/shorts_playlist_all.jsonl`）を走査。
2. **元動画URL抽出**: 各Shortsの概要欄（「元動画」「本編」「アーカイブ」等の記載）から元配信IDを特定。
3. **字幕アライメント**: Shortsと元配信の自動日本語字幕（VTT）を照合し、**フレーズ一致 5箇所以上**の一致区間 `[T_start, T_end]` を特定。
4. **自動逐次保存**: 1件一致するごとに以下に即座に保存・更新されます：
   - `data/processed/verified_alignments.json`（確定JSON）
   - `data/processed/verified_shorts_dataset.csv`（確定CSV）
   - `docs/paper/verification_report.html`（目視確認用レポート）

> [!NOTE]
> **中断・再開に対応しています**。途中で `Ctrl+C` 等で停止しても、次回実行時に処理済み動画は自動スキップして続きから再開します。

---

## 2. 移行先PCでのセットアップ手順

### ステップ 1: リポジトリのクローン・ブランチ切替

```bash
# 作業ディレクトリに移動してクローン（SSHまたはHTTPS）
git clone -b exp-nested-loocv git@github.com:ShotaMiwa/year1-research.git
cd year1-research

# すでにクローン済みの場合は最新を取得
git checkout exp-nested-loocv
git pull origin exp-nested-loocv
```

### ステップ 2: 仮想環境の作成と依存ライブラリのインストール

```bash
# Python仮想環境の作成
python3 -m venv .venv
source .venv/bin/activate

# 必要なライブラリのインストール
pip install yt-dlp pandas
```

---

## 3. データ収集バッチの実行

### ① 通常実行（フォアグラウンドで進捗確認しながら実行）

```bash
source .venv/bin/activate
python src/satisfaction_analysis/data_collection/collect_all_unlimited.py
```

### ② バックグラウンド実行（画面を閉じても動かし続けたい場合）

```bash
source .venv/bin/activate
nohup python -u src/satisfaction_analysis/data_collection/collect_all_unlimited.py > data/raw/chat/collect_unlimited.log 2>&1 &

# 進捗ログをリアルタイムで確認するコマンド
tail -f data/raw/chat/collect_unlimited.log
```

---

## 4. 目視確認（レポート閲覧・判定）のやり方

収集した結果を目視で確認・再生したい場合は、内蔵サーバーを起動します：

```bash
# 検証サーバーの起動（ポート 8765）
python3 src/satisfaction_analysis/data_collection/verification_server.py
```

ブラウザで以下のURLを開きます：
👉 **`http://localhost:8765/verification_report.html`**

- **特徴**:
  - 各Shorts動画と、元配信の該当秒数（埋め込みプレーヤー）が左右に並んで表示されます。
  - 初期状態はすべて「✅ 承認 (OK)」として保存されています。
  - 明らかにズレている動画があれば「❌ 却下 (NG)」を押すか、「⚠️ 区間を微調整」で秒数を直せます。
  - 変更内容は即座に `verified_alignments.json` に同期されます。

---

## 5. 出力ファイル一覧

| ファイルパス | 内容・用途 |
|---|---|
| `data/processed/verified_alignments.json` | 確定したShortsと元配信区間のメタデータ（JSON形式） |
| `data/processed/verified_shorts_dataset.csv` | モデル学習・分析用の確定データセット（CSV形式） |
| `docs/paper/verification_report.html` | ブラウザで目視・再生確認できるインタラクティブHTMLレポート |
| `data/raw/chat/shorts_playlist_all.jsonl` | 収集対象のShorts全776本のメタデータリスト |
