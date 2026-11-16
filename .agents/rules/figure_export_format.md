# 図表出力形式（PDF優先・ベクター画像標準化）ルール

論文執筆および研究報告における図表（グラフ、散布図、ヒートマップ、決定木構造図、棒グラフ等）の出力・保存形式について、以下のルールを厳格に適用すること。

---

## 1. 図表出力形式の標準化（PDF優先原則）

1. **基本出力形式（PDF標準）**:
   - 今後実行・作成するすべての可視化スクリプト（Python / matplotlib / seaborn 等）において、図表は**ベクター形式である PDF（`.pdf`）として出力・保存することを標準（必須）**とする。
   - 論文（LaTeX / Overleaf 等）への掲載時に拡大しても線や文字が劣化（ラスタライズ・ボケ）しない高品質なベクターグラフィックスを保証するため。

2. **Markdown / Webプレビュー用のPNG併録（Dual Output）**:
   - GitHub上での表示やMarkdownレポート（`.md`）内でのインライン画像プレビュー互換性を維持するため、**PDFと同時に同名の高解像度PNG（`dpi=300` 推奨）も併せて出力**する。
   ```python
   # 出力コードの標準パターン
   fig.savefig(out_dir / f"{fig_name}.pdf", bbox_inches="tight")
   fig.savefig(out_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
   ```

---

## 2. 保存先および命名規則

- **論文執筆用マスター**:
  - 保存先: `docs/paper/figures/`
  - 命名規則: `exp{番号}_{説明}.pdf`（および `.png`）
- **週次・進捗報告用**:
  - 保存先: `進捗報告用/YYYYMMDD/figures/`
  - 命名規則: 内容を表すスネークケース（例: `view_count_vs_like_rate.pdf`）

---

## 3. フォント埋め込みとスタイル仕様

- PDF出力時に文字化けやレイアウト崩れを防ぐため、日本語フォントのTrueType埋め込み（`pdf.fonttype: 42`）を設定すること。
```python
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['pdf.fonttype'] = 42  # TrueTypeフォント埋め込み（論文投稿標準）
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False
```

