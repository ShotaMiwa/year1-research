import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

try:
    import japanize_matplotlib
except ImportError:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "googlecolab"))

from evaluation_utils import generate_confusion_matrix_md

# ─── 統合評価データセットの構築 ────────────────────────────

def build_common_dataset() -> pd.DataFrame:
    """
    両モデルのアノテーションCSVを統合し、重複を排除した上で
    各コメントについてBERTとXLM-R双方の予測を補完した統合評価データを返します。
    """
    # 各アノテーションCSV読み込み
    df_bert = pd.read_csv(
        os.path.join(current_dir, "data", "annotations_日本語BERT.csv"),
        encoding="utf-8-sig"
    )
    df_xlmr = pd.read_csv(
        os.path.join(current_dir, "data", "annotations_多言語XLM-R.csv"),
        encoding="utf-8-sig"
    )

    # 不一致CSV読み込み（コメントのみ）
    dis_path = os.path.join(current_dir, "outputs", "run_20260624_071441", "sentiment_disagreements.csv")
    df_dis = pd.read_csv(dis_path, encoding="utf-8-sig")
    df_dis_com = df_dis[df_dis["種別"] == "コメント"].copy()
    df_dis_com["text_key"] = df_dis_com["テキスト"].str.strip()

    # BERT CSV を整形
    df_bert_clean = df_bert[["text", "label", "pred_label", "tier"]].copy()
    df_bert_clean.columns = ["text", "human_label", "bert_pred", "tier"]
    df_bert_clean["text_key"] = df_bert_clean["text"].str.strip()
    df_bert_clean["source"] = "BERT"

    # XLM-R CSV を整形
    df_xlmr_clean = df_xlmr[["text", "label", "pred_label", "tier"]].copy()
    df_xlmr_clean.columns = ["text", "human_label", "xlmr_pred", "tier"]
    df_xlmr_clean["text_key"] = df_xlmr_clean["text"].str.strip()
    df_xlmr_clean["source"] = "XLM-R"

    # 不一致CSVをdictに変換（高速検索用）
    dis_dict = {}
    for _, row in df_dis_com.iterrows():
        dis_dict[row["text_key"]] = {
            "bert": row["日本語BERT_判定"],
            "xlmr": row["多言語XLM-R_判定"]
        }

    # BERT CSV の各行にXLM-Rの予測を補完
    def get_xlmr_pred(row):
        key = row["text_key"]
        if key in dis_dict:
            return dis_dict[key]["xlmr"]   # 不一致CSVに載っている → XLM-Rの予測が異なった
        else:
            return row["bert_pred"]        # 不一致CSVに載っていない → 両モデルが一致

    df_bert_clean["xlmr_pred"] = df_bert_clean.apply(get_xlmr_pred, axis=1)

    # XLM-R CSV の各行にBERTの予測を補完
    def get_bert_pred(row):
        key = row["text_key"]
        if key in dis_dict:
            return dis_dict[key]["bert"]   # 不一致CSVに載っている → BERTの予測が異なった
        else:
            return row["xlmr_pred"]        # 不一致CSVに載っていない → 両モデルが一致

    df_xlmr_clean["bert_pred"] = df_xlmr_clean.apply(get_bert_pred, axis=1)

    # BERT CSV を統合ベースとして使用（全列揃える）
    df_bert_full = df_bert_clean[["text", "text_key", "human_label", "bert_pred", "xlmr_pred", "tier", "source"]]

    # XLM-R CSV は BERT にない行のみ追加（重複排除）
    bert_keys = set(df_bert_clean["text_key"])
    df_xlmr_new = df_xlmr_clean[~df_xlmr_clean["text_key"].isin(bert_keys)].copy()
    df_xlmr_new = df_xlmr_new[["text", "text_key", "human_label", "bert_pred", "xlmr_pred", "tier", "source"]]

    # 結合
    df_common = pd.concat([df_bert_full, df_xlmr_new], ignore_index=True)
    df_common = df_common.drop(columns=["text_key"])

    print(f"統合評価データセット: {len(df_common)} 件")
    print(f"  - 日本語BERTのみのデータ由来: {len(df_bert_full)} 件")
    print(f"  - 多言語XLM-Rのみのデータ由来 (重複除く): {len(df_xlmr_new)} 件")
    return df_common


# ─── 混同行列の画像描画 ────────────────────────────────────

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str],
                          save_path: str, title: str):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            matrix[idx[t], idx[p]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel="True Label", xlabel="Predicted Label")

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            row_sum = matrix[i].sum()
            pct = f"\n({val / row_sum * 100:.1f}%)" if row_sum > 0 else ""
            ax.text(j, i, f"{val}{pct}", ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─── 評価メイン ──────────────────────────────────────────

def run_common_evaluation():
    df = build_common_dataset()

    # CSVとして保存
    csv_out = os.path.join(current_dir, "data", "annotations_common.csv")
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n統合アノテーションCSVを保存しました: {csv_out}")

    classes = ["Positive", "Neutral", "Negative"]
    models = {
        "Japanese BERT":      (df["human_label"].tolist(), df["bert_pred"].tolist()),
        "Multilingual XLM-R": (df["human_label"].tolist(), df["xlmr_pred"].tolist()),
    }

    report_lines = []
    report_lines.append("# 感情分析 共通データセット 比較評価レポート\n\n")
    report_lines.append(f"- **評価サンプル数（共通）**: {len(df)} 件\n")
    report_lines.append(f"  - 日本語BERTアノテーション由来: {(df['source']=='BERT').sum()} 件\n")
    report_lines.append(f"  - 多言語XLM-Rアノテーション由来（重複除く）: {(df['source']=='XLM-R').sum()} 件\n\n")

    # 全体正解率の比較表
    report_lines.append("## ■ モデル間 全体正解率 比較\n\n")
    report_lines.append("| モデル | 全体正解率 (Accuracy) | 正解数/合計 |\n")
    report_lines.append("| :--- | :---: | :---: |\n")
    for model_name, (y_true, y_pred) in models.items():
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        total = len(y_true)
        acc = correct / total * 100
        report_lines.append(f"| {model_name} | {acc:.2f}% | {correct}/{total} |\n")
    report_lines.append("\n---\n\n")

    # 各モデルの詳細レポート
    for model_name, (y_true, y_pred) in models.items():
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        total = len(y_true)
        acc = correct / total * 100

        print("="*70)
        print(f" {model_name}")
        print("="*70)
        print(f"  全体正解率: {acc:.2f}% ({correct}/{total} 件)")

        # 層ごと
        print("  層ごとの正解率:")
        report_lines.append(f"## ■ {model_name}\n\n")
        report_lines.append(f"- **全体正解率**: **{acc:.2f}%** ({correct}/{total} 件)\n\n")
        report_lines.append("### 層（Tier）ごとの正解率\n\n")
        report_lines.append("| 確信度層 | 正解率 | 正解数/合計 |\n")
        report_lines.append("| :--- | :---: | :---: |\n")

        df_eval = df.copy()
        df_eval["y_pred"] = y_pred
        df_eval["correct"] = df_eval["human_label"] == df_eval["y_pred"]
        tiers = sorted(df_eval["tier"].unique())
        for t in tiers:
            td = df_eval[df_eval["tier"] == t]
            tc = td["correct"].sum()
            tt = len(td)
            ta = tc / tt * 100 if tt > 0 else 0.0
            print(f"    {t}: {ta:.2f}% ({tc}/{tt} 件)")
            report_lines.append(f"| {t} | {ta:.2f}% | {tc}/{tt} |\n")
        report_lines.append("\n")

        # 混同行列テキスト
        report_lines.append(generate_confusion_matrix_md(y_true, y_pred, model_name))
        report_lines.append("\n---\n\n")

        # 混同行列画像
        model_tag = "BERT" if "BERT" in model_name else "XLM-R"
        png_path = os.path.join(current_dir, "outputs", f"confusion_matrix_common_{model_tag}.png")
        plot_confusion_matrix(y_true, y_pred, classes, png_path, f"Confusion Matrix - {model_name}\n(Common Dataset, n={total})")
        print(f"  混同行列画像: {png_path}")

    # レポート保存
    report_path = os.path.join(current_dir, "outputs", "evaluation_report_common.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report_lines))
    print(f"\n共通評価レポートを保存しました: {report_path}")


if __name__ == "__main__":
    run_common_evaluation()
