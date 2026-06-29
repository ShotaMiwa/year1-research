import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

try:
    import japanize_matplotlib
except ImportError:
    pass

# パスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
# googlecolab ディレクトリの evaluation_utils をインポートするため
sys.path.append(os.path.join(current_dir, "googlecolab"))

from evaluation_utils import export_markdown_annotations_to_csv, generate_confusion_matrix_md

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str], save_path: str, title: str):
    """
    混同行列を論文風の美しいヒートマップ画像として出力・保存します。
    """
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            matrix[class_to_idx[t], class_to_idx[p]] += 1
            
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 軸の設定
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
           
    # テキストの配置
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            # 割合も計算する (その行の合計に対する割合)
            row_sum = matrix[i].sum()
            percent_str = f"\n({val / row_sum * 100:.1f}%)" if row_sum > 0 else ""
            ax.text(j, i, f"{val}{percent_str}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=12)
                    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_accuracy_metrics(df: pd.DataFrame) -> Dict:
    """
    拡張CSVのDataFrameから全体の正解率および各層(tier)ごとの正解率を計算します。
    """
    results = {}
    if df.empty:
        return results

    # 全体正解率
    correct = (df['label'] == df['pred_label']).sum()
    total = len(df)
    results['overall_accuracy'] = correct / total if total > 0 else 0.0
    results['overall_total'] = total

    # 層ごとの正解率
    results['tiers'] = {}
    # 層の順序をソート（層1, 層2...）
    tiers = sorted(df['tier'].unique())
    for t in tiers:
        tier_df = df[df['tier'] == t]
        t_correct = (tier_df['label'] == tier_df['pred_label']).sum()
        t_total = len(tier_df)
        results['tiers'][t] = {
            'accuracy': t_correct / t_total if t_total > 0 else 0.0,
            'correct': t_correct,
            'total': t_total
        }
    return results

def run_evaluation():
    models_config = {
        "日本語BERT": {
            "md_path": os.path.join(current_dir, "data", "BERT-日本語.md"),
            "csv_path": os.path.join(current_dir, "data", "annotations_日本語BERT.csv"),
            "report_path": os.path.join(current_dir, "outputs", "evaluation_report_日本語BERT.md"),
            "png_path": os.path.join(current_dir, "outputs", "confusion_matrix_日本語BERT.png")
        },
        "多言語XLM-R": {
            "md_path": os.path.join(current_dir, "data", "sentiment_samples_多言語XLM-R.md"),
            "csv_path": os.path.join(current_dir, "data", "annotations_多言語XLM-R.csv"),
            "report_path": os.path.join(current_dir, "outputs", "evaluation_report_多言語XLM-R.md"),
            "png_path": os.path.join(current_dir, "outputs", "confusion_matrix_多言語XLM-R.png")
        }
    }

    for model_name, paths in models_config.items():
        print("="*80)
        print(f" モデル評価: {model_name}")
        print("="*80)

        # Markdownが存在する場合はまずCSVを更新
        if os.path.exists(paths["md_path"]):
            print(f"[1] Markdownファイルからアノテーションデータを解析し、CSVを更新中...")
            df = export_markdown_annotations_to_csv(paths["md_path"], paths["csv_path"])
        else:
            print(f"[1] アノテーションMarkdownファイルが見つかりません。既存のCSVを読み込みます: {paths['csv_path']}")
            if os.path.exists(paths["csv_path"]):
                try:
                    df = pd.read_csv(paths["csv_path"], encoding="utf-8-sig")
                except Exception as e:
                    print(f"エラー: CSVのロードに失敗しました: {e}")
                    df = pd.DataFrame()
            else:
                print(f"警告: CSVファイルも見つかりません。スキップします。\n")
                continue

        if df.empty:
            print("データが空のため、評価をスキップします。\n")
            continue

        # 必要な列が存在するかチェック
        required_cols = ["text", "label", "pred_label", "tier"]
        if not all(col in df.columns for col in required_cols):
            print(f"警告: CSVに必要な列 {required_cols} が揃っていません。")
            print("Markdownから最新のCSVを再書き出しするか、CSVの形式を確認してください。\n")
            continue

        # 正解率などのメトリクス計算
        metrics = calculate_accuracy_metrics(df)

        # ターミナル表示
        print(f"\n[2] 評価結果:")
        overall_correct = (df['label'] == df['pred_label']).sum()
        print(f"  ● 全体正解率 (Accuracy): {metrics['overall_accuracy'] * 100:.2f}% ({overall_correct}/{metrics['overall_total']} 件)")
        print("\n  ● 層(Tier)ごとの正解率:")
        for t, data in metrics['tiers'].items():
            print(f"    - {t}: {data['accuracy'] * 100:.2f}% ({data['correct']}/{data['total']} 件)")

        # 混合行列のMarkdownテキストを生成
        y_true = df["label"].tolist()
        y_pred = df["pred_label"].tolist()
        confusion_md = generate_confusion_matrix_md(y_true, y_pred, model_name)

        # 混合行列を画像としてプロット・保存
        classes = ["Positive", "Neutral", "Negative"]
        model_name_en = "Japanese BERT" if model_name == "日本語BERT" else "Multilingual XLM-R"
        plot_confusion_matrix(y_true, y_pred, classes, paths["png_path"], f"Confusion Matrix - {model_name_en}")
        print(f"  ● 混合行列画像を保存しました: {paths['png_path']}")

        # 評価レポート用Markdownの構築
        overall_correct = (df['label'] == df['pred_label']).sum()
        report_content = []
        report_content.append(f"# 感情分析 評価詳細レポート - {model_name}\n\n")
        report_content.append("## ■ 正解率サマリー\n\n")
        report_content.append(f"- **全体正解率 (Accuracy)**: **{metrics['overall_accuracy'] * 100:.2f}%** ({overall_correct}/{metrics['overall_total']} 件)\n\n")
        
        report_content.append("### ● 層（Tier）ごとの正解率\n\n")
        report_content.append("| 確信度層 | 正解率 (Accuracy) | 正解数/合計 (サンプル数) |\n")
        report_content.append("| :--- | :---: | :---: |\n")
        for t, data in metrics['tiers'].items():
            report_content.append(f"| {t} | {data['accuracy'] * 100:.2f}% | {data['correct']}/{data['total']} |\n")
        report_content.append("\n---\n\n")
        
        # 混合行列と評価指標
        report_content.append(confusion_md)

        # レポートファイル書き出し
        try:
            with open(paths["report_path"], "w", encoding="utf-8") as f:
                f.write("".join(report_content))
            print(f"\n[3] 評価レポートMarkdownを保存しました: {paths['report_path']}")
        except Exception as e:
            print(f"エラー: レポートファイルの保存に失敗しました: {e}")
        
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    run_evaluation()
