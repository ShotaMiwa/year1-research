import os
import sys
import pandas as pd
from typing import Dict, List

# パスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
# googlecolab ディレクトリの evaluation_utils をインポートするため
sys.path.append(os.path.join(current_dir, "googlecolab"))

from evaluation_utils import export_markdown_annotations_to_csv, generate_confusion_matrix_md

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
            'total': t_total
        }
    return results

def run_evaluation():
    models_config = {
        "日本語BERT": {
            "md_path": os.path.join(current_dir, "data", "BERT-日本語.md"),
            "csv_path": os.path.join(current_dir, "data", "annotations_日本語BERT.csv"),
            "report_path": os.path.join(current_dir, "data", "evaluation_report_日本語BERT.md")
        },
        "多言語XLM-R": {
            "md_path": os.path.join(current_dir, "data", "sentiment_samples_多言語XLM-R.md"),
            "csv_path": os.path.join(current_dir, "data", "annotations_多言語XLM-R.csv"),
            "report_path": os.path.join(current_dir, "data", "evaluation_report_多言語XLM-R.md")
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
        print(f"  ● 全体サンプル数: {metrics['overall_total']} 件")
        print(f"  ● 全体正解率 (Accuracy): {metrics['overall_accuracy'] * 100:.2f}%")
        print("\n  ● 層(Tier)ごとの正解率:")
        for t, data in metrics['tiers'].items():
            print(f"    - {t}: {data['accuracy'] * 100:.2f}% (サンプル数: {data['total']} 件)")

        # 混合行列のMarkdownテキストを生成
        y_true = df["label"].tolist()
        y_pred = df["pred_label"].tolist()
        confusion_md = generate_confusion_matrix_md(y_true, y_pred, model_name)

        # 評価レポート用Markdownの構築
        report_content = []
        report_content.append(f"# 感情分析 評価詳細レポート - {model_name}\n\n")
        report_content.append("## ■ 正解率サマリー\n\n")
        report_content.append(f"- **全体サンプル数**: {metrics['overall_total']} 件\n")
        report_content.append(f"- **全体正解率 (Accuracy)**: **{metrics['overall_accuracy'] * 100:.2f}%**\n\n")
        
        report_content.append("### ● 層（Tier）ごとの正解率\n\n")
        report_content.append("| 確信度層 | 正解率 (Accuracy) | サンプル数 |\n")
        report_content.append("| :--- | :---: | :---: |\n")
        for t, data in metrics['tiers'].items():
            report_content.append(f"| {t} | {data['accuracy'] * 100:.2f}% | {data['total']} |\n")
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
