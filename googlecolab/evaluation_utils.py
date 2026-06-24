import os
import re
import pandas as pd
from typing import List, Dict, Tuple

def parse_annotated_markdown(md_path: str) -> Dict[str, str]:
    """
    手動アノテーションされたMarkdownファイルを読み込み、
    「コメント本文」と「正解ラベル (Positive/Neutral/Negative)」のペアを抽出して返します。
    """
    annotations = {}
    if not os.path.exists(md_path):
        print(f"[evaluation_utils] 警告: アノテーションファイルが見つかりません: {md_path}")
        return annotations

    # 1=Positive, 2=Neutral, 3=Negative
    label_map = {
        "1": "Positive",
        "2": "Neutral",
        "3": "Negative"
    }

    # 行の末尾が「半角スペース + 1or2or3」で終わるものを抽出する正規表現
    # 例: "   > おっおー！こんばんは♪ 1" -> コメント: "おっおー！こんばんは♪", ラベル: "Positive"
    pattern = re.compile(r'^\s*>\s*(.+?)\s+([123])\s*$')

    print(f"[evaluation_utils] アノテーションファイルを解析中: {md_path}")
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    text = match.group(1).strip()
                    label_code = match.group(2)
                    annotations[text] = label_map[label_code]
        print(f"[evaluation_utils] 解析完了。アノテーション数: {len(annotations)} 件")
    except Exception as e:
        print(f"[evaluation_utils] エラー: ファイル解析に失敗しました: {e}")
        
    return annotations


def calculate_metrics(matrix: Dict[str, Dict[str, int]], labels: List[str]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    混同行列の辞書からAccuracy、Precision、Recall、F1-scoreを算出します。
    """
    total = sum(sum(matrix[t][p] for p in labels) for t in labels)
    correct = sum(matrix[lbl][lbl] for lbl in labels)
    
    accuracy = correct / total if total > 0 else 0.0
    
    label_metrics = {}
    for lbl in labels:
        # 実際にそのラベルだった数 (行の合計)
        actual_total = sum(matrix[lbl][p] for p in labels)
        # そのラベルと予測された数 (列の合計)
        pred_total = sum(matrix[t][lbl] for t in labels)
        
        tp = matrix[lbl][lbl]
        
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / actual_total if actual_total > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_metrics[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total
        }
        
    return accuracy, label_metrics


def generate_confusion_matrix_md(y_true: List[str], y_pred: List[str], model_name: str) -> str:
    """
    正解ラベルと予測ラベルから混同行列を計算し、Markdown形式のレポートを作成します。
    """
    labels = ["Positive", "Neutral", "Negative"]
    
    # 混同行列の初期化 (行:正解, 列:予測)
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
            
    # 指標の計算
    accuracy, metrics = calculate_metrics(matrix, labels)
    total_samples = len(y_true)

    # Markdownテキストの生成
    md = []
    md.append(f"# 感情分析 評価レポート (Confusion Matrix) - {model_name}\n\n")
    md.append(f"アノテーションデータ（正解）とモデルによる予測結果の比較レポートです。\n")
    md.append(f"- 評価サンプル数: {total_samples} 件\n")
    md.append(f"- 全体正解率 (Accuracy): {accuracy * 100:.2f}%\n\n")
    
    # 1. 混同行列テーブル
    md.append("## ■ 混同行列 (Confusion Matrix)\n\n")
    md.append("| 正解 \\ 予測 | Positive | Neutral | Negative | 合計 (正解) |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: |\n")
    
    for t in labels:
        pos_val = matrix[t]["Positive"]
        neu_val = matrix[t]["Neutral"]
        neg_val = matrix[t]["Negative"]
        row_sum = sum(matrix[t].values())
        
        # 斜め成分（一致）を太字にする
        p_str = f"**{pos_val}**" if t == "Positive" else str(pos_val)
        neu_str = f"**{neu_val}**" if t == "Neutral" else str(neu_val)
        neg_str = f"**{neg_val}**" if t == "Negative" else str(neg_val)
        
        md.append(f"| **{t}** | {p_str} | {neu_str} | {neg_str} | {row_sum} |\n")
        
    col_sums = [sum(matrix[t][p] for t in labels) for p in labels]
    md.append(f"| **合計 (予測)** | {col_sums[0]} | {col_sums[1]} | {col_sums[2]} | {total_samples} |\n\n")
    
    # 2. 分類評価指標テーブル
    md.append("## ■ 分類評価指標 (Evaluation Metrics)\n\n")
    md.append("| 感情ラベル | 適合率 (Precision) | 再現率 (Recall) | F1-Score | サンプル数 (Support) |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: |\n")
    
    for lbl in labels:
        m = metrics[lbl]
        md.append(f"| **{lbl}** | {m['precision'] * 100:.2f}% | {m['recall'] * 100:.2f}% | {m['f1']:.4f} | {m['support']} |\n")
        
    md.append("\n---\n")
    md.append("### 【用語解説】\n")
    md.append("- **適合率 (Precision)**: モデルが「その感情である」と予測したうち、実際に正しかった割合（誤検知の少なさ）。\n")
    md.append("- **再現率 (Recall)**: 実際のデータの中に存在するその感情を、モデルがどれだけ漏らさず検出できたかの割合（見落としの少なさ）。\n")
    md.append("- **F1-Score**: 適合率と再現率の調和平均。バランスの良さを示します（最大1.0）。\n")
    
    return "".join(md)


def export_markdown_annotations_to_csv(md_path: str, csv_path: str) -> pd.DataFrame:
    """
    Markdownファイルからアノテーションデータを解析し、
    CSVファイル (text, label) として保存（または上書き）します。
    """
    annotations = parse_annotated_markdown(md_path)
    if not annotations:
        print(f"[evaluation_utils] アノテーションデータが空のため、CSVの書き出しはスキップします: {md_path}")
        return pd.DataFrame(columns=["text", "label"])
        
    df = pd.DataFrame(list(annotations.items()), columns=["text", "label"])
    try:
        # ディレクトリを作成
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # UTF-8-SIGで日本語文字化けを防ぎつつ保存
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[evaluation_utils] アノテーションCSVを書き出しました: {csv_path} ({len(df)} 件)")
    except Exception as e:
        print(f"[evaluation_utils] エラー: CSV書き出しに失敗しました: {e}")
        
    return df


def load_annotations_from_csv(csv_path: str) -> Dict[str, str]:
    """
    保存されたアノテーションCSVファイルを読み込み、
    「コメント本文」と「正解ラベル (Positive/Neutral/Negative)」のペアの辞書を返します。
    """
    annotations = {}
    if not os.path.exists(csv_path):
        print(f"[evaluation_utils] 警告: アノテーションCSVファイルが見つかりません: {csv_path}")
        return annotations
        
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # 列が存在するか確認
        if "text" in df.columns and "label" in df.columns:
            # NaNを取り除いて辞書化
            df = df.dropna(subset=["text", "label"])
            annotations = dict(zip(df["text"].astype(str), df["label"].astype(str)))
            print(f"[evaluation_utils] CSVから {len(annotations)} 件のアノテーションを読み込みました: {csv_path}")
        else:
            print(f"[evaluation_utils] エラー: CSVのフォーマットが正しくありません (text, label列が必要): {csv_path}")
    except Exception as e:
        print(f"[evaluation_utils] エラー: CSVの読み込みに失敗しました: {e}")
        
    return annotations

