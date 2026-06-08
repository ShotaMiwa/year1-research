import os
import sys
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Colab環境でのインポートエラーを防ぐため、パス調整用のコードを挿入
try:
    # ローカル実行時
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
    simplified_dir = os.path.abspath(os.path.join(src_dir, "data_creaters", "簡易化版"))
    sys.path.append(src_dir)
    sys.path.append(simplified_dir)
except NameError:
    # ColabのNotebook上でセルとして実行する場合
    # ユーザーが適宜 `/content/year1/src/data_creaters/簡易化版` などのパスを追加できるようにします
    pass

# プロジェクト内のデータ取得モジュールからインポートを試みる
# インポートに失敗した場合はエラーメッセージを表示し、必要な関数を案内
try:
    from common_transcript_processing import get_raw_transcript, basic_transcript_processing
    from test_window import get_comments, timestamp_to_seconds
except ImportError as e:
    print("Warning: プロジェクトのインポートパスを設定してください。")
    print("Google Colabでは以下を実行してからこのスクリプトをインポートしてください:")
    print("!git clone <repository_url>")
    print("sys.path.append('/content/year1/src/data_creaters/簡易化版')")
    raise e

# 感情分析ライブラリのロード
try:
    from transformers import pipeline
except ImportError:
    print("transformers がインストールされていません。!pip install transformers sentencepiece を実行してください。")
    raise


def ts_to_sec(ts: str) -> float:
    """'H:MM:SS' または 'M:SS' を秒数に変換"""
    parts = ts.strip().split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        return float(parts[0])


def sec_to_ts(sec: float) -> str:
    """秒数を 'H:MM:SS' に変換"""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def parse_timetable(timetable_raw: str) -> List[Tuple[float, float]]:
    """
    タイムテーブル文字列（'開始 終了' 形式の行リスト）を解析して
    (開始秒, 終了秒) のタプルリストを返します。
    """
    segments = []
    for line in timetable_raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            start_sec = ts_to_sec(parts[0])
            end_sec = ts_to_sec(parts[1])
            segments.append((start_sec, end_sec))
    return segments


def load_sentiment_pipeline(device: int = -1):
    """
    感情分析モデル 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual'
    のパイプラインをロードします。
    device: GPUを使用する場合は 0 (または適切なデバイス番号), CPUの場合は -1
    """
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    print(f"感情分析モデル {model_name} をロード中... (device={device})")
    
    # センチメント感情分析パイプライン
    # cardiffnlpのこのモデルはラベルが 0: negative, 1: neutral, 2: positive とマッピングされます
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device
    )
    return sentiment_pipeline


def map_label_name(hf_label: str) -> str:
    """
    Hugging Faceの感情ラベルを人間にわかりやすい文字列にマッピングします
    """
    label_lower = hf_label.lower()
    # モデルの出力形式 (LABEL_0, LABEL_1, LABEL_2 又は negative, neutral, positive) に対応
    if "negative" in label_lower or "label_0" in label_lower:
        return "Negative"
    elif "neutral" in label_lower or "label_1" in label_lower:
        return "Neutral"
    elif "positive" in label_lower or "label_2" in label_lower:
        return "Positive"
    return "Unknown"


def predict_sentiment_batch(texts: List[str], sentiment_pipeline, batch_size: int = 32) -> List[str]:
    """
    大量のテキストリストに対してバッチ処理で感情予測を実行します
    """
    if not texts:
        return []
    
    print(f"{len(texts)} 件のテキストを感情分析中（バッチサイズ={batch_size}）...")
    results = sentiment_pipeline(texts, batch_size=batch_size, truncation=True, max_length=128)
    
    mapped_labels = [map_label_name(res["label"]) for res in results]
    return mapped_labels


def calculate_sentiment_ratio(labels: List[str]) -> Tuple[float, float, float, int]:
    """
    感情ラベルのリストから Positive, Neutral, Negative の比率を算出します
    """
    total = len(labels)
    if total == 0:
        return 0.0, 0.0, 0.0, 0
    
    pos_count = labels.count("Positive")
    neu_count = labels.count("Neutral")
    neg_count = labels.count("Negative")
    
    pos_ratio = (pos_count / total) * 100
    neu_ratio = (neu_count / total) * 100
    neg_ratio = (neg_count / total) * 100
    
    return pos_ratio, neu_ratio, neg_ratio, total


def analyze_sentiment_by_segments(
    video_url: str,
    segments: List[Tuple[float, float]],
    sentiment_pipeline,
    enable_comments: bool = True,
    min_sentence_length: int = 6
) -> pd.DataFrame:
    """
    YouTube動画の字幕とコメントを取得し、各セグメントに分類して感情分析比率を計算します
    """
    # 1. 字幕の取得と前処理
    print("\n--- 字幕データの取得と前処理 ---")
    raw_chunks, raw_metadata = get_raw_transcript(video_url, language="ja")
    sentences, sentence_metadata = basic_transcript_processing(
        raw_chunks, raw_metadata, min_len=min_sentence_length
    )
    
    # 2. コメントの取得と秒数への変換
    comments = []
    if enable_comments:
        print("\n--- コメントデータの取得 ---")
        try:
            raw_comments = get_comments(video_url)
            for c in raw_comments:
                comments.append({
                    "text": c["text"],
                    "seconds": timestamp_to_seconds(c["time"])
                })
            print(f"取得したコメント総数: {len(comments)}")
        except Exception as e:
            print(f"Warning: コメント取得中にエラーが発生しました（スキップします）: {e}")
            enable_comments = False
    
    # 3. 各字幕とコメントをセグメントに分類
    print("\n--- 各セグメントへデータを振り分け中 ---")
    segment_data = []  # 各セグメントのテキスト群を保持
    for idx, (start, end) in enumerate(segments, 1):
        segment_data.append({
            "id": idx,
            "start": start,
            "end": end,
            "subtitles": [],
            "comments": []
        })
    
    # 字幕の分類 (開始秒がセグメント内にあるかで判定)
    unassigned_sub_count = 0
    for text, meta in zip(sentences, sentence_metadata):
        sub_start = meta["start"]
        assigned = False
        for seg in segment_data:
            if seg["start"] <= sub_start < seg["end"]:
                seg["subtitles"].append(text)
                assigned = True
                break
        if not assigned:
            unassigned_sub_count += 1
            
    # コメントの分類 (投稿秒がセグメント内にあるかで判定)
    unassigned_com_count = 0
    for c in comments:
        com_sec = c["seconds"]
        assigned = False
        for seg in segment_data:
            if seg["start"] <= com_sec < seg["end"]:
                seg["comments"].append(c["text"])
                assigned = True
                break
        if not assigned:
            unassigned_com_count += 1

    print(f"セグメント外の字幕数（集計除外）: {unassigned_sub_count}")
    if enable_comments:
        print(f"セグメント外のコメント数（集計除外）: {unassigned_com_count}")
        
    # 4. セグメントごとに感情分析を実行
    print("\n--- 感情分析と比率計算を実行中 ---")
    
    # 効率化のため、セグメントごとに分かれたテキストをフラットなリストにして一括推論する準備をします
    all_subs_flat = []
    all_coms_flat = []
    
    for seg in segment_data:
        all_subs_flat.extend(seg["subtitles"])
        all_coms_flat.extend(seg["comments"])
        
    # 一括バッチ推論の実行
    sub_labels_flat = predict_sentiment_batch(all_subs_flat, sentiment_pipeline)
    com_labels_flat = predict_sentiment_batch(all_coms_flat, sentiment_pipeline)
    
    # 推論結果を各セグメントに再マッピング
    sub_idx_offset = 0
    com_idx_offset = 0
    
    rows = []
    for seg in segment_data:
        num_subs = len(seg["subtitles"])
        num_coms = len(seg["comments"])
        
        # 字幕のラベル切り出し
        seg_sub_labels = sub_labels_flat[sub_idx_offset : sub_idx_offset + num_subs]
        sub_idx_offset += num_subs
        
        # コメントのラベル切り出し
        seg_com_labels = com_labels_flat[com_idx_offset : com_idx_offset + num_coms]
        com_idx_offset += num_coms
        
        # 感情比率の計算
        sub_pos, sub_neu, sub_neg, sub_count = calculate_sentiment_ratio(seg_sub_labels)
        com_pos, com_neu, com_neg, com_count = calculate_sentiment_ratio(seg_com_labels)
        
        rows.append({
            "セグメント": seg["id"],
            "開始時刻": sec_to_ts(seg["start"]),
            "終了時刻": sec_to_ts(seg["end"]),
            "長さ(秒)": int(seg["end"] - seg["start"]),
            "字幕数": sub_count,
            "字幕_Positive(%)": round(sub_pos, 2),
            "字幕_Neutral(%)": round(sub_neu, 2),
            "字幕_Negative(%)": round(sub_neg, 2),
            "コメント数": com_count,
            "コメント_Positive(%)": round(com_pos, 2),
            "コメント_Neutral(%)": round(com_neu, 2),
            "コメント_Negative(%)": round(com_neg, 2),
        })
        
    df_results = pd.DataFrame(rows)
    return df_results


# ── Google Colab で実行する際のお試し用のコード ────────────────────
if __name__ == "__main__":
    # テスト実行用の設定
    TEST_URL = "https://www.youtube.com/watch?v=pP2KLW-_7hQ"
    
    TEST_TIMETABLE_RAW = """
    0:01:49 0:03:44
    0:03:46 0:06:00
    0:06:56 0:08:47
    """
    
    print("=== セグメント感情分析テスト実行 ===")
    
    # 1. タイムテーブル解析
    segments = parse_timetable(TEST_TIMETABLE_RAW)
    print(f"解析したセグメント数: {len(segments)}")
    
    # 2. パイプライン準備 (GPUが使えれば device=0)
    import torch
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = load_sentiment_pipeline(device=device)
    
    # 3. 実行（テストのためコメント取得はオフにすることも可能）
    # ※コメント数が多い動画では PyTchat の取得に数分かかる場合があります。
    df_results = analyze_sentiment_by_segments(
        video_url=TEST_URL,
        segments=segments,
        sentiment_pipeline=sentiment_pipeline,
        enable_comments=True,
        min_sentence_length=6
    )
    
    # 4. 結果の出力
    print("\n--- 分析結果 ---")
    print(df_results.to_string(index=False))
    
    # CSVに保存
    output_csv = "./segment_sentiment_results.csv"
    df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"結果をCSVに保存しました: {output_csv}")
