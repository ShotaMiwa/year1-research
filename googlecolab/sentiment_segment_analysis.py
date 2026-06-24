import os
import sys
import re
import json
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


def extract_video_id(url: str) -> str:
    """YouTubeのURLから動画IDを抽出する"""
    pattern = r'(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/|^)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return "unknown_video"


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


def load_sentiment_pipeline(model_name: str = "LoneWolfgang/bert-for-japanese-twitter-sentiment", device: int = -1):
    """
    指定された感情分析モデルのパイプラインをロードします。
    device: GPUを使用する場合は 0 (または適切なデバイス番号), CPUの場合は -1
    """
    print(f"感情分析モデル {model_name} をロード中... (device={device})")
    
    # センチメント感情分析パイプライン
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


def predict_sentiment_batch(texts: List[str], sentiment_pipeline, batch_size: int = 32) -> Tuple[List[str], List[float]]:
    """
    大量のテキストリストに対してバッチ処理で感情予測を実行します。
    返り値: (ラベルリスト, 確信度スコアリスト)
    """
    if not texts:
        return [], []
    
    print(f"{len(texts)} 件のテキストを感情分析中（バッチサイズ={batch_size}）...")
    results = sentiment_pipeline(texts, batch_size=batch_size, truncation=True, max_length=128)
    
    mapped_labels = [map_label_name(res["label"]) for res in results]
    scores = [res["score"] for res in results]
    return mapped_labels, scores


def save_score_distribution_and_samples(
    model_name: str,
    texts: List[str],
    labels: List[str],
    scores: List[float],
    pdf_path: str,
    md_path: str
):
    """
    感情ラベルごとにスコアの分布をPDFで保存し、
    指定条件を満たすコメントのサンプルを抽出してコンソール表示およびMarkdownファイル保存します。
    """
    import pandas as pd
    import numpy as np
    
    # DataFrameの作成
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "score": scores
    })
    # 空白やNaNの除去
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    
    # 1. グラフ描画 (PDF出力)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Headless環境でのエラー防止
        import matplotlib.pyplot as plt
        try:
            import japanize_matplotlib
        except ImportError:
            print("Warning: japanize_matplotlib がインストールされていないため、PDFの日本語が文字化け（トーフ）する可能性があります。")
            print("文字化けを防ぐには、Google Colab上で !pip install japanize-matplotlib を実行してください。")
        
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        
        for ax, label in zip(axes, ["Positive", "Neutral", "Negative"]):
            label_df = df[df["label"] == label]
            if not label_df.empty:
                ax.hist(label_df["score"], bins=20, range=(0.0, 1.0), 
                        color=colors.get(label, "#3498db"), alpha=0.7, edgecolor='black')
                ax.set_title(f"{label}\n(Total: {len(label_df)})", fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"{label}\n(Total: 0)", fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Confidence Score", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Frequency", fontsize=10)
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.5)
            
        plt.suptitle(f"Sentiment Score Distribution (Comments) - {model_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"[{model_name}] スコア分布グラフをPDFに保存しました: {pdf_path}")
    except Exception as e:
        print(f"Warning: [{model_name}] グラフ生成中にエラーが発生しました: {e}")
        
    # 2. サンプルの抽出
    tiers = [
        {"name": "層1：0.33 以上 0.50 未満 （低確信度）", "min": 0.33, "max": 0.50},
        {"name": "層2：0.50 以上 0.65 未満", "min": 0.50, "max": 0.65},
        {"name": "層3：0.65 以上 0.80 未満", "min": 0.65, "max": 0.80},
        {"name": "層4：0.80 以上 0.90 未満", "min": 0.80, "max": 0.90},
        {"name": "層5：0.90 以上 1.01 未満 （高確信度）", "min": 0.90, "max": 1.01},
    ]
    
    sampled_tiers = []
    for tier in tiers:
        label_samples = {}
        for label in ["Positive", "Neutral", "Negative"]:
            filtered = df[(df["score"] >= tier["min"]) & (df["score"] < tier["max"]) & (df["label"] == label)]
            if len(filtered) <= 30:
                sampled = filtered
            else:
                sampled = filtered.sample(n=30, random_state=42)
            label_samples[label] = sampled.sort_values(by="score", ascending=False)
            
        sampled_tiers.append({
            "name": tier["name"],
            "samples": label_samples
        })
        
    # Markdown作成
    md_content = []
    md_content.append(f"# 感情分析 コメント目視確認レポート - {model_name}\n\n")
    md_content.append(f"このファイルは、感情分析結果の妥当性を目視確認するためのサンプルコメントです。\n\n")
    
    for tier_data in sampled_tiers:
        md_content.append(f"## ■ {tier_data['name']}\n\n")
        for label in ["Positive", "Neutral", "Negative"]:
            md_content.append(f"### ● {label} (最大30件)\n\n")
            samples = tier_data["samples"][label]
            if samples.empty:
                md_content.append("*該当するコメントは見つかりませんでした。*\n\n")
            else:
                for idx, row in samples.reset_index().iterrows():
                    md_content.append(f"{idx + 1}. **Score: {row['score']:.4f}**\n")
                    escaped_text = row['text'].replace('\n', ' ')
                    md_content.append(f"   > {escaped_text}\n\n")
                
    # Markdownファイル出力
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("".join(md_content))
        print(f"[{model_name}] 目視確認用サンプルファイルを保存しました: {md_path}")
    except Exception as e:
        print(f"Warning: [{model_name}] サンプルファイル保存中にエラーが発生しました: {e}")
        
    # コンソール出力
    print("\n" + "="*80)
    print(f"  感情分析コメントサンプル目視確認: {model_name}")
    print("="*80)
    for tier_data in sampled_tiers:
        print(f"\n【{tier_data['name']}】")
        for label in ["Positive", "Neutral", "Negative"]:
            print(f"\n● {label} (最大30件):")
            samples = tier_data["samples"][label]
            if samples.empty:
                print("  (該当なし)")
            else:
                for idx, row in samples.reset_index().iterrows():
                    print(f"  {idx + 1}. [{row['score']:.4f}] {row['text']}")
    print("\n" + "="*80 + "\n")


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
    min_sentence_length: int = 6,
    disagreement_csv_path: str = "sentiment_disagreements.csv",
    cache_dir: str = None
) -> pd.DataFrame:
    """
    YouTube動画の字幕とコメントを取得し、各セグメントに分類して感情分析比率を計算します。
    sentiment_pipeline に辞書形式 {"モデル名": pipeline} を渡すことで複数モデルを一度に比較できます。
    また、2つ以上のモデルがある場合、モデル間で判定が不一致だったテキストを CSV に書き出します。
    cache_dir が指定されている場合、そこから字幕・コメントのキャッシュをロード/保存します。
    """
    # 1. 動画IDの抽出とキャッシュパスの設定
    video_id = extract_video_id(video_url)
    sub_cache_path = None
    com_cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        sub_cache_path = os.path.join(cache_dir, f"subtitles_{video_id}.json")
        com_cache_path = os.path.join(cache_dir, f"comments_{video_id}.json")

    # 2. 字幕データの取得と前処理
    sentences = []
    sentence_metadata = []
    
    if sub_cache_path and os.path.exists(sub_cache_path):
        print(f"\n--- 字幕データをキャッシュからロード中 ({sub_cache_path}) ---")
        try:
            with open(sub_cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                sentences = cache_data["sentences"]
                sentence_metadata = cache_data["metadata"]
            print(f"キャッシュからロードした字幕文数: {len(sentences)}")
        except Exception as e:
            print(f"Warning: 字幕キャッシュのロードに失敗しました。再取得します: {e}")
            sub_cache_path = None
            
    if not sentences:
        print("\n--- 字幕データの取得と前処理 ---")
        raw_chunks, raw_metadata = get_raw_transcript(video_url, language="ja")
        sentences, sentence_metadata = basic_transcript_processing(
            raw_chunks, raw_metadata, min_len=min_sentence_length
        )
        if sub_cache_path:
            print(f"字幕データをキャッシュに保存中: {sub_cache_path}")
            try:
                with open(sub_cache_path, "w", encoding="utf-8") as f:
                    json.dump({"sentences": sentences, "metadata": sentence_metadata}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: 字幕キャッシュの保存に失敗しました: {e}")
    
    # 3. コメントの取得と秒数への変換
    comments = []
    if enable_comments:
        if com_cache_path and os.path.exists(com_cache_path):
            print(f"\n--- コメントデータをキャッシュからロード中 ({com_cache_path}) ---")
            try:
                with open(com_cache_path, "r", encoding="utf-8") as f:
                    comments = json.load(f)
                print(f"キャッシュからロードしたコメント総数: {len(comments)}")
            except Exception as e:
                print(f"Warning: コメントキャッシュのロードに失敗しました。再取得します: {e}")
                com_cache_path = None
                
        if not comments:
            print("\n--- コメントデータの取得 ---")
            try:
                raw_comments = get_comments(video_url)
                for c in raw_comments:
                    comments.append({
                        "text": c["text"],
                        "seconds": timestamp_to_seconds(c["time"])
                    })
                print(f"取得したコメント総数: {len(comments)}")
                if com_cache_path:
                    print(f"コメントデータをキャッシュに保存中: {com_cache_path}")
                    try:
                        with open(com_cache_path, "w", encoding="utf-8") as f:
                            json.dump(comments, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Warning: コメントキャッシュの保存に失敗しました: {e}")
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
    
    # Prepare flat lists for batch inference with segment-wise subtitle chunking.
    all_subs_flat = []
    all_coms_flat = []
    sub_segment_ids = []
    com_segment_ids = []

    max_chunk_chars = 120  # Japanese: ~1 char ≈ 1 token; 128 token limit → 120 chars as safe upper bound.
    sub_chunk_counts = []  # track number of subtitle chunks per segment

    for seg in segment_data:
        # Combine subtitles within the segment.
        combined_sub = " ".join(seg["subtitles"])
        # Split into chunks respecting the character limit.
        chunks = []
        if len(combined_sub) <= max_chunk_chars:
            chunks = [combined_sub] if combined_sub else []
        else:
            start = 0
            while start < len(combined_sub):
                chunk = combined_sub[start:start + max_chunk_chars]
                chunks.append(chunk)
                start += max_chunk_chars
        # Append chunks and associate with segment ID.
        all_subs_flat.extend(chunks)
        sub_segment_ids.extend([seg["id"]] * len(chunks))
        sub_chunk_counts.append(len(chunks))

        # Comments are processed per comment (no chunking).
        all_coms_flat.extend(seg["comments"])
        com_segment_ids.extend([seg["id"]] * len(seg["comments"]))
        
    # Prepare sub-slice indices based on chunked subtitles
    sub_slices = []
    com_slices = []
    curr_sub_idx = 0
    curr_com_idx = 0
    # Compute number of subtitle chunks per segment (may differ from original subtitle count)
    for seg in segment_data:
        # Number of subtitle chunks added for this segment
        num_sub_chunks = sub_segment_ids.count(seg["id"]) - curr_sub_idx
        num_coms = len(seg["comments"])
        sub_slices.append((curr_sub_idx, curr_sub_idx + num_sub_chunks))
        com_slices.append((curr_com_idx, curr_com_idx + num_coms))
        curr_sub_idx += num_sub_chunks
        curr_com_idx += num_coms

    is_multi_model = isinstance(sentiment_pipeline, dict)
    
    if is_multi_model:
        # 複数モデルの場合
        model_results = {}
        for model_alias, pipeline_obj in sentiment_pipeline.items():
            print(f"\nモデル '{model_alias}' での感情分析を実行します。")
            sub_labels, sub_scores = predict_sentiment_batch(all_subs_flat, pipeline_obj)
            com_labels, com_scores = predict_sentiment_batch(all_coms_flat, pipeline_obj)
            model_results[model_alias] = {
                "sub_labels": sub_labels,
                "sub_scores": sub_scores,
                "com_labels": com_labels,
                "com_scores": com_scores
            }
            
            # グラフ作成（PDF出力）とコメントサンプルの抽出（目視確認用）
            pdf_path = f"./sentiment_score_dist_{model_alias}.pdf"
            md_path = f"./sentiment_samples_{model_alias}.md"
            save_score_distribution_and_samples(
                model_name=model_alias,
                texts=all_coms_flat,
                labels=com_labels,
                scores=com_scores,
                pdf_path=pdf_path,
                md_path=md_path
            )
    else:
        # 単一モデルの場合
        sub_labels_flat, sub_scores_flat = predict_sentiment_batch(all_subs_flat, sentiment_pipeline)
        com_labels_flat, com_scores_flat = predict_sentiment_batch(all_coms_flat, sentiment_pipeline)
        
        # グラフ作成（PDF出力）とコメントサンプルの抽出（目視確認用）
        pdf_path = "./sentiment_score_dist.pdf"
        md_path = "./sentiment_samples.md"
        save_score_distribution_and_samples(
            model_name="SentimentModel",
            texts=all_coms_flat,
            labels=com_labels_flat,
            scores=com_scores_flat,
            pdf_path=pdf_path,
            md_path=md_path
        )
        
    # 推論結果を各セグメントに再マッピングして集計
    rows = []
    for idx, seg in enumerate(segment_data):
        num_subs = len(seg["subtitles"])
        num_coms = len(seg["comments"])
        
        row = {
            "セグメント": seg["id"],
            "開始時刻": sec_to_ts(seg["start"]),
            "終了時刻": sec_to_ts(seg["end"]),
            "長さ(秒)": int(seg["end"] - seg["start"]),
            "字幕数": num_subs,
        }
        
        sub_start, sub_end = sub_slices[idx]
        com_start, com_end = com_slices[idx]
        
        if is_multi_model:
            # 複数モデルのスコアを結合
            for model_alias, results in model_results.items():
                seg_sub_labels = results["sub_labels"][sub_start:sub_end]
                seg_com_labels = results["com_labels"][com_start:com_end]
                
                sub_pos, sub_neu, sub_neg, _ = calculate_sentiment_ratio(seg_sub_labels)
                com_pos, com_neu, com_neg, _ = calculate_sentiment_ratio(seg_com_labels)
                
                row[f"{model_alias}_字幕_Positive(%)"] = round(sub_pos, 2)
                row[f"{model_alias}_字幕_Neutral(%)"] = round(sub_neu, 2)
                row[f"{model_alias}_字幕_Negative(%)"] = round(sub_neg, 2)
                
                row[f"{model_alias}_コメント_Positive(%)"] = round(com_pos, 2)
                row[f"{model_alias}_コメント_Neutral(%)"] = round(com_neu, 2)
                row[f"{model_alias}_コメント_Negative(%)"] = round(com_neg, 2)
        else:
            # 単一モデルのスコアを結合
            seg_sub_labels = sub_labels_flat[sub_start:sub_end]
            seg_com_labels = com_labels_flat[com_start:com_end]
            
            sub_pos, sub_neu, sub_neg, _ = calculate_sentiment_ratio(seg_sub_labels)
            com_pos, com_neu, com_neg, _ = calculate_sentiment_ratio(seg_com_labels)
            
            row["字幕_Positive(%)"] = round(sub_pos, 2)
            row["字幕_Neutral(%)"] = round(sub_neu, 2)
            row["字幕_Negative(%)"] = round(sub_neg, 2)
            
            row["コメント_Positive(%)"] = round(com_pos, 2)
            row["コメント_Neutral(%)"] = round(com_neu, 2)
            row["コメント_Negative(%)"] = round(com_neg, 2)
            
        row["コメント数"] = num_coms
        rows.append(row)
        
    df_results = pd.DataFrame(rows)

    # 5. 複数モデル比較時の判定不一致データの抽出とCSV保存
    if is_multi_model and len(sentiment_pipeline) >= 2 and disagreement_csv_path:
        print("\n--- モデル間での判定不一致テキストを抽出中 ---")
        disagreement_rows = []
        model_names = list(model_results.keys())
        
        # 字幕の不一致抽出
        for idx, text in enumerate(all_subs_flat):
            labels = {model: model_results[model]["sub_labels"][idx] for model in model_names}
            # ユニークなラベル数が2以上あれば、いずれかのモデルで判定が異なる
            if len(set(labels.values())) > 1:
                row = {
                    "種別": "字幕",
                    "セグメントID": sub_segment_ids[idx],
                    "テキスト": text
                }
                for model, label in labels.items():
                    row[f"{model}_判定"] = label
                disagreement_rows.append(row)
                
        # コメントの不一致抽出
        for idx, text in enumerate(all_coms_flat):
            labels = {model: model_results[model]["com_labels"][idx] for model in model_names}
            if len(set(labels.values())) > 1:
                row = {
                    "種別": "コメント",
                    "セグメントID": com_segment_ids[idx],
                    "テキスト": text
                }
                for model, label in labels.items():
                    row[f"{model}_判定"] = label
                disagreement_rows.append(row)
                
        if disagreement_rows:
            df_disagree = pd.DataFrame(disagreement_rows)
            df_disagree.to_csv(disagreement_csv_path, index=False, encoding="utf-8-sig")
            print(f"モデル間で判定が異なったデータ（{len(df_disagree)}件）をCSVに保存しました: {disagreement_csv_path}")
        else:
            print("すべてのデータでモデル間の判定が一致しました。")
            
    return df_results


# ── Google Colab で実行する際のテスト用のコード ────────────────────
if __name__ == "__main__":
    # テスト実行用の設定
    TEST_URL = "https://www.youtube.com/watch?v=pP2KLW-_7hQ"
    
    TEST_TIMETABLE_RAW = """
    0:01:49 0:03:44
    0:03:46 0:06:00
    0:06:56 0:08:47
    0:08:47 0:09:51
    0:10:07 0:11:49
    0:11:50 0:14:22
    0:14:24 0:15:30
    0:16:10 0:17:36
    0:17:36 0:18:38
    0:20:53 0:21:57
    0:23:47 0:26:08
    0:28:34 0:32:21
    0:32:41 0:33:48
    0:34:30 0:36:42
    0:39:13 0:40:37
    0:45:51 0:47:08
    0:52:30 0:54:47
    0:55:17 0:57:42
    0:57:44 0:58:44
    1:00:35 1:03:03
    1:03:06 1:04:24
    1:04:47 1:06:43
    1:10:12 1:14:51
    """
    
    print("=== セグメント感情分析テスト実行 ===")
    
    # 1. タイムテーブル解析
    segments = parse_timetable(TEST_TIMETABLE_RAW)
    print(f"解析したセグメント数: {len(segments)}")
    
    # 2. パイプライン準備 (GPUが使えれば device=0)
    import torch
    device = 0 if torch.cuda.is_available() else -1
    
    # 2つのモデルをロードして比較用に格納
    models = {
        "日本語BERT": load_sentiment_pipeline("LoneWolfgang/bert-for-japanese-twitter-sentiment", device=device),
        "多言語XLM-R": load_sentiment_pipeline("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", device=device)
    }
    
    # 3. 実行（テストのためコメント取得はオフにすることも可能）
    df_results = analyze_sentiment_by_segments(
        video_url=TEST_URL,
        segments=segments,
        sentiment_pipeline=models, # 辞書を渡して比較実行
        enable_comments=True,
        min_sentence_length=6,
        disagreement_csv_path="./sentiment_disagreements.csv"
    )
    
    # 4. 結果の出力
    print("\n--- 分析結果 ---")
    print(df_results.to_string(index=False))
    
    # CSVに保存
    output_csv = "./segment_sentiment_comparison_results.csv"
    df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"結果をCSVに保存しました: {output_csv}")
