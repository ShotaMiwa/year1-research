# -*- coding: utf-8 -*-
"""
update_extended_features_77_with_unified_extractor.py

共通抽出モジュール UnifiedFeatureExtractor を用いて、
Shorts 77本の特徴量をアンケート23本と完全同一の定義・モデルで再抽出し、
extended_feature_matrix_77.csv を更新する。
"""
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/shota/work/year1")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.satisfaction_analysis.features.unified_feature_extractor import (
    UnifiedFeatureExtractor, parse_live_chat_json, parse_vtt_subtitles
)

PROJECT_ROOT = Path("/home/shota/work/year1")
ALIGNMENT_JSON = PROJECT_ROOT / "data/processed/verified_alignments.json"
INPUT_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
RAW_CHAT_DIR = PROJECT_ROOT / "data/raw/chat"


def main():
    print("=== 共通モジュールによる Shorts 77本 特徴量再抽出開始 ===")
    df_existing = pd.read_csv(INPUT_CSV)
    print(f"既存 77本 データ件数: {len(df_existing)}")

    with open(ALIGNMENT_JSON, "r", encoding="utf-8") as f:
        alignments = json.load(f)

    # カードIDをキーにしたマップ
    card_map = {item.get("card_id"): item for item in alignments if "card_id" in item}

    extractor = UnifiedFeatureExtractor(device="cpu")
    extractor.load_models()

    # キャッシュ用辞書 (配信ごとのパース結果)
    chat_cache = {}
    sub_cache = {}

    updated_rows = []
    t0 = time.time()

    for idx, row in df_existing.iterrows():
        card_id = row["card_id"]
        shorts_id = row["shorts_id"]
        origin_id = row["origin_id"]
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        dur = end_sec - start_sec

        # 配信チャットの取得・パース
        if origin_id not in chat_cache:
            chat_path = RAW_CHAT_DIR / f"{origin_id}.live_chat.json"
            chat_cache[origin_id] = parse_live_chat_json(chat_path)
        chats = chat_cache[origin_id]

        # 配信字幕の取得・パース
        if origin_id not in sub_cache:
            vtt_path = RAW_CHAT_DIR / f"{origin_id}.ja.vtt"
            sub_cache[origin_id] = parse_vtt_subtitles(vtt_path)
        subs, tokens = sub_cache[origin_id]

        # 共通抽出器で計算
        feats = extractor.extract_features(chats, subs, tokens, start_sec, end_sec)

        # 既存メタデータ + 新特徴量
        row_dict = dict(row)
        for k, v in feats.items():
            row_dict[k] = v

        # ヒートマップ特徴量は既存の値をそのまま維持
        # (hm_mean, hm_max, hm_volatility)
        updated_rows.append(row_dict)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_existing):
            elapsed = time.time() - t0
            print(f"進捗: {idx + 1}/{len(df_existing)} 完了 ({elapsed:.1f}s) - 最新: TTR={feats['comment_vocab_diversity']:.3f}, BERT_VAR={feats['comment_semantic_variance']:.1f}")

    df_updated = pd.DataFrame(updated_rows)

    # Z-score カラムの再計算
    numeric_feature_cols = [
        "comment_rate", "grass_ratio", "question_ratio", "exclamation_ratio",
        "sentiment_polarity", "subtitle_sentiment_polarity", "speech_rate",
        "pause_ratio", "sub_comment_similarity", "comment_semantic_variance",
        "comment_vocab_diversity"
    ]
    for col in numeric_feature_cols:
        std = df_updated[col].std(ddof=0)
        mean = df_updated[col].mean()
        df_updated[f"{col}_zscore"] = (df_updated[col] - mean) / (std if std > 1e-9 else 1.0)

    # CSV保存
    df_updated.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 更新完了: {OUTPUT_CSV} に保存しました。")
    print(f"カラム一覧: {list(df_updated.columns)}")


if __name__ == "__main__":
    main()
