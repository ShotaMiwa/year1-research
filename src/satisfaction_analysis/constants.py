# -*- coding: utf-8 -*-
"""
constants.py

視聴満足度推定プロジェクトにおける全特徴量の「公式定義書（Single Source of Truth）」。
各スクリプトで特徴量リストを手書きすることを禁止し、必ず本モジュールの定数を参照する。
"""
from typing import List, Dict, Any

# ==============================================================================
# 公式 全12特徴量リスト（順序・命名を完全固定）
# ==============================================================================
FEATURE_NAMES_12: List[str] = [
    # ① チャット・コメント系 (7個)
    "comment_rate",                # コメント密度 (件/秒)
    "grass_ratio",                 # 草・笑い率 (0.0 ~ 1.0)
    "question_ratio",              # 質問・疑問率 (0.0 ~ 1.0)
    "exclamation_ratio",           # 感嘆符率 (0.0 ~ 1.0)
    "sentiment_polarity",          # コメントBERT感情極性 (-1.0 ~ +1.0)
    "comment_vocab_diversity",     # 語彙多様性 TTR (0.0 ~ 1.0)
    "comment_semantic_variance",   # 意味多様性 BERT埋め込みL2分散 (約 40.0 ~ 80.0)

    # ② 字幕・音声系 (4個)
    "subtitle_sentiment_polarity", # 字幕BERT感情極性 (-1.0 ~ +1.0)
    "speech_rate",                 # 発話速度 (文字数/発話秒, 約 1.0 ~ 15.0)
    "pause_ratio",                 # 無音比率 (0.0 ~ 1.0)
    "sub_comment_similarity",      # 字幕-コメントBERTコサイン類似度 (-1.0 ~ +1.0, 通常 0.5 ~ 0.95)

    # ③ メタ情報 (1個)
    "segment_length_sec"           # 動画尺・セグメント秒数 (> 0.0)
]

# ==============================================================================
# 特徴量の仕様・期待値レンジ（バリデーション用）
# ==============================================================================
FEATURE_SPECS: Dict[str, Dict[str, Any]] = {
    "comment_rate": {"min": 0.0, "max": 50.0, "desc": "コメント密度 (件/秒)"},
    "grass_ratio": {"min": 0.0, "max": 1.0, "desc": "草・笑い率"},
    "question_ratio": {"min": 0.0, "max": 1.0, "desc": "質問率"},
    "exclamation_ratio": {"min": 0.0, "max": 1.0, "desc": "感嘆符率"},
    "sentiment_polarity": {"min": -1.0, "max": 1.0, "desc": "コメント感情極性"},
    "comment_vocab_diversity": {"min": 0.0, "max": 1.0, "desc": "語彙多様性 (TTR)"},
    "comment_semantic_variance": {"min": 20.0, "max": 120.0, "desc": "意味多様性 (BERT L2分散)"},
    "subtitle_sentiment_polarity": {"min": -1.0, "max": 1.0, "desc": "字幕感情極性"},
    "speech_rate": {"min": 0.0, "max": 25.0, "desc": "発話速度 (文字/発話秒)"},
    "pause_ratio": {"min": 0.0, "max": 1.0, "desc": "無音比率"},
    "sub_comment_similarity": {"min": -1.0, "max": 1.0, "desc": "字幕-コメントBERT類似度"},
    "segment_length_sec": {"min": 1.0, "max": 600.0, "desc": "動画尺 (秒)"},
}

# 除外対象の再生数外れ値 Shorts ID (4本)
OUTLIER_SHORTS_IDS = {
    "LD06GCUS6cI",  # 36万回再生（希釈）
    "Msd39_BhMqU",  # 11万回再生（希釈）
    "4KMwiyfGxA0",  # 374回再生（インフレ）
    "UEvV6gO5l88"   # 534回再生（インフレ）
}
