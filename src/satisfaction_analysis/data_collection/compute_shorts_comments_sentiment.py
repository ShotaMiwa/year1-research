# -*- coding: utf-8 -*-
"""
compute_shorts_comments_sentiment.py

取得した YouTube Shorts コメント（data/raw/shorts_comments/*.comments.json）に対し、
公式の感情分析パイプライン（UnifiedFeatureExtractor.sentiment_pipe）を用いて
感情極性スコア（(pos - neg) / (pos + neg + 1e-6)）を算出し、動画ごとの集計CSVを出力する。
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/home/shota/work/year1")
sys.path.insert(0, str(PROJECT_ROOT))

from src.satisfaction_analysis.features.unified_feature_extractor import UnifiedFeatureExtractor

SHORTS_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
COMMENTS_DIR = PROJECT_ROOT / "data/raw/shorts_comments"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/shorts_comments_sentiment_summary.csv"

def main():
    print("=== Shorts コメント感情極性スコア算出開始 ===")
    
    print("BERTモデルをロード中...")
    extractor = UnifiedFeatureExtractor(device="cpu")
    extractor.load_models()
    pipe = extractor.sentiment_pipe
    print("ロード完了。")

    df_shorts = pd.read_csv(SHORTS_CSV)
    shorts_ids = df_shorts['shorts_id'].unique().tolist()
    
    records = []
    
    for i, s_id in enumerate(shorts_ids, 1):
        json_path = COMMENTS_DIR / f"{s_id}.comments.json"
        
        if not json_path.exists():
            records.append({
                "shorts_id": s_id,
                "n_comments": 0,
                "sentiment_polarity": 0.0,
                "pos_count": 0,
                "neg_count": 0,
            })
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        comments = data.get("comments", [])
        texts = [c.get("text", "").strip() for c in comments if c.get("text", "").strip()]
        
        if not texts:
            records.append({
                "shorts_id": s_id,
                "n_comments": 0,
                "sentiment_polarity": 0.0,
                "pos_count": 0,
                "neg_count": 0,
            })
            continue
            
        # extract_features と完全同一の極性計算ロジック
        pipe_res = pipe(texts[:100])  # 上限100件
        pos = sum(1 for r in pipe_res if r["label"].upper() in ("POSITIVE", "POS"))
        neg = sum(1 for r in pipe_res if r["label"].upper() in ("NEGATIVE", "NEG"))
        polarity = (pos - neg) / (pos + neg + 1e-6)
        
        records.append({
            "shorts_id": s_id,
            "n_comments": len(texts),
            "sentiment_polarity": round(float(polarity), 6),
            "pos_count": pos,
            "neg_count": neg,
        })
        
        if i % 10 == 0 or i == len(shorts_ids):
            print(f"進捗: [{i:02d}/{len(shorts_ids):02d}] 完了")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 感情分析結果を保存しました: {OUTPUT_CSV}")
    print(df_out[['n_comments', 'sentiment_polarity', 'pos_count', 'neg_count']].describe())

if __name__ == "__main__":
    main()
