# -*- coding: utf-8 -*-
"""
filter_clean_extended_dataset.py

チャットログ取得でタイムアウト（n_comments == 0）となった2件を厳密に除外し、
全10特徴量＋いいね率がすべて「実測生データ」から計算された
クリーンな77件データセットを確定する。
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATASET_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_79.csv"
CLEAN_DATASET_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
CLEAN_TABLE_CSV = PROJECT_ROOT / "docs/paper/tables/exp_extended_features_77.csv"

def main():
    df = pd.read_csv(RAW_DATASET_CSV)
    print(f"元データ件数: {len(df)} 件")
    
    # チャットが実測できたもの（n_comments > 0）のみを抽出
    df_clean = df[df["n_comments"] > 0].copy().reset_index(drop=True)
    excluded = df[df["n_comments"] == 0]
    
    print(f"除外件数: {len(excluded)} 件")
    for _, row in excluded.iterrows():
        print(f"  [-] 除外: Card {row['card_id']} | Shorts: {row['shorts_id']} | 元配信: {row['origin_id']} (チャット未取得)")
        
    print(f"\n確定有効件数: {len(df_clean)} 件")
    
    # 77件で z-score（標準化）を再計算
    feat_cols = [
        "comment_rate", "grass_ratio", "question_ratio", "exclamation_ratio",
        "sentiment_polarity", "subtitle_sentiment_polarity",
        "speech_rate", "pause_ratio", "sub_comment_similarity", "comment_semantic_variance"
    ]
    for col in feat_cols:
        vals = df_clean[col]
        mu, sigma = vals.mean(), vals.std()
        df_clean[f"{col}_zscore"] = ((vals - mu) / sigma).round(6) if sigma > 0 else 0.0

    CLEAN_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df_clean.to_csv(CLEAN_DATASET_CSV, index=False, encoding="utf-8-sig")
    df_clean.to_csv(CLEAN_TABLE_CSV, index=False, encoding="utf-8-sig")
    
    print("=" * 60)
    print(f"✅ 完全実測データセット（N=77）を保存しました：")
    print(f"   CSV:   {CLEAN_DATASET_CSV}")
    print(f"   TABLE: {CLEAN_TABLE_CSV}")
    print("=" * 60)
    print("\n--- 確定77件の統計サマリー ---")
    print(df_clean[feat_cols + ["like_rate", "segment_length_sec"]].describe().round(4).to_string())

if __name__ == "__main__":
    main()

