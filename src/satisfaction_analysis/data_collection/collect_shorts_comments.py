# -*- coding: utf-8 -*-
"""
collect_shorts_comments.py

Shorts 77本の動画コメント欄（YouTube Shorts コメント）を取得し、
data/raw/shorts_comments/{shorts_id}.comments.json に個別保存するスクリプト。

【特徴】
- yt-dlp を使用して動画本体をダウンロードせずコメント情報のみ高速取得
- 取得済みファイルが存在する場合はスキップ（レジューム・中断再開機能）
- YouTubeサーバーへの負荷軽減・アクセス制限防止のため1秒のスリープを設ける
- 取得結果（コメント数、ステータス）を data/processed/shorts_comments_summary.csv に集計
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import yt_dlp

PROJECT_ROOT = Path("/home/shota/work/year1")
SHORTS_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/raw/shorts_comments"
SUMMARY_CSV = PROJECT_ROOT / "data/processed/shorts_comments_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_comments_for_shorts(shorts_id: str):
    """1本のShorts動画からコメントを取得してJSON保存"""
    out_json = OUTPUT_DIR / f"{shorts_id}.comments.json"
    
    if out_json.exists():
        # 既存ファイル読み込み
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("comments", [])), "skipped (already exists)"

    url = f"https://www.youtube.com/shorts/{shorts_id}"
    ydl_opts = {
        'getcomments': True,
        'extract_flat': False,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            comments = info.get("comments", [])
            
            # 必要なフィールドのみ抽出してファイルサイズ節約
            clean_comments = []
            for c in comments:
                clean_comments.append({
                    "id": c.get("id"),
                    "text": c.get("text"),
                    "like_count": c.get("like_count", 0),
                    "author": c.get("author"),
                    "timestamp": c.get("timestamp"),
                    "is_favorited": c.get("is_favorited", False),
                    "is_pinned": c.get("is_pinned", False),
                })
            
            save_payload = {
                "shorts_id": shorts_id,
                "title": info.get("title", ""),
                "view_count": info.get("view_count", 0),
                "comment_count": len(clean_comments),
                "comments": clean_comments
            }
            
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, ensure_ascii=False, indent=2)
                
            return len(clean_comments), "success"
            
    except Exception as e:
        err_msg = str(e)
        # エラー時も空のJSONを保存して次回無限ループ防止
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"shorts_id": shorts_id, "error": err_msg, "comments": []}, f, ensure_ascii=False, indent=2)
        return 0, f"error: {err_msg[:60]}"

def main():
    print("=== YouTube Shorts コメント取得バッチ開始 ===")
    df = pd.read_csv(SHORTS_CSV)
    shorts_ids = df['shorts_id'].unique().tolist()
    total = len(shorts_ids)
    print(f"取得対象: 全 {total} 本")

    records = []
    
    for i, s_id in enumerate(shorts_ids, 1):
        print(f"[{i:02d}/{total:02d}] {s_id} ... ", end="", flush=True)
        count, status = fetch_comments_for_shorts(s_id)
        print(f"{status} (コメント数: {count})")
        
        records.append({
            "shorts_id": s_id,
            "comment_count": count,
            "status": status
        })
        
        # 実際に取得通信を行った場合のみ少し待機
        if "skipped" not in status:
            time.sleep(1.5)

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 取得サマリーを保存しました: {SUMMARY_CSV}")
    print(f"総コメント数: {df_summary['comment_count'].sum()} 件")
    print(f"コメント取得成功動画数: {(df_summary['comment_count'] > 0).sum()} / {total} 本")

if __name__ == "__main__":
    main()
