# -*- coding: utf-8 -*-
"""
Shorts動画と、概要欄に記載された候補元配信URL（複数）の字幕を照合し、
元配信の開始秒・終了秒を自動特定するスクリプト。
"""

import os
import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript_text(video_id):
    """YouTube動画の字幕（タイムスタンプ付き）を取得"""
    try:
        # 日本語字幕（自動生成含む）を直接取得
        return YouTubeTranscriptApi.get_transcript(video_id, languages=['ja'])
    except Exception as e:
        # 他言語やフォールバック
        try:
            transcript_list = YouTubeTranscriptApi().list_transcripts(video_id)
            for t in transcript_list:
                return t.fetch()
        except Exception as e2:
            print(f"[{video_id}] 字幕取得失敗: {e2}")
            return None

def align_shorts_to_origin(shorts_id, candidate_ids):
    print(f"=== 1. Shorts動画 [{shorts_id}] の字幕を取得中... ===")
    shorts_transcript = get_transcript_text(shorts_id)
    if not shorts_transcript:
        print("Shortsの字幕が取得できませんでした。")
        return
    
    shorts_text = " ".join([t['text'] for t in shorts_transcript])
    print(f"\nShorts字幕（全文）:\n{shorts_text}\n")
    
    # 検索用キーワード（特徴的な語彙）
    # 例: 堀、ショートスリーパー、睡眠、等
    keywords = ["堀", "ショートスリーパー", "睡眠", "寝る"]
    
    print(f"=== 2. 元配信候補（{len(candidate_ids)}本）を走査中... ===")
    for origin_id in candidate_ids:
        print(f"\n--- チェック中: https://www.youtube.com/watch?v={origin_id} ---")
        origin_transcript = get_transcript_text(origin_id)
        if not origin_transcript:
            continue
        
        found = False
        for i, item in enumerate(origin_transcript):
            text = item['text']
            # 「堀」または「ショートスリーパー」が含まれるか
            if "堀" in text or "ショートスリーパー" in text:
                start_sec = int(item['start'])
                m, s = divmod(start_sec, 60)
                h, m = divmod(m, 60)
                time_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                
                # 前後3行のコンテキストを表示
                context = " ".join([origin_transcript[j]['text'] for j in range(max(0, i-1), min(len(origin_transcript), i+3))])
                
                print(f"\n🎯 【マッチ候補発見！】")
                print(f"元配信URL: https://www.youtube.com/watch?v={origin_id}&t={start_sec}s")
                print(f"該当時間: {time_str}（開始秒: {start_sec}秒）")
                print(f"元配信の字幕抜粋: {context}")
                found = True
        
        if not found:
            print("一致するセグメントは見つかりませんでした。")

if __name__ == '__main__':
    SHORTS_ID = "gbI8-X7YMBQ"
    CANDIDATES = [
        "bsIGUPuoRbk",
        "KmCUBDWi6T0",
        "daulqJwmooE"
    ]
    align_shorts_to_origin(SHORTS_ID, CANDIDATES)
