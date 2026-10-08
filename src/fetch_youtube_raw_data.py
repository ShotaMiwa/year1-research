#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: YouTube から字幕・チャットを取得し、セグメント別に整理する
"""
import os, sys, json, re, subprocess
import pandas as pd
import numpy as np
from pathlib import Path

VIDEO_ID = "pP2KLW-_7hQ"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
PROJECT_ROOT = Path("/home/shota/work/year1")
RAW_DIR = PROJECT_ROOT / "data/raw"
CHAT_DIR = RAW_DIR / "chat"
SUBTITLE_DIR = RAW_DIR / "subtitles"
CHAT_DIR.mkdir(parents=True, exist_ok=True)
SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = PROJECT_ROOT / ".venv/bin/python3"

# ============================================================
# 1) セグメント時刻の読み込み
# ============================================================
df_seg = pd.read_csv(PROJECT_ROOT / "data/processed/merged_survey_heatmap.csv")
print("=== Segments ===")
print(df_seg[['セグメント','開始時刻','終了時刻','開始(秒)','終了(秒)']].to_string())
print()

# ============================================================
# 2) 字幕取得（youtube_transcript_api）
# ============================================================
print("=== Fetching subtitles via youtube_transcript_api ===")
subtitle_script = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}/.venv/lib/python3.12/site-packages')
from youtube_transcript_api import YouTubeTranscriptApi
import json, csv

video_id = '{VIDEO_ID}'
# 日本語字幕を優先、なければ自動生成
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ja'])
    print('Found ja transcript', file=sys.stderr)
except Exception as e:
    print(f'ja failed: {{e}}', file=sys.stderr)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ja-JP'])
        print('Found ja-JP transcript', file=sys.stderr)
    except Exception as e2:
        print(f'ja-JP failed: {{e2}}', file=sys.stderr)
        # 自動生成字幕を取得
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        for t in transcript_list:
            print(f'Available: {{t.language_code}} generated={{t.is_generated}}', file=sys.stderr)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print('Found default transcript', file=sys.stderr)

# CSV に保存
output_path = '{SUBTITLE_DIR}/subtitles_raw.csv'
with open(output_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['start', 'duration', 'text'])
    writer.writeheader()
    for entry in transcript:
        writer.writerow(entry)
print(f'Saved {{len(transcript)}} subtitle entries to {{output_path}}')
"""

result = subprocess.run(
    [str(VENV_PYTHON), "-c", subtitle_script],
    capture_output=True, text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
if result.returncode != 0:
    print("ERROR: subtitle fetch failed")
else:
    print("Subtitle fetch OK")

print()

# ============================================================
# 3) チャット取得（yt-dlp）
# ============================================================
print("=== Fetching chat via yt-dlp ===")
YTDLP = PROJECT_ROOT / ".venv/bin/yt-dlp"
chat_json_path = CHAT_DIR / f"{VIDEO_ID}.live_chat.json"

# yt-dlp でチャットを JSON ダウンロード（動画本体はスキップ）
cmd = [
    str(YTDLP),
    "--skip-download",
    "--write-subs",
    "--write-auto-subs",
    "--sub-lang", "ja",
    "--write-comments",      # チャットリプレイ
    "-o", str(CHAT_DIR / f"{VIDEO_ID}.%(ext)s"),
    VIDEO_URL
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

# ダウンロードされたファイルを確認
print("\n=== Downloaded files in chat dir ===")
for f in sorted(CHAT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size} bytes)")
