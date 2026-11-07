# -*- coding: utf-8 -*-
"""
@hiroyukike チャンネルの全Shorts（776本）を対象に、
目標件数の制限なし（全件走査）で元配信アーカイブとの高精度アライメントを実行するスクリプト。

特徴:
- フレーズ一致閾値: 5箇所以上（前回の実績基準）
- 1件マッチするごとに即座に JSON / CSV / HTMLレポート を自動保存
- 中断・再開可能（すでに取得・マッチ済みのShortsは自動スキップ）
- 別のPCにGitクローンして即実行可能
"""

import os
import sys
import json
import csv
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.satisfaction_analysis.data_collection.align_origin_stream import (
    align_shorts_to_origin,
    format_seconds,
    download_vtt_subtitles,
    RAW_CHAT_DIR,
    YTDLP_BIN
)
from src.satisfaction_analysis.data_collection.generate_verification_report import (
    generate_verification_html
)

ALL_PLAYLIST = RAW_CHAT_DIR / "shorts_playlist_all.jsonl"
OUTPUT_JSON = PROJECT_ROOT / "data/processed/verified_alignments.json"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/verified_shorts_dataset.csv"
OUTPUT_HTML = PROJECT_ROOT / "docs/paper/verification_report.html"

def save_data(results):
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON保存（デフォルトOK）
    json_data = {}
    for idx, r in enumerate(results, 1):
        json_data[str(idx)] = {
            "status": "OK",
            "shorts_id": r['shorts_id'],
            "origin_id": r.get('origin_id', ''),
            "start_sec": r.get('start_sec', 0.0),
            "end_sec": r.get('end_sec', 0.0),
            "start_str": r.get('start_str', ''),
            "end_str": r.get('end_str', ''),
            "duration": r.get('duration', 0.0),
            "view_count": r.get('view_count', 0),
            "like_count": r.get('like_count', 0),
            "like_rate": r.get('like_rate', 0.0),
            "shorts_title": r.get('shorts_title', '')
        }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        
    # 2. CSV保存
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "card_id", "shorts_id", "title", "view_count", "like_count", 
            "like_rate", "status", "origin_id", "start_sec", "end_sec", 
            "start_str", "end_str", "duration"
        ])
        for idx, r in enumerate(results, 1):
            writer.writerow([
                idx,
                r['shorts_id'],
                r.get('shorts_title', ''),
                r.get('view_count', 0),
                r.get('like_count', 0),
                f"{r.get('like_rate', 0.0):.4f}",
                "OK",
                r.get('origin_id', ''),
                f"{r.get('start_sec', 0.0):.2f}",
                f"{r.get('end_sec', 0.0):.2f}",
                r.get('start_str', ''),
                r.get('end_str', ''),
                f"{r.get('duration', 0.0):.2f}"
            ])
            
    # 3. HTMLレポート生成
    generate_verification_html(results, OUTPUT_HTML)

def run_unlimited_collection():
    if not ALL_PLAYLIST.exists():
        print(f"❌ プレイリストが見つかりません: {ALL_PLAYLIST}")
        return

    items = []
    with open(ALL_PLAYLIST, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass

    print("=" * 65)
    print(f"🚀 無制限・全件データ拡張バッチ開始 (総対象: {len(items)}本)")
    print(f"   [基準]: フレーズ一致 >= 5 (前回実績設定) | 1件ごとに自動保存")
    print("=" * 65)

    # 既存の保存データがあれば引き継ぐ
    existing_results = []
    processed_shorts_ids = set()
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                for k, v in saved.items():
                    existing_results.append(v)
                    processed_shorts_ids.add(v['shorts_id'])
            print(f"ℹ️ 既存の保存済みデータ {len(existing_results)} 件を引き継ぎます。")
        except Exception:
            pass

    results = list(existing_results)

    for i, item in enumerate(items, 1):
        sid = item.get('id')
        title = item.get('title', '')
        
        if not sid or sid in processed_shorts_ids:
            continue

        print(f"\n[{i}/{len(items)}] 検査中: {sid} | {title[:40]}...")

        matched_res = align_shorts_to_origin(sid)
        
        if not matched_res or not matched_res.get('origin_id'):
            print("  [-] 一致なし / スキップ")
            continue

        hits = matched_res.get('hits', [])
        if len(hits) < 5:
            print(f"  [-] フレーズ一致不足: {len(hits)}個 (< 5個)")
            continue

        meta = matched_res.get('shorts_meta', {})
        vc = meta.get('view_count', 0)
        lc = meta.get('like_count', 0)
        lr = (lc / vc * 100) if vc and vc > 0 else 0.0

        record = {
            'shorts_id': sid,
            'shorts_title': title,
            'shorts_duration': meta.get('duration', 0),
            'view_count': vc,
            'like_count': lc,
            'like_rate': lr,
            'origin_id': matched_res['origin_id'],
            'start_sec': matched_res['start_sec'],
            'end_sec': matched_res['end_sec'],
            'duration': matched_res['duration'],
            'start_str': format_seconds(matched_res['start_sec']),
            'end_str': format_seconds(matched_res['end_sec']),
            'hits': list(hits)[:10],
            'shorts_text_snippet': matched_res.get('shorts_text_snippet', ''),
            'origin_text_snippet': matched_res.get('origin_text_snippet', '')
        }

        results.append(record)
        processed_shorts_ids.add(sid)

        # 1件ヒットするごとにリアルタイム自動保存
        save_data(results)
        print(f"  🎉 【照合成功】累計: {len(results)} 件 | 元配信: {record['origin_id']} ({record['start_str']}〜{record['end_str']}, 一致: {len(hits)}個)")

    print("\n" + "=" * 65)
    print(f"🏁 全件走査完了！ 合計 {len(results)} 件のマッチデータを保存しました。")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")
    print(f"HTML: {OUTPUT_HTML}")
    print("=" * 65)

if __name__ == '__main__':
    run_unlimited_collection()
