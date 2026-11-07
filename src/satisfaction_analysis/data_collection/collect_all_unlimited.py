# -*- coding: utf-8 -*-
"""
@hiroyukike チャンネルの全Shorts（777本）を対象に、
目標件数の制限なし（全件走査）で元配信アーカイブとの高精度アライメントを実行するスクリプト。

特徴:
- フレーズ一致閾値: 5箇所以上（前回の実績基準）
- 1件マッチするごとに即座に JSON / CSV / HTMLレポート を自動保存
- 中断・再開可能（すでに取得・マッチ済みのShortsは自動スキップ）
- 別のPCにGitクローンして即実行可能
"""

import os
import sys
import re
import json
import csv
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satisfaction_analysis.data_collection.align_origin_stream import (
    parse_vtt_cues,
    extract_robust_phrases,
    format_seconds,
    download_subtitles_ytdlp,
    extract_candidate_ids_from_description,
    fetch_video_metadata,
    RAW_CHAT_DIR
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


def align_single_shorts(sid: str, title: str):
    """1本のShortsに対して概要欄から候補元配信を特定し、字幕フレーズ照合を行う"""
    shorts_url = f"https://www.youtube.com/shorts/{sid}"
    meta = fetch_video_metadata(shorts_url)
    if not meta:
        return None

    desc = meta.get("description", "")
    vc = meta.get("view_count", 0)
    lc = meta.get("like_count", 0)
    lr = (lc / vc * 100) if vc > 0 else 0.0

    candidates = extract_candidate_ids_from_description(desc)
    if not candidates:
        return None

    # Shorts字幕取得
    svtt = download_subtitles_ytdlp(sid)
    if not svtt:
        return None

    scues = parse_vtt_cues(svtt)
    if not scues:
        return None

    sdur = scues[-1][1] - scues[0][0]
    if sdur < 30.0 or sdur > 240.0:
        return None

    phrases = extract_robust_phrases(scues)
    if len(phrases) < 5:
        return None

    # 各候補元配信との照合
    for cid in candidates:
        ovtt = download_subtitles_ytdlp(cid)
        if not ovtt:
            continue
        ocues = parse_vtt_cues(ovtt)
        if not ocues:
            continue

        matched_indices = []
        for idx, (_, _, txt) in enumerate(ocues):
            hits = [p for p in phrases if p in txt]
            if hits:
                matched_indices.append((idx, hits))

        clusters = []
        if matched_indices:
            cur = [matched_indices[0]]
            for item in matched_indices[1:]:
                if item[0] - cur[-1][0] <= 25:
                    cur.append(item)
                else:
                    uh = {h for _, hs in cur for h in hs}
                    if len(uh) >= 5:
                        clusters.append((cur, uh))
                    cur = [item]
            uh = {h for _, hs in cur for h in hs}
            if len(uh) >= 5:
                clusters.append((cur, uh))

        valid_clusters = []
        for cl, uh in clusters:
            s_idx = max(0, cl[0][0] - 2)
            e_idx = min(len(ocues) - 1, cl[-1][0] + 1)
            s_sec = ocues[s_idx][0]
            e_sec = ocues[e_idx][1]
            dur = e_sec - s_sec
            if 0.5 * sdur <= dur <= 2.0 * sdur:
                valid_clusters.append((cl, uh, s_sec, e_sec, dur, s_idx, e_idx))

        if valid_clusters:
            cl, uh, s_sec, e_sec, dur, s_idx, e_idx = valid_clusters[0]
            return {
                'shorts_id': sid,
                'shorts_title': meta.get('title', title),
                'shorts_duration': sdur,
                'view_count': vc,
                'like_count': lc,
                'like_rate': lr,
                'origin_id': cid,
                'start_sec': s_sec,
                'end_sec': e_sec,
                'duration': dur,
                'start_str': format_seconds(s_sec),
                'end_str': format_seconds(e_sec),
                'hits': list(uh)[:10],
                'shorts_text_snippet': " ".join([c[2] for c in scues[:6]]) + "...",
                'origin_text_snippet': " ".join([ocues[i][2] for i in range(s_idx, min(s_idx+6, len(ocues)))]) + "..."
            }

    return None


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
    print(f"   [基準]: フレーズ一致 >= 5 | 1件ごとに自動保存")
    print("=" * 65)

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

        print(f"\n[{i}/{len(items)}] 検査中: {sid} | {title[:35]}...")
        matched_res = align_single_shorts(sid, title)

        if not matched_res:
            print("  [-] 一致なし / スキップ")
            continue

        results.append(matched_res)
        processed_shorts_ids.add(sid)

        # 1件ヒットするごとにリアルタイム自動保存
        save_data(results)
        print(f"  🎉 【照合成功】累計: {len(results)} 件 | 元配信: {matched_res['origin_id']} ({matched_res['start_str']}〜{matched_res['end_str']}, 一致: {len(matched_res['hits'])}個)")

    print("\n" + "=" * 65)
    print(f"🏁 全件走査完了！ 合計 {len(results)} 件のマッチデータを保存しました。")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")
    print(f"HTML: {OUTPUT_HTML}")
    print("=" * 65)


if __name__ == '__main__':
    run_unlimited_collection()
