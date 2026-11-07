# -*- coding: utf-8 -*-
"""
@hiroyukike チャンネルの全Shorts（776本）を対象に、
目標件数の制限なし（全件走査）で元配信アーカイブとの高精度アライメントを実行するスクリプト。

特徴:
- フレーズ一致閾値: 5箇所以上（前回の実績基準）
- 尺: 40s〜240s
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
    fetch_video_metadata,
    extract_candidate_ids_from_description,
    download_subtitles_ytdlp,
    parse_vtt_cues,
    extract_robust_phrases,
    format_seconds,
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

def match_candidate(sid, scues, sdur, phrases, cid):
    ovtt = download_subtitles_ytdlp(cid)
    if not ovtt:
        return None
    ocues = parse_vtt_cues(ovtt)
    if not ocues:
        return None

    matched_cues_idx = []
    matched_phrases = set()

    for idx, c in enumerate(ocues):
        text = c[2]
        for p in phrases:
            if p in text:
                matched_cues_idx.append(idx)
                matched_phrases.add(p)
                break

    if len(matched_phrases) < 5:
        return None

    # クラスタリング（近接する字幕キューの連続区間を特定）
    clusters = []
    cur = [matched_cues_idx[0]]
    for idx in matched_cues_idx[1:]:
        if idx - cur[-1] <= 15:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)

    best_cluster = None
    best_unique_hits = set()

    for cl in clusters:
        uh = set()
        for idx in cl:
            for p in phrases:
                if p in ocues[idx][2]:
                    uh.add(p)
        if len(uh) >= 5 and len(uh) > len(best_unique_hits):
            best_unique_hits = uh
            best_cluster = cl

    if not best_cluster:
        return None

    s_idx = best_cluster[0]
    e_idx = best_cluster[-1]
    s_sec = ocues[s_idx][0]
    e_sec = ocues[e_idx][1]
    dur = e_sec - s_sec

    # 尺チェック: Shortsの尺の0.5倍〜2.0倍の範囲内
    if dur < sdur * 0.5 or dur > sdur * 2.0:
        return None

    return {
        'origin_id': cid,
        'start_sec': s_sec,
        'end_sec': e_sec,
        'duration': dur,
        'start_str': format_seconds(s_sec),
        'end_str': format_seconds(e_sec),
        'hits': list(best_unique_hits)[:10],
        'shorts_text_snippet': " ".join([c[2] for c in scues[:6]]) + "...",
        'origin_text_snippet': " ".join([ocues[i][2] for i in range(s_idx, min(s_idx+6, len(ocues)))]) + "..."
    }

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

        shorts_url = f"https://www.youtube.com/shorts/{sid}"
        meta = fetch_video_metadata(shorts_url)
        if not meta:
            print("  メタデータ取得失敗、スキップ")
            continue

        desc = meta.get("description", "")
        vc = meta.get("view_count", 0)
        lc = meta.get("like_count", 0)
        lr = (lc / vc * 100) if vc > 0 else 0.0

        candidates = extract_candidate_ids_from_description(desc)
        if not candidates:
            print("  概要欄に元動画リンクなし、スキップ")
            continue

        svtt = download_subtitles_ytdlp(sid)
        if not svtt:
            print("  Shorts字幕取得失敗、スキップ")
            continue

        scues = parse_vtt_cues(svtt)
        if not scues:
            print("  Shorts字幕パース失敗、スキップ")
            continue

        sdur = scues[-1][1] - scues[0][0]
        if sdur < 40.0 or sdur > 240.0:
            print(f"  尺対象外（{sdur:.1f}秒: 40s〜240s外）、スキップ")
            continue

        phrases = extract_robust_phrases(scues)
        if len(phrases) < 5:
            print(f"  フレーズ抽出不足（{len(phrases)}個）、スキップ")
            continue

        matched = None
        for cid in candidates:
            matched = match_candidate(sid, scues, sdur, phrases, cid)
            if matched:
                break

        if not matched:
            print("  一致セグメントなし（条件不一致）")
            continue

        record = {
            'shorts_id': sid,
            'shorts_title': title,
            'shorts_duration': sdur,
            'view_count': vc,
            'like_count': lc,
            'like_rate': lr,
            'origin_id': matched['origin_id'],
            'start_sec': matched['start_sec'],
            'end_sec': matched['end_sec'],
            'duration': matched['duration'],
            'start_str': matched['start_str'],
            'end_str': matched['end_str'],
            'hits': matched['hits'],
            'shorts_text_snippet': matched['shorts_text_snippet'],
            'origin_text_snippet': matched['origin_text_snippet']
        }

        results.append(record)
        processed_shorts_ids.add(sid)

        save_data(results)
        print(f"  🎉 【照合成功】累計: {len(results)} 件 | 元配信: {record['origin_id']} ({record['start_str']}〜{record['end_str']}, 一致: {len(record['hits'])}個)")

    print("\n" + "=" * 65)
    print(f"🏁 全件走査完了！ 合計 {len(results)} 件のマッチデータを保存しました。")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")
    print(f"HTML: {OUTPUT_HTML}")
    print("=" * 65)

if __name__ == '__main__':
    run_unlimited_collection()
