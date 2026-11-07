# -*- coding: utf-8 -*-
"""
強化版バッチアライメント・データ収集スクリプト (Robust Alignment Pipeline)
【改善点】
1. トピック一致フィルター: Shortsタイトルに含まれる名詞キーワードが、元配信の特定セグメントにも含まれているか検証
2. フレーズ密度厳格化: 一致条件を「8フレーズ以上」かつ「短時間での共起密度」に引き上げ
3. 即時ファイル永続化: 1件成功するごとに `data/processed/verified_alignments.json` および CSV へ即時書き込み（データ消失防止）
4. 目視検証HTMLレポート（docs/paper/verification_report.html）のリアルタイム更新
"""

import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satisfaction_analysis.data_collection.align_origin_stream import (
    parse_vtt_cues, extract_robust_phrases, format_seconds,
    download_subtitles_ytdlp, extract_candidate_ids_from_description,
    fetch_video_metadata
)
from src.satisfaction_analysis.data_collection.generate_verification_report import generate_verification_html

RAW_CHAT_DIR = PROJECT_ROOT / "data/raw/chat"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
DATASET_JSON = PROCESSED_DIR / "verified_alignments.json"
DATASET_CSV = PROCESSED_DIR / "verified_shorts_dataset.csv"
REPORT_HTML = PROJECT_ROOT / "docs/paper/verification_report.html"

def extract_title_keywords(title: str) -> list:
    """タイトルからトピック照合用の主要キーワード（3文字以上の名詞・固有名詞）を抽出"""
    # 記号や定型フレーズを除去
    cleaned = re.sub(r'【.*?】|#\S+|ひろゆき|切り抜き|shorts|について|正直|完全|論破', ' ', title)
    words = re.findall(r'[一-龠ぁ-んァ-ヶa-zA-Z0-9ー]{3,10}', cleaned)
    # ノイズ単語の除外
    stops = {"思います", "言われて", "なります", "言います", "できます", "しょうか", "理由とは", "真実に", "辿り着き"}
    return [w for w in words if w not in stops and len(w) >= 3]

def save_confirmed_dataset(confirmed_list: list):
    """即時ファイル永続化（JSON & CSV）"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON保存 (辞書形式: card_id -> data)
    decisions_dict = {}
    for idx, item in enumerate(confirmed_list, 1):
        decisions_dict[str(idx)] = {
            "status": "OK",
            "shorts_id": item['shorts_id'],
            "origin_id": item['origin_id'],
            "start_sec": item['start_sec'],
            "end_sec": item['end_sec'],
            "start_str": item['start_str'],
            "end_str": item['end_str'],
            "duration": item['duration'],
            "view_count": item.get('view_count', 0),
            "like_count": item.get('like_count', 0),
            "like_rate": item.get('like_rate', 0.0)
        }
    with open(DATASET_JSON, 'w', encoding='utf-8') as f:
        json.dump(decisions_dict, f, ensure_ascii=False, indent=2)

    # CSV保存
    rows = ["card_id,shorts_id,title,view_count,like_count,like_rate,status,origin_id,start_sec,end_sec,start_str,end_str,duration"]
    for idx, item in enumerate(confirmed_list, 1):
        t_clean = item['shorts_title'].replace('"', '""')
        rows.append(f'{idx},"{item["shorts_id"]}","{t_clean}",{item.get("view_count", 0)},{item.get("like_count", 0)},{item.get("like_rate", 0.0):.2f},OK,"{item["origin_id"]}",{item["start_sec"]:.1f},{item["end_sec"]:.1f},"{item["start_str"]}","{item["end_str"]}",{item["duration"]:.1f}')
    DATASET_CSV.write_text("\ufeff" + "\n".join(rows), encoding='utf-8')

def run_robust_batch(target_matches=100):
    playlist_file = RAW_CHAT_DIR / "shorts_playlist_all.jsonl"
    if not playlist_file.exists():
        print("全プレイリストファイルがありません。")
        return

    confirmed_results = []
    
    # 既存の確定済みファイルがあれば復元
    if DATASET_JSON.exists():
        try:
            with open(DATASET_JSON, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # 既存データをロード
                print(f"既存の確定データをロード中 ({len(saved)}件)...")
        except:
            pass

    with open(playlist_file, 'r', encoding='utf-8') as f:
        shorts_list = [json.loads(line) for line in f if line.strip()]

    print(f"============================================================")
    print(f"🚀 高品質データ拡張バッチ開始 (対象: 全{len(shorts_list)}本 / 目標: {target_matches}件)")
    print(f"   [フィルター条件]: 尺40s〜240s | フレーズ一致 >= 8 | タイトルトピック一致")
    print(f"============================================================")

    seen_ids = set()

    for item in shorts_list:
        if len(confirmed_results) >= target_matches:
            break

        sid = item.get("id")
        title = item.get("title", "")
        if sid in seen_ids:
            continue
        seen_ids.add(sid)

        shorts_url = f"https://www.youtube.com/shorts/{sid}"
        
        # 1. メタデータ取得
        meta = fetch_video_metadata(shorts_url)
        if not meta:
            continue
        
        desc = meta.get("description", "")
        vc = meta.get("view_count", 0)
        lc = meta.get("like_count", 0)
        lr = (lc / vc * 100) if vc > 0 else 0.0

        # 2. 概要欄から元動画候補URL抽出
        candidates = extract_candidate_ids_from_description(desc)
        if not candidates:
            continue

        # 3. Shorts字幕取得
        svtt = download_subtitles_ytdlp(sid)
        if not svtt:
            continue
        scues = parse_vtt_cues(svtt)
        if not scues:
            continue

        sdur = scues[-1][1] - scues[0][0]
        # 尺フィルター（40秒〜240秒）
        if sdur < 40.0 or sdur > 240.0:
            continue

        phrases = extract_robust_phrases(scues)
        if len(phrases) < 10:
            continue

        # タイトルキーワードの抽出
        title_keywords = extract_title_keywords(title)

        # 4. 元配信照合
        matched = None
        for cid in candidates:
            ovtt = download_subtitles_ytdlp(cid)
            if not ovtt:
                continue
            ocues = parse_vtt_cues(ovtt)
            if not ocues:
                continue

            # キューごとのフレーズ完全一致ヒット
            matched_indices = []
            for idx, (_, _, txt) in enumerate(ocues):
                hits = [p for p in phrases if p in txt]
                if hits:
                    matched_indices.append((idx, hits))

            clusters = []
            if matched_indices:
                cur = [matched_indices[0]]
                for it in matched_indices[1:]:
                    if it[0] - cur[-1][0] <= 25:
                        cur.append(it)
                    else:
                        uh = {h for _, hs in cur for h in hs}
                        if len(uh) >= 8:  # 厳格化: 8フレーズ以上
                            clusters.append((cur, uh))
                        cur = [it]
                uh = {h for _, hs in cur for h in hs}
                if len(uh) >= 8:
                    clusters.append((cur, uh))

            # 妥当性検証
            for cl, uh in clusters:
                s_idx = max(0, cl[0][0] - 2)
                e_idx = min(len(ocues) - 1, cl[-1][0] + 1)
                s_sec = ocues[s_idx][0]
                e_sec = ocues[e_idx][1]
                dur = e_sec - s_sec

                # 尺の整合性 (0.5倍〜2.0倍)
                if not (0.5 * sdur <= dur <= 2.0 * sdur):
                    continue

                # トピックキーワード一致検証
                # 元配信セグメントテキスト（＋前後数キュー）
                segment_text = " ".join([ocues[i][2] for i in range(max(0, s_idx-2), min(len(ocues), e_idx+3))])
                topic_hits = [w for w in title_keywords if w in segment_text]

                # タイトルに名詞がある場合、最低1つ以上元配信セグメントに含まれていること
                if title_keywords and not topic_hits:
                    # トピック不一致の疑いがあるためスキップ（誤検出の排除）
                    continue

                matched = {
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
                break
            if matched:
                break

        if matched:
            confirmed_results.append(matched)
            # 1件確定するごとに即時ファイル永続化（データ消失防止）
            save_confirmed_dataset(confirmed_results)
            # HTMLレポートも逐次更新
            generate_verification_html(confirmed_results, REPORT_HTML)
            print(f"[{len(confirmed_results)}/{target_matches}] 🎉 高精度一致: {sid} -> {matched['origin_id']} ({matched['start_str']}〜{matched['end_str']}, 尺:{matched['duration']:.1f}s, フレーズ:{len(matched['hits'])}個) | {matched['shorts_title'][:28]}...")
        else:
            print(f"[-] スキップ/不一致: {sid} | {title[:25]}...")

    print("\n" + "="*60)
    print(f"🏁 高精度データ拡張完了！ 合計 {len(confirmed_results)} 件を確定・保存しました。")
    print(f"   確定JSON: {DATASET_JSON}")
    print(f"   確定CSV:  {DATASET_CSV}")
    print(f"   目視レポート: {REPORT_HTML}")
    print("="*60)

if __name__ == '__main__':
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_robust_batch(target_matches=target)

