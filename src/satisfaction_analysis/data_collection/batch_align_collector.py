# -*- coding: utf-8 -*-
"""
チャンネル一覧から順次Shortsを検査し、
照合に成功した動画（目標N件）を収集して検証レポートHTMLを出力するバッチスクリプト。
"""

import json
import time
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CHAT_DIR = PROJECT_ROOT / "data/raw/chat"
REPORT_HTML = PROJECT_ROOT / "docs/paper/verification_report.html"

def run_pilot_batch(target_matches=100):
    playlist_file = RAW_CHAT_DIR / "shorts_playlist_300.jsonl"
    if not playlist_file.exists():
        playlist_file = RAW_CHAT_DIR / "shorts_playlist_100.jsonl"
    if not playlist_file.exists():
        print("プレイリストファイルがありません。")
        return

    confirmed_results = []
    
    # 既存の確定済み2件（gbI8-X7YMBQ, 06HqMnn9UPY）をまず追加
    confirmed_results.append({
        'shorts_id': 'gbI8-X7YMBQ',
        'shorts_title': '【ひろゆき】ショートスリーパー堀大輔について語るひろゆき #shorts',
        'shorts_duration': 57.8,
        'like_rate': 3.82,
        'origin_id': '9XlbakibD1Q',
        'start_sec': 9194.2,
        'end_sec': 9238.4,
        'duration': 44.2,
        'start_str': '02:33:14',
        'end_str': '02:33:58',
        'hits': ['コメンテーターとして', '質問してましたが', 'ショートスリーパー堀', 'カリフォルニアバーク', '生活をしてるかって', '一応質問をしてます'],
        'shorts_text_snippet': '安倍マネの睡眠についての中でひ樹さんも コメンテーターとして質問してましたが ショートスリーパー堀大輔さんを完全無視... カリフォルニアバークレ大学の睡眠学をちゃんとやってる先生が来たので...',
        'origin_text_snippet': 'の睡眠についての中で弘樹さんも コメンテーターとして質問してましたが ショートスリーパー堀大輔さんを完全った... え何時が前じゃなくて多分ね半年ぐらい前なやつですね一応質問をしてますけど...'
    })
    confirmed_results.append({
        'shorts_id': '06HqMnn9UPY',
        'shorts_title': '【ひろゆき】事態が急変しました…急いで●●の備蓄を済ませてください#shorts',
        'shorts_duration': 52.0,
        'like_rate': 0.37,
        'origin_id': 'pUsi5NQQgrA',
        'start_sec': 8611.6,
        'end_sec': 8669.8,
        'duration': 58.2,
        'start_str': '02:23:31',
        'end_str': '02:24:29',
        'hits': ['11月2日に防災庁を', '備蓄の緊急要請すると', '同じがやってるん', '備蓄というのを', '国民にやらせるより国', 'ゆとり世代サラリーマ', '民間持つのか'],
        'shorts_text_snippet': '11月2日に防災庁を設置するというニュースで備蓄の緊急要請すると... 国民にやらせるより国がやった方がいいんじゃないかなという風に思うんですけど...',
        'origin_text_snippet': '11月2日に防災庁を設置するというニュースで備蓄の緊急要請すると... 国民にやらせるより国がやった方がいいんじゃないかなという風に思うんですけど... 要はさっき言ったようなそのコインが...'
    })

    print(f"初期確定済み件数: {len(confirmed_results)}件。目標照合数: {target_matches}件")
    
    with open(playlist_file, 'r', encoding='utf-8') as f:
        shorts_list = [json.loads(line) for line in f if line.strip()]

    # 既に確定済みのIDを除外
    seen_ids = {r['shorts_id'] for r in confirmed_results}
    
    for item in shorts_list:
        if len(confirmed_results) >= target_matches:
            break
        
        sid = item.get("id")
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        
        shorts_url = f"https://www.youtube.com/shorts/{sid}"
        print(f"\n==========================================")
        print(f"🔍 調査中: {sid} | {item.get('title')[:30]}...")
        
        # メタデータ取得
        meta = fetch_video_metadata(shorts_url)
        if not meta:
            print("  メタデータ取得失敗、スキップ")
            continue
        
        desc = meta.get("description", "")
        vc = meta.get("view_count", 0)
        lc = meta.get("like_count", 0)
        lr = (lc / vc * 100) if vc > 0 else 0.0
        
        # 概要欄から候補抽出
        candidates = extract_candidate_ids_from_description(desc)
        if not candidates:
            print("  概要欄に候補URLなし、スキップ")
            continue
        print(f"  検出された候補元配信 ({len(candidates)}件): {candidates}")
        
        # 字幕取得
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
            print(f"  尺が対象外（{sdur:.1f}秒: 40s〜240s外）、スキップ")
            continue
        
        phrases = extract_robust_phrases(scues)
        if len(phrases) < 10:
            print(f"  抽出フレーズ僅少（{len(phrases)}個）、スキップ")
            continue
        
        # 照合
        matched = None
        for cid in candidates:
            ovtt = download_subtitles_ytdlp(cid)
            if not ovtt:
                continue
            ocues = parse_vtt_cues(ovtt)
            if not ocues:
                continue
            
            # キューごとのヒット
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
                        if len(uh) >= 5:
                            clusters.append((cur, uh))
                        cur = [it]
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
                matched = {
                    'shorts_id': sid,
                    'shorts_title': meta.get('title', item.get('title')),
                    'shorts_duration': sdur,
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
                print(f"  🎉 照合成功！ 元配信: {cid} ({matched['start_str']} 〜 {matched['end_str']}, 尺: {dur:.1f}s, 一致数: {len(uh)})")
                break
        
        if matched:
            confirmed_results.append(matched)
            # 1件成功するごとにレポートを逐次更新
            generate_verification_html(confirmed_results, REPORT_HTML)
            print(f"  現在の一致確定数: {len(confirmed_results)} / {target_matches}")
        else:
            print("  一致セグメントなし")

    print("\n" + "="*60)
    print(f"🏁 バッチ処理完了！ 合計 {len(confirmed_results)} 件の一致をレポートに出力しました。")
    print(f"レポート先: {REPORT_HTML}")
    print("="*60)

if __name__ == '__main__':
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_pilot_batch(target_matches=target)
