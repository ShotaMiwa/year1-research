# -*- coding: utf-8 -*-
"""
Shorts動画のURL/IDを渡すだけで、以下のパイプラインを全自動で実行するスクリプト：
1. Shortsのメタデータ（タイトル・再生数・いいね数・概要欄）を自動取得
2. 概要欄から「元動画」「本編」配下の候補URL（動画ID）を自動抽出
3. 候補元配信の自動字幕（VTT）を取得し、2段階ハイブリッド照合で該当区間 [T_start, T_end] を特定
4. （概要欄のコピペミス等で候補が不一致の場合）Shortsの特徴語から自動でYouTube検索フォールバック
"""

import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CHAT_DIR = PROJECT_ROOT / "data/raw/chat"
YTDLP_BIN = PROJECT_ROOT / ".venv/bin/yt-dlp"

def get_ytdlp_cmd():
    return str(YTDLP_BIN) if YTDLP_BIN.exists() else "yt-dlp"


def fetch_video_metadata(video_id_or_url: str) -> dict:
    """yt-dlp を用いて動画のメタデータ（タイトル、概要欄、いいね数、再生数等）を取得"""
    cmd = [
        get_ytdlp_cmd(),
        "--dump-json",
        "--skip-download",
        video_id_or_url
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception as e:
        print(f"[{video_id_or_url}] メタデータ取得エラー: {e}")
    return {}


def extract_candidate_ids_from_description(desc: str) -> list:
    """
    概要欄テキストから元動画・本編URL候補を抽出する。
    1. '元動画', '本編', '元配信', 'アーカイブ' などの見出し以降のURLを優先抽出
    2. 見出しがない場合は、概要欄内の全YouTube動画URL（watch?v= または live/）を抽出
    """
    if not desc:
        return []
    
    lines = desc.splitlines()
    in_origin_section = False
    priority_ids = []
    fallback_ids = []
    
    url_pattern = re.compile(r'https?://(?:www\.)?youtube\.com/(?:watch\?v=|live/)([a-zA-Z0-9_-]{11})')
    short_url_pattern = re.compile(r'https?://youtu\.be/([a-zA-Z0-9_-]{11})')
    
    for line in lines:
        if re.search(r'(?:元動画|本編|元配信|アーカイブ|切り抜き元|生放送URL|配信元)', line, re.IGNORECASE):
            in_origin_section = True
        
        matches = url_pattern.findall(line) + short_url_pattern.findall(line)
        if matches:
            if in_origin_section:
                priority_ids.extend(matches)
            else:
                fallback_ids.extend(matches)
    
    # 優先セクションがあればそれを採用、なければ全体のURLを採用
    chosen = priority_ids if priority_ids else fallback_ids
    # 重複除去（順序保持）
    return list(dict.fromkeys(chosen))


def search_youtube_fallback(query: str, max_results: int = 5) -> list:
    """概要欄から特定できなかった場合、YouTube検索で候補動画IDを自動取得"""
    cmd = [
        get_ytdlp_cmd(),
        "--dump-json",
        f"ytsearch{max_results}:{query}"
    ]
    candidate_ids = []
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                vid = d.get("id")
                # 概要欄に元動画URLが含まれている場合それも拾う
                desc = d.get("description", "")
                inner_ids = extract_candidate_ids_from_description(desc)
                if inner_ids:
                    candidate_ids.extend(inner_ids)
                if vid:
                    candidate_ids.append(vid)
            except:
                pass
    except Exception as e:
        print(f"検索フォールバックエラー: {e}")
    return list(dict.fromkeys(candidate_ids))[:10]


def download_subtitles_ytdlp(video_id: str) -> Path:
    """yt-dlp を使用して自動生成日本語字幕（VTT）を取得・保存する"""
    RAW_CHAT_DIR.mkdir(parents=True, exist_ok=True)
    vtt_path = RAW_CHAT_DIR / f"{video_id}.ja.vtt"
    if vtt_path.exists() and vtt_path.stat().st_size > 0:
        return vtt_path
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        get_ytdlp_cmd(),
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-lang", "ja",
        "-o", str(RAW_CHAT_DIR / f"{video_id}.%(ext)s"),
        video_url
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if vtt_path.exists() and vtt_path.stat().st_size > 0:
            return vtt_path
    except Exception as e:
        print(f"[{video_id}] 字幕ダウンロード失敗: {e}")
    return None


def parse_vtt_cues(vtt_path: Path):
    """VTT字幕ファイルをパースし、(start_sec, end_sec, clean_text) のリストを返す"""
    subs = []
    time_pat = re.compile(r'(\d+):(\d+):(\d+\.\d+)\s+-->\s+(\d+):(\d+):(\d+\.\d+)')
    cur_start, cur_end, cur_text = None, None, []

    with open(vtt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = time_pat.search(line)
            if m:
                if cur_start is not None and cur_text:
                    subs.append((cur_start, cur_end, ' '.join(cur_text)))
                h1, m1, s1, h2, m2, s2 = map(float, m.groups())
                cur_start = h1 * 3600 + m1 * 60 + s1
                cur_end   = h2 * 3600 + m2 * 60 + s2
                cur_text  = []
            elif line and not line.startswith('WEBVTT') and not line.isdigit() and 'align:' not in line and not line.startswith('Kind:'):
                cleaned = re.sub(r'<[^>]+>', '', line).strip()
                if cleaned and (not cur_text or cur_text[-1] != cleaned):
                    cur_text.append(cleaned)
    if cur_start is not None and cur_text:
        subs.append((cur_start, cur_end, ' '.join(cur_text)))
    return subs


def extract_robust_phrases(cues):
    """
    Shorts字幕の冒頭から末尾まで全体を網羅するように、
    6〜10文字の確実な部分文字列（完全一致照合用）を抽出する。
    1行丸ごとの完全一致は音声認識のゆらぎで失敗しやすいため、
    各文・各行から確実なまとまりをフレーズとしてサンプリング。
    """
    seen_texts = []
    for c in cues:
        txt = re.sub(r'\[.*?\]', '', c[2]).strip()
        if txt and (not seen_texts or txt != seen_texts[-1]):
            seen_texts.append(txt)
    
    full_text = ' '.join(seen_texts)
    # 漢字・ひらがな・カタカナ・英数の6〜10文字の塊を抽出
    raw_chunks = re.findall(r'[一-龠ぁ-んァ-ヶa-zA-Z0-9ー]{6,10}', full_text)
    
    # 助詞のみや定型句などのノイズを除外
    stops = {"ことについて", "ということは", "しているという", "質問してました"}
    unique_chunks = []
    for ch in raw_chunks:
        if ch not in stops and ch not in unique_chunks and not any(ch in u for u in unique_chunks):
            unique_chunks.append(ch)
    return unique_chunks


def format_seconds(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def auto_align_pipeline(shorts_input: str):
    """
    ShortsのURLまたはIDから、概要欄解析・字幕照合・元配信区間特定までを全自動実行
    """
    total_start = time.time()
    # Shorts IDの抽出
    m = re.search(r'([a-zA-Z0-9_-]{11})', shorts_input)
    shorts_id = m.group(1) if m else shorts_input
    shorts_url = f"https://www.youtube.com/shorts/{shorts_id}"
    
    print("="*60)
    print(f"🚀 全自動アライメントパイプライン開始: {shorts_url}")
    print("="*60)
    
    # 1. Shortsのメタデータ取得
    print("\n【Step 1: Shortsのメタデータ & 概要欄を取得中...】")
    meta = fetch_video_metadata(shorts_url)
    title = meta.get("title", "不明なタイトル")
    view_count = meta.get("view_count", 0)
    like_count = meta.get("like_count", 0)
    desc = meta.get("description", "")
    like_rate = (like_count / view_count * 100) if view_count > 0 else 0.0
    
    print(f"  タイトル: {title}")
    print(f"  再生回数: {view_count:,} 回 | いいね数: {like_count:,} 件 (いいね率: {like_rate:.2f}%)")
    
    # 2. 概要欄から元動画候補URLを自動抽出
    print("\n【Step 2: 概要欄から元動画URL候補を自動抽出中...】")
    candidates = extract_candidate_ids_from_description(desc)
    print(f"  概要欄から検出された候補ID ({len(candidates)}件): {candidates}")
    
    # 3. Shortsの字幕取得 & キーフレーズ生成
    print("\n【Step 3: Shorts字幕の取得 & キーフレーズ自動生成...】")
    shorts_vtt = download_subtitles_ytdlp(shorts_id)
    if not shorts_vtt:
        print("❌ Shortsの字幕取得に失敗しました。終了します。")
        return
    
    shorts_cues = parse_vtt_cues(shorts_vtt)
    if not shorts_cues:
        print("❌ Shortsの字幕パースに失敗しました。")
        return
    shorts_duration = shorts_cues[-1][1] - shorts_cues[0][0]
    
    # 尺のフィルタリング（設計方針: 40秒〜240秒）
    if shorts_duration < 40.0 or shorts_duration > 240.0:
        print(f"⚠️ Shorts尺が対象外（{shorts_duration:.1f}秒: 許容40s〜240s）です。スキップします。")
        return

    key_phrases = extract_robust_phrases(shorts_cues)
    print(f"  Shorts尺: {shorts_duration:.1f}秒, 抽出フレーズ数: {len(key_phrases)}個")
    
    # 4. 照合実行
    print(f"\n【Step 4: 候補元配信との高速完全一致照合を実行中...】")
    matched_segment = None
    
    def try_align(cand_list):
        for cid in cand_list:
            t0 = time.time()
            vtt = download_subtitles_ytdlp(cid)
            if not vtt:
                continue
            origin_cues = parse_vtt_cues(vtt)
            if not origin_cues:
                continue
            
            # 元配信キューごとの完全一致ヒットを走査
            matched_indices = []
            for idx, (_, _, txt) in enumerate(origin_cues):
                hits = [p for p in key_phrases if p in txt]
                if hits:
                    matched_indices.append((idx, hits))
            
            # クラスタリング（近傍25キュー以内のヒット群）
            clusters = []
            if matched_indices:
                cur = [matched_indices[0]]
                for item in matched_indices[1:]:
                    if item[0] - cur[-1][0] <= 25:
                        cur.append(item)
                    else:
                        u_hits = {h for _, hs in cur for h in hs}
                        if len(u_hits) >= 5:
                            clusters.append((cur, u_hits))
                        cur = [item]
                u_hits = {h for _, hs in cur for h in hs}
                if len(u_hits) >= 5:
                    clusters.append((cur, u_hits))
            
            el = time.time() - t0
            # 妥当性のあるクラスタ（Shorts尺との整合性チェック）
            valid_clusters = []
            for cl, u_hits in clusters:
                s_idx = max(0, cl[0][0] - 2)
                e_idx = min(len(origin_cues) - 1, cl[-1][0] + 1)
                s_sec = origin_cues[s_idx][0]
                e_sec = origin_cues[e_idx][1]
                dur = e_sec - s_sec
                # 尺の整合性: Shorts尺の0.5倍〜2.0倍の範囲内であること
                if 0.5 * shorts_duration <= dur <= 2.0 * shorts_duration:
                    valid_clusters.append((cl, u_hits, s_sec, e_sec, dur))
            
            if valid_clusters:
                cl, u_hits, s_sec, e_sec, dur = valid_clusters[0]
                return {
                    "origin_id": cid,
                    "start_sec": s_sec,
                    "end_sec": e_sec,
                    "duration": dur,
                    "hits": list(u_hits),
                    "elapsed": el
                }
            else:
                print(f"  [{cid}] 一致なし ({el:.3f}秒)")
        return None

    matched_segment = try_align(candidates)
    
    # 5. フォールバック（概要欄候補で不一致の場合）
    if not matched_segment:
        print("\n⚠️ 概要欄の候補URLでは一致が見つかりませんでした。")
        print("【Step 5: YouTube検索による自動フォールバック探索を実行中...】")
        # タイトルや特徴語から検索
        search_query = f"ひろゆき {' '.join(key_phrases[:3])}"
        print(f"  検索クエリ: {search_query}")
        fallback_candidates = search_youtube_fallback(search_query)
        print(f"  フォールバック候補ID ({len(fallback_candidates)}件): {fallback_candidates}")
        matched_segment = try_align(fallback_candidates)
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print(f"🎉 アライメント完了結果報告 (所要時間: {total_time:.2f}秒)")
    print("="*60)
    if matched_segment:
        res = matched_segment
        print(f"【特定成功】元配信動画ID: {res['origin_id']}")
        print(f"元配信URL: https://www.youtube.com/watch?v={res['origin_id']}&t={int(res['start_sec'])}s")
        print(f"特定区間: {format_seconds(res['start_sec'])} 〜 {format_seconds(res['end_sec'])} ({res['start_sec']:.1f}s 〜 {res['end_sec']:.1f}s)")
        print(f"セグメント尺: {res['duration']:.1f}秒 (Shorts尺: {shorts_duration:.1f}秒)")
        print(f"いいね率（擬似ラベル）: {like_rate:.2f}% (いいね: {like_count:,} / 再生: {view_count:,})")
        print(f"照合一致フレーズ数: {len(res['hits'])}個")
    else:
        print("❌ 元配信の特定に至りませんでした。")
    print("="*60)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/shorts/gbI8-X7YMBQ"
    auto_align_pipeline(target)
