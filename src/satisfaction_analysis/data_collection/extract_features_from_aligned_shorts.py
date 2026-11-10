# -*- coding: utf-8 -*-
"""
extract_features_from_aligned_shorts.py

verified_alignments.json（79件の確定アライメント）をもとに、
各元配信アーカイブの対応区間 [start_sec, end_sec] から以下の10特徴量を算出する：
  1. comment_rate              — チャットコメント数 / 秒
  2. grass_ratio               — 草・笑い比率
  3. question_ratio            — 疑問文比率
  4. exclamation_ratio         — 感嘆符比率
  5. sentiment_polarity        — コメントBERT感情極性 (pos-neg)/(pos+neg)
  6. subtitle_sentiment_polarity — 字幕BERT感情極性
  7. speech_rate               — 字幕ブロック数 / 秒
  8. pause_ratio               — 無音時間率
  9. sub_comment_similarity    — 字幕-コメント Jaccard類似度
 10. comment_semantic_variance — コメント意味多様性 (TTR)

さらに擬似ラベルとして:
  - like_rate  — いいね率（%）

出力:
  data/processed/extended_feature_matrix_79.csv
  docs/paper/tables/exp_extended_features_79.csv
"""

import re
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CHAT_DIR = PROJECT_ROOT / "data/raw/chat"
YTDLP_BIN = PROJECT_ROOT / ".venv/bin/yt-dlp"
ALIGNMENT_JSON = PROJECT_ROOT / "data/processed/verified_alignments.json"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/extended_feature_matrix_79.csv"
OUTPUT_TABLE_CSV = PROJECT_ROOT / "docs/paper/tables/exp_extended_features_79.csv"

# BERT感情分析（transformers）
try:
    from transformers import pipeline
    _SENTIMENT_PIPE = pipeline(
        "sentiment-analysis",
        model="koheiduck/bert-japanese-finetuned-sentiment",
        truncation=True,
        max_length=128,
        device=-1,  # CPU
    )
    BERT_AVAILABLE = True
    print("✅ BERT感情分析モデルをロードしました。")
except Exception as e:
    _SENTIMENT_PIPE = None
    BERT_AVAILABLE = False
    print(f"⚠️ BERTモデルが使えません（{e}）。ルールベース極性にフォールバックします。")


# =====================================================
# ユーティリティ関数
# =====================================================

GRASS_RE = re.compile(r'[wｗ草]+|笑+', re.IGNORECASE)
QUEST_RE = re.compile(r'[？?]+|何|どう|誰|いつ|どこ|なぜ|ですか|ますか')
EXCLA_RE = re.compile(r'[！!]+')
# ルールベース感情極性（BERT未利用時）
POS_RE = re.compile(r'すごい|最高|面白|草|笑|好き|ありがとう|神|感動|素晴らし|最強|楽し')
NEG_RE = re.compile(r'最悪|つまらない|嫌い|ひどい|つらい|辛い|悲しい|無理|最低|怖い')


def ytdlp_cmd():
    return str(YTDLP_BIN) if YTDLP_BIN.exists() else "yt-dlp"


def download_live_chat(video_id: str) -> Path:
    """yt-dlp でライブチャットJSONを取得してパスを返す"""
    out_path = RAW_CHAT_DIR / f"{video_id}.live_chat.json"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    RAW_CHAT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        ytdlp_cmd(),
        "--write-subs",
        "--sub-langs", "live_chat",
        "--skip-download",
        "-o", str(RAW_CHAT_DIR / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as e:
        print(f"  [chat] yt-dlp失敗 ({video_id}): {e}")
        return None
    return out_path if out_path.exists() else None


def download_vtt(video_id: str) -> Path:
    """yt-dlp で日本語VTT字幕を取得してパスを返す"""
    out_path = RAW_CHAT_DIR / f"{video_id}.ja.vtt"
    if out_path.exists() and out_path.stat().st_size > 100:
        return out_path
    RAW_CHAT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        ytdlp_cmd(),
        "--write-auto-sub",
        "--sub-langs", "ja",
        "--skip-download",
        "-o", str(RAW_CHAT_DIR / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as e:
        print(f"  [vtt] yt-dlp失敗 ({video_id}): {e}")
        return None
    return out_path if out_path.exists() else None


def parse_live_chat(chat_path: Path):
    """live_chat.json を解析して [(offset_sec, text), ...] を返す"""
    records = []
    if not chat_path or not chat_path.exists():
        return records
    with open(chat_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            actions = obj.get("replayChatItemAction", {}).get("actions", [])
            for action in actions:
                item = action.get("addChatItemAction", {}).get("item", {})
                renderer = (
                    item.get("liveChatTextMessageRenderer")
                    or item.get("liveChatPaidMessageRenderer")
                )
                if renderer:
                    runs = renderer.get("message", {}).get("runs", [])
                    text = "".join(r.get("text", "") for r in runs)
                    offset_ms = obj.get("videoOffsetTimeMsec") or obj.get(
                        "replayChatItemAction", {}
                    ).get("videoOffsetTimeMsec")
                    if offset_ms is not None and text.strip():
                        records.append((float(offset_ms) / 1000.0, text))
    return records


def vtt_to_sec(t: str) -> float:
    parts = t.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0


def parse_vtt(vtt_path: Path):
    """VTT字幕 -> [(start_sec, end_sec, text), ...]"""
    if not vtt_path or not vtt_path.exists():
        return []
    content = vtt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\n+", content)
    records = []
    for block in blocks:
        lines = block.strip().split("\n")
        time_line = next((l for l in lines if "-->" in l), None)
        if not time_line:
            continue
        parts = time_line.split("-->")
        if len(parts) < 2:
            continue
        try:
            s = vtt_to_sec(parts[0].strip().split()[-1])
            e = vtt_to_sec(parts[1].strip().split()[0])
        except Exception:
            continue
        text_lines = []
        past_time = False
        for l in lines:
            if "-->" in l:
                past_time = True
                continue
            if past_time:
                clean = re.sub(r"<[^>]+>", "", l).strip()
                if clean:
                    text_lines.append(clean)
        text = " ".join(text_lines).strip()
        if text and e > s:
            records.append((s, e, text))
    return records


def sentiment_polarity_bert(texts: list) -> float:
    """BERTで感情極性を計算"""
    if not texts or not BERT_AVAILABLE:
        return _rule_sentiment(texts)
    sample = texts[:30]  # 多すぎると重い
    try:
        results = _SENTIMENT_PIPE(sample)
        pos = sum(1 for r in results if r["label"].upper() in ("POSITIVE", "POS"))
        neg = sum(1 for r in results if r["label"].upper() in ("NEGATIVE", "NEG"))
        total = pos + neg
        return (pos - neg) / (total + 1e-9) if total > 0 else 0.0
    except Exception:
        return _rule_sentiment(texts)


def _rule_sentiment(texts: list) -> float:
    """ルールベース感情極性フォールバック"""
    if not texts:
        return 0.0
    pos = sum(1 for t in texts if POS_RE.search(t))
    neg = sum(1 for t in texts if NEG_RE.search(t))
    total = pos + neg
    return (pos - neg) / (total + 1e-9) if total > 0 else 0.0


def compute_features(origin_id: str, start_sec: float, end_sec: float):
    """
    元配信ID + 時間区間 -> 10特徴量の辞書を返す
    チャット・字幕がどちらも取得できない場合は None を返す
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return None

    # チャット取得
    chat_path = download_live_chat(origin_id)
    all_chats = parse_live_chat(chat_path)
    seg_chats = [t for s, t in all_chats if start_sec <= s < end_sec]
    n_comments = len(seg_chats)

    # 字幕取得
    vtt_path = download_vtt(origin_id)
    all_subs = parse_vtt(vtt_path)
    seg_subs = [(s, e, t) for s, e, t in all_subs if not (e <= start_sec or s >= end_sec)]
    n_subs = len(seg_subs)

    if n_comments == 0 and n_subs == 0:
        return None

    # ----- チャット系特徴量 -----
    comment_rate = n_comments / duration

    if n_comments > 0:
        grass_ratio = sum(1 for t in seg_chats if GRASS_RE.search(t)) / n_comments
        question_ratio = sum(1 for t in seg_chats if QUEST_RE.search(t)) / n_comments
        exclamation_ratio = sum(1 for t in seg_chats if EXCLA_RE.search(t)) / n_comments
        all_words = re.findall(r'\w+', ' '.join(seg_chats))
        comment_semantic_variance = (
            len(set(all_words)) / len(all_words) if all_words else 0.0
        )
        sentiment_polarity = sentiment_polarity_bert(seg_chats)
    else:
        grass_ratio = question_ratio = exclamation_ratio = 0.0
        comment_semantic_variance = 0.0
        sentiment_polarity = 0.0

    # ----- 字幕系特徴量 -----
    speech_rate = n_subs / duration if duration > 0 else 0.0

    if n_subs > 0:
        total_speech = sum(
            min(e, end_sec) - max(s, start_sec) for s, e, _ in seg_subs
        )
        pause_ratio = max(0.0, 1.0 - total_speech / duration)

        sub_chars = set(re.findall(r'\w', ' '.join(t for _, _, t in seg_subs)))
        chat_chars = set(re.findall(r'\w', ' '.join(seg_chats))) if seg_chats else set()
        union = sub_chars | chat_chars
        sub_comment_similarity = (
            len(sub_chars & chat_chars) / len(union) if union else 0.0
        )

        sub_texts = [t for _, _, t in seg_subs[:30]]
        subtitle_sentiment_polarity = sentiment_polarity_bert(sub_texts)
    else:
        pause_ratio = 1.0
        sub_comment_similarity = 0.0
        subtitle_sentiment_polarity = 0.0

    return {
        "comment_rate": round(comment_rate, 4),
        "grass_ratio": round(grass_ratio, 4),
        "question_ratio": round(question_ratio, 4),
        "exclamation_ratio": round(exclamation_ratio, 4),
        "sentiment_polarity": round(sentiment_polarity, 6),
        "subtitle_sentiment_polarity": round(subtitle_sentiment_polarity, 6),
        "speech_rate": round(speech_rate, 4),
        "pause_ratio": round(pause_ratio, 4),
        "sub_comment_similarity": round(sub_comment_similarity, 4),
        "comment_semantic_variance": round(comment_semantic_variance, 4),
        "n_comments": n_comments,
        "n_subtitles": n_subs,
    }


# =====================================================
# メイン処理
# =====================================================

def main():
    with open(ALIGNMENT_JSON, encoding="utf-8") as f:
        alignments = json.load(f)

    print(f"対象: {len(alignments)} 件のアライメント")
    print("=" * 60)

    rows = []
    errors = []
    seen_origins = {}  # origin_id -> (chat_records, vtt_records) キャッシュ

    for card_id, rec in alignments.items():
        if rec.get("status") == "NG":
            print(f"[{card_id}] NG判定につきスキップ")
            continue

        origin_id = rec["origin_id"]
        start_sec = float(rec["start_sec"])
        end_sec = float(rec["end_sec"])
        shorts_id = rec["shorts_id"]
        like_rate = float(rec.get("like_rate", 0.0))
        shorts_title = rec.get("shorts_title", "")
        duration = end_sec - start_sec

        print(f"\n[{card_id}] {shorts_id} -> 元配信: {origin_id} ({rec.get('start_str')}〜{rec.get('end_str')}, {duration:.0f}s)")

        feats = compute_features(origin_id, start_sec, end_sec)
        if feats is None:
            print(f"  ⚠️ チャット・字幕どちらも取得できませんでした。スキップ。")
            errors.append(card_id)
            continue

        row = {
            "card_id": card_id,
            "shorts_id": shorts_id,
            "origin_id": origin_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "segment_length_sec": round(duration, 2),
            "like_rate": round(like_rate, 4),
            "shorts_title": shorts_title,
        }
        row.update(feats)
        rows.append(row)
        print(f"  ✅ 特徴量抽出完了: comments={feats['n_comments']}, subs={feats['n_subtitles']}, "
              f"comment_rate={feats['comment_rate']}, grass={feats['grass_ratio']:.3f}, "
              f"question={feats['question_ratio']:.3f}")

        # レート制限回避
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ 有効なデータが1件もありませんでした。")
        return

    # z-score 正規化（10特徴量）
    feat_cols = [
        "comment_rate", "grass_ratio", "question_ratio", "exclamation_ratio",
        "sentiment_polarity", "subtitle_sentiment_polarity",
        "speech_rate", "pause_ratio", "sub_comment_similarity", "comment_semantic_variance"
    ]
    for col in feat_cols:
        vals = df[col]
        mu, sigma = vals.mean(), vals.std()
        df[f"{col}_zscore"] = ((vals - mu) / sigma).round(6) if sigma > 0 else 0.0

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUT_TABLE_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"🎉 特徴量抽出完了！")
    print(f"   有効件数: {len(df)} / {len(alignments)} 件")
    print(f"   スキップ: {len(errors)} 件 ({errors})")
    print(f"   出力: {OUTPUT_CSV}")
    print(f"   出力: {OUTPUT_TABLE_CSV}")
    print("=" * 60)
    print("\n--- 統計サマリー ---")
    print(df[feat_cols + ["like_rate", "segment_length_sec"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()

