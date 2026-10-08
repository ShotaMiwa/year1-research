#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実チャットログ・字幕VTTから全13特徴量を実計算して feature_matrix.csv を更新する
"""
import re, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/home/shota/work/year1")
CHAT_JSON    = PROJECT_ROOT / "data/raw/chat/pP2KLW-_7hQ.live_chat.json"
VTT_FILE     = PROJECT_ROOT / "data/raw/chat/pP2KLW-_7hQ.ja.vtt"
BERT_SEG_CSV = PROJECT_ROOT / "outputs/run_20260624_071441/segment_sentiment_comparison_results.csv"
SURVEY_CSV   = PROJECT_ROOT / "data/processed/merged_survey_heatmap.csv"
OUT_MATRIX   = PROJECT_ROOT / "data/processed/feature_matrix.csv"

# ============================================================
# 1) セグメント情報の読み込み
# ============================================================
df_survey = pd.read_csv(SURVEY_CSV)
segments = df_survey[['セグメント','開始(秒)','終了(秒)']].copy()
print(f"Segments: {len(segments)}")

# ============================================================
# 2) ライブチャットJSONの解析
# ============================================================
print("Parsing live_chat.json ...")
chat_records = []
with open(CHAT_JSON, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # yt-dlp の live_chat.json フォーマット
        # 各行が1イベント。replayChatItemAction の中に chatItem がある
        actions = obj.get('replayChatItemAction', {}).get('actions', [])
        for action in actions:
            item = action.get('addChatItemAction', {}).get('item', {})
            
            # 通常コメント
            renderer = item.get('liveChatTextMessageRenderer', {})
            if not renderer:
                renderer = item.get('liveChatPaidMessageRenderer', {})
            
            if renderer:
                # テキスト取得
                msg_runs = renderer.get('message', {}).get('runs', [])
                text = ''.join(r.get('text', '') for r in msg_runs)
                
                # 動画内オフセット時刻（マイクロ秒）
                offset_us = obj.get('videoOffsetTimeMsec')
                if offset_us is None:
                    offset_us = obj.get('replayChatItemAction', {}).get('videoOffsetTimeMsec')
                if offset_us is not None:
                    offset_sec = int(offset_us) / 1000.0
                else:
                    continue
                
                if text:
                    chat_records.append({'time_sec': offset_sec, 'text': text})

print(f"Parsed {len(chat_records)} chat messages")
df_chat = pd.DataFrame(chat_records)

if len(df_chat) == 0:
    print("WARNING: No chat messages parsed. Trying alternate format...")
    # フォールバック: 別フォーマット試行
    with open(CHAT_JSON, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # フラット形式
                if 'message' in obj and 'time_in_seconds' in obj:
                    chat_records.append({'time_sec': obj['time_in_seconds'], 'text': obj['message']})
            except:
                pass
    df_chat = pd.DataFrame(chat_records)
    print(f"Fallback: {len(df_chat)} chat messages")

# ============================================================
# 3) VTT字幕のパース
# ============================================================
print("Parsing VTT subtitles ...")

def vtt_time_to_sec(t):
    """HH:MM:SS.mmm -> seconds"""
    parts = t.strip().replace(',', '.').split(':')
    if len(parts) == 3:
        return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0])*60 + float(parts[1])
    return 0.0

subtitle_records = []
vtt_content = VTT_FILE.read_text(encoding='utf-8', errors='ignore')
# VTT ブロックの解析
blocks = re.split(r'\n\n+', vtt_content)
for block in blocks:
    lines = block.strip().split('\n')
    # タイムコード行を探す
    time_line = None
    for l in lines:
        if '-->' in l:
            time_line = l
            break
    if time_line is None:
        continue
    
    # タイムコードの解析
    time_parts = time_line.split('-->')
    if len(time_parts) < 2:
        continue
    start_str = time_parts[0].strip().split()[-1]  # align等の付属情報を除去
    end_str   = time_parts[1].strip().split()[0]
    
    try:
        start_sec = vtt_time_to_sec(start_str)
        end_sec   = vtt_time_to_sec(end_str)
    except:
        continue
    
    # テキスト行（タイムコード行以降）
    text_lines = []
    past_time = False
    for l in lines:
        if '-->' in l:
            past_time = True
            continue
        if past_time:
            # HTMLタグ除去
            clean = re.sub(r'<[^>]+>', '', l).strip()
            if clean:
                text_lines.append(clean)
    
    text = ' '.join(text_lines).strip()
    if text and end_sec > start_sec:
        subtitle_records.append({
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration': end_sec - start_sec,
            'text': text
        })

df_sub = pd.DataFrame(subtitle_records).drop_duplicates(subset=['start_sec', 'text'])
print(f"Parsed {len(df_sub)} subtitle entries")
if len(df_sub) > 0:
    print("Sample subtitles:")
    print(df_sub.head(5).to_string())

# ============================================================
# 4) セグメント別特徴量の計算
# ============================================================
print("\nComputing features per segment ...")

# BERT感情極性（実測）
df_bert = pd.read_csv(BERT_SEG_CSV)
print("BERT seg columns:", df_bert.columns.tolist())

# 草・笑い正規表現
GRASS_RE  = re.compile(r'[wｗ草]+|笑+|ｗ+', re.IGNORECASE)
QUEST_RE  = re.compile(r'[？?]+')
EXCLA_RE  = re.compile(r'[！!]+')

feature_rows = []
for _, seg in segments.iterrows():
    seg_id  = int(seg['セグメント'])
    t_start = seg['開始(秒)']
    t_end   = seg['終了(秒)']
    duration = t_end - t_start
    
    # ---- チャット系 ----
    seg_chat = df_chat[(df_chat['time_sec'] >= t_start) & (df_chat['time_sec'] < t_end)].copy()
    n_comments = len(seg_chat)
    comment_rate = n_comments / duration if duration > 0 else 0.0
    
    if n_comments > 0:
        texts = seg_chat['text'].fillna('').tolist()
        grass_ratio = sum(1 for t in texts if GRASS_RE.search(t)) / n_comments
        question_ratio = sum(1 for t in texts if QUEST_RE.search(t)) / n_comments
        exclamation_ratio = sum(1 for t in texts if EXCLA_RE.search(t)) / n_comments
    else:
        texts = []
        grass_ratio = question_ratio = exclamation_ratio = 0.0
    
    # コメント意味多様性: ユニーク文字n-gram の多様性（type-token ratio近似）
    if n_comments > 1:
        all_chars = ' '.join(texts)
        all_words = re.findall(r'\w+', all_chars)
        ttr = len(set(all_words)) / len(all_words) if all_words else 0.0
        comment_semantic_variance = round(ttr, 4)
    else:
        comment_semantic_variance = 0.0
    
    # ---- 字幕系 ----
    seg_sub = df_sub[(df_sub['start_sec'] >= t_start) & (df_sub['start_sec'] < t_end)].copy()
    n_subs = len(seg_sub)
    speech_rate = n_subs / duration if duration > 0 else 0.0
    
    if n_subs > 0:
        total_speech = seg_sub['duration'].sum()
        pause_ratio = max(0.0, 1.0 - total_speech / duration)
        
        # 字幕-コメント類似度: 字幕テキストとコメントの文字レベルJaccard類似度
        sub_text_chars = set(re.findall(r'\w', ' '.join(seg_sub['text'].tolist())))
        chat_text_chars = set(re.findall(r'\w', ' '.join(texts))) if texts else set()
        if sub_text_chars or chat_text_chars:
            intersection = len(sub_text_chars & chat_text_chars)
            union = len(sub_text_chars | chat_text_chars)
            sub_comment_similarity = intersection / union if union > 0 else 0.0
        else:
            sub_comment_similarity = 0.0
    else:
        pause_ratio = 1.0
        sub_comment_similarity = 0.0
    
    # ---- BERT感情極性（実測値） ----
    bert_row = df_bert[df_bert['セグメント'] == seg_id]
    if len(bert_row) > 0:
        bert_row = bert_row.iloc[0]
        # チャット感情極性
        p_pos_c = bert_row['日本語BERT_コメント_Positive(%)'] / 100.0
        p_neg_c = bert_row['日本語BERT_コメント_Negative(%)'] / 100.0
        sentiment_polarity = (p_pos_c - p_neg_c) / (p_pos_c + p_neg_c + 1e-9)
        
        # 字幕感情極性
        p_pos_s = bert_row['日本語BERT_字幕_Positive(%)'] / 100.0
        p_neg_s = bert_row['日本語BERT_字幕_Negative(%)'] / 100.0
        subtitle_sentiment_polarity = (p_pos_s - p_neg_s) / (p_pos_s + p_neg_s + 1e-9)
    else:
        sentiment_polarity = 0.0
        subtitle_sentiment_polarity = 0.0
    
    feature_rows.append({
        'セグメント': seg_id,
        'comment_rate': round(comment_rate, 4),
        'grass_ratio': round(grass_ratio, 4),
        'question_ratio': round(question_ratio, 4),
        'exclamation_ratio': round(exclamation_ratio, 4),
        'sentiment_polarity': round(sentiment_polarity, 6),
        'subtitle_sentiment_polarity': round(subtitle_sentiment_polarity, 6),
        'speech_rate': round(speech_rate, 4),
        'pause_ratio': round(pause_ratio, 4),
        'sub_comment_similarity': round(sub_comment_similarity, 4),
        'comment_semantic_variance': comment_semantic_variance,
        'n_comments': n_comments,
        'n_subtitles': n_subs,
    })

df_features_real = pd.DataFrame(feature_rows)
print("\n=== Real feature values ===")
print(df_features_real.to_string())

# ============================================================
# 5) 既存の feature_matrix.csv とマージして更新
# ============================================================
df_old = pd.read_csv(OUT_MATRIX)
# 実測値で上書きする列
real_cols = ['comment_rate','grass_ratio','question_ratio','exclamation_ratio',
             'sentiment_polarity','subtitle_sentiment_polarity',
             'speech_rate','pause_ratio','sub_comment_similarity','comment_semantic_variance']

df_merged = df_old.copy()
for col in real_cols:
    mapping = df_features_real.set_index('セグメント')[col]
    df_merged[col] = df_merged['セグメント'].map(mapping)

# ヒートマップ系はそのまま残す（hm_mean, hm_max, hm_volatility は実測済み）
# z-score を再計算
for col in real_cols + ['hm_mean','hm_max','hm_volatility']:
    vals = df_merged[col]
    mu, sigma = vals.mean(), vals.std()
    df_merged[f'{col}_zscore'] = ((vals - mu) / sigma).round(6) if sigma > 0 else 0.0

df_merged.to_csv(OUT_MATRIX, index=False, encoding='utf-8-sig')
print(f"\nSaved updated feature_matrix.csv ({len(df_merged)} rows)")

# 追記情報の保存
df_features_real.to_csv(
    PROJECT_ROOT / "data/raw/chat/real_chat_features.csv",
    index=False, encoding='utf-8-sig'
)
print("Saved real_chat_features.csv")
