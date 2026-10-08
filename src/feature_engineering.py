# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
feature_engineering.py
実生データ（生字幕・生チャット・実測ヒートマップ・アンケート実測値）から
厳密に全13特徴量を算出し feature_matrix.csv を作成・更新する。
※ シミュレーション・乱数・モック値は一切使用せず、生データ不在時はハードフェイルする。
"""
import os
import sys
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path("/home/shota/work/year1")
CHAT_JSON = PROJECT_ROOT / "data/raw/chat/pP2KLW-_7hQ.live_chat.json"
VTT_FILE = PROJECT_ROOT / "data/raw/chat/pP2KLW-_7hQ.ja.vtt"
BERT_SEG_CSV = PROJECT_ROOT / "outputs/run_20260624_071441/segment_sentiment_comparison_results.csv"
BERT_SIM_CSV = PROJECT_ROOT / "data/processed/bert_sub_comment_similarity.csv"
BERT_VAR_CSV = PROJECT_ROOT / "data/processed/bert_comment_variance.csv"
MERGED_CSV = PROJECT_ROOT / "data/processed/merged_survey_heatmap.csv"
OUTPUT_FEATURE_MATRIX = PROJECT_ROOT / "data/processed/feature_matrix.csv"


def validate_raw_data_exists():
    """生データファイルの実在と有効性を厳格に事前チェックする（ハードフェイル設計）"""
    checks = [
        (MERGED_CSV, "アンケート・ヒートマップ統合データ"),
        (CHAT_JSON, "YouTube生チャットJSONデータ"),
        (VTT_FILE, "YouTube生字幕VTTデータ"),
        (BERT_SEG_CSV, "日本語BERT実測感情分析CSV")
    ]
    for p, desc in checks:
        if not p.exists():
            raise FileNotFoundError(f"【FATAL DATA INTEGRITY ERROR】必要な実生データが存在しません: {desc} ({p})")
        if p.stat().st_size == 0:
            raise ValueError(f"【FATAL DATA INTEGRITY ERROR】生データファイルが空（0バイト）です: {desc} ({p})")
    print("✅ 全生データファイルの存在・サイズ検証をクリアしました（モック代替禁止）。")


def vtt_time_to_sec(t_str):
    parts = t_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0


def parse_vtt_tokens(vtt_path: Path):
    """VTT字幕から発話単語・文字トークンと発話開始秒を抽出する"""
    with open(vtt_path, encoding='utf-8') as f:
        content = f.read()
    cues = re.split(r'\n\n+', content)
    word_tokens = []
    time_pat = re.compile(r'(\d+:\d+:\d+\.\d+)\s+-->\s+(\d+:\d+:\d+\.\d+)')
    tag_pat = re.compile(r'<(\d+:\d+:\d+\.\d+)><c>(.*?)</c>')
    for c in cues:
        lines = c.strip().split('\n')
        if not lines:
            continue
        m = time_pat.search(lines[0])
        if not m:
            continue
        c_start = vtt_time_to_sec(m.group(1))
        c_end = vtt_time_to_sec(m.group(2))
        if c_end - c_start <= 0.02:
            continue
        for l in lines[1:]:
            if '<c>' in l:
                first_tag = re.search(r'<(\d+:\d+:\d+\.\d+)>', l)
                if first_tag:
                    prefix = l[:first_tag.start()]
                    clean_pre = re.sub(r'<[^>]+>', '', prefix).strip()
                    if clean_pre:
                        word_tokens.append((c_start, clean_pre))
                for tm, txt in tag_pat.findall(l):
                    sec = vtt_time_to_sec(tm)
                    clean_txt = txt.strip()
                    if clean_txt:
                        word_tokens.append((sec, clean_txt))
    return word_tokens


def parse_vtt(vtt_path: Path):
    """VTT字幕ファイルをパースして (start_sec, end_sec, text) のリストを返す"""
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
            elif line and not line.startswith('WEBVTT') and not line.isdigit() and 'align:' not in line:
                cleaned = re.sub(r'<[^>]+>', '', line)
                if cleaned:
                    cur_text.append(cleaned)
    if cur_start is not None and cur_text:
        subs.append((cur_start, cur_end, ' '.join(cur_text)))
    return subs


def parse_chat(chat_path: Path):
    """ライブチャットJSONをパースして (offset_sec, text) のリストを返す"""
    records = []
    with open(chat_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            actions = obj.get('replayChatItemAction', {}).get('actions', [])
            for action in actions:
                item = action.get('addChatItemAction', {}).get('item', {})
                renderer = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer')
                if renderer:
                    runs = renderer.get('message', {}).get('runs', [])
                    text = ''.join(r.get('text', '') for r in runs)
                    offset_us = obj.get('videoOffsetTimeMsec') or obj.get('replayChatItemAction', {}).get('videoOffsetTimeMsec')
                    if offset_us is not None:
                        sec = float(offset_us) / 1000.0
                        records.append((sec, text))
    return records


def run_strict_feature_pipeline():
    # 1. ハードフェイル検証
    validate_raw_data_exists()

    df_merged = pd.read_csv(MERGED_CSV)
    subtitles = parse_vtt(VTT_FILE)
    sub_tokens = parse_vtt_tokens(VTT_FILE)
    chats = parse_chat(CHAT_JSON)
    df_bert = pd.read_csv(BERT_SEG_CSV)
    df_bert_sim = pd.read_csv(BERT_SIM_CSV)
    df_bert_var = pd.read_csv(BERT_VAR_CSV)

    print(f"実測生データ読み込み完了: 字幕 {len(subtitles):,} 件, チャット {len(chats):,} 件, BERTセグメント {len(df_bert)} 件")

    # 因子目的変数の計算
    df = df_merged.copy()
    if '平均ヒートマップ値' in df.columns:
        df['hm_mean'] = df['平均ヒートマップ値']
        df['hm_max'] = df['最大ヒートマップ値']
        df['hm_volatility'] = (df['hm_max'] - df['hm_mean']) / (df['hm_mean'] + 1e-6)

    df['segment_length_sec'] = df['長さ(秒)'] if '長さ(秒)' in df.columns else 120.0
    df['Factor_Entertainment'] = df[['面白かった', '話に引き込まれた', 'テンポが良かった']].mean(axis=1)
    df['Factor_Information'] = df[['新しい情報を得られた', '有益な内容だった']].mean(axis=1)
    df['Factor_Relaxation'] = df[['気軽に視聴できた', '気分転換になった']].mean(axis=1)
    df['Factor_SocialShare'] = df['誰かに共有したいと思った']
    df['Target_Satisfaction'] = df['この動画に満足した']

    # 13特徴量計算
    feature_rows = []
    for _, seg in df.iterrows():
        seg_id = int(seg['セグメント'])
        t_start = float(seg['開始(秒)'])
        t_end   = float(seg['終了(秒)'])
        dur = max(t_end - t_start, 1.0)

        # 区間内チャット
        seg_chats = [t for s, t in chats if t_start <= s < t_end]
        n_c = len(seg_chats)
        comment_rate = n_c / dur

        if n_c > 0:
            grass_count = sum(1 for t in seg_chats if re.search(r'[wWｗ草笑]', t))
            grass_ratio = grass_count / n_c
            question_count = sum(1 for t in seg_chats if re.search(r'[?？]|何|どう|誰|いつ|どこ|なぜ|ですか|ますか', t))
            question_ratio = question_count / n_c
            exclamation_count = sum(1 for t in seg_chats if re.search(r'[!！]', t))
            exclamation_ratio = exclamation_count / n_c

            # 意味多様性（語彙多様性 TTR）
            all_words = []
            for t in seg_chats:
                all_words.extend([w for w in re.findall(r'[\u4e00-\u9fff]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+|[a-zA-Z0-9]+', t) if len(w) > 1])
            comment_semantic_variance = round(len(set(all_words)) / len(all_words), 4) if all_words else 0.5
        else:
            grass_ratio = question_ratio = exclamation_ratio = 0.0
            comment_semantic_variance = 0.5

        # 区間内字幕と発話速度（文字数 / 秒）
        seg_subs = [(s, e, t) for s, e, t in subtitles if not (e <= t_start or s >= t_end)]
        seg_words = [t for tm, t in sub_tokens if t_start <= tm < t_end]
        seg_spoken_text = ''.join(seg_words)
        speech_rate = len(seg_spoken_text) / dur if dur > 0 else 0.0

        total_speech_sec = sum(min(e, t_end) - max(s, t_start) for s, e, t in seg_subs)
        pause_sec = max(dur - total_speech_sec, 0.0)
        pause_ratio = pause_sec / dur

        # 字幕-コメント類似度 (BERT埋め込みベクトル平均コサイン類似度)
        b_sim_row = df_bert_sim[df_bert_sim['セグメント'] == seg_id]
        if not b_sim_row.empty:
            sub_comment_similarity = float(b_sim_row['sub_comment_bert_cosine'].values[0])
        else:
            raise ValueError(f"【DATA INTEGRITY ERROR】BERT類似度結果にセグメント {seg_id} が見つかりません。")

        # 日本語BERT感情極性
        b_row = df_bert[df_bert['セグメント'] == seg_id]
        if not b_row.empty:
            cp = float(b_row['日本語BERT_コメント_Positive(%)'].values[0])
            cn = float(b_row['日本語BERT_コメント_Negative(%)'].values[0])
            sentiment_polarity = (cp - cn) / (cp + cn + 1e-6)
            sp = float(b_row['日本語BERT_字幕_Positive(%)'].values[0])
            sn = float(b_row['日本語BERT_字幕_Negative(%)'].values[0])
            subtitle_sentiment_polarity = (sp - sn) / (sp + sn + 1e-6)
        else:
            raise ValueError(f"【DATA INTEGRITY ERROR】BERT感情分析結果にセグメント {seg_id} が見つかりません。")

        # BERT コメント意味多様性 (L2分散真値)
        b_var_row = df_bert_var[df_bert_var['セグメント'] == seg_id]
        if not b_var_row.empty:
            comment_semantic_variance = float(b_var_row['bert_comment_variance'].values[0])
        else:
            raise ValueError(f"【DATA INTEGRITY ERROR】BERT意味多様性結果にセグメント {seg_id} が見つかりません。")

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
            'comment_semantic_variance': round(comment_semantic_variance, 4),
        })

    df_feats = pd.DataFrame(feature_rows)
    for c in df_feats.columns:
        if c != 'セグメント':
            df[c] = df['セグメント'].map(df_feats.set_index('セグメント')[c])

    # z-score 標準化
    quantitative_features = [
        'comment_rate', 'grass_ratio', 'question_ratio', 'exclamation_ratio',
        'sentiment_polarity', 'subtitle_sentiment_polarity', 'pause_ratio', 'speech_rate',
        'sub_comment_similarity', 'comment_semantic_variance',
        'hm_mean', 'hm_max', 'hm_volatility'
    ]
    for col in quantitative_features:
        vals = df[col]
        mu, sigma = vals.mean(), vals.std()
        df[f'{col}_zscore'] = ((vals - mu) / sigma).round(6) if sigma > 0 else 0.0

    df.to_csv(OUTPUT_FEATURE_MATRIX, index=False, encoding='utf-8-sig')
    print(f"✅ 100%実生データによる feature_matrix.csv の更新が完了しました（{len(df)} 行）。")


if __name__ == '__main__':
    run_strict_feature_pipeline()
