import os
import csv
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
import unicodedata
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import MeCab


def seconds_to_hms(seconds: float) -> str:
    """
    秒数を時:分:秒形式に変換
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def hms_to_seconds(hms: str) -> float:
    """
    時:分:秒形式を秒数に変換
    """
    parts = hms.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    elif len(parts) == 1:
        return float(parts[0])
    else:
        return 0.0


def normalize_text(text: str) -> str:
    """テキストを正規化"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_raw_transcript(video_url: str, language: str = "ja") -> Tuple[List[str], List[Dict]]:
    """
    YouTubeから受け取ったままの未加工の字幕データを取得
    """
    print(f"未加工の字幕データを取得中: {video_url}")
    
    # 動画IDを取得
    yt = YouTube(video_url)
    video_id = yt.video_id
    
    # YouTubeTranscriptApiから生データを取得
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    
    # 指定言語の字幕を探す
    transcript_obj = next((t for t in transcript_list if t.language_code == language), None)
    if not transcript_obj:
        print(f"警告: {language}字幕が見つかりません。他の言語を検索します")
        transcript_obj = transcript_list[0]  # 最初の字幕を使用
    
    # 生データを取得
    transcript = transcript_obj.fetch().to_raw_data()
    
    raw_chunks = []
    raw_metadata = []
    
    for entry in transcript:
        # テキストは完全に未加工の状態で保存（音楽タグなども含む）
        text = entry['text'].replace("\n", " ").strip()
        raw_chunks.append(text)
        
        # メタデータ
        raw_metadata.append({
            'start': entry['start'],
            'duration': entry['duration'],
            'end': entry['start'] + entry['duration']
        })
    
    print(f"未加工字幕取得完了: {len(raw_chunks)} チャンク")
    return raw_chunks, raw_metadata


def basic_transcript_processing(
    raw_chunks: List[str], 
    raw_metadata: List[Dict],
    min_len: int = 6,
    max_len: int = 60,
    debug_mode: bool = False
) -> Tuple[List[str], List[Dict]]:
    """
    基本的な字幕処理（両ファイルで共通）：
    1. 特殊タグの除去
    2. 文頭句読点除去
    3. 短文結合（min_len未満を前の文に結合）
    4. 長文分割（max_len以上を分割）
    """
    print("基本的な字幕処理を開始します...")
    
    # 1. 特殊タグの除去と正規化
    print("1. 特殊タグ除去と正規化...")
    processed_chunks = []
    processed_metadata = []
    
    with tqdm(total=len(raw_chunks), desc="タグ除去") as pbar:
        for chunk, meta in zip(raw_chunks, raw_metadata):
            if not chunk.strip():
                pbar.update(1)
                continue
                
            # 特殊タグを除去
            text = re.sub(r'\[音楽\]', '', chunk)
            text = re.sub(r'\[拍手\]', '', text)
            text = re.sub(r'\[.*?\]', '', text)  # すべての[]タグを除去
            
            # 空白を正規化
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text:  # 空でない場合のみ追加
                processed_chunks.append(text)
                processed_metadata.append(meta)
            
            pbar.update(1)
    
    print(f"  タグ除去後: {len(processed_chunks)} チャンク")
    
    # 2. 文頭句読点除去
    print("2. 文頭句読点除去...")
    cleaned_chunks = []
    cleaned_metadata = []
    
    with tqdm(total=len(processed_chunks), desc="句読点除去") as pbar:
        for chunk, meta in zip(processed_chunks, processed_metadata):
            # 文頭の句読点を除去
            cleaned_chunk = re.sub(r'^[。．.！!？?、,]+', '', chunk)
            if cleaned_chunk.strip():
                cleaned_chunks.append(cleaned_chunk)
                cleaned_metadata.append(meta)
            pbar.update(1)
    
    print(f"  句読点除去後: {len(cleaned_chunks)} チャンク")
    
    # 3. 短文の結合処理
    print(f"3. 短文結合処理（{min_len}文字未満を結合）...")
    combined_sentences = []
    combined_metadata = []
    
    with tqdm(total=len(cleaned_chunks), desc="短文結合") as pbar:
        for i, (chunk, meta) in enumerate(zip(cleaned_chunks, cleaned_metadata)):
            if i == 0:
                # 最初のチャンクはそのまま追加
                combined_sentences.append(chunk)
                combined_metadata.append({
                    'start': meta['start'],
                    'end': meta['end'],
                    'original_start': meta['start'],
                    'original_end': meta['end']
                })
            else:
                # 短文（min_len未満）の場合、前の文に結合
                if len(chunk) < min_len and combined_sentences:
                    # 前の文に結合
                    combined_sentences[-1] += chunk
                    
                    # メタデータを更新（終了時間を延長）
                    combined_metadata[-1]['end'] = meta['end']
                    combined_metadata[-1]['original_end'] = meta['end']
                    
                    if debug_mode and i < 10:  # デバッグ用
                        print(f"  短文結合: 文{i}『{chunk}』({len(chunk)}文字) → 前の文に結合")
                else:
                    # 通常の文はそのまま追加
                    combined_sentences.append(chunk)
                    combined_metadata.append({
                        'start': meta['start'],
                        'end': meta['end'],
                        'original_start': meta['start'],
                        'original_end': meta['end']
                    })
            pbar.update(1)
    
    print(f"  結合後文数: {len(combined_sentences)}（結合前: {len(cleaned_chunks)}）")
    
    # 4. 長文分割処理（オプション）
    print(f"4. 長文分割処理（{max_len}文字以上を分割）...")
    final_sentences = []
    final_metadata = []
    
    with tqdm(total=len(combined_sentences), desc="長文分割") as pbar:
        for text, meta in zip(combined_sentences, combined_metadata):
            chunk_start = meta['start']
            chunk_duration = meta['end'] - meta['start']
            
            # max_lenを超える場合のみ分割
            if len(text) > max_len:
                # 可能な限り意味の切れ目で分割
                parts = []
                current_pos = 0
                
                while current_pos < len(text):
                    # 次の分割点を探す（句読点や自然な切れ目）
                    next_split = current_pos + max_len
                    
                    # 可能なら句読点で分割
                    if next_split < len(text):
                        # 句読点を探す
                        punct_positions = []
                        for punct in ['。', '.', '！', '!', '？', '?', '、', ',']:
                            pos = text.find(punct, current_pos + int(max_len * 0.5), min(next_split + 10, len(text)))
                            if pos != -1:
                                punct_positions.append(pos)
                        
                        if punct_positions:
                            next_split = min(punct_positions) + 1  # 句読点を含める
                    
                    part = text[current_pos:next_split]
                    
                    # 文頭の句読点を除去
                    part = re.sub(r'^[。．.！!？?、,]+', '', part.strip())
                    if part:
                        # 時間按分計算
                        part_ratio_start = current_pos / len(text)
                        part_ratio_end = next_split / len(text)
                        
                        part_start_time = chunk_start + (chunk_duration * part_ratio_start)
                        part_end_time = chunk_start + (chunk_duration * part_ratio_end)
                        
                        final_sentences.append(part)
                        final_metadata.append({
                            'start': part_start_time,
                            'end': part_end_time,
                            'original_start': meta['original_start'],
                            'original_end': meta['original_end']
                        })
                    
                    current_pos = next_split
            else:
                # 分割しない
                final_sentences.append(text)
                final_metadata.append(meta)
            
            pbar.update(1)
    
    # 5. 最終的な短文結合（分割後も短い文を結合）
    print("5. 最終短文結合処理...")
    very_final_sentences = []
    very_final_metadata = []
    
    for i, (text, meta) in enumerate(zip(final_sentences, final_metadata)):
        if i == 0:
            very_final_sentences.append(text)
            very_final_metadata.append(meta)
        else:
            if len(text) < min_len and very_final_sentences:
                very_final_sentences[-1] += text
                very_final_metadata[-1]['end'] = meta['end']
            else:
                very_final_sentences.append(text)
                very_final_metadata.append(meta)
    
    print(f"  最終文数: {len(very_final_sentences)}")
    
    # 統計情報
    combined_count = len(cleaned_chunks) - len(combined_sentences)
    split_count = len(final_sentences) - len(combined_sentences)
    
    if combined_count > 0:
        print(f"  結合された短文数: {combined_count}")
    if split_count > 0:
        print(f"  分割された長文数: {split_count}")
    
    # デバッグ情報
    if debug_mode:
        print("\n最初の5文:")
        for i, (sentence, meta) in enumerate(zip(very_final_sentences[:5], very_final_metadata[:5])):
            start_hms = seconds_to_hms(meta['start'])
            end_hms = seconds_to_hms(meta['end'])
            display_text = f"{sentence[:50]}..." if len(sentence) > 50 else sentence
            print(f"  {i+1}. [{start_hms} - {end_hms}] {display_text}")
        
        # 文字数分布
        lengths = [len(s) for s in very_final_sentences]
        if lengths:
            print(f"\n文字数分布:")
            print(f"  平均: {sum(lengths)/len(lengths):.1f}文字")
            print(f"  最小: {min(lengths)}文字")
            print(f"  最大: {max(lengths)}文字")
            print(f"  {min_len}文字未満: {sum(1 for l in lengths if l < min_len)}文")
            print(f"  {max_len}文字以上: {sum(1 for l in lengths if l > max_len)}文")
    
    return very_final_sentences, very_final_metadata


def save_transcript_to_csv(
    sentences: List[str],
    metadata: List[Dict],
    csv_path: str,
    title: str = "字幕テキスト"
):
    """
    字幕データをCSVに保存
    """
    print(f"字幕をCSVに保存: {csv_path}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['文番号', '開始時間', '終了時間', '開始時間(秒)', '終了時間(秒)', title])
        
        with tqdm(total=len(sentences), desc="CSV保存") as pbar:
            for i, (sentence, meta) in enumerate(zip(sentences, metadata)):
                start_seconds = meta['start']
                end_seconds = meta['end']
                
                start_hms = seconds_to_hms(start_seconds)
                end_hms = seconds_to_hms(end_seconds)
                
                writer.writerow([
                    i + 1,
                    start_hms,
                    end_hms,
                    f"{start_seconds:.2f}",
                    f"{end_seconds:.2f}",
                    sentence
                ])
                pbar.update(1)
    
    print(f"CSV保存完了: {csv_path}")