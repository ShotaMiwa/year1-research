import os
import csv
import json
import re
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
from test_window import (
    get_comments, 
    build_comment_vectors,
    generate_coherence_data,
    simcse_tokenizer,
    AutoTokenizer,
    YouTube,  # pytubeからYouTubeをインポート
    YouTubeTranscriptApi  # YouTubeTranscriptApiをインポート
)

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

def get_raw_transcript(video_url: str) -> Tuple[List[str], List[Dict]]:
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
    
    # 日本語字幕を探す
    transcript_obj = next((t for t in transcript_list if t.language_code == "ja"), None)
    if not transcript_obj:
        print(f"警告: 日本語字幕が見つかりません。他の言語を検索します")
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

def simple_transcript_processing(video_url: str, min_len: int = 6, debug_mode: bool = False) -> Tuple[List[str], List[Dict], List[str], List[Dict]]:
    """
    シンプルな3段階の字幕処理：
    1. 生チャンク取得（タグ除去・正規化）
    2. 文頭句読点除去
    3. 短文結合（min_len未満を前の文に結合）
    """
    print("シンプルな字幕処理を開始します...")
    
    # 1. 生チャンクの取得
    print("1. 生チャンクの取得...")
    raw_chunks, raw_metadata = get_raw_transcript(video_url)
    
    if not raw_chunks:
        raise ValueError("字幕データが取得できませんでした")
    
    print(f"  取得した生チャンク数: {len(raw_chunks)}")
    
    # 1.5. 特殊タグの除去と正規化（生チャンク取得時に実行）
    print("1.5. 特殊タグ除去と正規化...")
    processed_chunks = []
    processed_metadata = []
    
    for chunk, meta in zip(raw_chunks, raw_metadata):
        if not chunk.strip():
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
    
    print(f"  タグ除去後: {len(processed_chunks)} チャンク")
    
    # 2. 句読点分割のスキップ（文頭句読点のみ除去）
    print("2. 文頭句読点除去...")
    cleaned_chunks = []
    cleaned_metadata = []
    
    for chunk, meta in zip(processed_chunks, processed_metadata):
        # 文頭の句読点を除去
        cleaned_chunk = re.sub(r'^[。．.！!？?、,]+', '', chunk)
        if cleaned_chunk.strip():
            cleaned_chunks.append(cleaned_chunk)
            cleaned_metadata.append(meta)
    
    print(f"  句読点除去後: {len(cleaned_chunks)} チャンク")
    
    # 3. 短文の結合処理
    print(f"3. 短文結合処理（{min_len}文字未満を結合）...")
    final_sentences = []
    final_metadata = []
    
    for i, (chunk, meta) in enumerate(zip(cleaned_chunks, cleaned_metadata)):
        if i == 0:
            # 最初のチャンクはそのまま追加
            final_sentences.append(chunk)
            final_metadata.append({
                'start': meta['start'],
                'end': meta['end'],
                'original_start': meta['start'],
                'original_end': meta['end']
            })
        else:
            # 短文（min_len未満）の場合、前の文に結合
            if len(chunk) < min_len and final_sentences:
                # 前の文に結合
                final_sentences[-1] += chunk
                
                # メタデータを更新（終了時間を延長）
                final_metadata[-1]['end'] = meta['end']
                final_metadata[-1]['original_end'] = meta['end']
                
                if debug_mode and i < 10:  # デバッグ用
                    print(f"  短文結合: 文{i}『{chunk}』({len(chunk)}文字) → 前の文に結合")
            else:
                # 通常の文はそのまま追加
                final_sentences.append(chunk)
                final_metadata.append({
                    'start': meta['start'],
                    'end': meta['end'],
                    'original_start': meta['start'],
                    'original_end': meta['end']
                })
    
    print(f"  最終文数: {len(final_sentences)}（結合前: {len(cleaned_chunks)}）")
    
    # 結合された文の統計を表示
    combined_count = len(cleaned_chunks) - len(final_sentences)
    if combined_count > 0:
        print(f"  結合された短文数: {combined_count}")
    
    # 最初の5文を表示（デバッグ用）
    print("\n最初の5文:")
    for i, (sentence, meta) in enumerate(zip(final_sentences[:5], final_metadata[:5])):
        start_hms = seconds_to_hms(meta['start'])
        end_hms = seconds_to_hms(meta['end'])
        display_text = f"{sentence[:50]}..." if len(sentence) > 50 else sentence
        print(f"  {i+1}. [{start_hms} - {end_hms}] {display_text}")
    
    # 文字数分布を表示
    lengths = [len(s) for s in final_sentences]
    if lengths:
        print(f"\n文字数分布:")
        print(f"  平均: {sum(lengths)/len(lengths):.1f}文字")
        print(f"  最小: {min(lengths)}文字")
        print(f"  最大: {max(lengths)}文字")
        print(f"  {min_len}文字未満の文: {sum(1 for l in lengths if l < min_len)}")
    
    return final_sentences, final_metadata, raw_chunks, raw_metadata

def save_raw_transcript_to_csv(raw_chunks: List[str], raw_metadata: List[Dict], 
                               video_index: int, output_dir: str = "./raw_data"):
    """
    未加工の字幕データをCSVに保存
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = f"{output_dir}/video_{video_index}_raw_transcript.csv"
    
    print(f"未加工字幕をCSVに保存: {csv_path}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー書き込み
        writer.writerow([
            'チャンク番号', 
            '開始時間(時:分:秒)', 
            '終了時間(時:分:秒)',
            '開始時間(秒)', 
            '終了時間(秒)', 
            '期間(秒)',
            '未加工字幕テキスト',
            'メモ'
        ])
        
        # データを書き込み
        with tqdm(total=len(raw_chunks), desc="未加工データ保存") as pbar:
            for i, (chunk, meta) in enumerate(zip(raw_chunks, raw_metadata)):
                start_seconds = meta['start']
                end_seconds = meta['end']
                duration = meta['duration']
                
                # 秒を時:分:秒形式に変換
                start_hms = seconds_to_hms(start_seconds)
                end_hms = seconds_to_hms(end_seconds)
                
                # 特殊タグの有無をメモ欄に記録
                memo = ""
                if '[' in chunk and ']' in chunk:
                    memo = "音楽/効果音タグを含む"
                elif len(chunk) < 3:
                    memo = "短いチャンク"
                
                writer.writerow([
                    i + 1,
                    start_hms,
                    end_hms,
                    f"{start_seconds:.3f}",
                    f"{end_seconds:.3f}",
                    f"{duration:.3f}",
                    chunk,
                    memo
                ])
                pbar.update(1)
    
    print(f"未加工字幕CSV保存完了: {csv_path}")
    
    # 統計情報を表示
    total_chars = sum(len(chunk) for chunk in raw_chunks)
    avg_chars = total_chars / len(raw_chunks) if raw_chunks else 0
    
    print(f"統計情報:")
    print(f"  - 総チャンク数: {len(raw_chunks)}")
    print(f"  - 総文字数: {total_chars}")
    print(f"  - 平均文字数/チャンク: {avg_chars:.1f}")
    if raw_metadata:
        print(f"  - 時間範囲: {seconds_to_hms(raw_metadata[0]['start'])} ~ {seconds_to_hms(raw_metadata[-1]['end'])}")
    
    return csv_path

def parse_topic_timetable(timetable_text: str) -> List[Dict]:
    """
    時間ウィンドウ形式（開始時間 終了時間 トピック）のタイムテーブルを解析
    
    例：
    0:05:07 0:09:33 AI
    0:16:22 0:18:16 障害福祉の仕事
    0:18:16 0:22:42 高市総理
    """
    time_windows = []
    lines = timetable_text.strip().split('\n')
    
    print(f"時間ウィンドウ形式のタイムテーブルを解析中...")
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        # コメント行をスキップ
        if line.startswith('#'):
            continue
        
        # 時間ウィンドウ形式を解析（開始時間 終了時間 トピック）
        # 例：0:05:07 0:09:33 AI
        pattern = r'(\d+:\d+(?::\d+)?|\d+)\s+(\d+:\d+(?::\d+)?|\d+)\s+(.+)'
        match = re.match(pattern, line)
        
        if match:
            start_time_str = match.group(1)
            end_time_str = match.group(2)
            topic = match.group(3).strip()
            
            # 時間を秒に変換
            start_seconds = hms_to_seconds(start_time_str)
            end_seconds = hms_to_seconds(end_time_str)
            
            # 終了時間が開始時間より前の場合は警告
            if end_seconds <= start_seconds:
                print(f"警告（行{line_num}）: 終了時間が開始時間より前です: {start_time_str} - {end_time_str}")
                continue
            
            # ウィンドウを追加
            time_windows.append({
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'start_display': start_time_str,
                'end_display': end_time_str,
                'topic': topic,
                'duration': end_seconds - start_seconds
            })
            
            print(f"  ウィンドウ{len(time_windows)}: {start_time_str} - {end_time_str} ({topic})")
        else:
            print(f"警告（行{line_num}）: 解析できない行: {line}")
    
    # 開始時間でソート
    time_windows = sorted(time_windows, key=lambda x: x['start_seconds'])
    
    # ウィンドウの重複チェック
    for i in range(1, len(time_windows)):
        prev = time_windows[i-1]
        curr = time_windows[i]
        
        if curr['start_seconds'] < prev['end_seconds']:
            print(f"警告: ウィンドウ{i}とウィンドウ{i+1}が重複しています:")
            print(f"  {prev['start_display']}-{prev['end_display']} ({prev['topic']})")
            print(f"  {curr['start_display']}-{curr['end_display']} ({curr['topic']})")
    
    print(f"解析された時間ウィンドウ: {len(time_windows)} 件")
    return time_windows

def filter_by_time_windows(sentences: List[str], sentence_metadata: List[Dict], 
                          comments: List[Dict], time_windows: List[Dict]) -> Tuple[List[str], List[Dict], List[Dict]]:
    """
    時間ウィンドウに基づいて字幕とコメントをフィルタリング
    
    Args:
        sentences: 字幕文のリスト
        sentence_metadata: 字幕メタデータのリスト
        comments: コメントのリスト
        time_windows: 時間ウィンドウのリスト
    
    Returns:
        フィルタリング後の字幕、メタデータ、コメント
    """
    if not time_windows:
        print("時間ウィンドウが指定されていないため、全データを保持します")
        return sentences, sentence_metadata, comments
    
    print(f"時間ウィンドウによるフィルタリング開始: {len(time_windows)} ウィンドウ")
    
    # 字幕のフィルタリング
    filtered_sentences = []
    filtered_metadata = []
    sentence_window_indices = []  # 各文が属するウィンドウのインデックス
    
    for i, (sentence, metadata) in enumerate(zip(sentences, sentence_metadata)):
        sentence_start = metadata['start']
        sentence_end = metadata['end']
        
        # どのウィンドウに含まれるかをチェック
        in_any_window = False
        window_idx = -1
        
        for idx, window in enumerate(time_windows):
            # 文の開始時間がウィンドウ内にあるかチェック
            # または文がウィンドウと部分的に重なっているかチェック
            if window['start_seconds'] <= sentence_start < window['end_seconds']:
                in_any_window = True
                window_idx = idx
                break
        
        if in_any_window:
            filtered_sentences.append(sentence)
            filtered_metadata.append(metadata)
            sentence_window_indices.append(window_idx)
    
    print(f"字幕フィルタリング結果: {len(sentences)} -> {len(filtered_sentences)} 文")
    
    # 各ウィンドウごとの文の数をカウント
    window_counts = {}
    for idx in sentence_window_indices:
        window_counts[idx] = window_counts.get(idx, 0) + 1
    
    for idx, count in window_counts.items():
        if idx < len(time_windows):
            window_info = time_windows[idx]
            print(f"  ウィンドウ{idx+1} ({window_info['start_display']}-{window_info['end_display']}): {count} 文")
    
    # コメントのフィルタリング
    filtered_comments = []
    if comments:
        for comment in comments:
            # コメントの時間を秒数に変換
            comment_time = timestamp_to_seconds(comment['time'])
            
            # どのウィンドウに含まれるかをチェック
            in_any_window = False
            for window in time_windows:
                if window['start_seconds'] <= comment_time < window['end_seconds']:
                    in_any_window = True
                    break
            
            if in_any_window:
                filtered_comments.append(comment)
        
        print(f"コメントフィルタリング結果: {len(comments)} -> {len(filtered_comments)} 件")
    else:
        filtered_comments = []
    
    return filtered_sentences, filtered_metadata, filtered_comments

def validate_time_alignment(sentences: List[str], sentence_metadata: List[Dict], time_windows: List[Dict], video_index: int):
    """
    字幕修正後の時間と時間ウィンドウの対応を検証
    """
    print(f"\n=== 動画 {video_index+1} タイムアライメント検証 ===")
    
    if not time_windows or not sentences:
        print("検証対象データが不足しています")
        return
    
    # 字幕の最初と最後の時間
    first_sentence_time = sentence_metadata[0]['start'] if sentence_metadata else 0
    last_sentence_time = sentence_metadata[-1]['end'] if sentence_metadata else 0
    
    print(f"時間ウィンドウ情報:")
    for i, window in enumerate(time_windows):
        print(f"  ウィンドウ{i+1}: {window['start_display']} - {window['end_display']} ({window['topic']})")
    
    print(f"\n修正後字幕:")
    print(f"  最初の文: {first_sentence_time:.1f}秒 - 『{sentences[0][:50]}...』" 
          if sentences else "文なし")
    print(f"  最後の文: {last_sentence_time:.1f}秒 - 『{sentences[-1][:50]}...』" 
          if sentences else "文なし")
    
    # 各ウィンドウ内の文の数をカウント
    print(f"\n各ウィンドウ内の文の分布:")
    for i, window in enumerate(time_windows):
        window_start = window['start_seconds']
        window_end = window['end_seconds']
        
        # ウィンドウ内の文をカウント
        count_in_window = 0
        for meta in sentence_metadata:
            sentence_start = meta['start']
            if window_start <= sentence_start < window_end:
                count_in_window += 1
        
        print(f"  ウィンドウ{i+1} ({window['topic']}): {count_in_window} 文")
        
        # 最初の文と最後の文を表示
        if count_in_window > 0:
            first_in_window = next((s for s, m in zip(sentences, sentence_metadata) 
                                  if window_start <= m['start'] < window_end), None)
            last_in_window = next((s for s, m in reversed(list(zip(sentences, sentence_metadata))) 
                                 if window_start <= m['start'] < window_end), None)
            
            if first_in_window:
                print(f"    最初の文: 『{first_in_window[:60]}...』")
            if last_in_window:
                print(f"    最後の文: 『{last_in_window[:60]}...』")

def adjust_timetable_for_processed_transcript(time_windows: List[Dict], sentence_metadata: List[Dict]) -> List[Dict]:
    """
    修正処理後の字幕に合わせて時間ウィンドウを調整
    
    考え方:
    1. 各時間ウィンドウの開始・終了時間を、修正後字幕の文開始時間に最も近いものに調整
    2. 元の時間も保持して記録
    """
    print("時間ウィンドウの調整を開始...")
    
    if not time_windows or not sentence_metadata:
        print("調整対象データが不足しています")
        return time_windows
    
    adjusted_windows = []
    adjustment_stats = {
        'start_adjusted': 0,
        'end_adjusted': 0,
        'total_difference': 0.0
    }
    
    for window_idx, window in enumerate(time_windows):
        original_start = window['start_seconds']
        original_end = window['end_seconds']
        
        # 開始時間の調整
        closest_start_time = None
        closest_start_idx = None
        min_start_diff = float('inf')
        
        # 終了時間の調整
        closest_end_time = None
        closest_end_idx = None
        min_end_diff = float('inf')
        
        for idx, meta in enumerate(sentence_metadata):
            sentence_time = meta['start']
            
            # 開始時間に最も近い文を探す
            start_diff = abs(sentence_time - original_start)
            if start_diff < min_start_diff:
                min_start_diff = start_diff
                closest_start_time = sentence_time
                closest_start_idx = idx
            
            # 終了時間に最も近い文を探す
            end_diff = abs(sentence_time - original_end)
            if end_diff < min_end_diff:
                min_end_diff = end_diff
                closest_end_time = sentence_time
                closest_end_idx = idx
        
        # 調整後のウィンドウを作成
        adjusted_window = {
            **window,
            'original_start_seconds': original_start,
            'original_end_seconds': original_end,
            'start_adjusted': False,
            'end_adjusted': False,
            'start_adjustment_diff': min_start_diff,
            'end_adjustment_diff': min_end_diff
        }
        
        # 開始時間の調整（10秒以内の差なら調整）
        if closest_start_time is not None and min_start_diff < 10:
            adjusted_window['start_seconds'] = closest_start_time
            adjusted_window['start_display'] = seconds_to_hms(closest_start_time)
            adjusted_window['start_adjusted'] = True
            adjusted_window['closest_start_idx'] = closest_start_idx
            adjustment_stats['start_adjusted'] += 1
            adjustment_stats['total_difference'] += min_start_diff
        
        # 終了時間の調整（10秒以内の差なら調整）
        if closest_end_time is not None and min_end_diff < 10:
            adjusted_window['end_seconds'] = closest_end_time
            adjusted_window['end_display'] = seconds_to_hms(closest_end_time)
            adjusted_window['end_adjusted'] = True
            adjusted_window['closest_end_idx'] = closest_end_idx
            adjustment_stats['end_adjusted'] += 1
            adjustment_stats['total_difference'] += min_end_diff
        
        adjusted_windows.append(adjusted_window)
    
    # 統計情報を表示
    print(f"時間ウィンドウ調整完了:")
    print(f"  開始時間調整: {adjustment_stats['start_adjusted']}件")
    print(f"  終了時間調整: {adjustment_stats['end_adjusted']}件")
    if adjustment_stats['start_adjusted'] + adjustment_stats['end_adjusted'] > 0:
        avg_diff = adjustment_stats['total_difference'] / (adjustment_stats['start_adjusted'] + adjustment_stats['end_adjusted'])
        print(f"  平均調整差: {avg_diff:.2f}秒")
    
    # 最初の3つの調整結果を表示
    print(f"\n調整結果 (最初の3ウィンドウ):")
    for i, window in enumerate(adjusted_windows[:3]):
        start_status = "調整済み" if window.get('start_adjusted', False) else "未調整"
        end_status = "調整済み" if window.get('end_adjusted', False) else "未調整"
        
        if window.get('start_adjusted', False) or window.get('end_adjusted', False):
            print(f"  ウィンドウ{i+1}: {window['start_display']}-{window['end_display']}")
            print(f"    元: {seconds_to_hms(window['original_start_seconds'])}-{seconds_to_hms(window['original_end_seconds'])}")
            print(f"    開始: [{start_status}, 差: {window['start_adjustment_diff']:.1f}秒]")
            print(f"    終了: [{end_status}, 差: {window['end_adjustment_diff']:.1f}秒]")
        else:
            print(f"  ウィンドウ{i+1}: {window['start_display']}-{window['end_display']} [未調整]")
    
    return adjusted_windows

def assign_topic_labels_with_margin(sentences: List[str], sentence_metadata: List[Dict], time_windows: List[Dict], margin_seconds: float = 2.0) -> List[int]:
    """
    改善版：境界にマージンを設けてトピックラベルを割り当て
    """
    if not time_windows:
        print("警告: 時間ウィンドウが空のため、すべての文にラベル0を割り当てます")
        return [0] * len(sentences)
    
    print(f"境界マージン処理開始 (マージン: {margin_seconds}秒)")
    print(f"文の数: {len(sentences)}, 時間ウィンドウ数: {len(time_windows)}")
    
    # マージン付きの拡張ウィンドウを作成
    extended_windows = []
    for window in time_windows:
        extended_window = window.copy()
        extended_window['start_seconds'] = max(0, window['start_seconds'] - margin_seconds)
        extended_window['end_seconds'] = window['end_seconds'] + margin_seconds
        extended_windows.append(extended_window)
    
    labels = []
    boundary_counts = {i: 0 for i in range(len(time_windows))}
    unassigned_count = 0
    
    for i, (sentence, metadata) in enumerate(zip(sentences, sentence_metadata)):
        sentence_start = metadata['start']
        sentence_end = metadata['end']
        sentence_mid = (sentence_start + sentence_end) / 2  # 文中間時間
        
        assigned_label = -1
        
        # 方法1: 文中間時間で判定（優先）
        for window_idx, window in enumerate(time_windows):
            if window['start_seconds'] <= sentence_mid < window['end_seconds']:
                assigned_label = window_idx
                break
        
        # 方法2: 拡張ウィンドウで判定
        if assigned_label == -1:
            for window_idx, ext_window in enumerate(extended_windows):
                if ext_window['start_seconds'] <= sentence_start < ext_window['end_seconds']:
                    # 元のウィンドウ内かチェック
                    orig_window = time_windows[window_idx]
                    if orig_window['start_seconds'] <= sentence_start < orig_window['end_seconds']:
                        assigned_label = window_idx
                    else:
                        # マージン領域の場合は前後の文から推定
                        assigned_label = window_idx
                        boundary_counts[window_idx] += 1
                    break
        
        # 方法3: 文の開始時間で判定
        if assigned_label == -1:
            for window_idx, window in enumerate(time_windows):
                if window['start_seconds'] <= sentence_start < window['end_seconds']:
                    assigned_label = window_idx
                    break
        
        # 方法4: 前後の文から推定（境界付近の場合）
        if assigned_label == -1 and i > 0 and i < len(sentences)-1:
            prev_label = labels[-1] if labels else -1
            # 前の文のラベルが有効ならそれを使用
            if prev_label != -1:
                assigned_label = prev_label
        
        if assigned_label == -1:
            unassigned_count += 1
            # デバッグ情報
            if unassigned_count <= 5:  # 最初の5件のみ表示
                print(f"警告: 文{i} (時間: {sentence_start:.1f}秒) が未割り当て")
                for w_idx, window in enumerate(time_windows):
                    dist = min(abs(sentence_start - window['start_seconds']), 
                              abs(sentence_start - window['end_seconds']))
                    if dist < 5.0:
                        print(f"  近接ウィンドウ{w_idx}: {window['start_display']}-{window['end_display']} (距離: {dist:.1f}秒)")
        
        labels.append(assigned_label)
    
    # 境界付近の統計を表示
    total_boundary_assignments = sum(boundary_counts.values())
    if total_boundary_assignments > 0:
        print(f"\n境界マージン処理結果:")
        print(f"  マージン領域で割り当てられた文: {total_boundary_assignments}文")
        for window_idx, count in boundary_counts.items():
            if count > 0:
                window_info = time_windows[window_idx]
                print(f"  ウィンドウ{window_idx} ({window_info['topic']}): {count}文")
    
    if unassigned_count > 0:
        print(f"  未割り当て文: {unassigned_count}文")
    
    # 各ウィンドウに割り当てられた文の数を表示
    window_counts = {}
    for label in labels:
        window_counts[label] = window_counts.get(label, 0) + 1
    
    print(f"\nトピックラベル割り当て結果:")
    assigned_labels = sorted([k for k in window_counts.keys() if k != -1])
    
    for label in assigned_labels:
        if label < len(time_windows) and label >= 0:
            window_info = time_windows[label]
            print(f"  ウィンドウ{label} ({window_info['topic']}): {window_counts[label]} 文")
    
    if -1 in window_counts:
        print(f"  未割り当て (ラベル-1): {window_counts[-1]} 文")
    
    return labels

def create_boundary_labels(topic_labels: List[int]) -> List[int]:
    """
    トピックラベルから境界ラベルを生成
    """
    boundaries = [0] * len(topic_labels)  # 0: 境界でない, 1: 境界
    
    boundary_count = 0
    for i in range(1, len(topic_labels)):
        if topic_labels[i] != topic_labels[i-1]:
            boundaries[i] = 1
            boundary_count += 1
    
    # 動画の最後も境界とする
    if boundaries:
        boundaries[-1] = 1
        boundary_count += 1
    
    print(f"検出された境界数: {boundary_count}")
    return boundaries

def timestamp_to_seconds(t):
    """
    '1:23:45' → 5025.0
    '-16:04'  → -964.0
    '45'      → 45.0
    """
    try:
        sign = -1 if t.startswith('-') else 1
        t = t.lstrip('-')  # 負号を除去してから処理
        parts = [float(x) for x in t.split(":" )]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, *parts
        elif len(parts) == 1:
            h, m, s = 0, 0, parts[0]
        else:
            return 0.0
        return sign * (h * 3600 + m * 60 + s)
    except Exception:
        return 0.0

def save_inference_data(
    video_urls: List[str],  # 複数URL対応に変更
    timetable_texts: List[str],  # 複数タイムテーブル対応
    output_dir: str = "./inference_data",
    model_configs: Dict = None,
    use_comments: bool = True,  # コメント利用フラグを追加
    save_raw_data: bool = True,  # 未加工データ保存フラグを追加
    comment_window_start_offset: int = 10,  # コメント収集ウィンドウ開始オフセット（字幕開始からの秒数）
    comment_window_end_offset: int = 15,  # コメント収集ウィンドウ終了オフセット（字幕開始からの秒数）
    adjust_timetable: bool = False,  # タイムテーブル調整フラグ
    min_sentence_length: int = 6,  # 短文結合の閾値
    margin_seconds: float = 3.0  # 境界マージン秒数を追加
):
    """
    推論用データと正解ラベルを生成・保存（複数動画対応版）
    時間ウィンドウ形式（開始時間 終了時間 トピック）に対応
    
    Args:
        video_urls: YouTube動画URLのリスト
        timetable_texts: 各動画のタイムテーブルテキストのリスト（時間ウィンドウ形式）
        output_dir: 出力ディレクトリ
        model_configs: モデル設定
        use_comments: コメントデータを使用するかどうか
        save_raw_data: 未加工の字幕データを保存するかどうか
        comment_window_start_offset: コメント収集ウィンドウ開始オフセット（字幕開始からの秒数）
        comment_window_end_offset: コメント収集ウィンドウ終了オフセット（字幕開始からの秒数）
        adjust_timetable: タイムテーブルを字幕修正後の時間に合わせて調整するか
        min_sentence_length: 短文結合の最小文字数
        margin_seconds: 境界マージンの秒数（デフォルト: 3.0秒）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 推論データ生成を開始 ===")
    print(f"コメント利用: {'有効' if use_comments else '無効'}")
    print(f"未加工データ保存: {'有効' if save_raw_data else '無効'}")
    print(f"コメントウィンドウ: 字幕開始+{comment_window_start_offset}秒 ～ 字幕開始+{comment_window_end_offset}秒")
    print(f"タイムテーブル調整: {'有効' if adjust_timetable else '無効'}")
    print(f"短文結合閾値: {min_sentence_length}文字未満")
    print(f"境界マージン: {margin_seconds}秒")
    
    # デフォルトのモデル設定
    if model_configs is None:
        model_configs = {
            "default": {
                "coherence_model": "cl-tohoku/bert-base-japanese",
                "topic_model": "pkshatech/simcse-ja-bert-base-clcmlp"
            }
        }
    
    all_sentences = []
    all_sentence_metadata = []
    all_boundary_labels = []
    all_topic_labels = []
    all_time_windows = []  # 時間ウィンドウ情報を保存
    all_raw_chunks = []  # 未加工データ用
    all_raw_metadata = []  # 未加工データ用
    all_comment_vectors = []  # コメントベクトルを統合
    all_adjusted_windows = []  # 調整済みウィンドウを保存
    
    # 未加工データ保存用ディレクトリ
    raw_data_dir = f"{output_dir}/raw_transcripts"
    
    # 各動画ごとにデータを処理
    for video_idx, (video_url, timetable_text) in enumerate(zip(video_urls, timetable_texts)):
        print(f"\n--- 動画 {video_idx + 1}/{len(video_urls)} 処理中 ---")
        print(f"URL: {video_url}")
        
        # 0. シンプルな字幕処理（新規実装）
        print("0. シンプルな字幕処理実行...")
        try:
            sentences, sentence_metadata, raw_chunks, raw_metadata = simple_transcript_processing(
                video_url, 
                min_len=min_sentence_length,
                debug_mode=True
            )
            print(f"シンプル処理完了: {len(sentences)} 文")
        except Exception as e:
            print(f"シンプル処理エラー: {e}")
            print("デフォルトの処理で続行します")
            # エラー時はデフォルト処理にフォールバック
            from test_window import get_transcript
            sentences, sentence_metadata, raw_chunks, raw_metadata = get_transcript(video_url)
        
        # 1. 未加工データの保存（オプション）
        if save_raw_data:
            print("1. 未加工字幕データの保存...")
            try:
                save_raw_transcript_to_csv(raw_chunks, raw_metadata, video_idx + 1, raw_data_dir)
                all_raw_chunks.extend(raw_chunks)
                all_raw_metadata.extend(raw_metadata)
                print(f"未加工データ: {len(raw_chunks)} チャンク保存完了")
            except Exception as e:
                print(f"未加工データ保存エラー: {e}")
                print("未加工データなしで続行します")
        
        # 2. 時間ウィンドウ形式のタイムテーブルの解析
        print("2. 時間ウィンドウ形式のタイムテーブルを解析...")
        time_windows = parse_topic_timetable(timetable_text)
        
        # 3. 時間ウィンドウによるフィルタリング
        print("3. 時間ウィンドウによるフィルタリング...")
        filtered_sentences, filtered_metadata = sentences, sentence_metadata
        if time_windows:
            filtered_sentences, filtered_metadata, _ = filter_by_time_windows(
                sentences, sentence_metadata, [], time_windows
            )
        
        # 4. タイムアライメント検証（フィルタリング後）
        validate_time_alignment(filtered_sentences, filtered_metadata, time_windows, video_idx)
        
        # 5. 時間ウィンドウの調整（オプション）
        if adjust_timetable:
            print("5. 時間ウィンドウの調整...")
            adjusted_windows = adjust_timetable_for_processed_transcript(time_windows, filtered_metadata)
            time_windows = adjusted_windows  # 調整済みウィンドウを使用
            all_adjusted_windows.extend(adjusted_windows)
        else:
            print("5. 時間ウィンドウの調整をスキップ...")
        
        # 6. トピックラベルの割り当て（改善版: マージン処理を使用）
        print(f"6. トピックラベルの割り当て（マージン: {margin_seconds}秒）...")
        topic_labels = assign_topic_labels_with_margin(
            filtered_sentences, filtered_metadata, time_windows, margin_seconds
        )
        boundary_labels = create_boundary_labels(topic_labels)
        
        # 7. コメントデータの取得と処理（use_commentsフラグに基づいて実行）
        comment_vectors = []
        if use_comments:
            print("7. コメントデータの取得...")
            try:
                comments = get_comments(video_url)
                
                # 時間ウィンドウによるコメントフィルタリング
                if time_windows:
                    _, _, filtered_comments = filter_by_time_windows(
                        [], [], comments, time_windows
                    )
                    comments = filtered_comments
                
                # 修正: ウィンドウ設定を新しい形式で渡す
                comment_vectors = build_comment_vectors(
                    comments, 
                    filtered_metadata,
                    window_start_offset=comment_window_start_offset,
                    window_end_offset=comment_window_end_offset
                )
                print(f"コメントベクトル数: {len(comment_vectors)}")
                print(f"コメントウィンドウ設定: 字幕開始+{comment_window_start_offset}秒 ～ 字幕開始+{comment_window_end_offset}秒")
            except Exception as e:
                print(f"コメント取得エラー: {e}")
                print("コメントデータなしで続行します")
                comment_vectors = [torch.zeros(768) for _ in range(len(filtered_sentences))]
        else:
            print("7. コメントデータの取得をスキップ...")
            # ダミーのコメントベクトルを作成
            comment_vectors = [torch.zeros(768) for _ in range(len(filtered_sentences))]
        
        # データを全体リストに追加
        all_sentences.extend(filtered_sentences)
        all_sentence_metadata.extend(filtered_metadata)
        all_boundary_labels.extend(boundary_labels)
        all_comment_vectors.extend(comment_vectors)
        
        all_topic_labels.extend(topic_labels)
        
        # 時間ウィンドウ情報を統合（動画名を追加）
        for window in time_windows:
            all_time_windows.append({
                'start_seconds': window['start_seconds'],
                'end_seconds': window['end_seconds'],
                'start_display': window['start_display'],
                'end_display': window['end_display'],
                'topic': f"動画{video_idx+1}: {window['topic']}",
                'video_index': video_idx,
                'duration': window.get('duration', window['end_seconds'] - window['start_seconds']),
                'original_start_seconds': window.get('original_start_seconds', window['start_seconds']),
                'original_end_seconds': window.get('original_end_seconds', window['end_seconds']),
                'start_adjusted': window.get('start_adjusted', False),
                'end_adjusted': window.get('end_adjusted', False)
            })
        
        print(f"動画 {video_idx + 1} 処理完了:")
        print(f"  - フィルタリング後文数: {len(filtered_sentences)}文")
        print(f"  - 境界数: {sum(boundary_labels)}")
        print(f"  - 時間ウィンドウ数: {len(time_windows)}")
        print(f"  - 使用マージン: {margin_seconds}秒")
        print(f"  - コメントウィンドウ: 字幕開始+{comment_window_start_offset}秒 ～ 字幕開始+{comment_window_end_offset}秒")
    
    # 8. 各モデル設定ごとにデータを生成
    for config_name, config in model_configs.items():
        print(f"\n8. {config_name} モデル用データ生成...")
        
        # モデル設定に基づいてトークナイザーを準備
        coherence_tokenizer = AutoTokenizer.from_pretrained(config["coherence_model"])
        
        # Coherenceデータの生成
        coherence_data = generate_coherence_data(all_sentences, history=2, window_size=10)
        
        # Coherenceデータのトークン化
        coheren_inputs, coheren_masks, coheren_types = [], [], []
        
        with tqdm(total=len(coherence_data), desc=f"{config_name} Coherenceデータ変換") as pbar:
            for pos_pair, _ in coherence_data:
                context, cur = pos_pair
                text1 = " [SEP] ".join(context)
                text2 = cur[0] if cur else ""
                
                encoded = coherence_tokenizer(
                    text1, text2, 
                    truncation=True, 
                    max_length=512,
                    padding='max_length', 
                    return_tensors='pt'
                )
                
                coheren_inputs.append(encoded['input_ids'].squeeze(0).tolist())
                coheren_masks.append(encoded['attention_mask'].squeeze(0).tolist())
                coheren_types.append(encoded['token_type_ids'].squeeze(0).tolist())
                pbar.update(1)
        
        # Topicデータのトークン化
        topic_inputs, topic_masks = [], []
        
        # Topicモデル用トークナイザー
        topic_tokenizer = AutoTokenizer.from_pretrained(config["topic_model"])
        
        with tqdm(total=len(all_sentences)-1, desc=f"{config_name} Topicデータ変換") as pbar:
            for i in range(len(all_sentences)-1):
                context, cur = all_sentences[i], all_sentences[i+1]
                
                topic_con = topic_tokenizer(
                    context, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128, 
                    padding="max_length"
                )
                topic_cur = topic_tokenizer(
                    cur, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128, 
                    padding="max_length"
                )
                
                topic_inputs.append([
                    topic_con['input_ids'].squeeze(0).tolist(),
                    topic_cur['input_ids'].squeeze(0).tolist()
                ])
                topic_masks.append([
                    topic_con['attention_mask'].squeeze(0).tolist(),
                    topic_cur['attention_mask'].squeeze(0).tolist()
                ])
                pbar.update(1)
        
        # コメントベクトルの変換
        comment_vectors_list = [vec.tolist() for vec in all_comment_vectors]
        
        # 9. データの保存（ファイル名をモデル名に変更）
        print(f"9. {config_name} モデル用ファイルへの保存...")
        
        # 推論データの保存
        inference_data = {
            "coheren_inputs": coheren_inputs,
            "coheren_masks": coheren_masks,
            "coheren_types": coheren_types,
            "topic_inputs": topic_inputs,
            "topic_masks": topic_masks,
            "comment_vectors": comment_vectors_list,
            "sentences": all_sentences,
            "sentence_metadata": all_sentence_metadata,
            "model_config": config,
            "use_comments": use_comments,
            "save_raw_data": save_raw_data,
            "comment_window_start_offset": comment_window_start_offset,
            "comment_window_end_offset": comment_window_end_offset,
            "adjust_timetable": adjust_timetable,
            "min_sentence_length": min_sentence_length,
            "margin_seconds": margin_seconds,  # マージン情報を追加
            "time_windows": all_time_windows  # 時間ウィンドウ情報を追加
        }
        
        model_output_dir = f"{output_dir}/{config_name}"
        os.makedirs(model_output_dir, exist_ok=True)
        
        # ファイル名をモデル名に変更
        filename = "inference_data.json"
        with open(f"{model_output_dir}/{filename}", "w", encoding="utf-8") as f:
            json.dump(inference_data, f, indent=2, ensure_ascii=False)
        
        print(f"{config_name} モデル用推論データを保存: {model_output_dir}/{filename}")
    
    # 10. 正解ラベルの保存（全モデル共通）
    print("10. 正解ラベルの保存...")
    gold_labels = {
        "boundary_labels": all_boundary_labels,
        "topic_labels": all_topic_labels,
        "time_windows": all_time_windows,  # 時間ウィンドウ情報
        "sentences": all_sentences,
        "sentence_metadata": all_sentence_metadata,
        "video_count": len(video_urls),
        "use_comments": use_comments,
        "save_raw_data": save_raw_data,
        "comment_window_start_offset": comment_window_start_offset,
        "comment_window_end_offset": comment_window_end_offset,
        "adjust_timetable": adjust_timetable,
        "min_sentence_length": min_sentence_length,
        "margin_seconds": margin_seconds,  # マージン情報を追加
        "adjusted_windows": all_adjusted_windows if adjust_timetable else []
    }
    
    with open(f"{output_dir}/gold_labels.json", "w", encoding="utf-8") as f:
        json.dump(gold_labels, f, indent=2, ensure_ascii=False)
    
    # 11. 未加工データの統合CSVを作成（オプション）
    if save_raw_data and all_raw_chunks:
        print("11. 未加工データの統合CSVを作成...")
        raw_summary_csv = f"{raw_data_dir}/all_raw_transcripts_summary.csv"
        
        with open(raw_summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '動画番号', 'チャンク番号', '開始時間(時:分:秒)', 
                '終了時間(時:分:秒)', '開始時間(秒)', '終了時間(秒)',
                '期間(秒)', '未加工字幕テキスト', '文字数', 'メモ'
            ])
            
            chunk_counter = 0
            current_video = 1
            video_chunk_counts = {}
            
            for i, (chunk, meta) in enumerate(zip(all_raw_chunks, all_raw_metadata)):
                # 動画番号の判定（簡易的な方法）
                if i > 0 and i % 100 == 0:  # 適当な閾値
                    current_video += 1
                
                if current_video not in video_chunk_counts:
                    video_chunk_counts[current_video] = 0
                
                video_chunk_counts[current_video] += 1
                chunk_counter += 1
                
                start_hms = seconds_to_hms(meta['start'])
                end_hms = seconds_to_hms(meta['end'])
                
                # メモ欄
                memo = ""
                if '[' in chunk and ']' in chunk:
                    memo = "タグを含む"
                elif len(chunk) < 3:
                    memo = "短い"
                
                writer.writerow([
                    current_video,
                    video_chunk_counts[current_video],
                    start_hms,
                    end_hms,
                    f"{meta['start']:.3f}",
                    f"{meta['end']:.3f}",
                    f"{meta.get('duration', meta['end'] - meta['start']):.3f}",
                    chunk,
                    len(chunk),
                    memo
                ])
        
        print(f"未加工データ統合CSV保存完了: {raw_summary_csv}")
    
    # 12. CSV形式でも保存（可読性のため）
    print("12. 加工済みデータのCSV保存...")
    with open(f"{output_dir}/transcript_with_labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["動画番号", "文番号", "開始時間", "終了時間", "字幕テキスト", 
                        "トピック番号", "トピック名", "境界ラベル", "時間ウィンドウ", "調整済み", "マージン使用"])
        
        for i, (sentence, metadata, topic_idx, boundary) in enumerate(zip(
            all_sentences, all_sentence_metadata, all_topic_labels, all_boundary_labels
        )):
            start_time = seconds_to_hms(metadata['start'])
            end_time = seconds_to_hms(metadata['end'])
            
            # トピック情報の取得
            if topic_idx < len(all_time_windows):
                window_info = all_time_windows[topic_idx]
                topic_name = window_info['topic']
                time_window = f"{window_info['start_display']}-{window_info['end_display']}"
                is_adjusted = window_info.get('start_adjusted', False) or window_info.get('end_adjusted', False)
            else:
                topic_name = "Unknown"
                time_window = "Unknown"
                is_adjusted = False
            
            # 動画番号の判定
            video_num = 1
            cumulative_sentences = 0
            for video_idx in range(len(video_urls)):
                if i < cumulative_sentences + len(all_sentences) * (video_idx + 1) / len(video_urls):
                    video_num = video_idx + 1
                    break
                cumulative_sentences += len(all_sentences) * (video_idx + 1) / len(video_urls)
            
            writer.writerow([
                video_num,
                i + 1,
                start_time,
                end_time,
                sentence,
                topic_idx,
                topic_name,
                "○" if boundary == 1 else "",
                time_window,
                "○" if is_adjusted else "",
                "○"  # マージン使用フラグ
            ])
    
    print("\n=== 推論データ生成完了 ===")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"総文数（加工済み）: {len(all_sentences)}")
    print(f"境界数: {sum(all_boundary_labels)}")
    print(f"時間ウィンドウ数: {len(all_time_windows)}")
    print(f"動画数: {len(video_urls)}")
    print(f"コメント利用: {'有効' if use_comments else '無効'}")
    print(f"コメントウィンドウ: 字幕開始+{comment_window_start_offset}秒 ～ 字幕開始+{comment_window_end_offset}秒")
    print(f"未加工データ保存: {'有効' if save_raw_data else '無効'}")
    print(f"タイムテーブル調整: {'有効' if adjust_timetable else '無効'}")
    print(f"短文結合閾値: {min_sentence_length}文字")
    print(f"境界マージン: {margin_seconds}秒")
    if save_raw_data:
        print(f"未加工データディレクトリ: {raw_data_dir}")
        print(f"総未加工チャンク数: {len(all_raw_chunks)}")
    print(f"生成されたモデル設定: {list(model_configs.keys())}")

if __name__ == "__main__":
    # 複数の動画URLとタイムテーブルを定義
    VIDEO_URLS = [
        "https://www.youtube.com/watch?v=jmADUo9Vcho",
    
    ]
    
    # 各動画に対応するタイムテーブルを読み込む
    TIMETABLE_TEXTS = []
    for i in range(len(VIDEO_URLS)):
        try:
            with open(f"test_{i+1}.txt", "r", encoding="utf-8") as f:
                TIMETABLE_TEXTS.append(f.read())
        except FileNotFoundError:
            print(f"警告: test_{i+1}.txt が見つかりません。デフォルトのタイムテーブルを使用します。")
            # デフォルトの時間ウィンドウ形式
            TIMETABLE_TEXTS.append("0:00:00 1:00:00 デフォルトトピック")
    
    # 複数のモデル設定を定義
    MODEL_CONFIGS = {
        "default": {
            "coherence_model": "cl-tohoku/bert-base-japanese",
            "topic_model": "pkshatech/simcse-ja-bert-base-clcmlp"
        }
    }
    
    # コメント利用フラグ（True: コメントを使用, False: コメントを使用しない）
    USE_COMMENTS = True
    
    # 未加工データ保存フラグ（True: 未加工データを保存, False: 保存しない）
    SAVE_RAW_DATA = True
    
    # コメント収集ウィンドウ設定（秒）: 字幕開始+10秒 ～ 字幕開始+15秒
    COMMENT_WINDOW_START_OFFSET = 10   # 字幕開始からの開始オフセット
    COMMENT_WINDOW_END_OFFSET = 15     # 字幕開始からの終了オフセット
    
    # タイムテーブル調整フラグ
    ADJUST_TIMETABLE = True
    
    # 短文結合の閾値（文字数）
    MIN_SENTENCE_LENGTH = 6
    
    # 境界マージン秒数
    MARGIN_SECONDS = 3.0
    
    save_inference_data(
        video_urls=VIDEO_URLS,
        timetable_texts=TIMETABLE_TEXTS,
        output_dir="./inference_data",
        model_configs=MODEL_CONFIGS,
        use_comments=USE_COMMENTS,
        save_raw_data=SAVE_RAW_DATA,
        comment_window_start_offset=COMMENT_WINDOW_START_OFFSET,
        comment_window_end_offset=COMMENT_WINDOW_END_OFFSET,
        adjust_timetable=ADJUST_TIMETABLE,
        min_sentence_length=MIN_SENTENCE_LENGTH,
        margin_seconds=MARGIN_SECONDS
    )