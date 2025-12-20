import os
import csv
import re
import random
import pytchat
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForNextSentencePrediction
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from collections import Counter
import MeCab
import unicodedata
import time
from functools import lru_cache
import torch.nn.functional as F

from common_transcript_processing import (
    seconds_to_hms,
    get_raw_transcript,
    basic_transcript_processing,
    save_transcript_to_csv,
    hms_to_seconds
)

print("=== 初期化処理を開始します ===")


# NEologd辞書を使用するTaggerの設定
def create_neologd_tagger():
    """NEologd辞書を使用するMeCab Taggerを作成"""
    print("MeCab Taggerの初期化を開始します...")
    try:
        # NEologd辞書のパスを探す
        neologd_dic_path = None
        for root, dirs, files in os.walk('/usr/lib/x86_64-linux-gnu/mecab/dic'):
            if 'mecab-ipadic-neologd' in root:
                neologd_dic_path = root
                break
        
        if neologd_dic_path is None:
            # 見つからない場合はデフォルトパスを試す
            neologd_dic_path = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
        
        if os.path.exists(neologd_dic_path):
            tagger = MeCab.Tagger(f'-d {neologd_dic_path}')
            print(f"NEologd辞書を使用: {neologd_dic_path}")
        else:
            tagger = MeCab.Tagger('-Owakati')
            print("NEologd辞書が見つからないため、デフォルト辞書を使用します")
            
        return tagger
    except Exception as e:
        print(f"NEologd辞書の初期化に失敗: {e}")
        print("デフォルト辞書を使用します")
        return MeCab.Tagger('-Owakati')

# グローバルなTaggerインスタンスを作成
tagger = create_neologd_tagger()

# ==============================
# モデル準備
# ==============================
print("モデルの準備を開始します...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {DEVICE}")

print("トークナイザーとモデルの読み込み中...")
NSP_MODEL_NAME = "cl-tohoku/bert-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL_NAME)
nsp_tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL_NAME)
nsp_model = AutoModelForNextSentencePrediction.from_pretrained(NSP_MODEL_NAME).to(DEVICE)
nsp_model.eval()

SIMCSE_MODEL_NAME = "pkshatech/simcse-ja-bert-base-clcmlp"
simcse_tokenizer = AutoTokenizer.from_pretrained(SIMCSE_MODEL_NAME)
simcse_model = AutoModel.from_pretrained(SIMCSE_MODEL_NAME)
simcse_model.to(DEVICE)
simcse_model.eval()

LANGUAGE = "ja"


# ================================================
# 新しい字幕処理関数群（共通処理を使用するように修正）
# ================================================
def find_sentence_boundary_candidates(text: str, tagger) -> List[int]:
    """
    形態素解析して文末候補（発話が途切れそうな箇所）を抽出
    口語日本語の特性を考慮した文末候補を検出
    """
    candidates = []
    
    # テキストを形態素解析
    node = tagger.parseToNode(text)
    pos_list = []
    surfaces = []
    
    while node:
        if node.surface:
            feature = node.feature.split(',')
            pos = feature[0] if len(feature) > 0 else ''
            pos1 = feature[1] if len(feature) > 1 else ''
            surfaces.append(node.surface)
            pos_list.append((pos, pos1))
        node = node.next
    
    if not surfaces:
        return candidates
    
    current_length = 0
    for i, ((pos, pos1), surface) in enumerate(zip(pos_list, surfaces)):
        current_length += len(surface)
        
        # 文末候補の条件（口語日本語の特性を考慮）
        is_candidate = False
        
        # 1. 動詞で終わる（終止形・連体形）
        if pos == "動詞":
            is_candidate = True
        
        # 2. 助詞で終わる（特に終助詞・間投助詞）
        elif pos == "助詞":
            if pos1 in ["終助詞", "間投助詞"]:
                is_candidate = True
            # "ね"、"よ"、"よね"などの口語表現
            elif surface in ["ね", "よ", "よね", "か", "かな", "かしら"]:
                is_candidate = True
        
        # 3. 助動詞で終わる
        elif pos == "助動詞":
            is_candidate = True
        
        # 4. 名詞＋だ／です（断定）
        elif (pos == "名詞" and i > 0 and 
              pos_list[i-1][0] == "名詞" and
              surface in ["だ", "です", "である"]):
            is_candidate = True
        
        # 5. 感動詞で終わる
        elif pos == "感動詞":
            is_candidate = True
        
        # 6. 接続助詞の前（「て形」などで文が続く可能性が高い場所）
        elif pos == "助詞" and pos1 == "接続助詞":
            # 「て」、「たり」、「ながら」など
            if surface in ["て", "で", "たり", "ながら", "し"]:
                is_candidate = True
        
        # 最低文字数チェック（短すぎる文を防ぐ）
        if is_candidate and current_length >= 3:  # 最低3文字以上
            candidates.append(current_length)
    
    # 最後の位置も候補に追加（文全体）
    if len(text) >= 3:
        candidates.append(len(text))
    
    # 重複を除去して返す
    return sorted(list(set(candidates)))

def compute_nsp_score(sentence_a, sentence_b):
    """文A→文Bの自然なつながりを生スコア（logitsの差）で算出"""
    inputs = nsp_tokenizer(
        sentence_a, 
        sentence_b, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = nsp_model(**inputs).logits
        
        # softmax確率ではなく、生のlogitsの差を計算
        # logits[0]: IsNextスコア, logits[1]: NotNextスコア
        raw_score = logits[0, 0] - logits[0, 1]  # IsNext - NotNext
        
        return raw_score.item()  # 生スコアを返す

def split_combined_chunks_with_nsp(combined_text: str, tagger, debug_mode: bool = False) -> List[str]:
    """
    2チャンク結合テキストをNSPの生スコアで最適な境界で分割
    """
    if len(combined_text) <= 10:
        return [combined_text]
    
    candidates = find_sentence_boundary_candidates(combined_text, tagger)
    if not candidates:
        return [combined_text]
    
    candidate_scores = []
    for candidate_pos in candidates:
        if candidate_pos >= len(combined_text):
            continue
        left_text = combined_text[:candidate_pos]
        right_text = combined_text[candidate_pos:]
        
        # 生スコアを計算（softmax確率ではなく）
        nsp_raw_score = compute_nsp_score(left_text, right_text)
        candidate_scores.append((candidate_pos, nsp_raw_score, left_text, right_text))
    
    if not candidate_scores:
        return [combined_text]
    
    # 生スコアが最も高い候補を選択
    # 生スコアは正の値が大きいほど「つながりが自然」、負の値が大きいほど「つながらない」
    best_candidate = max(candidate_scores, key=lambda x: x[1])
    best_pos, best_raw_score, left_text, right_text = best_candidate
    
    if debug_mode:
        print(f"  候補スコア（生）:")
        for pos, score, l_text, r_text in candidate_scores:
            print(f"    位置{pos}: スコア={score:.4f}, 左:『{l_text[:20]}...』, 右:『{r_text[:20]}...』")
        print(f"  最良候補: 位置={best_pos}, 生スコア={best_raw_score:.4f}")
    
    # 閾値の調整：生スコアの場合は適切な閾値を設定する必要があります
    # 例: 0より大きければ結合が自然、0より小さければ不自然
    if best_raw_score < 0:  # 負の値は「つながらない」と判断
        return [combined_text]
    
    result = [left_text]
    if right_text.strip():
        result.extend(split_combined_chunks_with_nsp(right_text, tagger, debug_mode))
    
    return result
    
def analyze_chunk_characteristics(chunks: List[str]) -> Dict:
    """
    チャンクの統計的特性を分析
    """
    lengths = [len(chunk) for chunk in chunks]
    char_counts = [len(re.findall(r'[。．.！!？?]', chunk)) for chunk in chunks]
    
    return {
        'total_chunks': len(chunks),
        'avg_length': sum(lengths) / len(lengths),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'ending_punctuation_ratio': sum(1 for c in char_counts if c > 0) / len(chunks),
        'length_distribution': {
            'very_short': sum(1 for l in lengths if l <= 5),
            'short': sum(1 for l in lengths if 5 < l <= 15),
            'medium': sum(1 for l in lengths if 15 < l <= 30),
            'long': sum(1 for l in lengths if l > 30),
        }
    }

# ================================================
# 修正箇所: split_sentences_from_chunks関数を共通処理に置き換え
# ================================================
def split_sentences_from_chunks(raw_chunks: List[str], metadata_list: List[Dict], min_len: int = 6, max_len: int = 60, debug_nsp: bool = False):
    """基本的な字幕チャンク分割処理（共通処理を使用）"""
    
    print("基本的な字幕処理を開始します...")
    
    # 共通処理を使用して基本的な字幕処理を実行
    sentences, sentence_metadata = basic_transcript_processing(
        raw_chunks, 
        metadata_list,
        min_len=min_len,
        max_len=max_len,
        debug_mode=debug_nsp
    )
    
    print(f"基本的な処理完了: {len(sentences)} 文")
    
    # 従来の表示部分を保持
    print("=" * 60)
    print("処理後の字幕とタイムスタンプ")
    print("=" * 60)

    total_sentences = len(sentences)
    print(f"総文数: {total_sentences}")

    print("\n【最初の10件】")
    for i in range(min(10, total_sentences)):
        s = sentences[i]
        meta = sentence_metadata[i]
        start_hms = seconds_to_hms(meta['start'])
        end_hms = seconds_to_hms(meta['end'])
        print(f"{i+1:3d}. [{start_hms} - {end_hms}] {s}")

    if total_sentences > 10:
        print(f"\n【最後の10件】")
        for i in range(max(0, total_sentences - 10), total_sentences):
            s = sentences[i]
            meta = sentence_metadata[i]
            start_hms = seconds_to_hms(meta['start'])
            end_hms = seconds_to_hms(meta['end'])
            print(f"{i+1:3d}. [{start_hms} - {end_hms}] {s}")

    print("=" * 60)
    return sentences, sentence_metadata

def normalize_text(text):
    """テキストを正規化"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_pos_patterns(sentences: List[str]) -> Dict:
    """
    チャンク化された文章の最初の2品詞と最後の2品詞の組み合わせの割合を分析
    """
    print("品詞パターン分析を開始します...")
    first_pos_patterns = []
    last_pos_patterns = []
    
    # 特定の品詞パターンを持つ文章を記録
    target_patterns = []
    
    # プログレスバーで品詞分析の進捗を表示
    with tqdm(total=len(sentences), desc="品詞分析") as pbar:
        for sentence in sentences:
            if not sentence.strip():
                pbar.update(1)
                continue
                
            # テキストを正規化
            normalized_sentence = normalize_text(sentence)
            
            # MeCabで解析
            parsed = tagger.parse(normalized_sentence)
            node = tagger.parseToNode(normalized_sentence)
            nodes = []
            while node:
                if node.surface != "":  # 空白ノードを除外
                    feature = node.feature.split(',')
                    nodes.append({
                        'surface': node.surface,
                        'feature': feature,
                        'pos': feature[0] if len(feature) > 0 else '',
                        'pos1': feature[1] if len(feature) > 1 else '',
                        'pos2': feature[2] if len(feature) > 2 else '',
                        'conjtype': feature[4] if len(feature) > 4 else '',
                    })
                node = node.next
            
            if len(nodes) < 2:
                pbar.update(1)
                continue
                
            # 最初の2品詞の組み合わせ
            first_pos1 = nodes[0]['pos'] if nodes[0]['pos'] else ''
            first_pos2 = nodes[1]['pos'] if nodes[1]['pos'] else ''
            first_pattern = f"{first_pos1}+{first_pos2}"
            first_pos_patterns.append(first_pattern)
            
            # 最後の2品詞の組み合わせ
            last_pos1 = nodes[-2]['pos'] if nodes[-2]['pos'] else ''
            last_pos2 = nodes[-1]['pos'] if nodes[-1]['pos'] else ''
            last_pattern = f"{last_pos1}+{last_pos2}"
            last_pos_patterns.append(last_pattern)
            
            # 特定の品詞パターンを持つ文章を記録
            target_last_pattern1 = "助詞+助詞"
            target_last_pattern2 = "助詞+連体詞" 
            target_first_pattern = "助動詞+助詞"
            
            if last_pattern == target_last_pattern1:
                target_patterns.append({
                    'sentence': sentence,
                    'pattern_type': '最後の2品詞',
                    'pattern': target_last_pattern1,
                    'position': '末尾'
                })
            
            if last_pattern == target_last_pattern2:
                target_patterns.append({
                    'sentence': sentence,
                    'pattern_type': '最後の2品詞', 
                    'pattern': target_last_pattern2,
                    'position': '末尾'
                })
                
            if first_pattern == target_first_pattern:
                target_patterns.append({
                    'sentence': sentence,
                    'pattern_type': '最初の2品詞',
                    'pattern': target_first_pattern, 
                    'position': '先頭'
                })
            
            pbar.update(1)
    
    # 出現頻度を計算
    first_pos_counter = Counter(first_pos_patterns)
    last_pos_counter = Counter(last_pos_patterns)
    
    # 割合を計算
    total_sentences = len(first_pos_patterns)
    
    first_pos_ratios = {pattern: count/total_sentences for pattern, count in first_pos_counter.items()}
    last_pos_ratios = {pattern: count/total_sentences for pattern, count in last_pos_counter.items()}
    
    return {
        "first_pos_patterns": dict(first_pos_counter),
        "last_pos_patterns": dict(last_pos_counter),
        "first_pos_ratios": first_pos_ratios,
        "last_pos_ratios": last_pos_ratios,
        "total_analyzed_sentences": total_sentences,
        "target_pattern_sentences": target_patterns
    }

def print_pos_analysis(analysis_result: Dict):
    """
    品詞組み合わせの分析結果を表示
    """
    print("\n" + "="*80)
    print("品詞組み合わせ分析結果 (NEologd辞書使用)")
    print("="*80)
    
    print(f"分析対象文数: {analysis_result['total_analyzed_sentences']}")
    
    print("\n【最初の2品詞の組み合わせ（出現頻度順）】")
    print("-" * 50)
    sorted_first = sorted(analysis_result['first_pos_ratios'].items(), 
                         key=lambda x: x[1], reverse=True)
    for pattern, ratio in sorted_first[:10]:
        count = analysis_result['first_pos_patterns'][pattern]
        print(f"  {pattern}: {count}回 ({ratio*100:.2f}%)")
    
    print("\n【最後の2品詞の組み合わせ（出現頻度順）】")
    print("-" * 50)
    sorted_last = sorted(analysis_result['last_pos_ratios'].items(), 
                        key=lambda x: x[1], reverse=True)
    for pattern, ratio in sorted_last[:10]:
        count = analysis_result['last_pos_patterns'][pattern]
        print(f"  {pattern}: {count}回 ({ratio*100:.2f}%)")
    
    # 最も頻出する組み合わせを表示
    if sorted_first:
        most_common_first = sorted_first[0]
        print(f"\n★ 最も頻出する文頭品詞組み合わせ: '{most_common_first[0]}' ({most_common_first[1]*100:.2f}%)")
    
    if sorted_last:
        most_common_last = sorted_last[0]
        print(f"★ 最も頻出する文末品詞組み合わせ: '{most_common_last[0]}' ({most_common_last[1]*100:.2f}%)")
    
    # 特定の品詞パターンを持つ文章を表示
    target_sentences = analysis_result.get('target_pattern_sentences', [])
    if target_sentences:
        print("\n" + "="*80)
        print("特定品詞パターンを持つ文章")
        print("="*80)
        
        # パターンごとにグループ化
        pattern_groups = {}
        for item in target_sentences:
            pattern_key = f"{item['pattern_type']}: {item['pattern']}"
            if pattern_key not in pattern_groups:
                pattern_groups[pattern_key] = []
            pattern_groups[pattern_key].append(item['sentence'])
        
        for pattern_key, sentences in pattern_groups.items():
            print(f"\n【{pattern_key}】")
            print("-" * 50)
            for i, sentence in enumerate(sentences, 1):
                # 分かち書きと品詞情報を表示
                normalized_sentence = normalize_text(sentence)
                parsed = tagger.parse(normalized_sentence)
                node = tagger.parseToNode(normalized_sentence)
                
                wakati_words = []
                pos_info = []
                while node:
                    if node.surface != "":
                        feature = node.feature.split(',')
                        pos = feature[0] if len(feature) > 0 else ''
                        wakati_words.append(node.surface)
                        pos_info.append(f"{node.surface}({pos})")
                    node = node.next
                
                print(f"{i}. 文章全体: {sentence}")
                print(f"   分かち書き: {parsed.strip()}")
                print(f"   品詞情報: {' '.join(pos_info)}")
                print()
    
    print("="*80)

# ==============================
# 修正箇所: 字幕取得処理（共通処理を使用）
# ==============================
def get_transcript(video_url: str, debug_nsp: bool = False) -> Tuple[List[str], List[Dict]]:
    print("字幕取得処理を開始します...")
    
    # 共通処理を使用して生字幕データを取得
    raw_chunks, raw_metadata = get_raw_transcript(video_url, LANGUAGE)
    print(f"取得した生の字幕チャンク数: {len(raw_chunks)}")
    
    print("基本的な字幕処理を実行中...")
    # 共通処理を使用して基本的な字幕処理を実行
    sentences, sentence_metadata = basic_transcript_processing(
        raw_chunks, 
        raw_metadata,
        min_len=6,
        max_len=60,
        debug_mode=debug_nsp
    )

    print(f"字幕取得成功: {len(sentences)} 文を取得")
    return sentences, sentence_metadata, raw_chunks, raw_metadata

# ==============================
# コメント処理
# ==============================
def get_comments(video_url: str) -> List[Dict]:
    print("コメント取得を開始します...")
    video_id = YouTube(video_url).video_id
    chat = pytchat.create(video_id=video_id)
    comments = []
    
    print("リアルタイムコメントを収集中...")
    with tqdm(desc="コメント収集") as pbar:
        while chat.is_alive():
            for c in chat.get().items:
                comments.append({
                    "text": c.message,
                    "time": c.elapsedTime
                })
                pbar.update(1)
    
    print(f"コメント取得成功: {len(comments)} 件")
    return comments

def embed_comment(text: str) -> torch.Tensor:
    inputs = simcse_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = simcse_model(**inputs)
        # SimCSEは[CLS]ベクトルを利用
        emb = outputs.last_hidden_state[:, 0, :]
    return emb.squeeze(0).cpu()

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

def build_comment_vectors(comments: List[Dict], sentence_metadata: List[Dict], 
                         window_start_offset: int = 10, window_end_offset: int = 15) -> List[torch.Tensor]:
    """
    コメントベクトル構築（ウィンドウ設定を字幕開始からのオフセットで指定）
    
    Args:
        comments: コメントデータ
        sentence_metadata: 文のメタデータ（修正処理後の時間）
        window_start_offset: コメント収集開始オフセット（字幕開始からの秒数）
        window_end_offset: コメント収集終了オフセット（字幕開始からの秒数）
    """
    print("コメントベクトル構築を開始します...")
    print(f"ウィンドウ設定: 字幕開始+{window_start_offset}秒 ～ 字幕開始+{window_end_offset}秒")
    
    comment_vecs = []
    
    print("コメント埋め込みを計算中...")
    # 事前に全コメント埋め込みをキャッシュ
    with tqdm(total=len(comments), desc="コメント埋め込み") as pbar:
        for c in comments:
            c["vec"] = embed_comment(c["text"])
            # 時間を秒数に変換してキャッシュ
            c["timestamp_seconds"] = timestamp_to_seconds(c["time"])
            pbar.update(1)

    print("文ごとのコメントベクトルを構築中...")
    with tqdm(total=len(sentence_metadata), desc="コメントベクトル構築") as pbar:
        for i, meta in enumerate(sentence_metadata):
            sentence_start = meta["start"]
            
            # 字幕開始時間からのオフセットでウィンドウを計算
            window_start = sentence_start + window_start_offset  # 字幕開始+10秒
            window_end = sentence_start + window_end_offset      # 字幕開始+15秒
            
            # デバッグ情報（最初の数文のみ）
            if i < 5:
                print(f"文{i}: 字幕開始={sentence_start:.1f}秒, ウィンドウ={window_start:.1f}-{window_end:.1f}秒")
            
            # ウィンドウ内のコメントをフィルタリング
            window_comments = []
            for c in comments:
                comment_time = c["timestamp_seconds"]
                if window_start <= comment_time <= window_end:
                    window_comments.append(c["vec"])
            
            if window_comments:
                avg_vec = torch.stack(window_comments).mean(dim=0)
                if i < 5:  # デバッグ
                    print(f"  コメント数: {len(window_comments)}")
            else:
                avg_vec = torch.zeros(simcse_model.config.hidden_size)
                if i < 5:  # デバッグ
                    print(f"  コメント数: 0")
            
            comment_vecs.append(avg_vec)
            pbar.update(1)
            
    return comment_vecs

# ==============================
# Coherenceモデル用データ生成
# ==============================
def generate_coherence_data(sentences: List[str], history: int = 2, window_size: int = 10):
    """
    Coherenceモデル用のデータを生成
    - context: 現在発話の前後から取得
    - cur: 現在発話（正例）
    - neg: 現在位置から±window_size分外のランダム発話
    """
    print("Coherenceモデル用データを生成中...")
    data = []
    dial_len = len(sentences)
    
    with tqdm(total=dial_len, desc="Coherenceデータ生成") as pbar:
        for utt_idx in range(dial_len):
            context, cur, neg = [], [], []
            
            # neg_index: 現在位置から±window_size分外からランダム選択
            valid_neg_indices = []
            for i in range(dial_len):
                if i < utt_idx - window_size or i > utt_idx + window_size:
                    valid_neg_indices.append(i)
            
            if valid_neg_indices:
                neg_index = random.choice(valid_neg_indices)
            else:
                neg_index = random.randint(0, dial_len - 1)
            
            # contextの構築（双方向）
            l, r = utt_idx, utt_idx + 1
            for _ in range(history):
                if l > -1:
                    context.append(sentences[l])
                    l -= 1
                if r < dial_len:
                    cur.append(sentences[r])
                    r += 1
            
            context.reverse()
            
            if context and cur:  # 有効なデータのみ追加
                data.append([(context, cur), (context, [sentences[neg_index]])])
            
            pbar.update(1)
    
    return data

# ==============================
# 生チャンクをCSVに保存する関数
# ==============================
def save_raw_chunks_to_csv(raw_chunks: List[str], metadata_list: List[Dict], csv_save_path: str):
    """
    生チャンクをCSVファイルに保存する
    """
    print("生チャンクをCSVに保存します...")
    
    with open(csv_save_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダーを書き込み
        writer.writerow(['チャンク番号', '開始時間', '終了時間', '開始時間(秒)', '終了時間(秒)', '生チャンクテキスト'])
        
        # データを書き込み
        with tqdm(total=len(raw_chunks), desc="生チャンクCSV保存") as pbar:
            for i, (chunk, metadata) in enumerate(zip(raw_chunks, metadata_list)):
                start_seconds = metadata['start']
                end_seconds = metadata['start'] + metadata['duration']
                
                # 秒を時:分:秒形式に変換
                start_hms = seconds_to_hms(start_seconds)
                end_hms = seconds_to_hms(end_seconds)
                
                writer.writerow([
                    i + 1,
                    start_hms,
                    end_hms,
                    f"{start_seconds:.2f}",
                    f"{end_seconds:.2f}",
                    chunk
                ])
                pbar.update(1)
    
    print(f"生チャンクをCSVに保存しました: {csv_save_path}")

# ==============================
# メイン処理
# ==============================
def preprocess_and_save_with_csv(video_url: str, save_path: str, csv_save_path: str = None, history: int = 2, window_size: int = 10, debug_nsp: bool = True, enable_comments: bool = True, comment_window_start_offset: int = 10, comment_window_end_offset: int = 15):
    """
    既存の処理に加えて、字幕をCSVでも保存する拡張関数（デバッグモード追加）
    共通処理を使用するように修正
    
    Args:
        enable_comments: コメント取得を有効にするかどうか
        comment_window_start_offset: コメント収集ウィンドウ開始オフセット（字幕開始からの秒数）
        comment_window_end_offset: コメント収集ウィンドウ終了オフセット（字幕開始からの秒数）
    """
    print("=== 前処理を開始します ===")
    
    # データ取得
    print("1. 字幕データの取得を開始")
    sentences, sentence_metadata, raw_chunks, raw_metadata = get_transcript(video_url, debug_nsp)  # 生チャンクも取得
    
    # ★追加: 生チャンクをCSVに保存
    if csv_save_path:
        raw_chunks_csv_path = csv_save_path.replace('.csv', '_raw_chunks.csv')
        save_raw_chunks_to_csv(raw_chunks, raw_metadata, raw_chunks_csv_path)
    
    # ★追加: 品詞組み合わせの分析
    print("2. 品詞組み合わせの分析を開始")
    pos_analysis = analyze_pos_patterns(sentences)
    print_pos_analysis(pos_analysis)
    
    # CSV保存（オプション） - 共通処理を使用
    if csv_save_path:
        print(f"3. CSVファイルへの保存を開始: {csv_save_path}")
        save_transcript_to_csv(sentences, sentence_metadata, csv_save_path, "字幕テキスト")
    
    # コメント処理をオプション化
    if enable_comments:
        print("4. コメントデータの取得を開始")
        comments = get_comments(video_url)
        
        print("5. コメントベクトルの構築を開始")
        com_vecs = build_comment_vectors(comments, sentence_metadata, 
                                        window_start_offset=comment_window_start_offset, 
                                        window_end_offset=comment_window_end_offset)
    else:
        print("4. コメント取得はスキップします")
        comments = []
        # 空のコメントベクトルを作成
        com_vecs = [torch.zeros(simcse_model.config.hidden_size) for _ in range(len(sentences))]

    # Coherenceモデル用データ生成
    print("6. Coherenceモデル用データの生成を開始")
    coherence_data = generate_coherence_data(sentences, history, window_size)
    
    # NSP用トークン化 (Coherenceモデル用)
    print("7. トークン化処理を開始")
    coheren_inputs_pos, coheren_masks_pos, coheren_types_pos = [], [], []
    coheren_inputs_neg, coheren_masks_neg, coheren_types_neg = [], [], []

    with tqdm(total=len(coherence_data), desc="トークン化") as pbar:
        for pos_pair, neg_pair in coherence_data:
            # 正例（context, cur）
            context, response = pos_pair
            context_text = " [SEP] ".join(context)
            response_text = response[0] if response else ""
            encoded_pos = tokenizer(
                context_text,
                response_text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            coheren_inputs_pos.append(encoded_pos['input_ids'].squeeze(0))
            coheren_masks_pos.append(encoded_pos['attention_mask'].squeeze(0))
            coheren_types_pos.append(encoded_pos['token_type_ids'].squeeze(0))

            # 負例（context, neg）
            context, response = neg_pair
            context_text = " [SEP] ".join(context)
            response_text = response[0] if response else ""
            encoded_neg = tokenizer(
                context_text,
                response_text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            coheren_inputs_neg.append(encoded_neg['input_ids'].squeeze(0))
            coheren_masks_neg.append(encoded_neg['attention_mask'].squeeze(0))
            coheren_types_neg.append(encoded_neg['token_type_ids'].squeeze(0))
            
            pbar.update(1)

    # 正例・負例を結合して (N, 2, seq_len) へ整形
    print("8. テンソルの整形を開始")
    coheren_inputs = torch.stack([torch.stack([p, n]) for p, n in zip(coheren_inputs_pos, coheren_inputs_neg)])
    coheren_masks = torch.stack([torch.stack([p, n]) for p, n in zip(coheren_masks_pos, coheren_masks_neg)])
    coheren_types = torch.stack([torch.stack([p, n]) for p, n in zip(coheren_types_pos, coheren_types_neg)])

    # Topicモデル用データ
    print("9. Topicモデル用データの準備を開始")
    print("  - SimCSEトークン化を実行中...")
    sub_ids_simcse = []
    with tqdm(total=len(sentences), desc="SimCSEトークン化") as pbar:
        for s in sentences:
            sub_ids_simcse.append(simcse_tokenizer.encode(s, truncation=True, max_length=128))
            pbar.update(1)
            
    topic_num = [(len(sentences), i) for i in range(len(sentences))]
    
    # データ保存
    print("10. データの保存を開始")
    data = {
    # Coherenceモデル用（Tensorをlist化して軽量化）
    "coheren_inputs": [t.cpu().tolist() for t in coheren_inputs],
    "coheren_masks": [t.cpu().tolist() for t in coheren_masks],
    "coheren_types": [t.cpu().tolist() for t in coheren_types],
    
    # Topicモデル用
    "sub_ids_simcse": sub_ids_simcse,
    "com_vecs": com_vecs,  # コメント平均ベクトル
    "topic_num": topic_num,
    "sentences": sentences,  # デバッグ用
    # ★追加: 品詞分析結果も保存
    "pos_analysis": pos_analysis
 }

    torch.save({
    "coheren_inputs": coheren_inputs.cpu(),
    "coheren_masks": coheren_masks.cpu(),
    "coheren_types": coheren_types.cpu(),
    "sub_ids_simcse": sub_ids_simcse,
    "com_vecs": [v.cpu() for v in com_vecs],
    "topic_num": topic_num,
    "sentences": sentences,
    # ★追加: 品詞分析結果も保存
    "pos_analysis": pos_analysis
    }, save_path)

    print(f"保存完了（torch形式）: {save_path}")
    print(f"Coherenceデータ数: {len(coheren_inputs)}")
    print(f"Topicデータ数: {len(sentences)}")
    print("=== 前処理が完了しました ===")


# メイン処理
if __name__ == "__main__":
    print("プログラムを開始します")
    
    # 複数の動画URLをリストで指定
    video_urls = [
        "https://www.youtube.com/watch?v=RnG_iJ55evs",
        "https://www.youtube.com/watch?v=j96CiYjKpXk",
    ]
    
    for i, video_url in enumerate(video_urls):
        print(f"\n=== {i+1}/{len(video_urls)} 番目の動画を処理中 ===")
        save_path = f"/content/output_{i+1}.pt"  
        csv_save_path = f"/content/transcript_{i+1}.csv"
        
        preprocess_and_save_with_csv(
            video_url, 
            save_path, 
            csv_save_path, 
            history=2, 
            window_size=10, 
            debug_nsp=True, 
            enable_comments=False,  # コメント取得を無効化
            comment_window_start_offset=10,  # 字幕開始+10秒
            comment_window_end_offset=15     # 字幕開始+15秒
        )
    
    print("全ての動画処理が正常に終了しました")