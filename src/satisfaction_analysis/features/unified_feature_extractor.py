# -*- coding: utf-8 -*-
"""
unified_feature_extractor.py

アンケート23件とShorts 77本（および今後の全データ）で
完全に同一の定義・同一モデル・同一計算式で特徴量を抽出する共通モジュール。

マスター定義書: docs/paper/04_feature_definitions_and_formulation.md
"""
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


# ==============================================================================
# 共通正規表現パターン
# ==============================================================================
GRASS_RE = re.compile(r'[wWｗ草笑]')
QUEST_RE = re.compile(r'[?？]|何|どう|誰|いつ|どこ|なぜ|ですか|ますか')
EXCLA_RE = re.compile(r'[!！]')
WORD_RE  = re.compile(r'[\u4e00-\u9fff]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+|[a-zA-Z0-9]+')
TIME_PAT = re.compile(r'(\d+):(\d+):(\d+\.\d+)\s+-->\s+(\d+):(\d+):(\d+\.\d+)')
TAG_PAT  = re.compile(r'<(\d+:\d+:\d+\.\d+)><c>(.*?)</c>')


def vtt_time_to_sec(time_str: str) -> float:
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)
    return 0.0


# ==============================================================================
# パーサー関数
# ==============================================================================
def parse_live_chat_json(chat_path: Path) -> List[Tuple[float, str]]:
    """YouTube生チャットJSONをパースして (offset_sec, text) のリストを返す"""
    records = []
    if not chat_path.exists():
        return records
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


def parse_vtt_subtitles(vtt_path: Path) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, str]]]:
    """
    VTT字幕をパースして、
    (subtitles: [(start, end, text)], word_tokens: [(start, text)]) を返す
    """
    if not vtt_path.exists():
        return [], []

    subs = []
    cur_start, cur_end, cur_text = None, None, []
    with open(vtt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = TIME_PAT.search(line)
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

    # ワード/文字トークン抽出
    with open(vtt_path, encoding='utf-8') as f:
        content = f.read()
    cues = re.split(r'\n\n+', content)
    word_tokens = []
    t_pat = re.compile(r'(\d+:\d+:\d+\.\d+)\s+-->\s+(\d+:\d+:\d+\.\d+)')
    for c in cues:
        lines = c.strip().split('\n')
        if not lines:
            continue
        m = t_pat.search(lines[0])
        if not m:
            continue
        c_start = vtt_time_to_sec(m.group(1))
        c_end   = vtt_time_to_sec(m.group(2))
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
                for tm, txt in TAG_PAT.findall(l):
                    sec = vtt_time_to_sec(tm)
                    clean_txt = txt.strip()
                    if clean_txt:
                        word_tokens.append((sec, clean_txt))
    return subs, word_tokens


# ==============================================================================
# 特徴量抽出共通クラス
# ==============================================================================
class UnifiedFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = None
        self.bert_model = None
        self.sentiment_pipe = None

    def load_models(self):
        """BERTモデルの遅延ロード"""
        if self.bert_model is None:
            model_name = "cl-tohoku/bert-base-japanese"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.eval()
            if self.device != "cpu":
                self.bert_model.to(self.device)

        if self.sentiment_pipe is None:
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="koheiduck/bert-japanese-finetuned-sentiment",
                truncation=True,
                max_length=128,
                device=0 if self.device != "cpu" and torch.cuda.is_available() else -1,
            )

    def extract_features(
        self,
        chats: List[Tuple[float, str]],
        subtitles: List[Tuple[float, float, str]],
        sub_tokens: List[Tuple[float, str]],
        start_sec: float,
        end_sec: float,
    ) -> Dict[str, float]:
        """
        区間 [start_sec, end_sec] に対する全特徴量を計算
        """
        self.load_models()
        dur = max(end_sec - start_sec, 1.0)

        # 1. 区間内チャット
        seg_chats = [t for s, t in chats if start_sec <= s < end_sec]
        n_c = len(seg_chats)
        comment_rate = n_c / dur

        if n_c > 0:
            grass_count = sum(1 for t in seg_chats if GRASS_RE.search(t))
            grass_ratio = grass_count / n_c

            question_count = sum(1 for t in seg_chats if QUEST_RE.search(t))
            question_ratio = question_count / n_c

            exclamation_count = sum(1 for t in seg_chats if EXCLA_RE.search(t))
            exclamation_ratio = exclamation_count / n_c

            # 語彙多様性 (TTR: Type-Token Ratio)
            all_words = []
            for t in seg_chats:
                all_words.extend([w for w in WORD_RE.findall(t) if len(w) > 1])
            comment_vocab_diversity = len(set(all_words)) / len(all_words) if all_words else 0.5

            # BERT コメント埋め込み & L2意味分散
            inputs = self.tokenizer(
                seg_chats[:100],  # 上限100件で計算
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            )
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            chat_centroid = embeds.mean(axis=0)
            comment_semantic_variance = float(((embeds - chat_centroid) ** 2).sum(axis=1).mean())

            # コメント感情極性
            pipe_res = self.sentiment_pipe(seg_chats[:50])
            pos = sum(1 for r in pipe_res if r["label"].upper() in ("POSITIVE", "POS"))
            neg = sum(1 for r in pipe_res if r["label"].upper() in ("NEGATIVE", "NEG"))
            sentiment_polarity = (pos - neg) / (pos + neg + 1e-6)
        else:
            grass_ratio = question_ratio = exclamation_ratio = 0.0
            comment_vocab_diversity = 0.5
            comment_semantic_variance = 50.0  # デフォルト基準値
            sentiment_polarity = 0.0
            chat_centroid = None

        # 2. 区間内字幕・発話
        seg_subs = [(s, e, t) for s, e, t in subtitles if not (e <= start_sec or s >= end_sec)]
        seg_words = [t for tm, t in sub_tokens if start_sec <= tm < end_sec]
        seg_spoken_text = "".join(seg_words) if seg_words else "".join(t for _, _, t in seg_subs)

        total_speech_sec = sum(min(e, end_sec) - max(s, start_sec) for s, e, _ in seg_subs)
        speech_rate = len(seg_spoken_text) / total_speech_sec if total_speech_sec > 0 else 0.0

        pause_sec = max(dur - total_speech_sec, 0.0)
        pause_ratio = pause_sec / dur

        if seg_subs:
            # 字幕感情極性
            sub_texts = [t for _, _, t in seg_subs[:30]]
            sub_res = self.sentiment_pipe(sub_texts)
            sp = sum(1 for r in sub_res if r["label"].upper() in ("POSITIVE", "POS"))
            sn = sum(1 for r in sub_res if r["label"].upper() in ("NEGATIVE", "NEG"))
            subtitle_sentiment_polarity = (sp - sn) / (sp + sn + 1e-6)

            # 字幕-コメントコサイン類似度
            full_sub_text = " ".join(t for _, _, t in seg_subs)[:256]
            sub_input = self.tokenizer(
                [full_sub_text],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            )
            if self.device != "cpu":
                sub_input = {k: v.to(self.device) for k, v in sub_input.items()}
            with torch.no_grad():
                sub_out = self.bert_model(**sub_input)
                sub_embed = sub_out.last_hidden_state[:, 0, :].cpu().numpy()[0]

            if chat_centroid is not None:
                norm_sub = np.linalg.norm(sub_embed)
                norm_chat = np.linalg.norm(chat_centroid)
                if norm_sub > 0 and norm_chat > 0:
                    sub_comment_similarity = float(np.dot(sub_embed, chat_centroid) / (norm_sub * norm_chat))
                else:
                    sub_comment_similarity = 0.0
            else:
                sub_comment_similarity = 0.0
        else:
            subtitle_sentiment_polarity = 0.0
            sub_comment_similarity = 0.0

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
            "comment_vocab_diversity": round(comment_vocab_diversity, 4),
            "segment_length_sec": round(dur, 2),
        }
