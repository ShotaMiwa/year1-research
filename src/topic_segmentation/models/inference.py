"""
推論ロジックモジュール
モデルの推論処理を管理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from models.architecture import SegmentationModel
from utils.depth_score import DepthScoreCalculator


class InferenceWrapper:
    """
    推論処理を管理するラッパークラス
    """
    
    def __init__(
        self,
        model: SegmentationModel,
        use_comments: bool = True,
        fusion_method: str = 'average'
    ):
        """
        Args:
            model: セグメンテーションモデル
            use_comments: コメントを使用するか
            fusion_method: 融合方法
        """
        self.model = model
        self.use_comments = use_comments
        self.fusion_method = fusion_method
        self.depth_calculator = DepthScoreCalculator()
        
        # 評価モードに設定
        self.model.eval()
    
    @torch.no_grad()
    def predict_scores(
        self,
        sentences: List[str],
        tokenizer,
        comments: Optional[List[Dict]] = None,
        device: torch.device = torch.device('cuda')
    ) -> Tuple[List[float], Dict]:
        """
        境界スコアを予測
        
        Args:
            sentences: 発話のリスト
            tokenizer: トークナイザー
            comments: コメントデータ（オプション）
            device: デバイス
            
        Returns:
            (scores, debug_info)のタプル
        """
        num_sentences = len(sentences)
        
        # Coherenceスコア計算
        coherence_scores = self._compute_coherence_scores(
            sentences, tokenizer, device
        )
        
        # Topicスコア計算
        topic_scores, topic_debug = self._compute_topic_scores(
            sentences, tokenizer, comments, device
        )
        
        # スコアの結合
        final_scores = []
        for i in range(len(coherence_scores)):
            if i < len(topic_scores):
                combined = coherence_scores[i] + topic_scores[i]
            else:
                combined = coherence_scores[i]
            final_scores.append(combined)
        
        # デバッグ情報
        debug_info = {
            'coherence_raw_scores': coherence_scores,
            'topic_raw_scores': topic_scores,
            **topic_debug
        }
        
        return final_scores, debug_info
    
    def _compute_coherence_scores(
        self,
        sentences: List[str],
        tokenizer,
        device: torch.device
    ) -> List[float]:
        """
        Coherenceスコアを計算
        
        Args:
            sentences: 発話のリスト
            tokenizer: トークナイザー
            device: デバイス
            
        Returns:
            Coherenceスコアのリスト
        """
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            # 隣接する発話をペアにする
            sentence_a = sentences[i]
            sentence_b = sentences[i + 1]
            
            # トークン化
            encoded = tokenizer(
                sentence_a,
                sentence_b,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # デバイスに転送
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            token_type_ids = encoded['token_type_ids'].to(device)
            
            # Coherenceスコア取得
            scores, _ = self.model.encode_coherence(
                input_ids, attention_mask, token_type_ids
            )
            
            # Softmaxを適用して確率に変換
            probs = F.softmax(scores, dim=1)
            coherence_score = probs[0, 0].item()  # "is next"の確率
            
            coherence_scores.append(coherence_score)
        
        return coherence_scores
    
    def _compute_topic_scores(
        self,
        sentences: List[str],
        tokenizer,
        comments: Optional[List[Dict]],
        device: torch.device
    ) -> Tuple[List[float], Dict]:
        """
        Topicスコアを計算
        
        Args:
            sentences: 発話のリスト
            tokenizer: トークナイザー
            comments: コメントデータ（オプション）
            device: デバイス
            
        Returns:
            (topic_scores, debug_info)
        """
        # 発話をエンコード
        utterance_embeddings = self._encode_utterances(
            sentences, tokenizer, device
        )
        
        # コメントを使用する場合
        if self.use_comments and comments is not None:
            comment_embeddings = self._encode_comments(
                sentences, comments, tokenizer, device
            )
            # 融合
            fused_embeddings = []
            for utt_emb, com_emb in zip(utterance_embeddings, comment_embeddings):
                fused = self.model.fuse_vectors(utt_emb.unsqueeze(0), com_emb.unsqueeze(0))
                fused_embeddings.append(fused.squeeze(0))
        else:
            fused_embeddings = utterance_embeddings
        
        # トピックスコア計算
        topic_scores = []
        for i in range(1, len(fused_embeddings)):
            # 前後のコンテキストを平均
            context_start = max(0, i - 2)
            context_vec = torch.mean(
                torch.stack(fused_embeddings[context_start:i]), dim=0
            )
            
            current_end = min(len(fused_embeddings), i + 2)
            current_vec = torch.mean(
                torch.stack(fused_embeddings[i:current_end]), dim=0
            )
            
            # コサイン類似度
            similarity = F.cosine_similarity(
                context_vec.unsqueeze(0),
                current_vec.unsqueeze(0),
                dim=1
            ).item()
            
            topic_scores.append(similarity)
        
        debug_info = {
            'num_utterances': len(utterance_embeddings),
            'num_comments': len(comment_embeddings) if comments else 0,
            'used_fusion': self.use_comments and comments is not None
        }
        
        return topic_scores, debug_info
    
    def _encode_utterances(
        self,
        sentences: List[str],
        tokenizer,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        発話をエンコード
        
        Args:
            sentences: 発話のリスト
            tokenizer: トークナイザー
            device: デバイス
            
        Returns:
            埋め込みベクトルのリスト
        """
        embeddings = []
        
        for sentence in sentences:
            encoded = tokenizer(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            embedding = self.model.encode_topic(input_ids, attention_mask)
            embeddings.append(embedding.squeeze(0))
        
        return embeddings
    
    def _encode_comments(
        self,
        sentences: List[str],
        comments: List[Dict],
        tokenizer,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        コメントをエンコード
        
        Args:
            sentences: 発話のリスト
            comments: コメントデータ
            tokenizer: トークナイザー
            device: デバイス
            
        Returns:
            コメント埋め込みのリスト
        """
        # TODO: コメントの実際のエンコーディングロジックを実装
        # 仮実装: ゼロベクトルを返す
        embeddings = []
        hidden_dim = 768  # BERTの隠れ層次元
        
        for _ in sentences:
            # 実際にはコメントを処理してエンコード
            embedding = torch.zeros(hidden_dim, device=device)
            embeddings.append(embedding)
        
        return embeddings
    
    @torch.no_grad()
    def compute_depth_scores(
        self,
        raw_scores: List[float]
    ) -> Tuple[List[float], Dict]:
        """
        深度スコアを計算
        
        Args:
            raw_scores: 生スコアのリスト
            
        Returns:
            (depth_scores, statistics)
        """
        # Sigmoidを適用
        sigmoid_scores = [1 / (1 + np.exp(-s)) for s in raw_scores]
        
        # 深度スコア計算
        depth_scores = self.depth_calculator.calculate(sigmoid_scores)
        
        # 統計情報
        statistics = {
            'min': float(np.min(depth_scores)),
            'max': float(np.max(depth_scores)),
            'mean': float(np.mean(depth_scores)),
            'std': float(np.std(depth_scores))
        }
        
        return depth_scores, statistics