"""
モデルアーキテクチャモジュール
BERTベースのセグメンテーションモデルの定義
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForNextSentencePrediction, AutoModel
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertModel, BertOnlyNSPHead
)
from transformers.modeling_outputs import NextSentencePredictorOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple


class AverageFusionLayer(nn.Module):
    """
    推論専用: 発話ベクトルとコメントベクトルの単純平均を取る層
    """
    
    def __init__(self):
        super().__init__()
        # 学習パラメータなし
    
    def forward(self, utterance_vec: torch.Tensor, comment_vec: torch.Tensor) -> torch.Tensor:
        """
        発話ベクトルとコメントベクトルの要素ごとの平均を計算
        
        Args:
            utterance_vec: (batch_size, hidden_dim) 発話ベクトル
            comment_vec: (batch_size, hidden_dim) コメントベクトル
            
        Returns:
            fused: (batch_size, hidden_dim) 平均化されたベクトル
        """
        fused = (utterance_vec + comment_vec) / 2.0
        return fused


class SegmentationModel(nn.Module):
    """
    セグメンテーションモデルのアーキテクチャ
    CoherenceモデルとTopicモデルを統合
    """
    
    def __init__(
        self,
        coherence_model_name: str = "cl-tohoku/bert-base-japanese",
        topic_model_name: str = "pkshatech/simcse-ja-bert-base-clcmlp",
        use_comments_for_topic: bool = False,
        fusion_method: str = 'average'
    ):
        """
        Args:
            coherence_model_name: Coherenceモデル名
            topic_model_name: Topicモデル名
            use_comments_for_topic: 推論時にコメントを使用するか
            fusion_method: 融合方法 ('average' or 'linear')
        """
        super().__init__()
        
        self.coherence_model_name = coherence_model_name
        self.topic_model_name = topic_model_name
        self.use_comments_for_topic = use_comments_for_topic
        self.fusion_method = fusion_method
        
        # モデルのロード
        self._load_models()
        
        # 推論専用の融合層
        if use_comments_for_topic and fusion_method == 'average':
            self.comment_fusion = AverageFusionLayer()
        else:
            self.comment_fusion = None
    
    def _load_models(self):
        """モデルをロード"""
        try:
            # Topicモデル
            self.topic_model = AutoModel.from_pretrained(self.topic_model_name)
            
            # Coherenceモデル (Next Sentence Prediction対応)
            self.coherence_model = BertForNextSentencePrediction.from_pretrained(
                self.coherence_model_name,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=True
            )
            
            print(f"✅ モデルロード成功: Coherence={self.coherence_model_name}, Topic={self.topic_model_name}")
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            # フォールバック
            print("デフォルトモデルでフォールバック")
            self.topic_model = AutoModel.from_pretrained("pkshatech/simcse-ja-bert-base-clcmlp")
            self.coherence_model = BertForNextSentencePrediction.from_pretrained(
                "cl-tohoku/bert-base-japanese",
                num_labels=2,
                output_attentions=False,
                output_hidden_states=True
            )
    
    def encode_topic(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Topicモデルで発話をエンコード
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            encoded: (batch_size, hidden_dim) [CLS]トークンの表現
        """
        outputs = self.topic_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]
    
    def encode_coherence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coherenceモデルでスコアを計算
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            
        Returns:
            scores: (batch_size, 2) NSPスコア
            features: pooled output
        """
        outputs = self.coherence_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 新しいtransformersはNextSentencePredictorOutputを返す
        if hasattr(outputs, 'logits'):
            scores = outputs.logits
        else:
            scores = outputs[0]
        return scores, None
    
    def fuse_vectors(
        self,
        utterance_vec: torch.Tensor,
        comment_vec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        発話ベクトルとコメントベクトルを融合
        
        Args:
            utterance_vec: (batch_size, hidden_dim)
            comment_vec: (batch_size, hidden_dim) or None
            
        Returns:
            fused: (batch_size, hidden_dim)
        """
        if comment_vec is None or self.comment_fusion is None:
            return utterance_vec
        
        return self.comment_fusion(utterance_vec, comment_vec)
    
    def compute_topic_similarity(
        self,
        context_vec: torch.Tensor,
        target_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        コサイン類似度を計算
        
        Args:
            context_vec: (batch_size, hidden_dim)
            target_vec: (batch_size, hidden_dim)
            
        Returns:
            similarity: (batch_size,)
        """
        return F.cosine_similarity(context_vec, target_vec, dim=1, eps=1e-08)


class CustomBertForNSP(BertPreTrainedModel):
    """
    カスタムBERT NSPモデル
    隠れ層とpooled outputの両方を返す
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_feature: bool = False,
        **kwargs,
    ):
        """
        フォワードパス
        
        Returns:
            (NextSentencePredictorOutput, pooled_output)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)
        
        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_scores.view(-1, 2),
                labels.view(-1)
            )
        
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output
        
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), pooled_output