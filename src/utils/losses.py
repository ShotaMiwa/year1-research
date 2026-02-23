"""
損失関数モジュール
各種損失関数の定義
"""
import torch
import torch.nn as nn


class MarginRankingLoss(nn.Module):
    """マージンランキング損失"""
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: マージンの大きさ
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positive_scores: ポジティブサンプルのスコア
            negative_scores: ネガティブサンプルのスコア
            
        Returns:
            損失値
        """
        scores = self.margin - (positive_scores - negative_scores)
        scores = scores.clamp(min=0)
        return scores.mean()


class CombinedLoss:
    """複数の損失を組み合わせる"""
    
    def __init__(self, topic_weight: float = 1.0, margin_weight: float = 1.0):
        """
        Args:
            topic_weight: トピック損失の重み
            margin_weight: マージン損失の重み
        """
        self.topic_weight = topic_weight
        self.margin_weight = margin_weight
        self.topic_loss_fn = nn.CrossEntropyLoss()
        self.margin_loss_fn = MarginRankingLoss()
    
    def __call__(self, topic_loss: torch.Tensor, margin_loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            topic_loss: トピック損失
            margin_loss: マージン損失
            
        Returns:
            結合された損失
        """
        return self.topic_weight * topic_loss + self.margin_weight * margin_loss