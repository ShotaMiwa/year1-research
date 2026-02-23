"""
深度スコア計算モジュール
TETアルゴリズムによる深度スコア計算
"""
import torch
import numpy as np
from typing import List, Union


class DepthScoreCalculator:
    """深度スコア計算クラス"""
    
    @staticmethod
    def calculate(scores: Union[List[float], torch.Tensor, np.ndarray]) -> List[float]:
        """
        TETアルゴリズムで深度スコアを計算
        
        Args:
            scores: 入力スコアのリスト
            
        Returns:
            深度スコアのリスト
        """
        # Tensorの場合はCPUに移動
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().detach()
        
        # ndarrayの場合はそのまま使用
        if isinstance(scores, np.ndarray):
            scores = scores.tolist() if scores.ndim == 1 else scores
        
        output_scores = []
        
        for i in range(len(scores)):
            lflag = scores[i]
            rflag = scores[i]
            
            # エッジケース: 最初の要素
            if i == 0:
                # 右側のみを探索
                for r in range(i + 1, len(scores)):
                    if rflag <= scores[r]:
                        rflag = scores[r]
                    else:
                        break
            
            # エッジケース: 最後の要素
            elif i == len(scores) - 1:
                # 左側のみを探索
                for l in range(i - 1, -1, -1):
                    if lflag <= scores[l]:
                        lflag = scores[l]
                    else:
                        break
            
            # 通常ケース: 中間の要素
            else:
                # 右側を探索
                for r in range(i + 1, len(scores)):
                    if rflag <= scores[r]:
                        rflag = scores[r]
                    else:
                        break
                
                # 左側を探索
                for l in range(i - 1, -1, -1):
                    if lflag <= scores[l]:
                        lflag = scores[l]
                    else:
                        break
            
            # 深度スコアを計算
            depth_score = 0.5 * (lflag + rflag - 2 * scores[i])
            
            # Tensorの場合はスカラー値に変換
            if isinstance(depth_score, torch.Tensor):
                depth_score = depth_score.item()
            
            output_scores.append(float(depth_score))
        
        return output_scores
    
    @staticmethod
    def calculate_with_normalization(scores: Union[List[float], torch.Tensor, np.ndarray]) -> tuple:
        """
        深度スコアを計算し、Z-score正規化を適用
        
        Args:
            scores: 入力スコアのリスト
            
        Returns:
            (normalized_scores, raw_scores, mean, std)のタプル
        """
        raw_scores = DepthScoreCalculator.calculate(scores)
        
        # numpy配列に変換
        scores_array = np.array(raw_scores)
        
        # 統計量を計算
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        
        # Z-score正規化
        if std > 1e-8:
            normalized_scores = (scores_array - mean) / std
        else:
            normalized_scores = scores_array - mean
        
        return normalized_scores.tolist(), raw_scores, float(mean), float(std)


def normalize_scores(scores: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    スコアを正規化
    
    Args:
        scores: 入力スコア
        method: 正規化方法 ('zscore', 'minmax', 'sigmoid')
        
    Returns:
        正規化されたスコア
    """
    if method == 'zscore':
        mean = np.mean(scores)
        std = np.std(scores)
        if std > 1e-8:
            return (scores - mean) / std
        else:
            return scores - mean
    
    elif method == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val > 1e-8:
            return (scores - min_val) / (max_val - min_val)
        else:
            return scores - min_val
    
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-scores))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")