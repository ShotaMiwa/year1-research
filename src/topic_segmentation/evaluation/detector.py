"""
境界検出モジュール
深度スコアから境界を検出
"""
import numpy as np
from typing import List, Tuple, Dict


class BoundaryDetector:
    """
    境界検出アルゴリズム
    """
    
    @staticmethod
    def detect_adaptive(
        depth_scores: List[float],
        threshold_multiplier: float = 0.5
    ) -> List[int]:
        """
        適応的閾値による境界検出
        
        Args:
            depth_scores: 深度スコアのリスト
            threshold_multiplier: 閾値の乗数
            
        Returns:
            境界インデックスのリスト
        """
        depth_scores = np.array(depth_scores)
        
        # 統計量を計算
        mean_score = np.mean(depth_scores)
        std_score = np.std(depth_scores)
        
        # 閾値を設定
        threshold = mean_score + threshold_multiplier * std_score
        
        # 閾値を超えるインデックスを取得
        boundaries = np.where(depth_scores > threshold)[0]
        
        return sorted(boundaries.tolist())
    
    @staticmethod
    def detect_fixed(
        depth_scores: List[float],
        num_boundaries: int = None
    ) -> List[int]:
        """
        固定数の境界検出
        
        Args:
            depth_scores: 深度スコアのリスト
            num_boundaries: 検出する境界数（Noneの場合は自動）
            
        Returns:
            境界インデックスのリスト
        """
        depth_scores = np.array(depth_scores)
        
        # 境界数が指定されていない場合は自動計算
        if num_boundaries is None:
            num_boundaries = max(1, len(depth_scores) // 20)
        
        # スコアが高い上位N個を選択
        boundaries = np.argsort(depth_scores)[-num_boundaries:]
        
        return sorted(boundaries.tolist())
    
    @staticmethod
    def detect_threshold(
        depth_scores: List[float],
        threshold: float = 0.5
    ) -> List[int]:
        """
        固定閾値による境界検出
        
        Args:
            depth_scores: 深度スコアのリスト
            threshold: 閾値
            
        Returns:
            境界インデックスのリスト
        """
        depth_scores = np.array(depth_scores)
        
        # 閾値を超えるインデックスを取得
        boundaries = np.where(depth_scores > threshold)[0]
        
        return sorted(boundaries.tolist())
    
    @staticmethod
    def detect(
        depth_scores: List[float],
        method: str = 'adaptive',
        **kwargs
    ) -> List[int]:
        """
        指定された方法で境界検出
        
        Args:
            depth_scores: 深度スコアのリスト
            method: 検出方法 ('adaptive', 'fixed', 'threshold')
            **kwargs: 各メソッドの追加パラメータ
            
        Returns:
            境界インデックスのリスト
        """
        if method == 'adaptive':
            return BoundaryDetector.detect_adaptive(depth_scores, **kwargs)
        elif method == 'fixed':
            return BoundaryDetector.detect_fixed(depth_scores, **kwargs)
        elif method == 'threshold':
            return BoundaryDetector.detect_threshold(depth_scores, **kwargs)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    @staticmethod
    def boundaries_to_labels(
        boundaries: List[int],
        total_length: int
    ) -> List[int]:
        """
        境界インデックスを0/1ラベルに変換
        
        Args:
            boundaries: 境界インデックスのリスト
            total_length: 総長さ
            
        Returns:
            0/1ラベルのリスト
        """
        labels = [0] * total_length
        for b in boundaries:
            if 0 <= b < total_length:
                labels[b] = 1
        return labels


class MultimethodBoundaryDetector:
    """
    複数の方法で境界検出を試し、最良の結果を選択
    """
    
    def __init__(self, methods: List[str] = None):
        """
        Args:
            methods: 試行する検出方法のリスト
        """
        self.methods = methods or ['adaptive', 'fixed', 'threshold']
        self.detector = BoundaryDetector()
    
    def detect_best(
        self,
        depth_scores: List[float],
        gold_boundaries: List[int],
        metric_fn
    ) -> Tuple[List[int], str, float, Dict]:
        """
        複数の方法を試して最良の結果を返す
        
        Args:
            depth_scores: 深度スコアのリスト
            gold_boundaries: 正解境界（評価用）
            metric_fn: 評価関数（predicted, gold -> score）
            
        Returns:
            (best_boundaries, best_method, best_score, all_results)
        """
        best_score = float('inf')
        best_boundaries = []
        best_method = None
        all_results = {}
        
        for method in self.methods:
            # 境界を検出
            boundaries = self.detector.detect(depth_scores, method=method)
            
            # ラベルに変換
            predicted_labels = self.detector.boundaries_to_labels(
                boundaries, len(gold_boundaries)
            )
            
            # 評価
            score = metric_fn(predicted_labels, gold_boundaries)
            
            # 結果を保存
            all_results[method] = {
                'boundaries': boundaries,
                'score': score,
                'num_boundaries': len(boundaries)
            }
            
            # 最良スコアを更新
            if score < best_score:
                best_score = score
                best_boundaries = boundaries
                best_method = method
        
        return best_boundaries, best_method, best_score, all_results