"""
評価指標モジュール
各種評価メトリクスの計算
"""
import numpy as np
import segeval
from typing import List, Dict, Tuple


class MetricsCalculator:
    """
    評価指標を計算するクラス
    """
    
    @staticmethod
    def calculate_pk(
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        window_size: int = None
    ) -> float:
        """
        Pkスコアを計算
        
        Args:
            predicted_boundaries: 予測境界
            gold_boundaries: 正解境界
            window_size: ウィンドウサイズ（Noneの場合は自動計算）
            
        Returns:
            Pkスコア
        """
        # セグメント表現に変換
        seg_pred = MetricsCalculator._boundaries_to_segments(predicted_boundaries)
        seg_gold = MetricsCalculator._boundaries_to_segments(gold_boundaries)
        
        # Pkスコアを計算
        pk = segeval.pk(seg_pred, seg_gold, window_size=window_size)
        
        return float(pk)
    
    @staticmethod
    def calculate_window_diff(
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        window_size: int = None
    ) -> float:
        """
        WindowDiffスコアを計算
        
        Args:
            predicted_boundaries: 予測境界
            gold_boundaries: 正解境界
            window_size: ウィンドウサイズ
            
        Returns:
            WindowDiffスコア
        """
        # セグメント表現に変換
        seg_pred = MetricsCalculator._boundaries_to_segments(predicted_boundaries)
        seg_gold = MetricsCalculator._boundaries_to_segments(gold_boundaries)
        
        # WindowDiffスコアを計算
        wd = segeval.window_diff(seg_pred, seg_gold, window_size=window_size)
        
        return float(wd)
    
    @staticmethod
    def calculate_precision_recall_f1(
        predicted_boundaries: List[int],
        gold_boundaries: List[int]
    ) -> Dict[str, float]:
        """
        適合率、再現率、F1スコアを計算
        
        Args:
            predicted_boundaries: 予測境界
            gold_boundaries: 正解境界
            
        Returns:
            {'precision', 'recall', 'f1', 'correct', 'false_positive', 'false_negative'}
        """
        correct_detections = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gold in zip(predicted_boundaries, gold_boundaries):
            if pred == 1 and gold == 1:
                correct_detections += 1
            elif pred == 1 and gold == 0:
                false_positives += 1
            elif pred == 0 and gold == 1:
                false_negatives += 1
        
        # 適合率
        precision = correct_detections / sum(predicted_boundaries) \
            if sum(predicted_boundaries) > 0 else 0.0
        
        # 再現率
        recall = correct_detections / sum(gold_boundaries) \
            if sum(gold_boundaries) > 0 else 0.0
        
        # F1スコア
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'correct_detections': correct_detections,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    @staticmethod
    def _boundaries_to_segments(boundaries: List[int]) -> List[int]:
        """
        境界リストをセグメント長のリストに変換
        
        Args:
            boundaries: 境界リスト（0 or 1）
            
        Returns:
            セグメント長のリスト
        """
        segments = []
        tmp = 0
        
        for b in boundaries:
            tmp += 1
            if b == 1:
                segments.append(tmp)
                tmp = 0
        
        # 最後のセグメント
        if tmp > 0:
            segments.append(tmp)
        
        return segments
    
    @staticmethod
    def calculate_all_metrics(
        predicted_boundaries: List[int],
        gold_boundaries: List[int]
    ) -> Dict[str, float]:
        """
        全ての評価指標を計算
        
        Args:
            predicted_boundaries: 予測境界
            gold_boundaries: 正解境界
            
        Returns:
            全メトリクスの辞書
        """
        # ウィンドウサイズを計算
        seg_gold = MetricsCalculator._boundaries_to_segments(gold_boundaries)
        avg_segment_length = np.mean(seg_gold)
        window_size = int(avg_segment_length / 2)
        
        # 各メトリクスを計算
        pk = MetricsCalculator.calculate_pk(
            predicted_boundaries, gold_boundaries, window_size
        )
        wd = MetricsCalculator.calculate_window_diff(
            predicted_boundaries, gold_boundaries, window_size
        )
        prf_metrics = MetricsCalculator.calculate_precision_recall_f1(
            predicted_boundaries, gold_boundaries
        )
        
        return {
            'Pk': pk,
            'WindowDiff': wd,
            'window_size': window_size,
            **prf_metrics
        }


class RandomBaselineEvaluator:
    """
    ランダムベースラインの評価
    """
    
    @staticmethod
    def evaluate(
        gold_boundaries: List[int],
        num_trials: int = 100
    ) -> Dict[str, float]:
        """
        ランダム境界検出を評価
        
        Args:
            gold_boundaries: 正解境界
            num_trials: 試行回数
            
        Returns:
            統計情報
        """
        num_sentences = len(gold_boundaries)
        num_true_boundaries = sum(gold_boundaries)
        
        pk_scores = []
        wd_scores = []
        
        for _ in range(num_trials):
            # ランダムに境界を選択
            random_boundaries = np.random.choice(
                num_sentences,
                size=num_true_boundaries,
                replace=False
            )
            random_boundaries = sorted(random_boundaries)
            
            # 境界リストを作成
            predicted = [0] * num_sentences
            for b in random_boundaries:
                predicted[b] = 1
            
            # メトリクスを計算
            pk = MetricsCalculator.calculate_pk(predicted, gold_boundaries)
            wd = MetricsCalculator.calculate_window_diff(predicted, gold_boundaries)
            
            pk_scores.append(pk)
            wd_scores.append(wd)
        
        return {
            'pk_mean': float(np.mean(pk_scores)),
            'pk_std': float(np.std(pk_scores)),
            'wd_mean': float(np.mean(wd_scores)),
            'wd_std': float(np.std(wd_scores)),
            'num_trials': num_trials
        }