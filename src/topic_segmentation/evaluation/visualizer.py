"""
可視化モジュール
結果の可視化とグラフ生成
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


class ResultVisualizer:
    """
    結果を可視化するクラス
    """
    
    def __init__(self, save_dir: str = "./results"):
        """
        Args:
            save_dir: 保存ディレクトリ
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def create_score_histograms(
        self,
        result: Dict,
        filename: str = "score_histograms.png"
    ):
        """
        各種スコアのヒストグラムを作成
        
        Args:
            result: 結果の辞書
            filename: 保存ファイル名
        """
        debug_scores = result.get('debug_scores', {})
        
        # ヒストグラム用データの準備
        hist_data = {}
        
        # 各種スコアを収集
        if 'coherence_raw_scores' in debug_scores and debug_scores['coherence_raw_scores']:
            hist_data['Coherence Raw Scores'] = debug_scores['coherence_raw_scores']
        
        if 'topic_raw_scores' in debug_scores and debug_scores['topic_raw_scores']:
            hist_data['Topic Raw Scores'] = debug_scores['topic_raw_scores']
        
        if 'raw_scores' in result and result['raw_scores']:
            hist_data['Final Sigmoid Scores'] = result['raw_scores']
        
        if 'depth_scores' in result and result['depth_scores']:
            hist_data['Depth Scores'] = result['depth_scores']
        
        if not hist_data:
            print("⚠️ ヒストグラム用のスコアデータが見つかりません")
            return
        
        # ヒストグラムの作成
        num_plots = len(hist_data)
        fig, axes = plt.subplots(
            (num_plots + 2) // 3, 3,
            figsize=(18, 4 * ((num_plots + 2) // 3))
        )
        
        # 1次元配列に変換
        if num_plots <= 3:
            axes = axes.reshape(1, -1) if num_plots > 1 else np.array([[axes]])
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        for idx, (title, scores) in enumerate(hist_data.items()):
            row = idx // 3
            col = idx % 3
            
            ax = axes[row, col] if num_plots > 3 else axes[0, col] if num_plots > 1 else axes[0, 0]
            
            # ヒストグラムの描画
            ax.hist(scores, bins=30, alpha=0.7,
                   color=colors[idx % len(colors)], edgecolor='black')
            
            # 統計情報
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            median_val = np.median(scores)
            
            # タイトルと統計情報
            ax.set_title(f"{title}\nMean={mean_val:.3f}, Std={std_val:.3f}")
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.axvline(mean_val, color='red', linestyle='--', label='Mean')
            ax.axvline(median_val, color='green', linestyle='--', label='Median')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 未使用のサブプロットを非表示
        for idx in range(num_plots, (num_plots + 2) // 3 * 3):
            row = idx // 3
            col = idx % 3
            if num_plots > 3:
                axes[row, col].set_visible(False)
            elif num_plots > 1:
                axes[0, col].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ヒストグラムを保存: {save_path}")
    
    def visualize_boundary_detection(
        self,
        sentences: List[str],
        depth_scores: List[float],
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        filename: str = "boundary_visualization.png"
    ):
        """
        境界検出結果を可視化
        
        Args:
            sentences: 発話のリスト
            depth_scores: 深度スコア
            predicted_boundaries: 予測境界
            gold_boundaries: 正解境界
            filename: 保存ファイル名
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # 深度スコアのプロット
        x = np.arange(len(depth_scores))
        ax1.plot(x, depth_scores, 'b-', linewidth=2, label='Depth Score')
        ax1.set_xlabel('Sentence Index')
        ax1.set_ylabel('Depth Score')
        ax1.set_title('Depth Scores')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # 境界の可視化
        ax2.scatter(x, [1] * len(x), c='lightgray', s=100, alpha=0.5, label='Sentences')
        
        # 予測境界
        pred_indices = [i for i, b in enumerate(predicted_boundaries) if b == 1]
        if pred_indices:
            ax2.scatter(pred_indices, [1] * len(pred_indices),
                       c='red', s=200, marker='v', label='Predicted', alpha=0.7)
        
        # 正解境界
        gold_indices = [i for i, b in enumerate(gold_boundaries) if b == 1]
        if gold_indices:
            ax2.scatter(gold_indices, [0.5] * len(gold_indices),
                       c='green', s=200, marker='^', label='Gold', alpha=0.7)
        
        ax2.set_xlabel('Sentence Index')
        ax2.set_yticks([0.5, 1])
        ax2.set_yticklabels(['Gold', 'Predicted'])
        ax2.set_title('Boundary Detection Results')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 境界検出結果を保存: {save_path}")
    
    def plot_score_comparison(
        self,
        coherence_scores: List[float],
        topic_scores: List[float],
        combined_scores: List[float],
        filename: str = "score_comparison.png"
    ):
        """
        各種スコアの比較プロット
        
        Args:
            coherence_scores: コヒーレンススコア
            topic_scores: トピックスコア
            combined_scores: 結合スコア
            filename: 保存ファイル名
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        
        x = np.arange(len(coherence_scores))
        
        ax.plot(x, coherence_scores, 'b-', label='Coherence', linewidth=2, alpha=0.7)
        ax.plot(x, topic_scores, 'g-', label='Topic', linewidth=2, alpha=0.7)
        ax.plot(x, combined_scores, 'r-', label='Combined', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Sentence Boundary Index')
        ax.set_ylabel('Score')
        ax.set_title('Score Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ スコア比較を保存: {save_path}")


def save_results_to_csv(result: Dict, save_dir: str, filename: str = "boundary_results.csv"):
    """
    境界推定結果をCSVに保存
    
    Args:
        result: 結果の辞書
        save_dir: 保存ディレクトリ
        filename: ファイル名
    """
    import csv
    
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー行
        writer.writerow([
            'sentence_index',
            'sentence',
            'raw_score',
            'depth_score',
            'predicted_boundary',
            'gold_boundary',
            'is_correct'
        ])
        
        # データ行
        for i, (sentence, raw_score, depth_score, pred, gold) in enumerate(zip(
            result['sentences'],
            result['raw_scores'],
            result['depth_scores'],
            result['predicted_boundaries'],
            result['gold_boundaries']
        )):
            is_correct = "正解" if pred == gold else "不正解"
            writer.writerow([
                i,
                sentence,
                f"{raw_score:.6f}",
                f"{depth_score:.6f}",
                pred,
                gold,
                is_correct
            ])
    
    print(f"✅ 境界推定結果をCSVに保存: {csv_path}")