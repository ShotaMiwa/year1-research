import os
import json
import torch
import numpy as np
import segeval
from tqdm import tqdm
from transformers import set_seed
from model import SegModel
from decimal import Decimal
import csv  
import matplotlib.pyplot as plt  

# ===========================================================
# 設定部分
# ===========================================================
INFERENCE_DATA_BASE_DIR = "./inference_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./results"
os.makedirs(SAVE_PATH, exist_ok=True)

# モデル設定
MODEL_CONFIGS = {
    "trained-model": {
        "coherence_model": "cl-tohoku/bert-base-japanese",
        "topic_model": "pkshatech/simcse-ja-bert-base-clcmlp",
        "inference_data_path": f"{INFERENCE_DATA_BASE_DIR}/default/inference_data.json",
        "model_checkpoint": "/content/drive/MyDrive/seg_models/hiroyuki_model/epoch_4_step_918",  # 学習済みモデルのパス
        "use_comments_for_topic": True,  # テスト時はコメントを使用
        "fusion_method": "average"  # 平均融合のみを使用
    }
}

# ===========================================================
# 深度スコア計算
# ===========================================================
def depth_score_cal(scores):
    """深度スコアを計算"""
    scores_array = np.array(scores)
    output_scores = []
    
    for i in range(len(scores_array)):
        lflag = scores_array[i]
        rflag = scores_array[i]
        
        if i == 0:
            for r in range(i+1, len(scores_array)):
                if rflag <= scores_array[r]:
                    rflag = scores_array[r]
                else:
                    break
        elif i == len(scores_array)-1:
            for l in range(i-1, -1, -1):
                if lflag <= scores_array[l]:
                    lflag = scores_array[l]
                else:
                    break
        else:
            for r in range(i+1, len(scores_array)):
                if rflag <= scores_array[r]:
                    rflag = scores_array[r]
                else:
                    break
            for l in range(i-1, -1, -1):
                if lflag <= scores_array[l]:
                    lflag = scores_array[l]
                else:
                    break
        
        depth_score = 0.5 * (lflag + rflag - 2 * scores_array[i])
        output_scores.append(depth_score)
    
    return output_scores

# ===========================================================
# 境界検出関数
# ===========================================================
def detect_boundaries(depth_scores, method='adaptive', num_boundaries=None, threshold=0.5):
    """深度スコアから境界を検出"""
    depth_scores = np.array(depth_scores)
    
    if method == 'adaptive':
        mean_score = np.mean(depth_scores)
        std_score = np.std(depth_scores)
        threshold = mean_score + 0.5 * std_score
        boundaries = np.where(depth_scores > threshold)[0]
        
    elif method == 'fixed':
        if num_boundaries is None:
            num_boundaries = max(1, len(depth_scores) // 20)
        boundaries = np.argsort(depth_scores)[-num_boundaries:]
        
    elif method == 'threshold':
        boundaries = np.where(depth_scores > threshold)[0]
    
    else:
        raise ValueError("Unknown method")
    
    return sorted(boundaries)

# ===========================================================
# カスタムJSONエンコーダー
# ===========================================================
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Decimal, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)

# ===========================================================
# CSV保存関数
# ===========================================================
def save_results_to_csv(result, save_path):
    """境界推定結果をCSVに保存"""
    csv_path = f"{save_path}/boundary_results.csv"
    
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
    
    print(f"境界推定結果をCSVに保存: {csv_path}")

# ===========================================================
# スコアヒストグラム関数
# ===========================================================
def create_score_histograms(result, save_path):
    """各種スコアのヒストグラムを作成"""
    debug_scores = result.get('debug_scores', {})
    
    # ヒストグラム用データの準備
    hist_data = {}
    
    # コヒーレンス生スコア
    if 'coherence_raw_scores' in debug_scores and debug_scores['coherence_raw_scores']:
        hist_data['Coherence Raw Scores'] = debug_scores['coherence_raw_scores']
    
    # トピック生スコア
    if 'topic_raw_scores' in debug_scores and debug_scores['topic_raw_scores']:
        hist_data['Topic Raw Scores'] = debug_scores['topic_raw_scores']
    
    # コヒーレンス正規化後スコア
    if 'coherence_normalized_scores' in debug_scores and debug_scores['coherence_normalized_scores']:
        hist_data['Coherence Normalized Scores'] = debug_scores['coherence_normalized_scores']
    
    # トピック正規化後スコア
    if 'topic_normalized_scores' in debug_scores and debug_scores['topic_normalized_scores']:
        hist_data['Topic Normalized Scores'] = debug_scores['topic_normalized_scores']
    
    # 最終生スコア
    if 'final_raw' in debug_scores and debug_scores['final_raw']:
        hist_data['Final Raw Scores'] = debug_scores['final_raw']
    
    # sigmoid後最終スコア
    if 'raw_scores' in result and result['raw_scores']:
        hist_data['Final Sigmoid Scores'] = result['raw_scores']
    
    # 深度スコア
    if 'depth_scores' in result and result['depth_scores']:
        hist_data['Depth Scores'] = result['depth_scores']
    
    if not hist_data:
        print("!!!ヒストグラム用のスコアデータが見つかりません!!!")
        return
    
    # ヒストグラムの作成
    num_plots = len(hist_data)
    fig, axes = plt.subplots((num_plots + 2) // 3, 3, figsize=(18, 4 * ((num_plots + 2) // 3)))
    
    # 1次元配列に変換（サブプロットが1行の場合）
    if num_plots <= 3:
        axes = axes.reshape(1, -1)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
    
    for idx, (title, scores) in enumerate(hist_data.items()):
        row = idx // 3
        col = idx % 3
        
        ax = axes[row, col] if num_plots > 3 else axes[col]
        
        # ヒストグラムの描画
        n, bins, patches = ax.hist(scores, bins=30, alpha=0.7, color=colors[idx % len(colors)], edgecolor='black')
        
        # 統計情報の計算
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        median_val = np.median(scores)
        
        # 統計情報をプロットに追加
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        
        # タイトルとラベル
        ax.set_title(f'{title}\nDistribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Score Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 統計情報をテキストボックスで表示
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nMin: {min(scores):.4f}\nMax: {max(scores):.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
    
    # 未使用のサブプロットを非表示
    for idx in range(num_plots, axes.size):
        row = idx // 3
        col = idx % 3
        if num_plots > 3:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    # ヒストグラムを保存
    hist_path = f"{save_path}/score_histograms.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"スコアヒストグラムを保存: {hist_path}")

# ===========================================================
# スコア可視化関数
# ===========================================================
def visualize_scores(result, save_path):
    """コヒーレンス生スコア、トピック生スコア、正規化後のスコアを可視化"""
    # デバッグスコアデータの取得
    debug_scores = result.get('debug_scores', {})
    
    # 各種スコアを取得
    coherence_raw = debug_scores.get('coherence_raw_scores', [])
    topic_raw = debug_scores.get('topic_raw_scores', [])
    coherence_normalized = debug_scores.get('coherence_normalized_scores', [])
    topic_normalized = debug_scores.get('topic_normalized_scores', [])
    
    if not coherence_raw and not topic_raw and not coherence_normalized and not topic_normalized:
        print("⚠️ 可視化用のスコアデータが見つかりません")
        return
    
    # 文の数に応じてグラフサイズを調整
    num_sentences = len(result['raw_scores'])
    
    # 文の数に基づいて動的にサイズ調整
    if num_sentences <= 50:
        fig_width = 12
        font_size = 10
    elif num_sentences <= 100:
        fig_width = 16
        font_size = 9
    elif num_sentences <= 200:
        fig_width = 20
        font_size = 8
    else:
        fig_width = 24
        font_size = 7
    
    # サブプロットの数を決定（正規化後のスコアがある場合は5つ、ない場合は3つ）
    has_normalized = coherence_normalized and topic_normalized
    num_subplots = 5 if has_normalized else 3
    fig_height = 4 * num_subplots  # 各サブプロットの高さを4インチに設定
    
    line_width = 1.5 if num_sentences <= 100 else 1.0
    
    # グラフの作成
    plt.figure(figsize=(fig_width, fig_height))
    
    # 文のインデックス
    x = range(num_sentences)
    
    subplot_idx = 1
    
    # サブプロット1: コヒーレンス生スコア
    if coherence_raw:
        plt.subplot(num_subplots, 1, subplot_idx)
        plt.plot(x, coherence_raw[:num_sentences], 'b-', 
                label='Coherence Raw Scores', linewidth=line_width)
        plt.title('Coherence Raw Scores by Sentence Order', fontsize=font_size+2)
        plt.xlabel('Sentence Index', fontsize=font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=font_size)
        subplot_idx += 1
        
        # x軸の目盛り間隔を調整
        if num_sentences > 50:
            plt.xticks(range(0, num_sentences, max(1, num_sentences//20)))
    
    # サブプロット2: トピック生スコア
    if topic_raw:
        plt.subplot(num_subplots, 1, subplot_idx)
        plt.plot(x, topic_raw[:num_sentences], 'g-', 
                label='Topic Raw Scores', linewidth=line_width)
        plt.title('Topic Raw Scores by Sentence Order', fontsize=font_size+2)
        plt.xlabel('Sentence Index', fontsize=font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=font_size)
        subplot_idx += 1
        
        # x軸の目盛り間隔を調整
        if num_sentences > 50:
            plt.xticks(range(0, num_sentences, max(1, num_sentences//20)))
    
    # サブプロット3: コヒーレンス正規化後スコア（追加）
    if coherence_normalized:
        plt.subplot(num_subplots, 1, subplot_idx)
        plt.plot(x, coherence_normalized[:num_sentences], 'c-', 
                label='Coherence Normalized Scores', linewidth=line_width)
        plt.title('Coherence Normalized Scores by Sentence Order', fontsize=font_size+2)
        plt.xlabel('Sentence Index', fontsize=font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=font_size)
        subplot_idx += 1
        
        # x軸の目盛り間隔を調整
        if num_sentences > 50:
            plt.xticks(range(0, num_sentences, max(1, num_sentences//20)))
    
    # サブプロット4: トピック正規化後スコア（追加）
    if topic_normalized:
        plt.subplot(num_subplots, 1, subplot_idx)
        plt.plot(x, topic_normalized[:num_sentences], 'y-', 
                label='Topic Normalized Scores', linewidth=line_width)
        plt.title('Topic Normalized Scores by Sentence Order', fontsize=font_size+2)
        plt.xlabel('Sentence Index', fontsize=font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=font_size)
        subplot_idx += 1
        
        # x軸の目盛り間隔を調整
        if num_sentences > 50:
            plt.xticks(range(0, num_sentences, max(1, num_sentences//20)))
    
    # サブプロット5: 最終スコアと深度スコア
    plt.subplot(num_subplots, 1, subplot_idx)
    plt.plot(x, result['raw_scores'], 'r-', 
            label='Final Scores', linewidth=line_width, alpha=0.7)
    plt.plot(x, result['depth_scores'], 'm-', 
            label='Depth Scores', linewidth=line_width, alpha=0.7)
    
    # 予測境界を縦線で表示
    pred_boundaries = [i for i, val in enumerate(result['predicted_boundaries']) if val == 1]
    for boundary in pred_boundaries:
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 正解境界を縦線で表示
    gold_boundaries = [i for i, val in enumerate(result['gold_boundaries']) if val == 1]
    for boundary in gold_boundaries:
        plt.axvline(x=boundary, color='green', linestyle='--', alpha=0.7, linewidth=1)
    
    # 凡例に境界情報を追加
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', lw=2, alpha=0.7, label='Final Scores'),
        Line2D([0], [0], color='m', lw=2, alpha=0.7, label='Depth Scores'),
        Line2D([0], [0], color='red', linestyle='--', lw=1, label='Predicted Boundaries'),
        Line2D([0], [0], color='green', linestyle='--', lw=1, label='Gold Boundaries')
    ]
    
    plt.title('Final Scores and Depth Scores with Boundaries', fontsize=font_size+2)
    plt.xlabel('Sentence Index', fontsize=font_size)
    plt.ylabel('Score', fontsize=font_size)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, fontsize=font_size)
    
    # x軸の目盛り間隔を調整
    if num_sentences > 50:
        plt.xticks(range(0, num_sentences, max(1, num_sentences//20)))
    
    plt.tight_layout()
    
    # グラフを高解像度で保存
    plot_path = f"{save_path}/score_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"スコア可視化グラフを保存: {plot_path}")
    
    # ヒストグラムを作成
    create_score_histograms(result, save_path)
    
    # 文数が多い場合は追加のズームイングラフも作成
    if num_sentences > 100:
        create_zoomed_plots(result, save_path, num_sentences, has_normalized)

def create_zoomed_plots(result, save_path, num_sentences, has_normalized=False):
    """文数が多い場合にズームインしたグラフを作成"""
    debug_scores = result.get('debug_scores', {})
    coherence_raw = debug_scores.get('coherence_raw_scores', [])
    topic_raw = debug_scores.get('topic_raw_scores', [])
    coherence_normalized = debug_scores.get('coherence_normalized_scores', [])
    topic_normalized = debug_scores.get('topic_normalized_scores', [])
    
    # ズームインする区間を設定（例: 50文ごと）
    chunk_size = 50
    num_chunks = (num_sentences + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_sentences)
        
        if end_idx - start_idx < 10:  # 小さすぎるチャンクはスキップ
            continue
            
        # ズームイングラフの作成
        num_subplots = 5 if has_normalized else 3
        plt.figure(figsize=(12, 4 * num_subplots))
        
        x_zoom = range(start_idx, end_idx)
        subplot_idx = 1
        
        # サブプロット1: コヒーレンス生スコア（ズームイン）
        if coherence_raw:
            plt.subplot(num_subplots, 1, subplot_idx)
            plt.plot(x_zoom, coherence_raw[start_idx:end_idx], 'b-', 
                    label='Coherence Raw Scores', linewidth=1.5)
            plt.title(f'Coherence Raw Scores (Sentences {start_idx}-{end_idx-1})', fontsize=12)
            plt.xlabel('Sentence Index', fontsize=10)
            plt.ylabel('Score', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            subplot_idx += 1
        
        # サブプロット2: トピック生スコア（ズームイン）
        if topic_raw:
            plt.subplot(num_subplots, 1, subplot_idx)
            plt.plot(x_zoom, topic_raw[start_idx:end_idx], 'g-', 
                    label='Topic Raw Scores', linewidth=1.5)
            plt.title(f'Topic Raw Scores (Sentences {start_idx}-{end_idx-1})', fontsize=12)
            plt.xlabel('Sentence Index', fontsize=10)
            plt.ylabel('Score', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            subplot_idx += 1
        
        # サブプロット3: コヒーレンス正規化後スコア（ズームイン、追加）
        if coherence_normalized:
            plt.subplot(num_subplots, 1, subplot_idx)
            plt.plot(x_zoom, coherence_normalized[start_idx:end_idx], 'c-', 
                    label='Coherence Normalized Scores', linewidth=1.5)
            plt.title(f'Coherence Normalized Scores (Sentences {start_idx}-{end_idx-1})', fontsize=12)
            plt.xlabel('Sentence Index', fontsize=10)
            plt.ylabel('Score', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            subplot_idx += 1
        
        # サブプロット4: トピック正規化後スコア（ズームイン、追加）
        if topic_normalized:
            plt.subplot(num_subplots, 1, subplot_idx)
            plt.plot(x_zoom, topic_normalized[start_idx:end_idx], 'y-', 
                    label='Topic Normalized Scores', linewidth=1.5)
            plt.title(f'Topic Normalized Scores (Sentences {start_idx}-{end_idx-1})', fontsize=12)
            plt.xlabel('Sentence Index', fontsize=10)
            plt.ylabel('Score', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            subplot_idx += 1
        
        # サブプロット5: 最終スコアと深度スコア（ズームイン）
        plt.subplot(num_subplots, 1, subplot_idx)
        plt.plot(x_zoom, result['raw_scores'][start_idx:end_idx], 'r-', 
                label='Final Scores', linewidth=1.5, alpha=0.7)
        plt.plot(x_zoom, result['depth_scores'][start_idx:end_idx], 'm-', 
                label='Depth Scores', linewidth=1.5, alpha=0.7)
        
        # 境界線（ズームイン範囲内のみ表示）
        pred_boundaries = [i for i, val in enumerate(result['predicted_boundaries']) 
                          if val == 1 and start_idx <= i < end_idx]
        for boundary in pred_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        gold_boundaries = [i for i, val in enumerate(result['gold_boundaries']) 
                          if val == 1 and start_idx <= i < end_idx]
        for boundary in gold_boundaries:
            plt.axvline(x=boundary, color='green', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.title(f'Final and Depth Scores (Sentences {start_idx}-{end_idx-1})', fontsize=12)
        plt.xlabel('Sentence Index', fontsize=10)
        plt.ylabel('Score', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        
        # ズームイングラフを保存
        zoom_plot_path = f"{save_path}/score_visualization_zoom_{start_idx}_{end_idx-1}.png"
        plt.savefig(zoom_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ズームイングラフを保存: {zoom_plot_path}")

# ===========================================================
# 単一モデル推論関数
# ===========================================================
def run_inference_for_model(model_name, model_config):
    """単一モデルでの推論実行"""
    print(f"\n{'='*60}")
    print(f"モデル: {model_name}")
    print(f"{'='*60}")
    
    # 推論データの読み込み
    possible_paths = [
        model_config["inference_data_path"],
        f"{INFERENCE_DATA_BASE_DIR}/{model_name}/inference_data_{model_name}.json",
        f"{INFERENCE_DATA_BASE_DIR}/{model_name}/inference_data.json"
    ]
    
    inference_data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            inference_data_path = path
            break
    
    if inference_data_path is None:
        print(f"❌ 推論データが見つかりません。以下のパスを確認してください:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    gold_labels_path = f"{INFERENCE_DATA_BASE_DIR}/gold_labels.json"
    
    print(f"推論データ: {inference_data_path}")
    
    # 推論データの読み込み
    with open(inference_data_path, "r", encoding="utf-8") as f:
        inference_data = json.load(f)
    
    with open(gold_labels_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    
    # データの抽出
    coheren_inputs = inference_data["coheren_inputs"]
    coheren_masks = inference_data["coheren_masks"]
    coheren_types = inference_data["coheren_types"]
    topic_inputs = inference_data["topic_inputs"]
    topic_masks = inference_data["topic_masks"]
    comment_vectors = inference_data["comment_vectors"]
    sentences = inference_data["sentences"]
    
    boundary_labels = gold_data["boundary_labels"]
    
    # 動画境界情報を取得（複数動画対応）
    video_count = gold_data.get("video_count", 1)
    
    print(f"データ統計:")
    print(f"  文章数: {len(sentences)}")
    print(f"  境界数: {sum(boundary_labels)}")
    print(f"  動画数: {video_count}")
    print(f"  バッチ数: {len(coheren_inputs)}")
    print(f"  コメント使用フラグ: {model_config.get('use_comments_for_topic', True)}")
    print(f"  融合方法: {model_config.get('fusion_method', 'average')}")
    
    # テンソルに変換
    coheren_inputs_tensor = [torch.tensor(x) for x in coheren_inputs]
    coheren_masks_tensor = [torch.tensor(x) for x in coheren_masks]
    coheren_types_tensor = [torch.tensor(x) for x in coheren_types]
    comment_vectors_tensor = [torch.tensor(x) for x in comment_vectors]
    
    # =======================================================
    # モデルロード（ファインチューニング後orファインチューニングなし）
    # =======================================================
    print("モデルロード中...")
    try:
        # 学習済みモデルチェックポイントがある場合の処理
        if "model_checkpoint" in model_config and os.path.exists(model_config["model_checkpoint"]):
            print(f"学習済みモデルをロード: {model_config['model_checkpoint']}")
            model = SegModel(
                use_pretrained_only=False,  # 学習済みモデルを使用
                coherence_model_name=model_config["coherence_model"],
                topic_model_name=model_config["topic_model"],
                use_comments_for_topic=model_config.get("use_comments_for_topic", True),  # コメント使用フラグ
                fusion_method=model_config.get("fusion_method", "average")  # 平均融合のみ
            )
            # 学習済み重みをロード
            model.load_state_dict(torch.load(model_config["model_checkpoint"], map_location=DEVICE))
            print("✅ 学習済みモデルをロードしました")
        else:
            # 通常の事前学習モデル
            model = SegModel(
                use_pretrained_only=False,
                coherence_model_name=model_config["coherence_model"],
                topic_model_name=model_config["topic_model"],
                use_comments_for_topic=model_config.get("use_comments_for_topic", True),  # コメント使用フラグ
                fusion_method=model_config.get("fusion_method", "average")  # 平均融合のみ
            )
            print(f"✅ 事前学習モデルをロード: {model_config['coherence_model']}, {model_config['topic_model']}")
        
        model.to(DEVICE)
        model.eval()
        
    except Exception as e:
        print(f"❌ モデルロードエラー: {e}")
        return None
    
    # =======================================================
    # 全データで推論実行
    # =======================================================
    print("全データ推論実行中...")

    # まず全データの統計を収集
    print("グローバル統計収集中...")
    all_coherence_raw = []
    all_topic_raw = []

    with torch.no_grad():
        for i in tqdm(range(len(coheren_inputs_tensor)), desc="統計収集"):
            try:
                # coherenデータの準備
                coheren_input = coheren_inputs_tensor[i].unsqueeze(0).to(DEVICE)
                coheren_mask = coheren_masks_tensor[i].unsqueeze(0).to(DEVICE)
                coheren_type = coheren_types_tensor[i].unsqueeze(0).to(DEVICE)
                
                # topicデータの準備
                topic_input_0 = torch.tensor(topic_inputs[i][0]).unsqueeze(0).to(DEVICE)
                topic_input_1 = torch.tensor(topic_inputs[i][1]).unsqueeze(0).to(DEVICE)
                topic_mask_0 = torch.tensor(topic_masks[i][0]).unsqueeze(0).to(DEVICE)
                topic_mask_1 = torch.tensor(topic_masks[i][1]).unsqueeze(0).to(DEVICE)
                
                topic_i = [topic_input_0, topic_input_1]
                topic_m = [topic_mask_0, topic_mask_1]
                
                # コメントベクトルの準備
                if i < len(comment_vectors_tensor) - 1:
                    topic_comments = [
                        comment_vectors_tensor[i].unsqueeze(0).to(DEVICE),
                        comment_vectors_tensor[i+1].unsqueeze(0).to(DEVICE)
                    ]
                else:
                    topic_comments = [
                        comment_vectors_tensor[i].unsqueeze(0).to(DEVICE),
                        comment_vectors_tensor[i].unsqueeze(0).to(DEVICE)
                    ]
                
                # Topic_numの準備
                topic_num = [[1], [1]]
                
                # 統計収集用の推論（Z-score正規化なし）
                s = model.infer(
                    coheren_input, coheren_mask, coheren_type, 
                    topic_i, topic_m, topic_comments, topic_num,
                    use_comments_for_topic=model_config.get("use_comments_for_topic", True),
                    fusion_method=model_config.get("fusion_method", "average")
                )
                
                # 生スコアを収集
                if hasattr(model, 'last_inference_debug_info'):
                    debug_info = model.last_inference_debug_info
                    if 'coherence_raw' in debug_info:
                        all_coherence_raw.extend(debug_info['coherence_raw'])
                    if 'topic_raw' in debug_info:
                        all_topic_raw.extend(debug_info['topic_raw'])
                        
            except Exception as e:
                print(f"\n統計収集: バッチ {i} でエラー: {e}")
                continue

    # グローバル統計を計算
    global_coherence_mean = np.mean(all_coherence_raw) if all_coherence_raw else 0.0
    global_coherence_std = np.std(all_coherence_raw) if all_coherence_raw else 1.0
    global_topic_mean = np.mean(all_topic_raw) if all_topic_raw else 0.0
    global_topic_std = np.std(all_topic_raw) if all_topic_raw else 1.0

    print(f"グローバル統計:")
    print(f"  コヒーレンス: mean={global_coherence_mean:.6f}, std={global_coherence_std:.6f}")
    print(f"  トピック: mean={global_topic_mean:.6f}, std={global_topic_std:.6f}")

    # グローバル統計を使用して本推論を実行
    print("本推論実行中...")
    scores = []
    nan_count = 0

    # デバッグ用：スコア範囲の統計情報を収集
    coherence_raw_scores = []
    topic_raw_scores = []
    final_raw_scores = []
    coherence_normalized_scores = []
    topic_normalized_scores = []
    final_sigmoid_scores = []  # sigmoid後のスコアを追加

    with torch.no_grad():
        for i in tqdm(range(len(coheren_inputs_tensor)), desc=f"{model_name} 推論"):
            try:
                # coherenデータの準備
                coheren_input = coheren_inputs_tensor[i].unsqueeze(0).to(DEVICE)
                coheren_mask = coheren_masks_tensor[i].unsqueeze(0).to(DEVICE)
                coheren_type = coheren_types_tensor[i].unsqueeze(0).to(DEVICE)
                
                # topicデータの準備
                topic_input_0 = torch.tensor(topic_inputs[i][0]).unsqueeze(0).to(DEVICE)
                topic_input_1 = torch.tensor(topic_inputs[i][1]).unsqueeze(0).to(DEVICE)
                topic_mask_0 = torch.tensor(topic_masks[i][0]).unsqueeze(0).to(DEVICE)
                topic_mask_1 = torch.tensor(topic_masks[i][1]).unsqueeze(0).to(DEVICE)
                
                topic_i = [topic_input_0, topic_input_1]
                topic_m = [topic_mask_0, topic_mask_1]
                
                # コメントベクトルの準備
                if i < len(comment_vectors_tensor) - 1:
                    topic_comments = [
                        comment_vectors_tensor[i].unsqueeze(0).to(DEVICE),
                        comment_vectors_tensor[i+1].unsqueeze(0).to(DEVICE)
                    ]
                else:
                    topic_comments = [
                        comment_vectors_tensor[i].unsqueeze(0).to(DEVICE),
                        comment_vectors_tensor[i].unsqueeze(0).to(DEVICE)
                    ]
                
                # Topic_numの準備
                topic_num = [[1], [1]]
                
                # モデル推論（グローバル統計を渡す）
                s = model.infer(
                    coheren_input, coheren_mask, coheren_type, 
                    topic_i, topic_m, topic_comments, topic_num,
                    use_comments_for_topic=model_config.get("use_comments_for_topic", True),
                    fusion_method=model_config.get("fusion_method", "average"),  # 平均融合のみ
                    global_coherence_mean=global_coherence_mean,
                    global_coherence_std=global_coherence_std,
                    global_topic_mean=global_topic_mean,
                    global_topic_std=global_topic_std
                )
                
                # スコアの処理
                processed_scores = []
                for score in s:
                    if torch.is_tensor(score):
                        score_val = score.item()
                    else:
                        score_val = score
                    
                    if np.isnan(score_val) or np.isinf(score_val):
                        score_val = 0.5
                        nan_count += 1
                    
                    processed_scores.append(score_val)
                
                scores.extend(processed_scores)
                
                # デバッグ情報：生スコア範囲の収集
                if hasattr(model, 'last_inference_debug_info'):
                    debug_info = model.last_inference_debug_info
                    if 'coherence_raw' in debug_info:
                        coherence_raw_scores.extend(debug_info['coherence_raw'])
                    if 'topic_raw' in debug_info:
                        topic_raw_scores.extend(debug_info['topic_raw'])
                    if 'final_raw' in debug_info:
                        final_raw_scores.extend(debug_info['final_raw'])
                    if 'coherence_normalized' in debug_info:
                        coherence_normalized_scores.extend(debug_info['coherence_normalized'])
                    if 'topic_normalized' in debug_info:
                        topic_normalized_scores.extend(debug_info['topic_normalized'])
                
            except Exception as e:
                print(f"\nバッチ {i} でエラー: {e}")
                # エラー時は前のバッチのスコアを使用またはデフォルト値
                if scores:
                    default_scores = [scores[-1]] * 1
                else:
                    default_scores = [0.5] * 1
                scores.extend(default_scores)
                continue

    # sigmoid後のスコアを計算（最終出力スコア）
    final_sigmoid_scores = scores  # model.infer()ですでにsigmoidを通した値が返されている

    # =======================================================
    # デバッグ情報：スコア範囲の出力
    # =======================================================
    print(f"\n🔍 スコア範囲デバッグ情報:")
    if coherence_raw_scores:
        coherence_arr = np.array(coherence_raw_scores)
        print(f"  コヒーレンス生スコア範囲: min={coherence_arr.min():.6f}, max={coherence_arr.max():.6f}, mean={coherence_arr.mean():.6f}")
    else:
        print(f"  コヒーレンス生スコア: データなし")

    if topic_raw_scores:
        topic_arr = np.array(topic_raw_scores)
        print(f"  トピック生スコア範囲: min={topic_arr.min():.6f}, max={topic_arr.max():.6f}, mean={topic_arr.mean():.6f}")
    else:
        print(f"  トピック生スコア: データなし")

    # 正規化後のスコア範囲を追加
    if coherence_normalized_scores:
        coherence_norm_arr = np.array(coherence_normalized_scores)
        print(f"  コヒーレンス正規化後スコア範囲: min={coherence_norm_arr.min():.6f}, max={coherence_norm_arr.max():.6f}, mean={coherence_norm_arr.mean():.6f}")
    else:
        print(f"  コヒーレンス正規化後スコア: データなし")

    if topic_normalized_scores:
        topic_norm_arr = np.array(topic_normalized_scores)
        print(f"  トピック正規化後スコア範囲: min={topic_norm_arr.min():.6f}, max={topic_norm_arr.max():.6f}, mean={topic_norm_arr.mean():.6f}")
    else:
        print(f"  トピック正規化後スコア: データなし")

    if final_raw_scores:
        final_arr = np.array(final_raw_scores)
        print(f"  最終生スコア範囲: min={final_arr.min():.6f}, max={final_arr.max():.6f}, mean={final_arr.mean():.6f}")
    else:
        print(f"  最終生スコア: データなし")

    # sigmoid後のスコア範囲を追加
    if final_sigmoid_scores:
        sigmoid_arr = np.array(final_sigmoid_scores)
        print(f"  sigmoid後最終スコア範囲: min={sigmoid_arr.min():.6f}, max={sigmoid_arr.max():.6f}, mean={sigmoid_arr.mean():.6f}")
    else:
        print(f"  sigmoid後最終スコア: データなし")
    
    # =======================================================
    # 結果分析
    # =======================================================
    print(f"\n推論結果:")
    print(f"総スコア数: {len(scores)}")
    print(f"NaN/Infの数: {nan_count}")
    print(f"スコア統計: min={min(scores):.6f}, max={max(scores):.6f}, mean={np.mean(scores):.6f}")
    
    # 深度スコア計算
    depth_scores = depth_score_cal(scores)
    depth_array = np.array(depth_scores)
    print(f"深度スコア統計: min={depth_array.min():.6f}, max={depth_array.max():.6f}, mean={depth_array.mean():.6f}")
    
    # =======================================================
    # 境界検出と評価
    # =======================================================
    print(f"境界検出中...")
    
    methods = ['adaptive', 'fixed', 'threshold']
    best_pk = float('inf')
    best_method = None
    best_boundaries = None
    best_seg_pred = None
    best_window_size = None
    
    # ランダム境界検出の評価用変数を追加
    random_pk_scores = []
    random_wd_scores = []
    
    for method in methods:
        if method == 'adaptive':
            boundaries = detect_boundaries(depth_scores, method='adaptive')
        elif method == 'fixed':
            boundaries = detect_boundaries(depth_scores, method='fixed', 
                                         num_boundaries=sum(boundary_labels))
        elif method == 'threshold':
            threshold = depth_array.mean() + 0.5 * depth_array.std()
            boundaries = detect_boundaries(depth_scores, method='threshold', threshold=threshold)
        
        seg_pred = [0] * len(sentences)
        for b in boundaries:
            if b < len(seg_pred):
                seg_pred[b] = 1
        
        # 評価指標計算
        seg_r = []
        tmp = 0
        for g in boundary_labels:
            tmp += 1
            if g == 1:
                seg_r.append(tmp)
                tmp = 0
        if tmp > 0:
            seg_r.append(tmp)
        
        seg_p = []
        tmp = 0
        for p in seg_pred:
            tmp += 1
            if p == 1:
                seg_p.append(tmp)
                tmp = 0
        if tmp > 0:
            seg_p.append(tmp)
        
        # ウィンドウサイズ情報の計算
        avg_segment_length = np.mean(seg_r)
        window_size = int(avg_segment_length / 2)
        
        pk = segeval.pk(seg_p, seg_r)
        wd = segeval.window_diff(seg_p, seg_r)
        
        print(f"  {method}: 境界数={sum(seg_pred)}, Pk={pk:.4f}, WD={wd:.4f}, ウィンドウサイズ={window_size}")
        
        if pk < best_pk:
            best_pk = pk
            best_method = method
            best_boundaries = boundaries
            best_seg_pred = seg_pred
            best_window_size = window_size
    
    # ランダム境界検出の評価（100回試行）
    print(f"\nランダム境界検出評価中...")
    num_sentences = len(sentences)
    num_true_boundaries = sum(boundary_labels)
    
    for i in range(100):
        # ランダムに境界を選択
        random_boundaries = np.random.choice(num_sentences, size=num_true_boundaries, replace=False)
        random_boundaries = sorted(random_boundaries)
        
        seg_pred_random = [0] * num_sentences
        for b in random_boundaries:
            seg_pred_random[b] = 1
        
        # 評価指標計算
        seg_r = []
        tmp = 0
        for g in boundary_labels:
            tmp += 1
            if g == 1:
                seg_r.append(tmp)
                tmp = 0
        if tmp > 0:
            seg_r.append(tmp)
        
        seg_p = []
        tmp = 0
        for p in seg_pred_random:
            tmp += 1
            if p == 1:
                seg_p.append(tmp)
                tmp = 0
        if tmp > 0:
            seg_p.append(tmp)
        
        pk_random = segeval.pk(seg_p, seg_r)
        wd_random = segeval.window_diff(seg_p, seg_r)
        
        random_pk_scores.append(pk_random)
        random_wd_scores.append(wd_random)
    
    # ランダム境界検出の統計を計算
    random_pk_mean = np.mean(random_pk_scores)
    random_pk_std = np.std(random_pk_scores)
    random_wd_mean = np.mean(random_wd_scores)
    random_wd_std = np.std(random_wd_scores)
    
    print(f"ランダム境界検出結果 (100回試行):")
    print(f"  Pk: 平均={random_pk_mean:.4f}, 標準偏差={random_pk_std:.4f}")
    print(f"  WD: 平均={random_wd_mean:.4f}, 標準偏差={random_wd_std:.4f}")
    
    # 最良の結果を使用
    seg_pred = best_seg_pred
    boundaries = best_boundaries
    
    # 詳細評価
    correct_detections = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, gold in zip(seg_pred, boundary_labels):
        if pred == 1 and gold == 1:
            correct_detections += 1
        elif pred == 1 and gold == 0:
            false_positives += 1
        elif pred == 0 and gold == 1:
            false_negatives += 1
    
    precision = correct_detections / sum(seg_pred) if sum(seg_pred) > 0 else 0.0
    recall = correct_detections / sum(boundary_labels) if sum(boundary_labels) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"詳細評価:")
    print(f"  適合率: {precision:.4f}")
    print(f"  再現率: {recall:.4f}") 
    print(f"  F1スコア: {f1:.4f}")
    print(f"  Pkスコア: {best_pk:.4f}")
    print(f"  使用ウィンドウサイズ: {best_window_size}")
    
    # 結果を返す
    result = {
        "model_name": model_name,
        "model_config": model_config,
        "Pk": float(best_pk),
        "WD": float(segeval.window_diff(seg_p, seg_r)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "predicted_boundaries": seg_pred,
        "gold_boundaries": boundary_labels,
        "depth_scores": [float(score) for score in depth_scores],
        "raw_scores": [float(score) for score in scores],
        "sentences": sentences,
        "best_method": best_method,
        "detected_boundary_count": sum(seg_pred),
        "true_boundary_count": sum(boundary_labels),
        "correct_detections": correct_detections,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "window_size": int(best_window_size),
        "random_boundary_evaluation": {  # ランダム境界検出の結果を追加
            "pk_mean": float(random_pk_mean),
            "pk_std": float(random_pk_std),
            "wd_mean": float(random_wd_mean),
            "wd_std": float(random_wd_std),
            "num_trials": 100
        },
        "score_statistics": {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        },
        "depth_score_statistics": {
            "min": float(depth_array.min()),
            "max": float(depth_array.max()),
            "mean": float(depth_array.mean()),
            "std": float(depth_array.std())
        },
        # デバッグ情報も保存（可視化用に生スコアデータを追加）
        "debug_scores": {
            "coherence_raw_scores": coherence_raw_scores[:len(scores)],  # 可視化用に追加
            "topic_raw_scores": topic_raw_scores[:len(scores)],  # 可視化用に追加
            "coherence_normalized_scores": coherence_normalized_scores[:len(scores)],  # 正規化後スコアを追加
            "topic_normalized_scores": topic_normalized_scores[:len(scores)],  # 正規化後スコアを追加
            "coherence_raw_range": {
                "min": float(coherence_arr.min()) if coherence_raw_scores else 0.0,
                "max": float(coherence_arr.max()) if coherence_raw_scores else 0.0,
                "mean": float(coherence_arr.mean()) if coherence_raw_scores else 0.0
            },
            "topic_raw_range": {
                "min": float(topic_arr.min()) if topic_raw_scores else 0.0,
                "max": float(topic_arr.max()) if topic_raw_scores else 0.0,
                "mean": float(topic_arr.mean()) if topic_raw_scores else 0.0
            },
            "final_raw_range": {
                "min": float(final_arr.min()) if final_raw_scores else 0.0,
                "max": float(final_arr.max()) if final_raw_scores else 0.0,
                "mean": float(final_arr.mean()) if final_raw_scores else 0.0
            },
            "sigmoid_final_range": {
                "min": float(sigmoid_arr.min()) if final_sigmoid_scores else 0.0,
                "max": float(sigmoid_arr.max()) if final_sigmoid_scores else 0.0,
                "mean": float(sigmoid_arr.mean()) if final_sigmoid_scores else 0.0
            }
        }
    }
    
    # CSVに結果を保存
    save_results_to_csv(result, SAVE_PATH)
    
    # スコアを可視化
    visualize_scores(result, SAVE_PATH)
    
    return result

def main():
    print("=== 学習済みモデル推論開始 ===")
    
    all_results = {}
    model_name = "trained-model"
    model_config = MODEL_CONFIGS[model_name]
    
    result = run_inference_for_model(model_name, model_config)
    if result is not None:
        all_results[model_name] = result
        
        # 結果表示
        print(f"\n{'='*60}")
        print("推論結果サマリー")
        print(f"{'='*60}")
        print(f"モデル名: {model_name}")
        print(f"Pkスコア: {result['Pk']:.4f}")
        print(f"F1スコア: {result['f1_score']:.4f}")
        print(f"適合率: {result['precision']:.4f}")
        print(f"再現率: {result['recall']:.4f}")
        print(f"検出境界数: {result['detected_boundary_count']}")
        print(f"正解境界数: {result['true_boundary_count']}")
        print(f"最適な境界検出方法: {result['best_method']}")
        print(f"使用ウィンドウサイズ: {result['window_size']}")
        print(f"コメント使用フラグ: {model_config.get('use_comments_for_topic', True)}")
        print(f"融合方法: {model_config.get('fusion_method', 'average')}")
    
    # 結果保存
    final_results = {
        "model_results": all_results,
        "timestamp": str(np.datetime64('now'))
    }
    
    # 結果を保存
    result_save_path = f"{SAVE_PATH}/trained_model_results.json"
    with open(result_save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    
    print(f"\n結果を保存: {result_save_path}")
    print("=== 推論完了 ===")

if __name__ == "__main__":
    set_seed(3407)
    main()