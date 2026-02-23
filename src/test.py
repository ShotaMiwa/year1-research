"""
評価スクリプト
学習済みモデルの評価を実行
"""
import os
import json
import torch
import argparse
from transformers import AutoTokenizer, set_seed
from typing import Dict

from config import Config, ModelConfig, InferenceConfig, EvaluationConfig
from models.architecture import SegmentationModel
from models.inference import InferenceWrapper
from data.dataset import InferenceDataset
from evaluation.metrics import MetricsCalculator, RandomBaselineEvaluator
from evaluation.detector import BoundaryDetector, MultimethodBoundaryDetector
from evaluation.visualizer import ResultVisualizer, save_results_to_csv


def load_model(
    checkpoint_path: str,
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    device: torch.device
) -> InferenceWrapper:
    """
    学習済みモデルをロード
    
    Args:
        checkpoint_path: チェックポイントのパス
        model_config: モデル設定
        inference_config: 推論設定
        device: デバイス
        
    Returns:
        InferenceWrapper
    """
    # モデルを作成
    model = SegmentationModel(
        coherence_model_name=model_config.coherence_model_name,
        topic_model_name=model_config.topic_model_name,
        use_comments_for_topic=inference_config.use_comments_for_topic,
        fusion_method=inference_config.fusion_method
    ).to(device)
    
    # チェックポイントをロード
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # state_dictのキーを修正（TrainingWrapperでラップされている場合）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  # 'model.'を削除
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # 推論ラッパーでラップ
    inference_wrapper = InferenceWrapper(
        model=model,
        use_comments=inference_config.use_comments_for_topic,
        fusion_method=inference_config.fusion_method
    )
    
    return inference_wrapper


def evaluate_model(
    inference_wrapper: InferenceWrapper,
    dataset: InferenceDataset,
    tokenizer,
    eval_config: EvaluationConfig,
    device: torch.device
) -> Dict:
    """
    モデルを評価
    
    Args:
        inference_wrapper: 推論ラッパー
        dataset: データセット
        tokenizer: トークナイザー
        eval_config: 評価設定
        device: デバイス
        
    Returns:
        評価結果の辞書
    """
    # データを取得
    sentences = dataset.get_sentences()
    gold_boundaries = dataset.get_gold_boundaries()
    comments = dataset.get_comments() if inference_wrapper.use_comments else None
    
    print(f"発話数: {len(sentences)}")
    print(f"正解境界数: {sum(gold_boundaries)}")
    
    # スコアを予測
    print("スコアを計算中...")
    raw_scores, debug_info = inference_wrapper.predict_scores(
        sentences=sentences,
        tokenizer=tokenizer,
        comments=comments,
        device=device
    )
    
    # 深度スコアを計算
    print("深度スコアを計算中...")
    depth_scores, depth_stats = inference_wrapper.compute_depth_scores(raw_scores)
    
    # 複数の方法で境界検出を試行
    print("境界を検出中...")
    detector = MultimethodBoundaryDetector(eval_config.boundary_detection_methods)
    
    def pk_metric(pred, gold):
        return MetricsCalculator.calculate_pk(pred, gold)
    
    best_boundaries, best_method, best_pk, all_results = detector.detect_best(
        depth_scores=depth_scores,
        gold_boundaries=gold_boundaries,
        metric_fn=pk_metric
    )
    
    print(f"\n最適な検出方法: {best_method}")
    print(f"検出境界数: {len(best_boundaries)}")
    
    # 境界ラベルを作成
    predicted_labels = BoundaryDetector.boundaries_to_labels(
        best_boundaries, len(gold_boundaries)
    )
    
    # 評価指標を計算
    print("評価指標を計算中...")
    metrics = MetricsCalculator.calculate_all_metrics(
        predicted_boundaries=predicted_labels,
        gold_boundaries=gold_boundaries
    )
    
    # ランダムベースラインを評価
    print("ランダムベースラインを評価中...")
    random_metrics = RandomBaselineEvaluator.evaluate(
        gold_boundaries=gold_boundaries,
        num_trials=eval_config.num_random_trials
    )
    
    # 結果をまとめる
    result = {
        'sentences': sentences,
        'raw_scores': raw_scores,
        'depth_scores': depth_scores,
        'predicted_boundaries': predicted_labels,
        'gold_boundaries': gold_boundaries,
        'best_method': best_method,
        'metrics': metrics,
        'random_baseline': random_metrics,
        'debug_scores': debug_info,
        'all_detection_results': all_results
    }
    
    return result


def print_results(result: Dict):
    """
    結果を表示
    
    Args:
        result: 評価結果
    """
    print("\n" + "="*60)
    print("評価結果サマリー")
    print("="*60)
    
    metrics = result['metrics']
    print(f"Pkスコア: {metrics['Pk']:.4f}")
    print(f"WindowDiff: {metrics['WindowDiff']:.4f}")
    print(f"適合率: {metrics['precision']:.4f}")
    print(f"再現率: {metrics['recall']:.4f}")
    print(f"F1スコア: {metrics['f1']:.4f}")
    print(f"検出境界数: {sum(result['predicted_boundaries'])}")
    print(f"正解境界数: {sum(result['gold_boundaries'])}")
    print(f"最適な検出方法: {result['best_method']}")
    print(f"ウィンドウサイズ: {metrics['window_size']}")
    
    print("\nランダムベースライン:")
    random = result['random_baseline']
    print(f"  Pk: {random['pk_mean']:.4f} ± {random['pk_std']:.4f}")
    print(f"  WD: {random['wd_mean']:.4f} ± {random['wd_std']:.4f}")
    
    # 各検出方法の結果を表示
    print("\n全ての検出方法の結果:")
    for method, res in result['all_detection_results'].items():
        print(f"  {method}: Pk={res['score']:.4f}, 境界数={res['num_boundaries']}")


def main(args):
    """
    メイン評価関数
    
    Args:
        args: コマンドライン引数
    """
    # シードを設定
    set_seed(args.seed)
    
    # デバイスを設定
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 設定を作成
    model_config = ModelConfig()
    inference_config = InferenceConfig(
        use_comments_for_topic=args.use_comments,
        fusion_method=args.fusion_method,
        device=str(device)
    )
    eval_config = EvaluationConfig(
        inference_data_path=args.data_path,
        model_checkpoint=args.checkpoint,
        save_path=args.save_path
    )
    
    # データセットをロード
    print(f"Loading data: {args.data_path}")
    dataset = InferenceDataset(args.data_path)
    
    # トークナイザーをロード
    print(f"Loading tokenizer: {model_config.topic_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.topic_model_name)
    
    # モデルをロード
    inference_wrapper = load_model(
        checkpoint_path=args.checkpoint,
        model_config=model_config,
        inference_config=inference_config,
        device=device
    )
    
    # 評価を実行
    print("\n" + "="*60)
    print("評価を開始")
    print("="*60)
    
    result = evaluate_model(
        inference_wrapper=inference_wrapper,
        dataset=dataset,
        tokenizer=tokenizer,
        eval_config=eval_config,
        device=device
    )
    
    # 結果を表示
    print_results(result)
    
    # 結果を保存
    os.makedirs(args.save_path, exist_ok=True)
    
    # JSONに保存
    result_path = os.path.join(args.save_path, "evaluation_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        # debug_scoresは大きいので除外
        save_result = {k: v for k, v in result.items() if k != 'debug_scores'}
        json.dump(save_result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 結果を保存: {result_path}")
    
    # CSVに保存
    save_results_to_csv(result, args.save_path)
    
    # 可視化
    if not args.no_visualization:
        print("\n結果を可視化中...")
        visualizer = ResultVisualizer(args.save_path)
        
        # ヒストグラム作成
        visualizer.create_score_histograms(result)
        
        # 境界検出結果を可視化
        visualizer.visualize_boundary_detection(
            sentences=result['sentences'],
            depth_scores=result['depth_scores'],
            predicted_boundaries=result['predicted_boundaries'],
            gold_boundaries=result['gold_boundaries']
        )
        
        # スコア比較プロット
        if 'coherence_raw_scores' in result['debug_scores'] and 'topic_raw_scores' in result['debug_scores']:
            visualizer.plot_score_comparison(
                coherence_scores=result['debug_scores']['coherence_raw_scores'],
                topic_scores=result['debug_scores']['topic_raw_scores'],
                combined_scores=result['raw_scores']
            )
    
    print("\n✅ 評価完了!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    
    # 必須引数
    parser.add_argument("--data_path", required=True, help="Path to inference data (JSON)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    
    # オプション引数
    parser.add_argument("--save_path", default="./results", help="Path to save results")
    parser.add_argument("--use_comments", action='store_true', help="Use comments for topic modeling")
    parser.add_argument("--fusion_method", default="average", choices=['average', 'linear'], help="Fusion method")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--no_visualization", action='store_true', help="Disable visualization")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Evaluation Arguments:")
    print(args)
    print("="*60)
    
    main(args)