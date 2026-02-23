"""
学習スクリプト
モデルの学習を実行
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from torch.cuda import amp
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, set_seed
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from models.architecture import SegmentationModel
from models.training import TrainingWrapper
from data.loader import get_train_dataloader


def setup_device(args):
    """
    デバイスの設定
    
    Args:
        args: コマンドライン引数
        
    Returns:
        (device, n_gpu)
    """
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    
    return device, n_gpu


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    config,
    epoch
):
    """
    1エポックの学習
    
    Args:
        model: モデル
        dataloader: データローダー
        optimizer: オプティマイザー
        scheduler: スケジューラー
        scaler: AMPスケーラー
        device: デバイス
        config: 設定
        epoch: 現在のエポック
        
    Returns:
        損失の辞書
    """
    model.train()
    
    total_loss = 0
    total_margin_loss = 0
    total_topic_loss = 0
    
    epoch_iterator = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=config.training.local_rank not in [-1, 0]
    )
    
    for step, batch in enumerate(epoch_iterator):
        # データをデバイスに転送
        input_data = {
            'coheren_inputs': batch['coheren_inputs'].to(device),
            'coheren_mask': batch['coheren_mask'].to(device),
            'coheren_type': batch['coheren_type'].to(device),
            'topic_context': batch['topic_context'].to(device),
            'topic_pos': batch['topic_pos'].to(device),
            'topic_neg': batch['topic_neg'].to(device),
            'topic_context_mask': batch['topic_context_mask'].to(device),
            'topic_pos_mask': batch['topic_pos_mask'].to(device),
            'topic_neg_mask': batch['topic_neg_mask'].to(device),
            'topic_context_num': batch['topic_context_num'],
            'topic_pos_num': batch['topic_pos_num'],
            'topic_neg_num': batch['topic_neg_num'],
            'topic_train': batch['topic_train'].to(device),
            'topic_train_mask': batch['topic_train_mask'].to(device),
            'topic_num': batch['topic_num']
        }
        
        model.zero_grad()
        
        # フォワードパス
        with amp.autocast(enabled=(not config.training.no_amp)):
            loss, margin_loss, topic_loss = model(
                input_data,
                window_size=config.model.window_size
            )
        
        # 分散学習の場合は平均を取る
        if config.training.local_rank != -1:
            loss = loss.mean()
            margin_loss = margin_loss.mean() if margin_loss is not None else torch.tensor(0)
            topic_loss = topic_loss.mean() if topic_loss is not None else torch.tensor(0)
        
        # 損失を累積
        total_loss += loss.item()
        total_margin_loss += margin_loss.item() if margin_loss is not None else 0
        total_topic_loss += topic_loss.item() if topic_loss is not None else 0
        
        # バックワードパス
        if not config.training.no_amp:
            scaler.scale(loss).backward()
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
        
        epoch_iterator.set_description(f"Loss: {loss.item():.4f}")
    
    # 平均損失を計算
    avg_loss = total_loss / len(dataloader)
    avg_margin_loss = total_margin_loss / len(dataloader)
    avg_topic_loss = total_topic_loss / len(dataloader)
    
    return {
        'total_loss': avg_loss,
        'margin_loss': avg_margin_loss,
        'topic_loss': avg_topic_loss
    }


def main(args):
    """
    メイン学習関数
    
    Args:
        args: コマンドライン引数
        
    Returns:
        エポックごとの損失
    """
    # 設定を作成
    config = Config.from_args(args)
    
    # シードを設定
    set_seed(config.training.seed)
    
    # デバイスを設定
    device, n_gpu = setup_device(args)
    print(f"Using device: {device}")
    print(f"Number of GPUs: {n_gpu}")
    
    # データローダーを作成
    train_dataloader = get_train_dataloader(
        data_path=config.data.data_path,
        batch_size=config.training.batch_size,
        local_rank=config.training.local_rank
    )
    
    # モデルを作成
    base_model = SegmentationModel(
        coherence_model_name=config.model.coherence_model_name,
        topic_model_name=config.model.topic_model_name,
        use_comments_for_topic=False,  # 学習時はコメント不使用
        fusion_method='average'
    ).to(device)
    
    # 学習ラッパーでラップ
    model = TrainingWrapper(
        model=base_model,
        margin=config.model.margin,
        train_split=config.model.train_split,
        window_size=config.model.window_size
    ).to(device)
    
    # チェックポイントから再開
    if config.training.resume and config.training.checkpoint_path:
        print(f"Resuming from checkpoint: {config.training.checkpoint_path}")
        model.load_state_dict(
            torch.load(config.training.checkpoint_path, map_location=device),
            strict=False
        )
    
    # 分散学習の設定
    if config.training.local_rank != -1:
        model = DDP(
            model,
            device_ids=[config.training.local_rank],
            output_device=config.training.local_rank,
            find_unused_parameters=True
        )
    
    # オプティマイザーとスケジューラー
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * config.training.epochs
    num_warmup_steps = int(total_steps * config.training.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    # AMPスケーラー
    scaler = amp.GradScaler(enabled=(not config.training.no_amp))
    
    # 出力ディレクトリ
    out_path = os.path.join(config.data.root_dir, 'model', config.data.save_model_name)
    os.makedirs(out_path, exist_ok=True)
    
    # 学習ループ
    epoch_losses = {}
    
    for epoch in range(config.training.epochs):
        print(f'\n======== Epoch {epoch + 1} / {config.training.epochs} ========')
        
        # 1エポック学習
        losses = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch + 1
        )
        
        epoch_losses[epoch] = losses
        
        # 損失を表示
        if config.training.local_rank in [-1, 0]:
            print(f'Total Loss: {losses["total_loss"]:.4f}')
            print(f'Margin Loss: {losses["margin_loss"]:.4f}')
            print(f'Topic Loss: {losses["topic_loss"]:.4f}')
            
            # モデルを保存
            model_to_save = model.module if hasattr(model, 'module') else model
            save_path = os.path.join(out_path, f'epoch_{epoch}_step_{len(train_dataloader)}')
            
            print(f'Saving model to {save_path}')
            torch.save(model_to_save.state_dict(), save_path)
    
    # 損失を保存
    if config.training.local_rank in [-1, 0]:
        with open(os.path.join(out_path, 'loss.json'), 'w') as f:
            json.dump(epoch_losses, f, indent=2)
    
    return epoch_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model')
    
    # データ関連
    parser.add_argument("--data_path", required=True, help="Path to data files")
    parser.add_argument("--save_model_name", required=True, help="Model save name")
    parser.add_argument("--root", default='.', help="Root directory")
    
    # モデルパラメータ
    parser.add_argument("--margin", type=int, default=1, help="Margin for ranking loss")
    parser.add_argument("--train_split", type=int, default=5, help="Number of splits for training")
    parser.add_argument("--window_size", type=int, default=5, help="Window size")
    
    # 学習パラメータ
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    
    # チェックポイント
    parser.add_argument("--resume", action='store_true', help="Resume from checkpoint")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    
    # デバイス設定
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--no_amp", action='store_true', help="Disable AMP")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Arguments:")
    print(args)
    print("="*60)
    
    main(args)
    
    print("\n✅ Training completed!")
