import re
import os
import json
import torch
import random
import pickle
import argparse
import torch.nn as nn
from tqdm import tqdm
from model import SegModel
from torch.cuda import amp
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertForNextSentencePrediction, BertConfig, get_linear_schedule_with_warmup, set_seed, AutoModel
from torch.optim import AdamW  
import glob

def get_mask(tensor):
    attention_masks = []
    for sent in tensor:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(attention_masks)

class ourdataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def collect_fn(self, examples):
        batch_size = len(examples)
        
        # --- Coherence ---
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)

        # --- Topicデータ（コメントなし）---
        topic_context = pad_sequence([torch.tensor(ex[3]) for ex in examples], batch_first=True)
        topic_pos = pad_sequence([torch.tensor(ex[4]) for ex in examples], batch_first=True)
        topic_neg = pad_sequence([torch.tensor(ex[5]) for ex in examples], batch_first=True)
        
        topic_context_num = [ex[6] for ex in examples]
        topic_pos_num = [ex[7] for ex in examples]
        topic_neg_num = [ex[8] for ex in examples]

        topic_context_mask, topic_pos_mask, topic_neg_mask = (
            get_mask(topic_context),
            get_mask(topic_pos),
            get_mask(topic_neg)
        )

        topic_train = pad_sequence([torch.tensor(ids) for ex in examples for ids in ex[9]], batch_first=True)
        topic_train_mask = pad_sequence(
            [torch.ones(len(ids), dtype=torch.long) for ex in examples for ids in ex[9]],
            batch_first=True
        )

        topic_num = [ex[10] for ex in examples]

        return (
            coheren_inputs, coheren_mask, coheren_type, 
            topic_context, topic_pos, topic_neg,
            topic_context_mask, topic_pos_mask, topic_neg_mask,
            topic_context_num, topic_pos_num, topic_neg_num,
            topic_train, topic_train_mask, topic_num
        )

class MultiFileDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.loaded_data_list = []
        
        print(f"Loading {len(data_paths)} data files...")
        for path in tqdm(data_paths, desc="Loading data files"):
            try:
                loaded_data = torch.load(path, map_location="cpu")
                self.loaded_data_list.append(loaded_data)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.loaded_data_list)} data files")
        
        # 各ファイルからサンプルを生成
        self.all_samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        """すべてのデータファイルからサンプルを準備（コメントなし）"""
        total_samples = 0
        
        for file_idx, loaded_data in enumerate(self.loaded_data_list):
            # データ構造の検証
            if 'sentences' not in loaded_data:
                print(f"Warning: File {file_idx} missing 'sentences' key, skipping")
                continue
                
            total_utterances = len(loaded_data["sentences"])
            coherence_data_len = len(loaded_data["coheren_inputs"]) if "coheren_inputs" in loaded_data else total_utterances
            
            print(f"File {file_idx}: {total_utterances} utterances, {coherence_data_len} coherence samples")
            
            for i in range(coherence_data_len):
                current_utt = i % total_utterances

                # --- 正例・負例の構成 ---
                if "coheren_inputs" in loaded_data:
                    coheren_input = loaded_data["coheren_inputs"][i]
                    coheren_mask = loaded_data["coheren_masks"][i]
                    coheren_type = loaded_data["coheren_types"][i]
                else:
                    # フォールバック: ダミーデータ生成
                    coheren_input = torch.zeros(2, 512, dtype=torch.long)
                    coheren_mask = torch.zeros(2, 512, dtype=torch.long)
                    coheren_type = torch.zeros(2, 512, dtype=torch.long)

                # NSP部分のpos/neg構造に合わせる
                pos_idx = i % total_utterances
                neg_idx = random.randint(0, total_utterances - 1)
                while neg_idx == pos_idx:
                    neg_idx = random.randint(0, total_utterances - 1)

                # --- SimCSE入力（コメントなし）---
                context_ids = loaded_data["sub_ids_simcse"][pos_idx]
                pos_ids = loaded_data["sub_ids_simcse"][min(pos_idx + 1, total_utterances - 1)]
                neg_ids = loaded_data["sub_ids_simcse"][neg_idx]

                sample = (
                    coheren_input,
                    coheren_mask,
                    coheren_type,
                    context_ids,
                    pos_ids,
                    neg_ids,
                    1,  # topic_context_num
                    1,  # topic_pos_num  
                    1,  # topic_neg_num
                    loaded_data["sub_ids_simcse"],  # topic_train (全発話)
                    (total_utterances, current_utt)  # topic_num
                )

                self.all_samples.append(sample)
                total_samples += 1
        
        print(f"Total samples prepared: {total_samples}")
    
    def __getitem__(self, idx):
        return self.all_samples[idx]

    def __len__(self):
        return len(self.all_samples)
    
    def collect_fn(self, examples):
        # collect_fnの実装
        batch_size = len(examples)
        
        # --- Coherence ---
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)

        # --- Topicデータ（コメントなし）---
        topic_context = pad_sequence([torch.tensor(ex[3]) for ex in examples], batch_first=True)
        topic_pos = pad_sequence([torch.tensor(ex[4]) for ex in examples], batch_first=True)
        topic_neg = pad_sequence([torch.tensor(ex[5]) for ex in examples], batch_first=True)
        
        topic_context_num = [ex[6] for ex in examples]
        topic_pos_num = [ex[7] for ex in examples]
        topic_neg_num = [ex[8] for ex in examples]

        topic_context_mask, topic_pos_mask, topic_neg_mask = (
            get_mask(topic_context),
            get_mask(topic_pos),
            get_mask(topic_neg)
        )

        topic_train = pad_sequence([torch.tensor(ids) for ex in examples for ids in ex[9]], batch_first=True)
        topic_train_mask = pad_sequence(
            [torch.ones(len(ids), dtype=torch.long) for ex in examples for ids in ex[9]],
            batch_first=True
        )

        topic_num = [ex[10] for ex in examples]

        return (
            coheren_inputs, coheren_mask, coheren_type, 
            topic_context, topic_pos, topic_neg,
            topic_context_mask, topic_pos_mask, topic_neg_mask,
            topic_context_num, topic_pos_num, topic_neg_num,
            topic_train, topic_train_mask, topic_num
        )

def main(args):
    # データパスの解決
    data_paths = []
    if os.path.isfile(args.data_path):
        data_paths = [args.data_path]
    elif os.path.isdir(args.data_path):
        data_paths = glob.glob(os.path.join(args.data_path, "*.pt"))
        if not data_paths:
            raise ValueError(f"No .pt files found in directory: {args.data_path}")
    else:
        data_paths = glob.glob(args.data_path)
        if not data_paths:
            raise ValueError(f"No files match pattern: {args.data_path}")
    
    print(f"Found {len(data_paths)} data files")
    
    # マルチファイルデータセットの作成
    train_data = MultiFileDataset(data_paths)
     
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_data.collect_fn)

    scaler = amp.GradScaler(enabled=(not args.no_amp))
    
    # モデル初期化 - 学習時はコメント不使用
    model = SegModel(
        margin=args.margin, 
        train_split=args.train_split, 
        window_size=args.window_size,
        use_pretrained_only=False,
        coherence_model_name="cl-tohoku/bert-base-japanese",
        topic_model_name="pkshatech/simcse-ja-bert-base-clcmlp",
        use_comments_for_topic=False,  # 学習時はFalse
        fusion_method="average"  # 推論用（学習時は使用しない）
    ).to(args.device)
    
    if args.resume and args.ckpt:
        print(f"Resuming from checkpoint: {args.ckpt}")
        model.load_state_dict(torch.load(f'{args.root}/model/{args.ckpt}', map_location=args.device), strict=False)
    
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    epoch_loss = {}
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    for epoch_i in tqdm(range(args.epochs)):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        total_loss = 0
        total_margin_loss = 0
        total_topic_loss = 0
        
        model.train()
        epoch_iterator = tqdm(train_dataloader, disable=args.local_rank not in [-1, 0])
        window_size = args.window_size

        for step, batch in enumerate(epoch_iterator):
            # 入力データ構築（コメントなし）
            input_data = {
                'coheren_inputs': batch[0].to(args.device),
                'coheren_mask': batch[1].to(args.device), 
                'coheren_type': batch[2].to(args.device),
                'topic_context': batch[3].to(args.device),
                'topic_pos': batch[4].to(args.device),
                'topic_neg': batch[5].to(args.device),
                'topic_context_mask': batch[6].to(args.device),
                'topic_pos_mask': batch[7].to(args.device),
                'topic_neg_mask': batch[8].to(args.device),
                'topic_context_num': batch[9],
                'topic_pos_num': batch[10],
                'topic_neg_num': batch[11],
                'topic_train': batch[12].to(args.device),
                'topic_train_mask': batch[13].to(args.device),
                'topic_num': batch[14]
            }

            model.zero_grad()

            with autocast(enabled=(not args.no_amp)):
                loss, margin_loss, topic_loss = model(input_data, window_size)

            if args.n_gpu > 1:
                loss = loss.mean()
                margin_loss = margin_loss.mean() if margin_loss is not None else torch.tensor(0)
                topic_loss = topic_loss.mean() if topic_loss is not None else torch.tensor(0)

            total_loss += loss.item()
            total_margin_loss += margin_loss.item() if margin_loss is not None else 0
            total_topic_loss += topic_loss.item() if topic_loss is not None else 0

            if not args.no_amp:
                scaler.scale(loss).backward()
                if (step + 1) % args.accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if (step + 1) % args.accum == 0:
                    optimizer.step()
                    scheduler.step()

            epoch_iterator.set_description(f"Loss: {loss.item():.4f}")

        # エポック終了処理
        avg_train_loss = total_loss / len(train_dataloader)
        avg_margin_loss = total_margin_loss / len(train_dataloader)
        avg_topic_loss = total_topic_loss / len(train_dataloader)
        
        epoch_loss[epoch_i] = {
            'total_loss': avg_train_loss,
            'margin_loss': avg_margin_loss, 
            'topic_loss': avg_topic_loss
        }
        
        if args.local_rank in [-1, 0]:
            print(f'=========== Epoch {epoch_i} ===========')
            print(f'Total Loss: {avg_train_loss:.4f}')
            print(f'Margin Loss: {avg_margin_loss:.4f}')
            print(f'Topic Loss: {avg_topic_loss:.4f}')
            
            # モデル保存
            model_to_save = model.module if hasattr(model, 'module') else model
            save_path = f'{args.root}/model/{args.save_model_name}/epoch_{epoch_i}_step_{len(train_dataloader)}'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            print(f'Saving model to {save_path}')
            torch.save(model_to_save.state_dict(), save_path)

    return epoch_loss    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # データ関連
    parser.add_argument("--data_path", required=True, help="Path to preprocessed data file, directory, or pattern")
    parser.add_argument("--save_model_name", required=True, help="Name for saving the model")
    
    # モデルパラメータ
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--train_split", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)
    
    # 訓練パラメータ
    parser.add_argument("--ckpt", help="Checkpoint path to resume from")
    parser.add_argument("--root", default='.', help="Root directory for outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--resume", action='store_true', help="Resume training from checkpoint")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup proportion")
    
    # デバイス設定
    parser.add_argument("--no_amp", action='store_true', help="Disable automatic mixed precision")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # デバイス設定
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    
    args.device = device
    
    # 出力ディレクトリ作成
    out_path = f'{args.root}/model/{args.save_model_name}'
    os.makedirs(out_path, exist_ok=True) 
    
    print(f"Using device: {device}")
    print(f"Number of GPUs: {args.n_gpu}")
    print(f"Training arguments: {args}")
    print(f"学習時: コメント完全不使用モード")
    
    epoch_loss = main(args)
    
    # 損失の保存
    with open(f'{out_path}/loss.json', 'w') as f:
        json.dump(epoch_loss, f, indent=2)
    
    print("Training completed!")