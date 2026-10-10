"""
データコレータモジュール
バッチ生成処理
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Any


def get_attention_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    テンソルからアテンションマスクを生成
    
    Args:
        tensor: 入力テンソル
        
    Returns:
        アテンションマスク
    """
    attention_masks = []
    for sent in tensor:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(attention_masks)


class SegmentationDataCollator:
    """
    セグメンテーションタスク用のデータコレータ
    バッチ生成とパディングを担当
    """
    
    def __call__(self, examples: List[Tuple]) -> dict:
        """
        サンプルのリストからバッチを作成
        
        Args:
            examples: サンプルのリスト
            
        Returns:
            バッチデータの辞書
        """
        batch_size = len(examples)
        
        # --- Coherenceデータ ---
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)
        
        # --- Topicデータ（コメントなし）---
        topic_context = pad_sequence(
            [torch.tensor(ex[3]) for ex in examples],
            batch_first=True
        )
        topic_pos = pad_sequence(
            [torch.tensor(ex[4]) for ex in examples],
            batch_first=True
        )
        topic_neg = pad_sequence(
            [torch.tensor(ex[5]) for ex in examples],
            batch_first=True
        )
        
        # 数値情報
        topic_context_num = [ex[6] for ex in examples]
        topic_pos_num = [ex[7] for ex in examples]
        topic_neg_num = [ex[8] for ex in examples]
        
        # アテンションマスク生成
        topic_context_mask = get_attention_mask(topic_context)
        topic_pos_mask = get_attention_mask(topic_pos)
        topic_neg_mask = get_attention_mask(topic_neg)
        
        # 学習用トピックデータ
        topic_train = pad_sequence(
            [torch.tensor(ids) for ex in examples for ids in ex[9]],
            batch_first=True
        )
        topic_train_mask = pad_sequence(
            [torch.ones(len(ids), dtype=torch.long) for ex in examples for ids in ex[9]],
            batch_first=True
        )
        
        # topic_num に発話数も含める → training.py でオフセット計算に使う
        topic_num = [(ex[10][0], ex[10][1], len(ex[9])) for ex in examples]
        
        return {
            'coheren_inputs': coheren_inputs,
            'coheren_mask': coheren_mask,
            'coheren_type': coheren_type,
            'topic_context': topic_context,
            'topic_pos': topic_pos,
            'topic_neg': topic_neg,
            'topic_context_mask': topic_context_mask,
            'topic_pos_mask': topic_pos_mask,
            'topic_neg_mask': topic_neg_mask,
            'topic_context_num': topic_context_num,
            'topic_pos_num': topic_pos_num,
            'topic_neg_num': topic_neg_num,
            'topic_train': topic_train,
            'topic_train_mask': topic_train_mask,
            'topic_num': topic_num
        }


class SimpleDataCollator:
    """
    シンプルなデータコレータ
    基本的なパディングのみ
    """
    
    def __call__(self, examples: List[Any]) -> dict:
        """
        サンプルのリストからバッチを作成
        
        Args:
            examples: サンプルのリスト
            
        Returns:
            バッチデータの辞書
        """
        if isinstance(examples[0], dict):
            # 辞書形式の場合
            batch = {}
            for key in examples[0].keys():
                values = [ex[key] for ex in examples]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = pad_sequence(values, batch_first=True)
                else:
                    batch[key] = values
            return batch
        else:
            # タプル形式の場合
            return SegmentationDataCollator()(examples)