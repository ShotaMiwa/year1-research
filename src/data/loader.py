"""
データローダーモジュール
データローダーの生成と管理
"""
import glob
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Optional

from data.dataset import MultiFileDataset, SegmentationDataset, InferenceDataset
from data.collator import SegmentationDataCollator


class DataLoaderFactory:
    """
    データローダーを生成するファクトリークラス
    """
    
    @staticmethod
    def create_train_dataloader(
        data_path: str,
        batch_size: int = 12,
        local_rank: int = -1,
        num_workers: int = 0
    ) -> DataLoader:
        """
        学習用データローダーを作成
        
        Args:
            data_path: データパスまたはパターン
            batch_size: バッチサイズ
            local_rank: 分散学習用のローカルランク
            num_workers: ワーカー数
            
        Returns:
            DataLoader
        """
        # データファイルのパスを取得
        data_paths = DataLoaderFactory._get_data_paths(data_path)
        
        # データセットを作成
        dataset = MultiFileDataset(data_paths)
        
        # サンプラーを作成
        if local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        
        # コレータを作成
        collator = SegmentationDataCollator()
        
        # データローダーを作成
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    @staticmethod
    def create_eval_dataloader(
        data_path: str,
        batch_size: int = 1,
        num_workers: int = 0
    ) -> DataLoader:
        """
        評価用データローダーを作成
        
        Args:
            data_path: データパス
            batch_size: バッチサイズ
            num_workers: ワーカー数
            
        Returns:
            DataLoader
        """
        # データセットを作成
        dataset = InferenceDataset(data_path)
        
        # サンプラーを作成（順次サンプリング）
        sampler = SequentialSampler(dataset)
        
        # データローダーを作成
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        return dataloader
    
    @staticmethod
    def _get_data_paths(data_path: str) -> List[str]:
        """
        データファイルのパスリストを取得
        
        Args:
            data_path: データパスまたはパターン
            
        Returns:
            ファイルパスのリスト
        """
        # ディレクトリの場合
        if os.path.isdir(data_path):
            data_paths = glob.glob(os.path.join(data_path, "*.pt"))
            if not data_paths:
                raise ValueError(f"No .pt files found in directory: {data_path}")
        
        # パターンの場合
        else:
            data_paths = glob.glob(data_path)
            if not data_paths:
                raise ValueError(f"No files match pattern: {data_path}")
        
        print(f"Found {len(data_paths)} data files")
        
        return data_paths


def get_train_dataloader(
    data_path: str,
    batch_size: int = 12,
    local_rank: int = -1
) -> DataLoader:
    """
    学習用データローダーを取得（簡易版）
    
    Args:
        data_path: データパス
        batch_size: バッチサイズ
        local_rank: ローカルランク
        
    Returns:
        DataLoader
    """
    return DataLoaderFactory.create_train_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        local_rank=local_rank
    )


def get_eval_dataloader(data_path: str) -> DataLoader:
    """
    評価用データローダーを取得（簡易版）
    
    Args:
        data_path: データパス
        
    Returns:
        DataLoader
    """
    return DataLoaderFactory.create_eval_dataloader(data_path=data_path)