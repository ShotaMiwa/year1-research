"""
データセットモジュール
データセットクラスの定義
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import random
from tqdm import tqdm


class SegmentationDataset(Dataset):
    """
    セグメンテーション用データセット
    """
    
    def __init__(self, samples: List[Tuple]):
        """
        Args:
            samples: サンプルのリスト
        """
        self.samples = samples
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        指定インデックスのサンプルを取得
        
        Args:
            idx: インデックス
            
        Returns:
            サンプル
        """
        return self.samples[idx]
    
    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.samples)


class MultiFileDataset(Dataset):
    """
    複数ファイルからデータを読み込むデータセット
    """
    
    def __init__(self, data_paths: List[str]):
        """
        Args:
            data_paths: データファイルのパスのリスト
        """
        self.data_paths = data_paths
        self.loaded_data_list = []
        self.all_samples = []
        
        # データファイルをロード
        self._load_data_files()
        
        # サンプルを準備
        self._prepare_samples()
    
    def _load_data_files(self):
        """データファイルをロード"""
        print(f"Loading {len(self.data_paths)} data files...")
        
        for path in tqdm(self.data_paths, desc="Loading data files"):
            try:
                loaded_data = torch.load(path, map_location="cpu")
                self.loaded_data_list.append(loaded_data)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.loaded_data_list)} data files")
    
    def _prepare_samples(self):
        """すべてのデータファイルからサンプルを準備"""
        total_samples = 0
        
        for file_idx, loaded_data in enumerate(self.loaded_data_list):
            # データ構造の検証
            if 'sentences' not in loaded_data:
                print(f"Warning: File {file_idx} missing 'sentences' key, skipping")
                continue
            
            total_utterances = len(loaded_data["sentences"])
            coherence_data_len = len(loaded_data["coheren_inputs"]) \
                if "coheren_inputs" in loaded_data else total_utterances
            
            print(f"File {file_idx}: {total_utterances} utterances, "
                  f"{coherence_data_len} coherence samples")
            
            # サンプル生成
            for i in range(coherence_data_len):
                sample = self._create_sample(loaded_data, i, total_utterances)
                self.all_samples.append(sample)
                total_samples += 1
        
        print(f"Total samples prepared: {total_samples}")
    
    def _create_sample(
        self,
        loaded_data: Dict,
        idx: int,
        total_utterances: int
    ) -> Tuple:
        """
        1つのサンプルを作成
        
        Args:
            loaded_data: ロードされたデータ
            idx: インデックス
            total_utterances: 総発話数
            
        Returns:
            サンプル
        """
        current_utt = idx % total_utterances
        
        # Coherenceデータ
        if "coheren_inputs" in loaded_data:
            coheren_input = loaded_data["coheren_inputs"][idx]
            coheren_mask = loaded_data["coheren_masks"][idx]
            coheren_type = loaded_data["coheren_types"][idx]
        else:
            # フォールバック: ダミーデータ
            coheren_input = torch.zeros(2, 512, dtype=torch.long)
            coheren_mask = torch.zeros(2, 512, dtype=torch.long)
            coheren_type = torch.zeros(2, 512, dtype=torch.long)
        
        # 正例・負例の選択
        pos_idx = idx % total_utterances
        neg_idx = random.randint(0, total_utterances - 1)
        while neg_idx == pos_idx:
            neg_idx = random.randint(0, total_utterances - 1)
        
        # SimCSE入力（コメントなし）
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
        
        return sample
    
    def __getitem__(self, idx: int) -> Tuple:
        """サンプルを取得"""
        return self.all_samples[idx]
    
    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.all_samples)


class InferenceDataset(Dataset):
    """
    推論用データセット
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: データファイルのパス
        """
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> Dict:
        """
        データをロード
        
        Args:
            data_path: データパス
            
        Returns:
            ロードされたデータ
        """
        import json
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_sentences(self) -> List[str]:
        """発話のリストを取得"""
        return self.data.get('sentences', [])
    
    def get_gold_boundaries(self) -> List[int]:
        """正解境界のリストを取得"""
        return self.data.get('boundary_labels', [])
    
    def get_comments(self) -> List[Dict]:
        """コメントデータを取得"""
        return self.data.get('comments', [])
    
    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.get_sentences())
    
    def __getitem__(self, idx: int) -> str:
        """指定インデックスの発話を取得"""
        return self.get_sentences()[idx]