"""
設定管理モジュール
全ての設定値を一元管理
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """モデルアーキテクチャの設定"""
    coherence_model_name: str = "cl-tohoku/bert-base-japanese"
    topic_model_name: str = "pkshatech/simcse-ja-bert-base-clcmlp"
    margin: int = 1
    train_split: int = 5
    window_size: int = 5
    
    def __post_init__(self):
        """設定値の検証"""
        if self.margin <= 0:
            raise ValueError("margin must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


@dataclass
class TrainingConfig:
    """学習の設定"""
    batch_size: int = 12
    learning_rate: float = 3e-5
    epochs: int = 10
    warmup_proportion: float = 0.1
    seed: int = 3407
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # デバイス設定
    no_cuda: bool = False
    no_amp: bool = False
    local_rank: int = -1
    
    # チェックポイント設定
    resume: bool = False
    checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        """設定値の検証"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.warmup_proportion <= 1:
            raise ValueError("warmup_proportion must be between 0 and 1")


@dataclass
class InferenceConfig:
    """推論の設定"""
    use_comments_for_topic: bool = True
    fusion_method: str = "average"  # 'average' or 'linear'
    device: str = "cuda"
    
    def __post_init__(self):
        """設定値の検証"""
        if self.fusion_method not in ["average", "linear"]:
            raise ValueError("fusion_method must be 'average' or 'linear'")


@dataclass
class DataConfig:
    """データの設定"""
    data_path: str = ""
    save_model_name: str = ""
    root_dir: str = "."
    
    def __post_init__(self):
        """設定値の検証"""
        if not self.data_path:
            raise ValueError("data_path is required")
        if not self.save_model_name:
            raise ValueError("save_model_name is required")


@dataclass
class EvaluationConfig:
    """評価の設定"""
    inference_data_path: str = ""
    model_checkpoint: str = ""
    save_path: str = "./results"
    
    # 境界検出方法
    boundary_detection_methods: list = field(default_factory=lambda: [
        'adaptive', 'fixed', 'threshold'
    ])
    
    # ランダムベースライン
    num_random_trials: int = 100
    
    def __post_init__(self):
        """設定値の検証"""
        if not self.inference_data_path:
            raise ValueError("inference_data_path is required")
        if not self.model_checkpoint:
            raise ValueError("model_checkpoint is required")


@dataclass
class Config:
    """全体の設定を統合"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_args(cls, args):
        """argparseの引数から設定を作成"""
        model_config = ModelConfig(
            margin=getattr(args, 'margin', 1),
            train_split=getattr(args, 'train_split', 5),
            window_size=getattr(args, 'window_size', 5)
        )
        
        training_config = TrainingConfig(
            batch_size=getattr(args, 'batch_size', 12),
            learning_rate=getattr(args, 'lr', 3e-5),
            epochs=getattr(args, 'epochs', 10),
            warmup_proportion=getattr(args, 'warmup_proportion', 0.1),
            seed=getattr(args, 'seed', 3407),
            gradient_accumulation_steps=getattr(args, 'accum', 1),
            no_cuda=getattr(args, 'no_cuda', False),
            no_amp=getattr(args, 'no_amp', False),
            local_rank=getattr(args, 'local_rank', -1),
            resume=getattr(args, 'resume', False),
            checkpoint_path=getattr(args, 'ckpt', None)
        )
        
        data_config = DataConfig(
            data_path=getattr(args, 'data_path', ''),
            save_model_name=getattr(args, 'save_model_name', ''),
            root_dir=getattr(args, 'root', '.')
        )
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config
        )