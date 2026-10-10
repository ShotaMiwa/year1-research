import os
from pathlib import Path

# プロジェクトルート (src/utils/config.py から見て2つ上の階層)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# データ関連のパス
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# アノテーションデータのパス
ANNOTATIONS_DIR = DATA_RAW_DIR / "annotations"

# 出力関連のパス
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUT_CORRELATION_DIR = OUTPUTS_DIR / "correlation"
OUT_SATISFACTION_DIR = OUTPUTS_DIR / "explore_satisfaction"
OUT_EVALUATION_DIR = OUTPUTS_DIR / "model_evaluation"

def init_directories():
    """必要なディレクトリを作成します。"""
    dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        ANNOTATIONS_DIR,
        OUTPUTS_DIR,
        OUT_CORRELATION_DIR,
        OUT_SATISFACTION_DIR,
        OUT_EVALUATION_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
