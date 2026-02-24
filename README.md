# year1-research

## Purpose
Baseline implementation for topic boundary detection using
YouTube transcript data.

This repository contains a **minimal, reproducible reference
implementation** used as the main branch.

## Main branch policy
- `main` branch contains only:
  - Verified working code
  - Fixed hyperparameters
  - No experimental hacks
- All experiments are conducted in `exp-*` branches.

## Code structure
```
year1/
├── src/
│   ├── config.py
│   ├── train.py
│   ├── test.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── training.py
│   │   └── inference.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── collator.py
│   │   └── loader.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── metrics.py
│   │   └── visualizer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── depth_score.py
│   │   └── losses.py
│   └── data_creaters/
│       └── 簡易化版/
│           ├── common_transcript_processing.py
│           ├── create_inference_data.py
│           └── test_window.py   ← データ作成スクリプト
├── data/
├── outputs/
└── README.md
```
## Data
- Data files are NOT tracked by git.
- Place all data under `data/`.

## Baseline specification (main)
- Window size: 固定（現行実装）
- Coherence MODEL:cl-tohoku/bert-base-japanese
- Topic MODEL:pkshatech/simcse-ja-bert-base-clcmlp
- Input: json (comments optional)
- Output: 

## Environment
- OS: Ubuntu (WSL)
- Python: 3.12.12
- Framework: PyTorch, Transformers
