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
src/
├─ train.py # model training
├─ model.py # model definition
├─ test3_csv_rand.py # evaluation / inference
└─ data_creaters/
└─ 簡易化版/
├─ create_inference_data.py
├─ common_transcript_processing.py
└─ test_window.py

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
