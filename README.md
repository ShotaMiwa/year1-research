# year1-research

## Overview
This repository contains code for training a dialogue/topic segmentation model.
The model jointly optimizes:
- Coherence modeling based on sentence order (NSP-style)
- Topic consistency modeling using utterance and comment representations

## Method
The model jointly learns:
1. Coherence loss based on sentence adjacency (NSP-style input)
2. Topic loss using contrastive learning over utterance representations
3. (Optional) Comment-aware topic modeling using averaged comment embeddings

## Data Format
Training data must be provided as `.pt` files saved with `torch.save`.

Each file should contain a dictionary with at least the following keys:
- sentences: list of utterances
- coheren_inputs: tensor for NSP-style coherence modeling
- coheren_masks
- coheren_types
- sub_ids_simcse: utterance-level token ids (SimCSE)
- com_vecs: comment embedding vectors per utterance

## Training
The training script supports:
- single file input
- directory input (multiple `.pt` files)
- wildcard patterns

## Environment
- OS: Ubuntu (WSL)
- Python: 3.12.12
- Framework: PyTorch, Transformers

## Running on Google Colab
This repository is designed to run on Google Colab
without modifying the directory structure.


## 1. Clone the repository 

Run the follwing commands in a Colab cell:

```bash
!git clone git@github.com:ShotaMiwa/year1-research.git
%cd year1-research