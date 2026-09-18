#!/bin/bash
# PAD multimodal fine-tune with DermFM-Zero (v15) visual+text encoder.
set -euo pipefail
CUDA_DEVICES="${CUDA_DEVICES:-0}"
SEED="${SEED:-122}"
PY="${PY:-python}"
OUT_DIR="${OUT_DIR:-../multimodal_finetune-result/PAD/DermFM-Zero/}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/multimodal_finetune"
export PYTHONPATH="${REPO_ROOT}/src:.${PYTHONPATH:+:${PYTHONPATH}}"
# Multimodal finetune trains the encoder, so ImageNet normalization (the
# dataloader default) reproduces the best results (Derm7pt F1 0.766 / PAD 0.834).
# To force CLIP-style 0.5 norm instead: PANDERM_NORM_MEAN=0.5,0.5,0.5 PANDERM_NORM_STD=0.5,0.5,0.5
[ -n "${PANDERM_NORM_MEAN:-}" ] && export PANDERM_NORM_MEAN
[ -n "${PANDERM_NORM_STD:-}" ] && export PANDERM_NORM_STD

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PY}" train.py \
    --model_name 'DermFM-Zero' \
    --dataset_name 'PAD' --class_num 6 \
    --epochs 50 --batch_size 32 --accum_freq 2 --hidden_dim 1024 --learning_rate 1e-5 \
    --cuda True --use_cli --use_meta --use_text_encoder --meta_dim 768 \
    --num_head 8 --att_depth 4 --meta_num_head 8 --meta_att_depth 2 \
    --fusion 'cross attention' --meta_fusion_mode 'cross attention' \
    --encoder_pool 'mean' --out 'mlp' --seed "${SEED:-122}" \
    --output_dir "${OUT_DIR}"
