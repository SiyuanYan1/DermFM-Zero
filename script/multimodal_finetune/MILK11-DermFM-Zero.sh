#!/bin/bash
# MILK-11 multimodal fine-tune with DermFM-Zero (v15) visual encoder.
set -euo pipefail
CUDA_DEVICES="${CUDA_DEVICES:-0}"
PY="${PY:-/mnt/hdd/sda/xjli/miniconda3/envs_old/PanDerm-v2/bin/python}"
OUT_DIR="${OUT_DIR:-../multimodal_finetune-result/MILK-11/DermFM-Zero_weighted_sampler/}"

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
    --dataset_name 'MILK-11' --class_num 11 \
    --epochs 50 --batch_size 32 --accum_freq 2 --hidden_dim 1024 --learning_rate 5e-6 \
    --cuda True --use_derm --use_cli \
    --num_head 8 --att_depth 4 --fusion 'cross attention' \
    --encoder_pool 'mean' --out 'mlp' --patience 25 --monitor 'loss' --weight_sampler \
    --output_dir "${OUT_DIR}"
