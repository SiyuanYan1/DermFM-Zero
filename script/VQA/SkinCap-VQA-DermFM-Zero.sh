#!/bin/bash
# SkinCap-VQA fine-tune with DermFM-Zero (v15) visual+text encoder.
set -euo pipefail
CUDA_DEVICES="${CUDA_DEVICES:-0}"
PY="${PY:-/mnt/hdd/sda/xjli/miniconda3/envs_old/PanDerm-v2/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-../VQA-result/SkinCap-VQA/DermFM-Zero/}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/VQA"
export PYTHONPATH="${REPO_ROOT}/src:.${PYTHONPATH:+:${PYTHONPATH}}"
# VQA finetunes the encoder → ImageNet norm (dataloader default) like multimodal.
[ -n "${PANDERM_NORM_MEAN:-}" ] && export PANDERM_NORM_MEAN
[ -n "${PANDERM_NORM_STD:-}" ] && export PANDERM_NORM_STD
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PY}" train.py \
    --model_name 'DermFM-Zero' \
    --dataset_name 'SkinCap-VQA' --class_num 188 \
    --epochs 50 --batch_size 32 --accum_freq 2 --hidden_dim 1024 --learning_rate 1e-5 \
    --cuda True --use_derm --use_meta --use_text_encoder --meta_dim 768 \
    --num_head 8 --att_depth 4 --meta_num_head 8 --meta_att_depth 4 \
    --fusion 'cross attention' --meta_fusion_mode 'cross attention' \
    --encoder_pool 'mean' --out 'mlp' --use_visual_embedding_layer \
    --output_dir "$OUTPUT_DIR"
