#!/bin/bash
# Derm7pt-VQA — fine-tune DermFM-Zero on the held-out-image VQA benchmark
# and report metrics.
#
# Data:    data/VQA/derm7pt-VQA/meta/{train,val,test}.csv
# Model:   checkpoints/DermFM-Zero/open_clip_pytorch_model.bin
#          (download via huggingface-cli download redlessone/DermFM-Zero)
#
# Run from `VQA/` so the relative `--pretrain_path` and the
# `../data/...` CSV paths resolve correctly:
#   cd VQA
#   bash ../script/VQA/Derm7pt-VQA.sh

set -e

OUTPUT_DIR="../VQA-result/derm7pt-VQA/DermFM-Zero/"
mkdir -p "$OUTPUT_DIR"

# Step 1 (Optional) — build the Derm7pt-VQA train/val/test CSVs
# Skipped if the splits already exist.
META_DIR="../data/VQA/derm7pt-VQA/meta"
if [ ! -f "$META_DIR/train.csv" ] || [ ! -f "$META_DIR/val.csv" ] || [ ! -f "$META_DIR/test.csv" ]; then
    echo "[preprocess] regenerating Derm7pt-VQA splits..."
    ( cd preprocessing && python build_derm7pt_vqa.py )
fi

# Step 2 — train + evaluate.
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name 'PanDerm-Large-VL' \
    --pretrain_path '../checkpoints/DermFM-Zero/open_clip_pytorch_model.bin' \
    --dataset_name 'Derm7pt-VQA' \
    --class_num 49 \
    --epochs 50 \
    --batch_size 32 \
    --accum_freq 2 \
    --hidden_dim 1024 \
    --learning_rate 1e-5 \
    --cuda True \
    --use_derm \
    --use_meta \
    --use_text_encoder \
    --meta_dim 768 \
    --num_head 8 \
    --att_depth 4 \
    --meta_num_head 8 \
    --meta_att_depth 4 \
    --fusion 'cross attention' \
    --meta_fusion_mode 'cross attention' \
    --encoder_pool 'mean' \
    --out 'mlp' \
    --use_visual_embedding_layer \
    --output_dir "$OUTPUT_DIR"
