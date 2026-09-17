#!/bin/bash
# SkinCap-VQA — fine-tune DermFM-Zero on the SkinCap VQA benchmark
# and report metrics.
#
# Data:    data/VQA/SkinCap-VQA/meta/{train,val,test}.csv
# Model:   hf-hub:redlessone/DermFM-Zero (loaded directly from the HF Hub)
#
# Run from `VQA/` so the `../data/...` CSV paths resolve correctly:
#   cd VQA
#   bash ../script/VQA/SkinCap-VQA.sh

set -e

OUTPUT_DIR="../VQA-result/SkinCap-VQA/DermFM-Zero/"
mkdir -p "$OUTPUT_DIR"

# Step 1 (Optional) — build the SkinCap-VQA train/val/test CSVs
# Skipped if the splits already exist.
META_DIR="../data/VQA/SkinCap-VQA/meta"
if [ ! -f "$META_DIR/train.csv" ] || [ ! -f "$META_DIR/val.csv" ] || [ ! -f "$META_DIR/test.csv" ]; then
    echo "[preprocess] regenerating SkinCap-VQA splits..."
    ( cd preprocessing && python build_skincap_vqa.py )
fi

# Step 2 — train + evaluate.
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name 'DermFM-Zero' \
    --dataset_name 'SkinCap-VQA' \
    --class_num 188 \
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
