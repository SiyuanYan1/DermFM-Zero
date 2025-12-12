#!/bin/bash

################################################################################
# Linear Probing Evaluation Script
# 
# This script evaluates multiple vision models on dermatology datasets using
# linear probing. It tests models with different amounts of training data to
# assess data efficiency and representation quality.
################################################################################

# Clean previous logs (optional)
#rm -r linear_probing_logs/*

################################################################################
# MODELS CONFIGURATION
################################################################################
# List of models to evaluate
# Supported model types:
#   - 'PanDerm': Custom PanDerm model (requires local checkpoint)
#   - 'MONET': MONET dermatology model
#   - 'biomedclip': BiomedCLIP model
#   - 'open_clip_*': OpenCLIP models (various architectures)
#   - 'dinov3-*': DINOv3 self-supervised models
#   - 'open_clip_hf-hub:*': Models from Hugging Face Hub

models=('PanDerm' 'MONET' 'biomedclip' 'open_clip_vit_large_14' 'dinov3-l16' 'dinov3-7b' 'open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256')

################################################################################
# MODEL CHECKPOINTS
################################################################################
# Checkpoint paths for each model
# - Set to 'None' for models that load from Hugging Face/online repositories
# - Set to local path for custom models (e.g., PanDerm)
# 
# Usage:
#   ['MODEL_NAME']='path/to/checkpoint.pth'  # Local checkpoint
#   ['MODEL_NAME']='None'                     # Online/HF model

declare -A checkpoints=(
  ['PanDerm']='/mnt/hdd/sdc/syyan/My_Code/PanDerm/classification/panderm_ll_data6_checkpoint-499.pth' # Path to PanDerm checkpoint in your local workstation
  ['MONET']='None'
  ['biomedclip']='None'
  ['open_clip_vit_large_14']='None'
  ['dinov3-l16']='None'
  ['dinov3-7b']='None'
  ['open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256']='None'
)

################################################################################
# DATA PERCENTAGES
################################################################################
# Percentage of training data to use for evaluation
# Tests model performance with varying amounts of labeled data
# Values: 0.0-1.0, where 1.0 = 100% of training data
# 
# Example:
#   0.1  = 10% of training data (few-shot scenario)
#   0.3  = 30% of training data
#   0.5  = 50% of training data
#   1.0  = 100% of training data (full supervision)
percent_data_values=(0.1 0.3 0.5 1.0) 

################################################################################
# DATASETS CONFIGURATION
################################################################################
# Dataset names (used for logging and organization)
# Each dataset should have a corresponding CSV metadata file
datasets=('HAM' 'PAD' 'SD128' 'ISIC2020')

# CSV metadata paths for each dataset
# Each CSV should contain columns:
#   - image_path: Image paths
#   - label: Labels/classes
#   - split: train/val/test
# 
# Format: One-to-one correspondence with 'datasets' array
csv_paths=(
  'data/linear_probe/HAM-official-7-lp.csv'
  'data/linear_probe/pad-lp-ws0.csv'
  'data/linear_probe/sd-128.csv'
  'data/linear_probe/isic2020-2-lp.csv'
)

for percent_data in "${percent_data_values[@]}"; do
  echo "Running experiments with percent_data: ${percent_data}"
  for model in "${models[@]}"; do
    checkpoint="${checkpoints[$model]}"
    # Clean special characters from model name
    clean_model_name=$(echo "$model" | sed 's/:/_/g' | sed 's/\//_/g')
    # Check if checkpoint file exists (if specified)

    for i in "${!datasets[@]}"; do
      dataset="${datasets[$i]}"
      csv_path="${csv_paths[$i]}"
      # Check if CSV file exists
      if [ ! -f "$csv_path" ]; then
        echo "CSV file not found: $csv_path, skipping dataset: $dataset"
        continue
      fi
      # Create unified format using cleaned model name, including percent_data subdirectory
      output_dir="linear-probing-logs/percent_data_${percent_data}/${dataset}/${model}"
      csv_filename="${model}_${dataset}_results.csv"
      echo "Processing model: ${model}, dataset: ${dataset}, percent_data: ${percent_data}"
      # Build checkpoint argument
      if [ -n "$checkpoint" ]; then
        checkpoint_arg="--checkpoint ${checkpoint}"
      else
        checkpoint_arg=""
      fi
      # Run the linear_eval.py script
      CUDA_VISIBLE_DEVICES=1 python linear_probe/linear_eval.py \
        --batch_size 256  \
        --model "${model}" \
        --csv_filename "${csv_filename}" \
        --output_dir "${output_dir}" \
        --csv_path "${csv_path}" \
        --image_key "image_path" \
        --percent_data ${percent_data} \
        ${checkpoint_arg}
    done
  done
done
wait