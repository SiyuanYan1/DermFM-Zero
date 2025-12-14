#!/bin/bash
#rm -r linear_probing_logs/*

################################################################################
# MODELS CONFIGURATION - PanDerm-2(From huggingface API)
################################################################################
models=('open_clip_hf-hub:redlessone/PanDerm2')
declare -A checkpoints=(
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
  'data/linear_probe/HAM-official-7-lp/meta.csv'
  'data/linear_probe/pad-lp/meta.csv'
  'data/linear_probe/sd-128-lp/meta.csv'
  'data/linear_probe/isic2020-2-lp/meta.csv'
)


for percent_data in "${percent_data_values[@]}"; do
  echo "Running experiments with percent_data: ${percent_data}"
  for model in "${models[@]}"; do
    checkpoint="${checkpoints[$model]}"
    # Clean special characters from model name
    clean_model_name=$(echo "$model" | sed 's/:/_/g' | sed 's/\//_/g')
    # Check if checkpoint file exists (if specified)
    if [ -n "$checkpoint" ] && [ ! -f "$checkpoint" ]; then
      echo "Checkpoint file not found: $checkpoint, skipping model: $model"
      continue
    fi
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