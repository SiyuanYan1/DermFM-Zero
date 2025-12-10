#!/bin/bash
#rm -r linear_probing_logs/*
models=('PanDerm' 'MONET' 'biomedclip' 'open_clip_vit_large_14' 'dinov3-l16' 'dinov3-7b' 'open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256')
declare -A checkpoints=(
  ['PanDerm']='/mnt/hdd/sdc/syyan/My_Code/PanDerm/classification/panderm_ll_data6_checkpoint-499.pth' # Path to PanDerm checkpoint in your local workstation
  ['MONET']='None'
  ['biomedclip']='None'
  ['open_clip_vit_large_14']='None'
  ['dinov3-l16']='None'
  ['dinov3-7b']='None'
  ['open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256']='None'
)

percent_data_values=(0.1 0.3 0.5 1)
datasets=('HAM' 'PAD' 'SD128' 'ISIC2020')
csv_paths=(
  'meta-files/lp/HAM-official-7-lp.csv'
  'meta-files/lp/pad-lp-ws0.csv'
  'meta-files/lp/sd-128.csv'
  'meta-files/lp/isic2020-2-lp.csv'
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