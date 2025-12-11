# Export visual feature on SAE Dataset
python export_visual_features.py \
    --model_name hf-hub:redlessone/PanDerm2 \
    --csv_path /mnt/hdd/sdd/siyuanyan/Derm1M_v2/csv/Derm1M_v2_pretrain.csv \
    --data_root /mnt/hdd/sdd/siyuanyan/Derm1M_v2/ \
    --batch_size 2048 \
    --num_workers 16 \
    --device cuda \
    --output_dir ../automated-concept-discovery-result/SAE-embeddings/

# SAE Training
python train_sae.py \
    --data Derm1M_v2 \
    --backbone PanDerm-v2 \
    --save_dir '../automated-concept-discovery-result/SAE-embeddings/'