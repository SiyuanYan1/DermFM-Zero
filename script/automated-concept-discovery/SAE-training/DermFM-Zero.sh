# Export visual feature on SAE Dataset
python export_visual_features.py \
    --model_name hf-hub:redlessone/DermFM-Zero \
    --csv_path data/Derm1M/pretrain.csv \
    --data_root data/Derm1M/ \
    --batch_size 2048 \
    --num_workers 16 \
    --device cuda \
    --output_dir ../automated-concept-discovery-result/SAE-embeddings/

# SAE Training
python train_sae.py \
    --data Derm1M_v2 \
    --backbone DermFM-Zero \
    --save_dir '../automated-concept-discovery-result/SAE-embeddings/'