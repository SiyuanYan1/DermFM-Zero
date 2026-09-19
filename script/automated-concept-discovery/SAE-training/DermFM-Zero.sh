# Run from the repository root. Point --csv_path/--data_root at your local copy of the Derm1M pretraining set.
cd src
# Export visual features on the SAE dataset
python export_visual_features.py \
    --model_name hf-hub:redlessone/DermFM-Zero \
    --csv_path ../data/Derm1M/pretrain.csv \
    --data_root ../data/Derm1M/ \
    --batch_size 2048 \
    --num_workers 16 \
    --device cuda \
    --output_dir ../automated-concept-discovery-result/SAE-embeddings/
cd ..

# SAE training
python automated-concept-discovery/train_sae.py \
    --data Derm1M_v2 \
    --backbone DermFM-Zero \
    --save_dir automated-concept-discovery-result/SAE-embeddings/
