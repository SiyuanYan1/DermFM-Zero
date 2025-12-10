cd src
# PanDermv2 - Export image feature
CUDA_VISIBLE_DEVICES=2 python export_visual_features.py \
  --model_name 'hf-hub:redlessone/PanDerm2' \
  --csv ../meta-files/automated-concept-discovery/ISIC_ink_bias/train.csv \
  --img_col 'ImageID' \
  --data_root ../meta-files/automated-concept-discovery/ISIC_ink_bias/final_images/ \
  --output ../automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/train/

CUDA_VISIBLE_DEVICES=2 python export_visual_features.py \
  --model_name 'hf-hub:redlessone/PanDerm2' \
  --csv ../meta-files/automated-concept-discovery/ISIC_ink_bias/test.csv \
  --img_col 'ImageID' \
  --data_root ../meta-files/automated-concept-discovery/ISIC_ink_bias/final_images/ \
  --output ../automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/test/

cd ..
# Extract SAE concept
python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/train/all_embeddings.npy \
  --output automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/train/learned_activation.npy

python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/test/all_embeddings.npy \
  --output automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/test/learned_activation.npy

# build CBM based on SAE concept
python automated-concept-discovery/3_train_biased-cbm_binary-class.py \
  --train_csv meta-files/automated-concept-discovery/ISIC_ink_bias/train.csv \
  --test_csv meta-files/automated-concept-discovery/ISIC_ink_bias/test.csv \
  --train_embeddings automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/train/learned_activation.npy \
  --test_embeddings automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/test/learned_activation.npy \
  --image_root '../meta-files/automated-concept-discovery/ISIC_ink_bias/final_images/' \
  --image_col 'ImageID' \
  --output 'automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/before-intervention'

# Intervention
python automated-concept-discovery/4_intervention_CBM.py \
    --top_n 5 \
    --concept 'purple ink' \
    --method matching \
    --split test \
    --device cuda \
    --data_dir automated-concept-discovery-result/ISIC_ink_bias/ \
    --result_dir automated-concept-discovery-result/ISIC_ink_bias/PanDermv2 \
    --sae_checkpoint automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
    --clip_model_name 'hf-hub:redlessone/PanDerm2'

# build CBM based on SAE concept
python automated-concept-discovery/3_train_biased-cbm_binary-class.py \
  --train_csv meta-files/automated-concept-discovery/ISIC_ink_bias/train.csv \
  --test_csv meta-files/automated-concept-discovery/ISIC_ink_bias/test.csv \
  --train_embeddings automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/train/learned_activation.npy \
  --test_embeddings automated-concept-discovery-result/ISIC_ink_bias/PanDermv2/test/learned_activation_after_intervention.npy \
  --image_root '../meta-files/automated-concept-discovery/ISIC_ink_bias/final_images/' \
  --image_col 'ImageID'