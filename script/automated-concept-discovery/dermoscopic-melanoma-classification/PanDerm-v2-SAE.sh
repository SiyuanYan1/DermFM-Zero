cd src
# extract visual feature for malignancy dataset
python export_visual_features.py \
    --model_name hf-hub:redlessone/PanDerm2 \
    --csv_path ../meta-files/automated-concept-discovery/dermoscopic-melanoma/meta.csv \
    --data_root ../meta-files/automated-concept-discovery/dermoscopic-melanoma/final_images/ \
    --img_col 'ImageID' \
    --batch_size 2048 \
    --num_workers 16 \
    --device cuda \
    --output_dir ../automated-concept-discovery-result/dermoscopic-melanoma/

cd ..
# Extract SAE concept (checked)
python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint /mnt/hdd/sdc/syyan/My_Code/PanDerm-X/automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/dermoscopic-melanoma/all_embeddings.npy \
  --output automated-concept-discovery-result/dermoscopic-melanoma/learned_activation.npy 

# Extract SAE concept (checked)
python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint /mnt/hdd/sdc/syyan/My_Code/PanDerm-X/automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/dermoscopic-melanoma/all_embeddings.npy \
  --output automated-concept-discovery-result/dermoscopic-melanoma/learned_activation.npy

# build CBM based on SAE concept (checked)
python automated-concept-discovery/1_train_clf_binary-class.py \
  --csv meta-files/automated-concept-discovery/dermoscopic-melanoma/meta.csv \
  --embeddings automated-concept-discovery-result/dermoscopic-melanoma/learned_activation.npy \
  --image_col ImageID \
  --image_dir meta-files/automated-concept-discovery/dermoscopic-melanoma/final_images \
  --gpu 2 \
  --output automated-concept-discovery-result/dermoscopic-melanoma/

# build classifier based on vision feature
python automated-concept-discovery/1_train_clf_binary-class.py \
  --csv meta-files/automated-concept-discovery/dermoscopic-melanoma/meta.csv \
  --embeddings automated-concept-discovery-result/dermoscopic-melanoma/all_embeddings.npy \
  --image_col ImageID \
  --image_dir meta-files/automated-concept-discovery/dermoscopic-melanoma/final_images \
  --gpu 2