# DermPrime - Final + Weighted Sampler
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name 'PanDerm-v2' \
    --dataset_name 'MILK-11' \
    --class_num 11 \
    --epochs 50 \
    --batch_size 32 \
    --accum_freq 2 \
    --hidden_dim 1024 \
    --learning_rate 5e-6 \
    --cuda True \
    --use_derm \
    --use_cli \
    --num_head 8 \
    --att_depth 4 \
    --fusion 'cross attention' \
    --encoder_pool 'mean' \
    --out 'mlp' \
    --patience 25 \
    --monitor 'loss' \
    --weight_sampler \
    --output_dir '../multi-skin-modality-finetune-result/MILK-11/PanDerm-v2_weighted_sampler/'

CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_name 'PanDerm-v2' \
    --model_path '../multi-skin-modality-finetune-result/MILK-11/PanDerm-v2_weighted_sampler/bestvalloss_model_21.pth' \
    --dataset_name 'MILK-11' \
    --class_num 11 \
    --epochs 50 \
    --batch_size 32 \
    --hidden_dim 1024 \
    --cuda True \
    --use_derm \
    --use_cli \
    --num_head 8 \
    --att_depth 4 \
    --fusion 'cross attention' \
    --encoder_pool 'mean' \
    --out 'mlp' \
    --output_dir '../multi-skin-modality-finetune-result/MILK-11/PanDerm-v2_weighted_sampler/'