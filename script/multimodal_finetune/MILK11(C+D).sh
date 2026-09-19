# DermPrime - Final + Weighted Sampler
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name 'DermFM-Zero' \
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
    --output_dir '../multimodal_finetune-result/MILK-11/DermFM-Zero_weighted_sampler/'

CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_name 'DermFM-Zero' \
    --model_path "$(ls -t ../multimodal_finetune-result/MILK-11/DermFM-Zero_weighted_sampler/bestvalloss_model_*.pth | head -1)"   # checkpoint saved by train.py (epoch varies per run) \
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
    --output_dir '../multimodal_finetune-result/MILK-11/DermFM-Zero_weighted_sampler/'