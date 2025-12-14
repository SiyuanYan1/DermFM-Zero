# Cross Retrieval - SkinCap
python src/main.py \
    --val-data="data/zero-shot-retrieval/skincap/meta.csv"  \
    --csv-caption-key 'caption_zh_polish_en' \
    --dataset-type "csv" \
    --batch-size=256 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/PanDerm2'

# Cross Retrieval - Derm1M-holdout
python src/main.py \
    --val-data="data/zero-shot-retrieval/Derm1M-hold_out/meta.csv"  \
    --csv-caption-key 'truncated_caption' \
    --dataset-type "csv" \
    --batch-size=256 \
    --csv-label-key label \
    --csv-img-key image_path \
    --model 'hf-hub:redlessone/PanDerm2'