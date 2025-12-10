conda activate PanDerm-v2

# Disease Classification
python src/main.py \
   --val-data=""  \
   --dataset-type "csv" \
   --batch-size=1024 \
   --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
   --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
   --zeroshot-eval3=meta-files/zs/snu-134-zero-shot-test.csv \
   --zeroshot-eval4=meta-files/zs/sd-128-zero-shot-test.csv \
   --zeroshot-eval5=meta-files/zs/daffodil-5-zero-shot-test.csv \
   --zeroshot-eval6=meta-files/zs/ph2-2-zero-shot-test.csv \
   --zeroshot-eval7=meta-files/zs/isic2020-2-zero-shot-test.csv \
   --csv-label-key label \
   --csv-img-key image_path \
   --csv-caption-key 'truncated_caption' \
   --model 'hf-hub:redlessone/PanDerm2'

# Cross Retrieval - SkinCap
python src/main.py \
    --val-data="meta-files/retrieval/skincap.csv"  \
    --csv-caption-key 'caption_zh_polish_en' \
    --dataset-type "csv" \
    --batch-size=256 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/PanDerm2'

# Cross Retrieval - Derm1M-holdout
python src/main.py \
    --val-data="meta-files/retrieval/Derm1M-hold_out.csv"  \
    --csv-caption-key 'truncated_caption' \
    --dataset-type "csv" \
    --batch-size=256 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/PanDerm2'