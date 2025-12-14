# Disease Classification
python src/main.py \
   --val-data=""  \
   --dataset-type "csv" \
   --batch-size=1024 \
   --zeroshot-eval1=data/zero-shot-classification/pad-zero-shot-test.csv \
   --zeroshot-eval2=data/zero-shot-classification/HAM-official-7-zero-shot-test.csv \
   --zeroshot-eval3=data/zero-shot-classification/snu-134-zero-shot-test.csv \
   --zeroshot-eval4=data/zero-shot-classification/sd-128-zero-shot-test.csv \
   --zeroshot-eval5=data/zero-shot-classification/daffodil-5-zero-shot-test.csv \
   --zeroshot-eval6=data/zero-shot-classification/ph2-2-zero-shot-test.csv \
   --zeroshot-eval7=data/zero-shot-classification/isic2020-2-zero-shot-test.csv \
   --csv-label-key label \
   --csv-img-key image_path \
   --model 'hf-hub:redlessone/PanDerm2'

# Run PanDerm-2 on your dataset
python src/main.py \
   --val-data=""  \
   --dataset-type "csv" \
   --batch-size=1024 \
   --csv-label-key label \
   --csv-img-key image_path \
   --zeroshot_eval_custom examples/toy_dataset.csv \
   --model 'hf-hub:redlessone/PanDerm2'