# Data Deduplication / Leakage Analysis

Pipeline that quantifies **data leakage** between the DermFM-Zero pre-training
corpus and the downstream zero-shot / reader-study evaluation sets, using
**SSCD** copy-detection embeddings + top-1 cosine search.

> **No data is shipped.** This folder contains only the *pipeline code* and the
> *aggregate overlap statistics*. The pre-training corpus and its images are
> **private** and are supplied by the user at runtime via command-line paths.
> The per-pair `overlaps.csv` files here have had the pretrain-side columns
> removed, and no image visualisations are included.

## Method

1. **Embed.** Every image (pretrain + each evaluation set) is resized to
   320×320, ImageNet-normalised, and forwarded through the public
   `sscd_disc_mixup` TorchScript model (ResNet-50 + GeM, 512-d output).
   Model: [facebookresearch/sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection).
2. **Overlap.** Embeddings are L2-normalised. For every evaluation image we
   compute its top-1 cosine match against the pretrain bank (batched). A pair
   is flagged as leakage when `cosine_similarity >= 0.75`.
3. **Report.** Per-dataset overlap counts/rates (`overlap_summary.csv`) and the
   list of flagged evaluation images (`overlaps.csv`, pretrain side redacted).

## Files

```
data_deduplication/
├── embed.py            # SSCD embedder (csv | folder modes)
├── overlap.py          # overlap detector (downstream | reader_study modes)
├── run.sh              # end-to-end driver (edit the paths at the top)
├── requirements.txt
└── results/
    ├── zero-shot-benchmark/
    │   ├── overlap_summary.csv         # per-dataset totals + overlap rates
    │   └── <dataset>/overlaps.csv      # flagged eval images (pretrain side redacted)
    └── reader-study/
        ├── overlap_summary.csv
        └── RS*_images/overlaps.csv
```

## Run it on your own data

```bash
pip install -r requirements.txt
# download the SSCD model into models/ (see link above), then:
#   - point PRETRAIN_CSV at a CSV with an `image_path` column
#   - point DOWNSTREAM_ROOT / RS_ROOT at your evaluation images
bash run.sh
```

Individual stages (see `run.sh` for the full set of flags):

```bash
# pretrain embeddings
python embed.py csv --csv meta/pretrain.csv --image_column image_path \
    --model_path models/sscd_disc_mixup.torchscript.pt \
    --output_dir embeddings --output_name pretrain

# downstream overlap detection
python overlap.py downstream \
    --pretrain_emb embeddings/pretrain_embeddings.npy \
    --pretrain_csv meta/pretrain.csv \
    --downstream_root ./downstream_root \
    --embedding_dir embeddings --output_dir results/zero-shot-benchmark \
    --threshold 0.75
```

## Reported overlap rates (threshold 0.75)

Downstream zero-shot:

| Dataset                   | Total | Overlap | Rate    |
|---------------------------|------:|--------:|--------:|
| daffodil-5-zero-shot      | 1,910 |     259 | 13.56 % |
| ph2-2-zero-shot           |   200 |      25 | 12.50 % |
| sd-128-zero-shot          | 1,405 |      22 |  1.57 % |
| HAM-official-7-zero-shot  | 1,503 |      15 |  1.00 % |
| isic2020-2-zero-shot      | 4,969 |      38 |  0.76 % |
| pad-zero-shot             |   461 |       1 |  0.22 % |
| snu-134-zero-shot         | 2,101 |       3 |  0.14 % |

Reader study:

| Dataset      | Total | Overlap | Rate    |
|--------------|------:|--------:|--------:|
| RS1_images   |   146 |      20 | 13.70 % |
| RS2_images   | 2,096 |      14 |  0.67 % |
| RS3_images   | 3,142 |       4 |  0.13 % |
