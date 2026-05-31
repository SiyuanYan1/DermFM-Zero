#!/bin/bash
# Full data-leakage pipeline: pretrain embeddings -> downstream + reader-study
# overlap detection. Steps are skipped when their .npy already exists.
#
# You supply your own data (none of it ships with this repo):
#   PRETRAIN_CSV : a CSV with an `image_path` column for your pretraining images
#   DOWNSTREAM_ROOT : root holding data/zero-shot-classification/<ds>/meta.csv
#   RS_ROOT      : folder of reader-study image subfolders
#   MODEL        : SSCD model (public: facebookresearch/sscd-copy-detection,
#                  sscd_disc_mixup.torchscript.pt)

set -e
cd "$(dirname "$0")"

PRETRAIN_CSV="meta/pretrain.csv"            # <-- your pretrain metadata (image_path column)
PRETRAIN_NAME="pretrain"
MODEL="models/sscd_disc_mixup.torchscript.pt"
EMB_DIR="embeddings"
RESULTS_DIR="results"
DOWNSTREAM_ROOT="./downstream_root"         # <-- holds data/zero-shot-classification/<ds>/meta.csv
RS_ROOT="reader-study-meta"                 # <-- folders of reader-study images
THRESHOLD=0.75

# Optionally hide pretrain rows of a given `source` value from the viz PNGs
# (CSVs are unaffected). Leave empty to disable.
VIZ_EXCLUDE_ARGS=()
# e.g. VIZ_EXCLUDE_ARGS=(--viz_exclude_source <source_name>)

mkdir -p "$EMB_DIR" "$RESULTS_DIR"

# Step 1 — Pretrain embeddings (~30-90 min on a single GPU for ~1M images).
if [ ! -f "${EMB_DIR}/${PRETRAIN_NAME}_embeddings.npy" ]; then
  echo "[1/4] Pretrain embeddings"
  python embed.py csv \
      --csv "$PRETRAIN_CSV" --image_column image_path \
      --model_path "$MODEL" --output_dir "$EMB_DIR" --output_name "$PRETRAIN_NAME" \
      --batch_size 512 --num_workers 16
else
  echo "[1/4] Pretrain embeddings already exist — skipping."
fi

# Step 2 — Downstream embeddings (one CSV per zero-shot dataset).
DOWNSTREAM=(
  "HAM-official-7-zero-shot" "ph2-2-zero-shot" "isic2020-2-zero-shot"
  "snu-134-zero-shot" "daffodil-5-zero-shot" "pad-zero-shot" "sd-128-zero-shot"
)
echo "[2/4] Downstream embeddings"
for DS in "${DOWNSTREAM[@]}"; do
  [ -f "${EMB_DIR}/${DS}_embeddings.npy" ] && { echo "  $DS: exists — skip"; continue; }
  python embed.py csv \
      --csv "${DOWNSTREAM_ROOT}/data/zero-shot-classification/${DS}/meta.csv" \
      --image_column image_path --image_root "$DOWNSTREAM_ROOT" \
      --model_path "$MODEL" --output_dir "$EMB_DIR" --output_name "$DS" \
      --batch_size 256 --num_workers 16
done

# Step 3 — Reader study embeddings (folder mode).
echo "[3/4] Reader study embeddings"
for RS in RS1_images RS2_images RS3_images; do
  [ -f "${EMB_DIR}/${RS}_embeddings.npy" ] && { echo "  $RS: exists — skip"; continue; }
  python embed.py folder \
      --folder "${RS_ROOT}/${RS}" \
      --model_path "$MODEL" --output_dir "$EMB_DIR" --output_name "$RS" \
      --batch_size 256 --num_workers 16
done

# Step 4 — Overlap detection (downstream + reader study).
echo "[4/4] Overlap detection"
python overlap.py downstream \
    --pretrain_emb "${EMB_DIR}/${PRETRAIN_NAME}_embeddings.npy" \
    --pretrain_csv "$PRETRAIN_CSV" --downstream_root "$DOWNSTREAM_ROOT" \
    --embedding_dir "$EMB_DIR" --output_dir "${RESULTS_DIR}/zero-shot-benchmark" \
    --threshold "$THRESHOLD" --max_viz_pairs 20 "${VIZ_EXCLUDE_ARGS[@]}"

python overlap.py reader_study \
    --pretrain_emb "${EMB_DIR}/${PRETRAIN_NAME}_embeddings.npy" \
    --pretrain_csv "$PRETRAIN_CSV" --reader_study_root "$RS_ROOT" \
    --embedding_dir "$EMB_DIR" --output_dir "${RESULTS_DIR}/reader-study" \
    --threshold "$THRESHOLD" --max_viz_pairs 20 "${VIZ_EXCLUDE_ARGS[@]}"

echo "Done — results in ${RESULTS_DIR}/"
