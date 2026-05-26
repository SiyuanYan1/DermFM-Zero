# Preprocessing — multimodal_finetune

Scripts that build the per-dataset VQA splits used in `train.py`.

| Dataset       | Script | Inputs | Outputs |
|---|---|---|---|
| Derm7pt-VQA   | `build_derm7pt_vqa.py`   | `data/derm7pt/meta/meta.csv` + (optional) `data/multimodal_finetune/derm7pt-VQA/case_split.csv` | `data/multimodal_finetune/derm7pt-VQA/meta/{train,val,test}.csv` |
| SkinCap-VQA   | `build_skincap_vqa.py`   | `train_public_MCQA.json` + `test_public_MCQA.json` (DermVQA4 reference layout) | `data/multimodal_finetune/SkinCap-VQA/meta/{train,val,test}.csv` |

Both scripts assume they are run from `multimodal_finetune/preprocessing/` and
write CSVs whose `image_path` column is *relative* to `multimodal_finetune/`
(i.e. starts with `../data/...`).

## Derm7pt-VQA

```bash
cd multimodal_finetune/preprocessing
# 1) reproduce the paper splits exactly
python build_derm7pt_vqa.py \
    --case_split ../../data/multimodal_finetune/derm7pt-VQA/case_split.csv

# 2) generate a fresh seeded build from raw meta only
python build_derm7pt_vqa.py --seed 42
```

The case-level partition used in our paper is published as
`data/multimodal_finetune/derm7pt-VQA/case_split.csv` (one row per `image_id`).
Pipeline: load `meta.csv` → templated VQA → drop rare answers → balanced 4k
subset (40/35/15/10 % across question groups) → case-level split → build
`answer_id` from train uniques.

## SkinCap-VQA

```bash
cd multimodal_finetune/preprocessing
python build_skincap_vqa.py
```

Pipeline: parse the public MCQA conversation JSONs → keep SkinCap rows →
drop answers occurring ≤ 5 times → image-level random 60/20/20 (seed=42) →
restrict to answers present in all three splits → rewrite image_path → build
`answer_id` from the train→val→test row order.

## Notes

- All randomness is seeded via `--seed`; re-runs are deterministic.
- `data/derm7pt` and `data/SkinCap-VQA` are expected to live at the repo
  root (symlink your local copy if needed). Both VQA CSVs reference images
  through that prefix.
