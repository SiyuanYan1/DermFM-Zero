# Preprocessing — multimodal_finetune_VQA

Scripts that build the per-dataset VQA splits used in `train.py`.

| Dataset       | Script | Inputs | Outputs |
|---|---|---|---|
| Derm7pt-VQA   | `build_derm7pt_vqa.py`   | `data/multimodal_finetune_VQA/preprocessing_inputs/derm7pt-VQA/derm7pt-meta.csv` + `case_split.csv` | `data/multimodal_finetune_VQA/derm7pt-VQA/meta/{train,val,test}.csv` |
| SkinCap-VQA   | `build_skincap_vqa.py`   | `data/multimodal_finetune_VQA/preprocessing_inputs/SkinCap-VQA/{train,test}_public_MCQA.json` (DermVQA4 reference layout) | `data/multimodal_finetune_VQA/SkinCap-VQA/meta/{train,val,test}.csv` |

The `preprocessing_inputs/` subtree holds the official upstream artefacts
(meta CSV / case-split / DermVQA4 MCQA JSONs); the rest of
`multimodal_finetune_VQA/<dataset>/` holds the preprocessed splits.

Both scripts assume they are run from `multimodal_finetune/preprocessing/` and
write CSVs whose `image_path` column is *relative* to `multimodal_finetune/`
(i.e. starts with `../data/...`).

## Derm7pt-VQA

```bash
cd multimodal_finetune/preprocessing
# 1) reproduce the paper splits exactly (default --case_split already points
#    to the published case split)
python build_derm7pt_vqa.py

# 2) generate a fresh seeded build from raw meta only
python build_derm7pt_vqa.py --case_split "" --seed 42
```

Pipeline: load `meta.csv` → templated VQA → drop rare answers → balanced 4k
subset (40/35/15/10 % across question groups) → case-level split → build
`answer_id` from train uniques.

## SkinCap-VQA

```bash
cd multimodal_finetune/preprocessing
# SkinCap-VQA source images live outside the repo; override --image_root if
# your local copy is not at the default path.
python build_skincap_vqa.py
```

Pipeline: parse the public MCQA conversation JSONs → keep SkinCap rows →
drop answers occurring ≤ 5 times → image-level random 60/20/20 (seed=42) →
restrict to answers present in all three splits → rewrite image_path → build
`answer_id` from the train→val→test row order.

## Notes

- All randomness is seeded via `--seed`; re-runs are deterministic.
- The `multimodal_finetune_VQA/` task folder is self-contained: each dataset
  carries its own `images/` subfolder so users who only download the VQA
  bundle can train and evaluate without pulling any other task folder.
