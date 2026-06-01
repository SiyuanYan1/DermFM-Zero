# Benchmark Bootstrap CI

Provides a single command-line bootstrap-CI pipeline for the zero-shot classification and linear-probing benchmark tables. Includes example prediction CSVs and reference outputs so the pipeline can be validated end-to-end before being applied to a full prediction set.

## 📋 Tasks at a glance

| Task | Input format | Example datasets | Output |
|------|--------------|------------------|--------|
| `zero_shot` | per-image softmax CSV per `<dataset>/<model>.csv` | HAM, PAD | `model_comparison_results_comprehensive.csv` |
| `linear_probe` | per-image softmax CSV per `<dataset>_<pct>pct/<model>/` folder | HAM, PAD | `lp_results_<pct>percent_bootstrap.csv` |

Both tasks share one bootstrap routine: 1,000 percentile resamples, 95% CI, and 7 metrics (accuracy, balanced accuracy, macro/weighted F1, sensitivity, specificity, AUROC).

## 📂 Repository Structure

```
statistic_reproduce/
├── bootstrap_ci.py                       # unified script, --task {zero_shot, lp}
├── examples/
│   ├── zero_shot/
│   │   ├── HAM/<model>.csv               # per-image prediction CSVs
│   │   ├── PAD/<model>.csv
│   │   └── class2label/{HAM,PAD}.csv
│   └── linear_probe/
│       ├── HAM_100pct/<model>/           # one results CSV per model folder
│       ├── PAD_100pct/<model>/
│       └── class2label/{HAM,PAD}.csv
├── reference_outputs/
│   ├── zero_shot/model_comparison_results_comprehensive.csv
│   └── linear_probe/lp_results_100percent_bootstrap.csv
└── README.md                             # this file
```

## 🚀 Quick Start

```bash
pip install pandas numpy scipy scikit-learn

# Zero-shot
python bootstrap_ci.py --task zero_shot \
    --data-root ./examples/zero_shot \
    --output-dir ./out_zero_shot

# Linear probing (100% data fraction)
python bootstrap_ci.py --task lp \
    --data-root ./examples/linear_probe \
    --output-dir ./out_linear_probe \
    --fractions 100
```

Outputs land in `./out_<task>/`; cross-check against `reference_outputs/<task>/` to confirm the pipeline behaves as expected on your machine before pointing it at a larger prediction set.

## 🔬 Input format

Per-image prediction CSV, one row per image:

```
image_path, true_label, predicted_label, probability_class_0, probability_class_1, ...
```

The linear-probing CSVs use `filename` instead of `image_path`; either column name is accepted. Class indexing follows the corresponding `class2label/<dataset>.csv` lookup table.

## ⚙️ Options

| Flag | Default | Notes |
|------|---------|-------|
| `--task` | required | `zero_shot` or `lp` |
| `--data-root` | required | dataset / prediction directory |
| `--output-dir` | required | where result CSVs are written |
| `--n-bootstrap` | 1000 | bootstrap replicates |
| `--target-model` | task-specific | anchor model for pairwise p-values |
| `--fractions` | 10,30,50,100 | LP data-fraction sweep |
| `--seed` | 42 | RNG seed |

Run `python bootstrap_ci.py --help` for the full list.

## 📊 Running on the full prediction set

Once the example run matches `reference_outputs/`, point `--data-root` at your own complete prediction directory (same layout as `examples/`):

```bash
# Zero-shot, full prediction set
python bootstrap_ci.py --task zero_shot \
    --data-root /path/to/full/zero_shot_predictions \
    --output-dir ./out_zero_shot_full

# Linear probing, all data fractions
python bootstrap_ci.py --task lp \
    --data-root /path/to/full/linear_probe_predictions \
    --output-dir ./out_linear_probe_full \
    --fractions 10,30,50,100
```

## 📄 Citation

If you use this code, please cite the main DermFM-Zero paper (see the top-level repository [README](../README.md#citation)). Released under CC-BY-NC-ND.
