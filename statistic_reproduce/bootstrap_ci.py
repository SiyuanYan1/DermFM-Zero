#!/usr/bin/env python3
"""
bootstrap_ci.py
===============

Unified bootstrap 95% CI script for the zero-shot classification and
linear-probing benchmark tables.

Two modes:
  --task zero_shot   per-image softmax CSVs under <dataset>/<model>.csv
  --task lp          per-image softmax CSVs under <dataset>_<pct>pct/<model>/

Bootstrap protocol: 1000 reps, percentile [2.5, 97.5], output CSV
format "mean (ci_lower-ci_upper)".  See README.md for input format,
CLI options, and example commands.

CLI
---
python3 bootstrap_ci.py --task zero_shot --data-root DIR --output-dir DIR
python3 bootstrap_ci.py --task lp        --data-root DIR --output-dir DIR
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "sensitivity",
    "specificity",
    "auroc",
]

# Column-name spellings used in the output CSV (upper-case)
METRIC_COLS = [m.upper() for m in METRIC_KEYS]


# --------------------------------------------------------------------- #
# Metric helpers (copied verbatim from both originals; they agree)      #
# --------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_proba: np.ndarray | None,
                    num_classes: int) -> dict[str, float]:
    out: dict[str, float] = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    out["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")

    if num_classes == 2:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception:
            out["sensitivity"] = 0.0
            out["specificity"] = 0.0
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            try:
                out["auroc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                out["auroc"] = 0.0
        else:
            out["auroc"] = 0.0
    else:
        cm = confusion_matrix(y_true, y_pred)
        senss: list[float] = []
        specs: list[float] = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            senss.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        out["sensitivity"] = float(np.mean(senss))
        out["specificity"] = float(np.mean(specs))
        if y_proba is not None and y_proba.ndim == 2:
            try:
                out["auroc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except Exception:
                out["auroc"] = 0.0
        else:
            out["auroc"] = 0.0
    return out


def bootstrap_one(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_proba: np.ndarray | None,
                  indices: np.ndarray,
                  num_classes: int) -> dict[str, float] | None:
    try:
        bt = y_true[indices]
        bp = y_pred[indices]
        bpr = y_proba[indices] if y_proba is not None else None
        return compute_metrics(bt, bp, bpr, num_classes)
    except Exception:
        return None


def run_bootstrap(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_proba: np.ndarray | None,
                  n_bootstrap: int,
                  rng_seed: int) -> dict[str, dict[str, Any]]:
    """Sequential bootstrap (matches the original two scripts'
    `np.random.choice(n, n, replace=True)` per iteration).

    rng_seed seeds NumPy's legacy global RNG before sampling, so that
    the index sequence is identical across runs.
    """
    num_classes = len(np.unique(y_true))
    n_samples = len(y_true)
    np.random.seed(rng_seed)
    boot_indices = [
        np.random.choice(n_samples, size=n_samples, replace=True)
        for _ in range(n_bootstrap)
    ]

    raw: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for idx in boot_indices:
        r = bootstrap_one(y_true, y_pred, y_proba, idx, num_classes)
        if r is None:
            continue
        for k in METRIC_KEYS:
            raw[k].append(r[k])

    summary: dict[str, dict[str, Any]] = {}
    for k in METRIC_KEYS:
        vals = np.asarray(raw[k], dtype=float)
        if vals.size == 0:
            summary[k] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                          "values": np.zeros(n_bootstrap)}
        else:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "ci_lower": float(np.percentile(vals, 2.5)),
                "ci_upper": float(np.percentile(vals, 97.5)),
                "values": vals,
            }
    return summary


def maybe_floor_zero_ci(summary: dict[str, dict[str, Any]],
                        enabled: bool) -> None:
    """Reproduces the zero-shot script's hack: if a CI lower bound is
    exactly 0.0 while the mean is > 0, lift it to max(0.1*mean, 1e-6)."""
    if not enabled:
        return
    for k in METRIC_KEYS:
        if summary[k]["ci_lower"] == 0.0 and summary[k]["mean"] > 0.0:
            summary[k]["ci_lower"] = max(summary[k]["mean"] * 0.1, 1e-6)


def ci_string(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} ({lo:.4f}-{hi:.4f})"


# --------------------------------------------------------------------- #
# CSV loaders                                                           #
# --------------------------------------------------------------------- #
def load_zs_predictions(path: Path) -> dict[str, Any] | None:
    """Zero-shot CSV: true_label, predicted_label, probability_class_*"""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  load error {path}: {e}")
        return None
    if "true_label" not in df.columns or "predicted_label" not in df.columns:
        return None
    prob_cols = [c for c in df.columns if c.startswith("probability_class_")]
    proba = df[prob_cols].values if prob_cols else None
    return {
        "y_true": df["true_label"].values,
        "y_pred": df["predicted_label"].values,
        "y_proba": proba,
        "df": df,
    }


def load_lp_predictions(path: Path) -> dict[str, Any] | None:
    """LP CSV: flexible matching for true/pred/prob columns."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  load error {path}: {e}")
        return None
    # prefer exact-match column names
    y_true_col = "true_label" if "true_label" in df.columns else None
    y_pred_col = "predicted_label" if "predicted_label" in df.columns else None
    prob_cols: list[str] = []
    for c in df.columns:
        cl = c.lower()
        if y_true_col is None and any(x in cl for x in
                                      ["true_label", "actual", "ground_truth"]):
            y_true_col = c
        if y_pred_col is None and any(x in cl for x in
                                      ["predicted_label", "prediction"]):
            y_pred_col = c
        if any(x in cl for x in ["probability", "prob", "score"]):
            prob_cols.append(c)
    if y_true_col is None or y_pred_col is None:
        return None
    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    proba: np.ndarray | None = None
    if prob_cols:
        if len(prob_cols) >= n_classes:
            proba = df[prob_cols[:n_classes]].values
        elif n_classes == 2 and len(prob_cols) == 1:
            v = df[prob_cols[0]].values
            proba = np.stack([1 - v, v], axis=1)
    return {"y_true": y_true, "y_pred": y_pred, "y_proba": proba, "df": df}


def find_lp_file(model_folder: Path) -> Path | None:
    """Pick the per-image-prediction CSV inside an LP model folder."""
    if not model_folder.exists():
        return None
    csvs = [f for f in model_folder.iterdir() if f.is_file() and f.suffix == ".csv"]
    if not csvs:
        return None
    lp_files = [f for f in csvs if "lp" in f.name.lower()]
    if lp_files:
        non_res = [f for f in lp_files if "results" not in f.name.lower()]
        return non_res[0] if non_res else lp_files[0]
    non_res = [f for f in csvs if "results" not in f.name.lower()]
    return non_res[0] if non_res else csvs[0]


def extract_model_name_from_folder(folder_name: str, dataset_name: str) -> str:
    name = folder_name
    for pat in (
        f"_{dataset_name}",
        f"-{dataset_name}",
        f"_{dataset_name.upper()}",
        f"-{dataset_name.upper()}",
        f"_{dataset_name.lower()}",
        f"-{dataset_name.lower()}",
    ):
        if name.endswith(pat):
            name = name[: -len(pat)]
            break
    return name.rstrip("_-")


# --------------------------------------------------------------------- #
# P-value calculators                                                   #
# --------------------------------------------------------------------- #
def pvalue_paired_ttest_zs(target_df: pd.DataFrame,
                           other_df: pd.DataFrame) -> float | None:
    """Faithful reproduction of zero-shot-plot1.py's
    `calculate_pvalue_for_metric` -- note that for ALL metrics the
    original code derived per-sample scores from accuracy, so the
    resulting p-value is the same for every metric in a row."""
    try:
        if "image_path" in target_df.columns and "image_path" in other_df.columns:
            merged = pd.merge(target_df, other_df, on="image_path",
                              suffixes=("_a", "_b"))
            if len(merged) == 0:
                return None
            s1 = (merged["true_label_a"] == merged["predicted_label_a"]).astype(int)
            s2 = (merged["true_label_b"] == merged["predicted_label_b"]).astype(int)
        else:
            if len(target_df) != len(other_df):
                return None
            s1 = (target_df["true_label"] == target_df["predicted_label"]).astype(int)
            s2 = (other_df["true_label"] == other_df["predicted_label"]).astype(int)
        _, p = stats.ttest_rel(s1, s2)
        return float(p)
    except Exception:
        return None


def pvalue_mannwhitney_one_sided(target_vals: np.ndarray,
                                 other_vals: np.ndarray) -> float | None:
    """LP-style: one-tailed Mann-Whitney U on the two bootstrap
    distributions, alternative='greater'."""
    try:
        _, p = stats.mannwhitneyu(target_vals, other_vals, alternative="greater")
        return float(p)
    except Exception:
        try:
            _, p = stats.ttest_ind(target_vals, other_vals)
            return float(p) / 2.0
        except Exception:
            return None


def format_pvalue_lp(p: float | None) -> str:
    if p is None:
        return ""
    if p >= 0.01:
        return f"{p:.4f}"
    return f"{p:.2e}"


def format_pvalue_zs(p: float | None) -> str:
    if p is None:
        return ""
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"


# --------------------------------------------------------------------- #
# Output writer                                                         #
# --------------------------------------------------------------------- #
def build_output_row(dataset: str,
                     model: str,
                     boot: dict[str, dict[str, Any]],
                     pvalues: dict[str, float | None],
                     fmt_p) -> dict[str, str]:
    row: dict[str, str] = {"Dataset": dataset, "Model": model}
    for k, col in zip(METRIC_KEYS, METRIC_COLS):
        b = boot[k]
        row[col] = ci_string(b["mean"], b["ci_lower"], b["ci_upper"])
        row[f"{col}_p_value"] = fmt_p(pvalues.get(k))
    return row


def write_output_csv(rows: list[dict[str, str]], path: Path) -> None:
    cols = ["Dataset", "Model"]
    for c in METRIC_COLS:
        cols.extend([c, f"{c}_p_value"])
    df = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


# --------------------------------------------------------------------- #
# Zero-shot pipeline                                                    #
# --------------------------------------------------------------------- #
def run_zero_shot(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted([d.name for d in data_root.iterdir()
                       if d.is_dir() and d.name != "class2label"])
    models_set: set[str] = set()
    for d in datasets:
        for f in (data_root / d).glob("*.csv"):
            models_set.add(f.stem)
    models = sorted(models_set)
    print(f"datasets ({len(datasets)}): {datasets}")
    print(f"models ({len(models)}): {models}")
    print(f"target model: {args.target_model}")

    boot_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    pred_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for dataset in datasets:
        for model in models:
            csv_path = data_root / dataset / f"{model}.csv"
            if not csv_path.exists():
                continue
            preds = load_zs_predictions(csv_path)
            if preds is None:
                continue
            pred_cache[(dataset, model)] = preds
            print(f"  {dataset} / {model}  N={len(preds['y_true'])}  "
                  f"bootstrap n={args.n_bootstrap}...")
            boot = run_bootstrap(preds["y_true"], preds["y_pred"],
                                 preds["y_proba"], args.n_bootstrap,
                                 rng_seed=args.seed)
            maybe_floor_zero_ci(boot, args.floor_zero_ci)
            boot_cache[(dataset, model)] = boot

    rows: list[dict[str, str]] = []
    for dataset in datasets:
        for model in models:
            if (dataset, model) not in boot_cache:
                continue
            boot = boot_cache[(dataset, model)]
            pvals: dict[str, float | None] = {k: None for k in METRIC_KEYS}
            if model != args.target_model and (dataset, args.target_model) in pred_cache:
                target_df = pred_cache[(dataset, args.target_model)]["df"]
                other_df = pred_cache[(dataset, model)]["df"]
                if args.pvalue_method == "paired_ttest":
                    p = pvalue_paired_ttest_zs(target_df, other_df)
                    pvals = {k: p for k in METRIC_KEYS}
                else:
                    tboot = boot_cache[(dataset, args.target_model)]
                    for k in METRIC_KEYS:
                        pvals[k] = pvalue_mannwhitney_one_sided(
                            tboot[k]["values"], boot[k]["values"]
                        )
            rows.append(build_output_row(dataset, model, boot, pvals,
                                         format_pvalue_zs))

    out_csv = output_dir / "model_comparison_results_comprehensive.csv"
    write_output_csv(rows, out_csv)

    # summary CSV (mirrors the original) ------------------------------- #
    summary_rows: list[dict[str, str]] = []
    for dataset in datasets:
        n_classes = None
        for model in models:
            if (dataset, model) in pred_cache:
                n_classes = len(np.unique(pred_cache[(dataset, model)]["y_true"]))
                break
        primary_key = "auroc" if n_classes == 2 else "accuracy"
        for model in models:
            if (dataset, model) not in boot_cache:
                continue
            b = boot_cache[(dataset, model)][primary_key]
            summary_rows.append({
                "Dataset": dataset,
                "Model": model,
                "Primary_Metric": primary_key.upper(),
                "Score": ci_string(b["mean"], b["ci_lower"], b["ci_upper"]),
            })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "model_comparison_results_summary.csv", index=False)
    print(f"  wrote {output_dir / 'model_comparison_results_summary.csv'}")


# --------------------------------------------------------------------- #
# LP pipeline                                                           #
# --------------------------------------------------------------------- #
def discover_lp_tasks(data_root: Path,
                      fractions: list[int],
                      flat_examples: bool) -> list[dict[str, Any]]:
    """Walk either layout:

    Original:  data_root / percent_data_{0.1|0.3|0.5|1} / <dataset> / <model_folder>
    Flat:      data_root / <dataset>_{10|30|50|100}pct / <model_folder>
    """
    frac_to_dirname = {10: "0.1", 30: "0.3", 50: "0.5", 100: "1"}
    tasks: list[dict[str, Any]] = []

    # Original layout
    for f in fractions:
        sub = data_root / f"percent_data_{frac_to_dirname[f]}"
        if not sub.exists():
            continue
        for dataset_dir in sub.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name == "summary_results":
                continue
            ds = dataset_dir.name
            for mdir in dataset_dir.iterdir():
                if not mdir.is_dir():
                    continue
                lp = find_lp_file(mdir)
                if lp is None:
                    continue
                tasks.append({
                    "fraction": f,
                    "dataset": ds,
                    "model_folder": mdir.name,
                    "model_name": extract_model_name_from_folder(mdir.name, ds),
                    "lp_file": lp,
                })

    # Flat-examples layout
    if flat_examples or not tasks:
        for child in data_root.iterdir():
            if not child.is_dir() or child.name == "class2label":
                continue
            name = child.name
            for f in fractions:
                suffix = f"_{f}pct"
                if name.endswith(suffix):
                    ds = name[: -len(suffix)]
                    for mdir in child.iterdir():
                        if not mdir.is_dir():
                            continue
                        lp = find_lp_file(mdir)
                        if lp is None:
                            continue
                        tasks.append({
                            "fraction": f,
                            "dataset": ds,
                            "model_folder": mdir.name,
                            "model_name": extract_model_name_from_folder(mdir.name, ds),
                            "lp_file": lp,
                        })
                    break
    return tasks


def run_lp(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = args.fractions
    tasks = discover_lp_tasks(data_root, fractions, args.lp_flat_examples)
    print(f"found {len(tasks)} LP tasks")
    for t in tasks[:5]:
        print(f"  e.g. frac={t['fraction']}  {t['dataset']} / "
              f"{t['model_folder']}  -> {t['model_name']}")

    if args.interactive and tasks:
        resp = input(
            f"continue with {len(tasks)} tasks? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborted by user")
            return

    by_frac: dict[int, list[dict[str, Any]]] = {}
    for t in tasks:
        by_frac.setdefault(t["fraction"], []).append(t)

    # Reproduce the original LP seeding behaviour: the script called
    # np.random.seed(42) once at start of main(), then each
    # bootstrap_metrics_parallel call did
    #   base_seed = np.random.randint(0, 10000)
    # so each task's bootstrap is reproducible given a single global
    # seed.  We mirror that by drawing per-task seeds from a Generator.
    rng = np.random.default_rng(args.seed)

    for frac in sorted(by_frac.keys()):
        rows: list[dict[str, str]] = []
        boot_per_dataset: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

        print(f"\n[fraction {frac}%] {len(by_frac[frac])} tasks")
        for t in by_frac[frac]:
            preds = load_lp_predictions(t["lp_file"])
            if preds is None:
                print(f"  skip {t['lp_file']} (could not parse)")
                continue
            task_seed = int(rng.integers(0, 10_000))
            print(f"  {t['dataset']} / {t['model_name']}  "
                  f"N={len(preds['y_true'])}  seed={task_seed}")
            boot = run_bootstrap(preds["y_true"], preds["y_pred"],
                                 preds["y_proba"], args.n_bootstrap,
                                 rng_seed=task_seed)
            maybe_floor_zero_ci(boot, args.floor_zero_ci)
            boot_per_dataset.setdefault(t["dataset"], {})[t["model_name"]] = boot

        for ds, model_to_boot in boot_per_dataset.items():
            target_boot = model_to_boot.get(args.target_model)
            for mname, boot in model_to_boot.items():
                pvals: dict[str, float | None] = {k: None for k in METRIC_KEYS}
                if target_boot is not None and mname != args.target_model:
                    for k in METRIC_KEYS:
                        if args.pvalue_method == "mannwhitney":
                            pvals[k] = pvalue_mannwhitney_one_sided(
                                target_boot[k]["values"], boot[k]["values"]
                            )
                        else:
                            try:
                                _, p = stats.ttest_rel(
                                    target_boot[k]["values"], boot[k]["values"]
                                )
                                pvals[k] = float(p)
                            except Exception:
                                pvals[k] = None
                rows.append(build_output_row(ds, mname, boot, pvals,
                                             format_pvalue_lp))

        out_csv = output_dir / f"lp_results_{frac}percent_bootstrap.csv"
        write_output_csv(rows, out_csv)


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
def parse_fractions(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unified bootstrap-CI table builder for the "
                    "zero-shot (Task 1) and linear-probe (Task 3) "
                    "benchmarking tables.")
    p.add_argument("--task", required=True, choices=["zero_shot", "lp"],
                   help="which manuscript task to reproduce")
    p.add_argument("--data-root", required=True,
                   help="root directory containing per-image prediction CSVs")
    p.add_argument("--output-dir", required=True,
                   help="output directory for CSV tables")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="number of bootstrap reps (default: 1000)")
    p.add_argument("--target-model", default=None,
                   help="target model name (default: PanDerm-large-w-PubMed-256 "
                        "for zero_shot, DermFM-Zero for lp)")
    p.add_argument("--pvalue-method", default=None,
                   choices=["paired_ttest", "mannwhitney"],
                   help="how to compute p-values (default: paired_ttest for "
                        "zero_shot, mannwhitney for lp)")
    p.add_argument("--floor-zero-ci", dest="floor_zero_ci",
                   action="store_true", default=None,
                   help="enable the 'floor 0.0 CI to 0.1*mean' hack")
    p.add_argument("--no-floor-zero-ci", dest="floor_zero_ci",
                   action="store_false",
                   help="disable the floor-zero-CI hack")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default: 42)")
    p.add_argument("--fractions", type=parse_fractions,
                   default=[10, 30, 50, 100],
                   help="LP-only: comma-separated percent fractions to process "
                        "(default: 10,30,50,100)")
    p.add_argument("--lp-flat-examples", action="store_true",
                   help="LP-only: enable flat <dataset>_<pct>pct/<model> layout")
    p.add_argument("--interactive", action="store_true",
                   help="LP-only: prompt for confirmation before bulk run")
    args = p.parse_args()

    # apply per-task defaults
    if args.task == "zero_shot":
        if args.target_model is None:
            args.target_model = "DermFM-Zero"
        if args.pvalue_method is None:
            args.pvalue_method = "paired_ttest"
        if args.floor_zero_ci is None:
            args.floor_zero_ci = True
    else:
        if args.target_model is None:
            args.target_model = "DermFM-Zero"
        if args.pvalue_method is None:
            args.pvalue_method = "mannwhitney"
        if args.floor_zero_ci is None:
            args.floor_zero_ci = False
        # always also walk the flat layout (covers scaffold examples)
        args.lp_flat_examples = True

    t0 = time.time()
    if args.task == "zero_shot":
        run_zero_shot(args)
    else:
        run_lp(args)
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
