"""
RS1 — BCD vs Agent Inter-Rater Reliability
================================================================

Quantifies agreement between the LLM/agent grading pipeline (AI-only) and the
BCD-reviewed corrected grades. For every (case, reader) paired observation, we
compare the four structured grading dimensions:

  - Unaided_Dx_Score     (ordinal 1..5)
  - Assisted_Dx_Score    (ordinal 1..5)
  - Unaided_Mgmt_Grade   (ordinal 4..1; Perfect / Adequate / Inadequate-harmless / Inadequate-dangerous)
  - Assisted_Mgmt_Grade  (ordinal 4..1)

Metrics per dimension:
  * Exact-match rate
  * Cohen's quadratic-weighted kappa
  * Bootstrap 95% CI (1000 resamples, 2-sided percentile)
  * Correction proportion (% of obs where AI != BCD)

Also stratified by cohort (CN / EN) and condition (Unaided / Assisted), and
combined across all four dimensions.

Outputs:
  real_output/bcd_agent_irr.json
  real_output/bcd_agent_irr_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# File-pair filenames relative to <grading_dir>/batch{1,2}/... — the actual
# grading_dir is resolved from CLI (see main()).
FILE_PAIR_SPECS: List[Dict[str, str]] = [
    {
        "label": "CN_batch1",
        "cohort": "CN",
        "ai":  "batch1/PanDerm2_grade_china.xlsx",
        "bcd": "batch1/PanDerm2_grade_china_rev_cc.xlsx",
    },
    {
        "label": "EN_6GP",
        "cohort": "EN",
        "ai":  "batch2/PanDerm2_AI_Scored_6GP_v3.xlsx",
        "bcd": "batch2/PanDerm2_AI_Scored_6GP_v3_rev_cc.xlsx",
    },
    {
        "label": "EN_others",
        "cohort": "EN",
        "ai":  "batch2/PanDerm2_AI_Scored_others_v3 (1).xlsx",
        "bcd": "batch2/PanDerm2_AI_Scored_others_v3_rev_cc.xlsx",
    },
]

# Mgmt ordinal mapping (higher = better)
MGMT_MAP: Dict[str, int] = {
    "Perfect": 4,
    "Adequate": 3,
    "Inadequate but harmless": 2,
    "Inadequate and dangerous": 1,
}

DIMENSIONS = [
    ("Unaided_Dx_Score",    "dx",   "unaided",  "ordinal_dx"),
    ("Assisted_Dx_Score",   "dx",   "assisted", "ordinal_dx"),
    ("Unaided_Mgmt_Grade",  "mgmt", "unaided",  "ordinal_mgmt"),
    ("Assisted_Mgmt_Grade", "mgmt", "assisted", "ordinal_mgmt"),
]

KEY_COLS = ["Case ID", "Responder_ID"]

BOOTSTRAP_N = 1000
RNG_SEED = 20260529


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce(values: pd.Series, kind: str) -> pd.Series:
    """Convert a column to its integer ordinal representation."""
    if kind == "ordinal_dx":
        return pd.to_numeric(values, errors="coerce")
    if kind == "ordinal_mgmt":
        return values.map(MGMT_MAP)
    raise ValueError(kind)


def load_pair(ai_path: str, bcd_path: str) -> pd.DataFrame:
    """Inner-join AI and BCD files on (Case ID, Responder_ID)."""
    ai = pd.read_excel(ai_path, engine="openpyxl")
    bcd = pd.read_excel(bcd_path, engine="openpyxl")

    keep_cols = KEY_COLS + [d[0] for d in DIMENSIONS]
    ai_slim  = ai[keep_cols].copy()
    bcd_slim = bcd[keep_cols].copy()

    merged = ai_slim.merge(
        bcd_slim,
        on=KEY_COLS,
        how="inner",
        suffixes=("_ai", "_bcd"),
        validate="one_to_one",
    )
    return merged, len(ai), len(bcd)


def bootstrap_kappa_ci(
    ai: np.ndarray,
    bcd: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = RNG_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(ai)
    if n < 2:
        return (float("nan"), float("nan"))
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = ai[idx]
        b = bcd[idx]
        if len(np.unique(a)) < 2 and len(np.unique(b)) < 2 and np.all(a == b):
            samples[i] = 1.0
            continue
        try:
            samples[i] = cohen_kappa_score(a, b, weights="quadratic")
        except Exception:
            samples[i] = np.nan
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def agreement_block(ai: np.ndarray, bcd: np.ndarray) -> Dict[str, float]:
    mask = ~(pd.isna(ai) | pd.isna(bcd))
    a = np.asarray(ai)[mask].astype(int)
    b = np.asarray(bcd)[mask].astype(int)
    n = int(len(a))
    if n == 0:
        return {
            "n": 0,
            "exact_match_rate": float("nan"),
            "weighted_kappa": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "correction_proportion": float("nan"),
            "n_changed": 0,
        }
    exact = float(np.mean(a == b))
    changed = int(np.sum(a != b))
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        kappa = 1.0 if np.all(a == b) else 0.0
    else:
        kappa = float(cohen_kappa_score(a, b, weights="quadratic"))
    lo, hi = bootstrap_kappa_ci(a, b)
    return {
        "n": n,
        "exact_match_rate": exact,
        "weighted_kappa": kappa,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "correction_proportion": float(changed) / n,
        "n_changed": changed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="RS1 BCD vs Agent inter-rater reliability.")
    ap.add_argument(
        "--grading_dir",
        type=Path,
        default=here / "real_data" / "grading_files",
        help="Directory containing batch1/ and batch2/ AI vs BCD XLSX files.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=here / "real_output",
        help="Directory to write bcd_agent_irr.json and bcd_agent_irr_summary.md.",
    )
    ap.add_argument(
        "--real",
        action="store_true",
        help="Required: run on real graded data. Without --real the script exits cleanly.",
    )
    args = ap.parse_args()

    if not args.real:
        print(
            "This analysis requires real graded data; "
            "pass --real and --grading_dir to run."
        )
        sys.exit(0)

    grading_dir: Path = args.grading_dir
    out_dir: Path = args.out_dir

    if not grading_dir.exists():
        sys.exit(
            f"Grading directory not found: {grading_dir}. "
            "Real data are not shipped publicly; obtain on request."
        )

    # Resolve absolute file pair paths and check existence.
    FILE_PAIRS: List[Dict[str, str]] = []
    for spec in FILE_PAIR_SPECS:
        ai_path = grading_dir / spec["ai"]
        bcd_path = grading_dir / spec["bcd"]
        for p in (ai_path, bcd_path):
            if not p.exists():
                sys.exit(
                    f"Required grading file missing: {p}. "
                    "Real data are not shipped publicly; obtain on request."
                )
        FILE_PAIRS.append({
            "label": spec["label"],
            "cohort": spec["cohort"],
            "ai":  str(ai_path),
            "bcd": str(bcd_path),
        })

    OUT_JSON = out_dir / "bcd_agent_irr.json"
    OUT_MD   = out_dir / "bcd_agent_irr_summary.md"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("RS1 BCD-Agent IRR")
    print("=" * 78)

    # Diff column structure of one AI vs BCD file
    ai_demo  = pd.read_excel(FILE_PAIRS[0]["ai"],  engine="openpyxl")
    bcd_demo = pd.read_excel(FILE_PAIRS[0]["bcd"], engine="openpyxl")
    cols_ai, cols_bcd = list(ai_demo.columns), list(bcd_demo.columns)
    only_ai = sorted(set(cols_ai) - set(cols_bcd))
    only_bcd = sorted(set(cols_bcd) - set(cols_ai))
    print(f"\nColumn diff (CN_batch1):")
    print(f"  AI-only columns:  {only_ai}")
    print(f"  BCD-only columns: {only_bcd}")
    print(f"  AI n_cols={len(cols_ai)}  BCD n_cols={len(cols_bcd)}")

    # Aggregate per-pair merges
    per_pair_meta = {}
    merged_rows: List[pd.DataFrame] = []
    for pair in FILE_PAIRS:
        merged, n_ai, n_bcd = load_pair(pair["ai"], pair["bcd"])
        merged["__cohort__"] = pair["cohort"]
        merged["__source__"] = pair["label"]
        merged_rows.append(merged)
        per_pair_meta[pair["label"]] = {
            "cohort": pair["cohort"],
            "n_ai_rows": n_ai,
            "n_bcd_rows": n_bcd,
            "n_aligned": int(len(merged)),
            "n_readers_aligned": int(merged["Responder_ID"].nunique()),
            "n_cases_aligned":   int(merged["Case ID"].nunique()),
        }
        print(
            f"\n[{pair['label']}] AI={n_ai}  BCD={n_bcd}  "
            f"aligned={len(merged)}  readers={merged['Responder_ID'].nunique()}  "
            f"cases={merged['Case ID'].nunique()}"
        )

    full = pd.concat(merged_rows, ignore_index=True)
    total_aligned = int(len(full))
    total_readers = int(full["Responder_ID"].nunique())
    print(f"\nTOTAL aligned paired observations: {total_aligned}")
    print(f"TOTAL unique readers (Responder_ID): {total_readers}")

    # -----------------------------------------------------------------------
    # Per-dimension metrics
    # -----------------------------------------------------------------------
    results: Dict[str, Dict] = {
        "meta": {
            "bootstrap_n": BOOTSTRAP_N,
            "rng_seed": RNG_SEED,
            "ci_method": "percentile_2sided_95",
            "kappa_weights": "quadratic",
            "mgmt_ordinal_mapping": MGMT_MAP,
            "file_pairs": per_pair_meta,
            "total_aligned": total_aligned,
            "total_unique_readers": total_readers,
        },
        "per_dimension": {},
        "by_cohort": {},
        "by_condition": {},
        "by_dim_x_cohort": {},
        "overall": {},
    }

    # Build a long-format frame for the "overall combined" view (all 4 dims stacked)
    long_rows: List[pd.DataFrame] = []

    for col, family, condition, kind in DIMENSIONS:
        ai_col = f"{col}_ai"
        bcd_col = f"{col}_bcd"
        a = _coerce(full[ai_col], kind).to_numpy()
        b = _coerce(full[bcd_col], kind).to_numpy()
        block = agreement_block(a, b)
        block.update({"family": family, "condition": condition, "ordinal_kind": kind})
        results["per_dimension"][col] = block
        print(
            f"  [{col:>22s}] n={block['n']:4d}  "
            f"exact={block['exact_match_rate']:.4f}  "
            f"kappa_w={block['weighted_kappa']:.4f}  "
            f"95%CI=[{block['ci95_lo']:.4f},{block['ci95_hi']:.4f}]  "
            f"corr={block['correction_proportion']:.4f} (n_changed={block['n_changed']})"
        )

        df_long = pd.DataFrame({
            "ai":        a,
            "bcd":       b,
            "cohort":    full["__cohort__"].to_numpy(),
            "dimension": col,
            "family":    family,
            "condition": condition,
        })
        long_rows.append(df_long)

        # by cohort x dimension
        for cohort in ["CN", "EN"]:
            sub = df_long[df_long["cohort"] == cohort]
            block_c = agreement_block(sub["ai"].to_numpy(), sub["bcd"].to_numpy())
            results["by_dim_x_cohort"].setdefault(col, {})[cohort] = block_c

    long_all = pd.concat(long_rows, ignore_index=True)

    # -----------------------------------------------------------------------
    # By cohort (collapsing all 4 dimensions)
    # -----------------------------------------------------------------------
    for cohort in ["CN", "EN"]:
        sub = long_all[long_all["cohort"] == cohort]
        results["by_cohort"][cohort] = agreement_block(
            sub["ai"].to_numpy(), sub["bcd"].to_numpy()
        )

    # -----------------------------------------------------------------------
    # By condition (Unaided vs Assisted, both families pooled)
    # -----------------------------------------------------------------------
    for condition in ["unaided", "assisted"]:
        sub = long_all[long_all["condition"] == condition]
        results["by_condition"][condition] = agreement_block(
            sub["ai"].to_numpy(), sub["bcd"].to_numpy()
        )

    # By family (Dx vs Mgmt)
    results["by_family"] = {}
    for family in ["dx", "mgmt"]:
        sub = long_all[long_all["family"] == family]
        results["by_family"][family] = agreement_block(
            sub["ai"].to_numpy(), sub["bcd"].to_numpy()
        )

    # -----------------------------------------------------------------------
    # Overall combined (all dims stacked). Note: Dx (1..5) and Mgmt (1..4) live
    # on different ordinal scales, so the pooled kappa is reported alongside,
    # but the by_family breakdown is the cleaner read.
    # -----------------------------------------------------------------------
    results["overall"] = agreement_block(
        long_all["ai"].to_numpy(),
        long_all["bcd"].to_numpy(),
    )
    results["overall"]["n_dimensions"] = len(DIMENSIONS)
    print("\n" + "-" * 78)
    print(f"OVERALL (pooled across 4 dims): n={results['overall']['n']}  "
          f"exact={results['overall']['exact_match_rate']:.4f}  "
          f"kappa_w={results['overall']['weighted_kappa']:.4f}  "
          f"95%CI=[{results['overall']['ci95_lo']:.4f},"
          f"{results['overall']['ci95_hi']:.4f}]  "
          f"corr={results['overall']['correction_proportion']:.4f}")

    # -----------------------------------------------------------------------
    # JSON dump
    # -----------------------------------------------------------------------
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote: {OUT_JSON}")

    # -----------------------------------------------------------------------
    # Markdown summary
    # -----------------------------------------------------------------------
    md: List[str] = []
    md.append("# RS1 — BCD vs Agent Inter-Rater Reliability\n")
    md.append("Quadratic-weighted Cohen's kappa "
              "with 1000-resample percentile 95% CI.\n")
    md.append("## Input alignment\n")
    md.append("| File pair | Cohort | AI rows | BCD rows | Aligned | Readers | Cases |")
    md.append("|---|---|---|---|---|---|---|")
    for label, meta in per_pair_meta.items():
        md.append(
            f"| {label} | {meta['cohort']} | {meta['n_ai_rows']} | "
            f"{meta['n_bcd_rows']} | {meta['n_aligned']} | "
            f"{meta['n_readers_aligned']} | {meta['n_cases_aligned']} |"
        )
    md.append(
        f"| **TOTAL** | -- | -- | -- | **{total_aligned}** | "
        f"{total_readers} | -- |\n"
    )

    md.append("## Per-dimension agreement\n")
    md.append("| Dimension | n | Exact match | Weighted kappa | 95% CI | Correction prop. | n changed |")
    md.append("|---|---|---|---|---|---|---|")
    for col, _, _, _ in DIMENSIONS:
        b = results["per_dimension"][col]
        md.append(
            f"| `{col}` | {b['n']} | {b['exact_match_rate']:.4f} | "
            f"{b['weighted_kappa']:.4f} | "
            f"[{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}] | "
            f"{b['correction_proportion']:.4f} | {b['n_changed']} |"
        )

    md.append("\n## By cohort (pooled across 4 dimensions)\n")
    md.append("| Cohort | n | Exact | Weighted kappa | 95% CI | Correction prop. |")
    md.append("|---|---|---|---|---|---|")
    for cohort, b in results["by_cohort"].items():
        md.append(
            f"| {cohort} | {b['n']} | {b['exact_match_rate']:.4f} | "
            f"{b['weighted_kappa']:.4f} | "
            f"[{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}] | "
            f"{b['correction_proportion']:.4f} |"
        )

    md.append("\n## By condition (pooled Dx + Mgmt)\n")
    md.append("| Condition | n | Exact | Weighted kappa | 95% CI | Correction prop. |")
    md.append("|---|---|---|---|---|---|")
    for condition, b in results["by_condition"].items():
        md.append(
            f"| {condition} | {b['n']} | {b['exact_match_rate']:.4f} | "
            f"{b['weighted_kappa']:.4f} | "
            f"[{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}] | "
            f"{b['correction_proportion']:.4f} |"
        )

    md.append("\n## By family\n")
    md.append("| Family | n | Exact | Weighted kappa | 95% CI | Correction prop. |")
    md.append("|---|---|---|---|---|---|")
    for family, b in results["by_family"].items():
        md.append(
            f"| {family} | {b['n']} | {b['exact_match_rate']:.4f} | "
            f"{b['weighted_kappa']:.4f} | "
            f"[{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}] | "
            f"{b['correction_proportion']:.4f} |"
        )

    md.append("\n## Per-dimension stratified by cohort\n")
    md.append("| Dimension | Cohort | n | Exact | Weighted kappa | 95% CI | Correction prop. |")
    md.append("|---|---|---|---|---|---|---|")
    for col, _, _, _ in DIMENSIONS:
        for cohort in ["CN", "EN"]:
            b = results["by_dim_x_cohort"][col][cohort]
            md.append(
                f"| `{col}` | {cohort} | {b['n']} | "
                f"{b['exact_match_rate']:.4f} | "
                f"{b['weighted_kappa']:.4f} | "
                f"[{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}] | "
                f"{b['correction_proportion']:.4f} |"
            )

    md.append("\n## Overall (all 4 dimensions stacked)\n")
    b = results["overall"]
    md.append(
        f"- n = **{b['n']}** grading instances "
        f"(= {total_aligned} paired observations x {len(DIMENSIONS)} dimensions, "
        "minus missing values).\n"
        f"- Exact-match rate: **{b['exact_match_rate']:.4f}**\n"
        f"- Quadratic-weighted kappa: **{b['weighted_kappa']:.4f}** "
        f"(95% CI [{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}])\n"
        f"- Correction proportion (AI != BCD): **{b['correction_proportion']:.4f}** "
        f"(n_changed = {b['n_changed']})\n"
    )
    md.append("\n_Note: Dx scores (1..5) and Mgmt grades (mapped to 1..4) live "
              "on different ordinal scales, so the pooled kappa here is "
              "informative but the per-dimension and by-family breakdowns are "
              "the cleaner reads._\n")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
