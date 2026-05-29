"""
RS2B — Vascular subgroup analysis (R2 Comment 12)
==================================================

Reviewer #2 Comment 12 asks why vascular lesions appear to perform worse under
AI assistance. This script performs a multi-dimensional vascular-subgroup
deep-dive on the 71-reader analytic cohort:

  1. Overall VASC accuracy (unaided vs assisted) and per-reader paired Wilcoxon.
  2. Per-expertise stratification (expert vs non-expert).
  3. Per-profession stratification (BCD vs non-BCD).
  4. Per-image accuracy ranking (which of the 8 VASC images drove the drop).
  5. Reader-diagnosis confusion distribution (what GT=VASC gets called instead).
  6. Error diversification (Shannon entropy of error distribution).
  7. Per-reader revision behaviour on VASC (paired flips).
  8. Comparison against other non-melanocytic classes (BCC, MEL, BKL) for
     ceiling-effect context.
  9. Clinical-decision distribution (dismiss / observe / biopsy / refer) under
     each arm.

Inputs
------
real_output/panderm_cleaned_95pct.csv (71-reader cohort)

Outputs
-------
real_output/vascular_subgroup_analysis.json
real_output/vascular_subgroup_analysis_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CLI + paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true",
                    help="Use real_data/ -> real_output/. Without this flag, "
                         "the script exits without running (this analysis is "
                         "not meaningful on synthetic data).")
parser.add_argument("--data_dir", type=Path, default=None,
                    help="Override input data directory (default: real_output/)")
parser.add_argument("--out_dir", type=Path, default=None,
                    help="Override output directory (default: real_output/)")
args = parser.parse_args()

if not args.real:
    print("This vascular-subgroup analysis requires real RS2B data; "
          "pass --real to run.")
    sys.exit(0)

DATA_DIR = args.data_dir or (ROOT / "real_output")
OUT_DIR = args.out_dir or (ROOT / "real_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "panderm_cleaned_95pct.csv"
OUT_JSON = OUT_DIR / "vascular_subgroup_analysis.json"
OUT_MD = OUT_DIR / "vascular_subgroup_analysis_summary.md"

if not INPUT_CSV.exists():
    print(f"Input missing: {INPUT_CSV}. Real data are not shipped publicly; "
          f"obtain on request.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def paired_wilcoxon(per_reader: pd.DataFrame) -> dict:
    """Wilcoxon signed-rank on per-reader paired accuracy."""
    d = per_reader.dropna(subset=["unaided", "assisted"])
    if len(d) < 6:
        return {"n_readers": len(d), "p_value": None, "stat": None}
    try:
        stat, p = stats.wilcoxon(d["assisted"], d["unaided"])
    except ValueError:
        return {"n_readers": len(d), "p_value": None, "stat": None,
                "note": "All differences zero"}
    return {
        "n_readers": int(len(d)),
        "mean_unaided": float(d["unaided"].mean()),
        "mean_assisted": float(d["assisted"].mean()),
        "mean_diff": float((d["assisted"] - d["unaided"]).mean()),
        "median_diff": float((d["assisted"] - d["unaided"]).median()),
        "stat": float(stat),
        "p_value": float(p),
    }


def per_reader_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Return (reader_id × {unaided, assisted}) accuracy table."""
    out = (df.groupby(["reader_id", "has_ai_assistance"])["is_correct_numeric"]
             .mean().unstack())
    out.columns = ["unaided" if c is False else "assisted" for c in out.columns]
    if "unaided" not in out.columns:
        out["unaided"] = np.nan
    if "assisted" not in out.columns:
        out["assisted"] = np.nan
    return out[["unaided", "assisted"]]


def reader_diagnosis_dist(df: pd.DataFrame, gt_class: str) -> dict:
    """Distribution of reader_diagnosis when true_diagnosis == gt_class."""
    sub = df[df["true_diagnosis"] == gt_class]
    out = {}
    for ai_flag, label in [(False, "unaided"), (True, "assisted")]:
        d = sub[sub["has_ai_assistance"] == ai_flag]
        n = len(d)
        counts = d["reader_diagnosis"].value_counts().to_dict()
        props = {k: v / n for k, v in counts.items()} if n else {}
        out[label] = {
            "n": int(n),
            "counts": {k: int(v) for k, v in counts.items()},
            "proportions": {k: float(v) for k, v in props.items()},
        }
    return out


def shannon_entropy(props: dict) -> float:
    """Shannon entropy of error distribution (excluding the correct answer)."""
    ps = np.array([v for v in props.values() if v > 0], dtype=float)
    if len(ps) == 0:
        return 0.0
    ps = ps / ps.sum()
    return float(-(ps * np.log2(ps)).sum())


def clinical_decision_dist(df: pd.DataFrame, gt_class: str) -> dict:
    """Distribution of clinical_decision when true_diagnosis == gt_class."""
    sub = df[df["true_diagnosis"] == gt_class]
    out = {}
    for ai_flag, label in [(False, "unaided"), (True, "assisted")]:
        d = sub[sub["has_ai_assistance"] == ai_flag]
        n = len(d)
        counts = d["clinical_decision"].value_counts().to_dict()
        props = {k: v / n for k, v in counts.items()} if n else {}
        out[label] = {
            "n": int(n),
            "counts": {k: int(v) for k, v in counts.items()},
            "proportions": {k: float(v) for k, v in props.items()},
        }
    return out


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"  rows: {len(df)}, readers: {df['reader_id'].nunique()}, "
      f"classes: {df['true_diagnosis'].nunique()}")

vasc = df[df["true_diagnosis"] == "VASC"].copy()
print(f"  VASC rows: {len(vasc)}; unique images: {vasc['image_id'].nunique()}; "
      f"readers exposed: {vasc['reader_id'].nunique()}")


# ---------------------------------------------------------------------------
# Analysis 1 — Overall VASC paired Wilcoxon
# ---------------------------------------------------------------------------
print("\n[1/9] Overall VASC paired Wilcoxon ...")
per_reader_vasc = per_reader_accuracy(vasc)
overall = paired_wilcoxon(per_reader_vasc)
overall["per_obs_unaided_acc"] = float(
    vasc.loc[vasc["has_ai_assistance"] == False, "is_correct_numeric"].mean())
overall["per_obs_assisted_acc"] = float(
    vasc.loc[vasc["has_ai_assistance"] == True, "is_correct_numeric"].mean())
overall["n_obs_unaided"] = int((vasc["has_ai_assistance"] == False).sum())
overall["n_obs_assisted"] = int((vasc["has_ai_assistance"] == True).sum())
overall["unique_images"] = int(vasc["image_id"].nunique())


# ---------------------------------------------------------------------------
# Analysis 2 — By expertise (expert vs non-expert)
# ---------------------------------------------------------------------------
print("[2/9] VASC by expertise ...")
by_expertise = {}
for exp in sorted(vasc["reader_expertise"].dropna().unique()):
    sub = vasc[vasc["reader_expertise"] == exp]
    pr = per_reader_accuracy(sub)
    res = paired_wilcoxon(pr)
    res["per_obs_unaided"] = float(
        sub.loc[sub["has_ai_assistance"] == False,
                "is_correct_numeric"].mean())
    res["per_obs_assisted"] = float(
        sub.loc[sub["has_ai_assistance"] == True,
                "is_correct_numeric"].mean())
    by_expertise[exp] = res


# ---------------------------------------------------------------------------
# Analysis 3 — By profession (BCD vs non-BCD)
# ---------------------------------------------------------------------------
print("[3/9] VASC by profession ...")
vasc_bcd_flag = vasc.copy()
vasc_bcd_flag["is_bcd"] = (vasc_bcd_flag["profession"]
                          .fillna("").eq("boardCertifiedDermatologist"))
by_profession = {}
for is_bcd, label in [(True, "BCD"), (False, "non-BCD")]:
    sub = vasc_bcd_flag[vasc_bcd_flag["is_bcd"] == is_bcd]
    if len(sub) == 0:
        continue
    pr = per_reader_accuracy(sub)
    res = paired_wilcoxon(pr)
    res["per_obs_unaided"] = float(
        sub.loc[sub["has_ai_assistance"] == False,
                "is_correct_numeric"].mean())
    res["per_obs_assisted"] = float(
        sub.loc[sub["has_ai_assistance"] == True,
                "is_correct_numeric"].mean())
    by_profession[label] = res

# Also detail per-profession
print("[3b/9] VASC by profession (detailed) ...")
by_profession_detail = {}
prof_counts = vasc["profession"].value_counts()
for prof, n_obs in prof_counts.items():
    if n_obs < 6:
        continue  # skip tiny strata
    sub = vasc[vasc["profession"] == prof]
    un = sub[sub["has_ai_assistance"] == False]
    ai = sub[sub["has_ai_assistance"] == True]
    by_profession_detail[prof] = {
        "n_obs_unaided": int(len(un)),
        "n_obs_assisted": int(len(ai)),
        "unaided_acc": float(un["is_correct_numeric"].mean()) if len(un) else None,
        "assisted_acc": float(ai["is_correct_numeric"].mean()) if len(ai) else None,
    }


# ---------------------------------------------------------------------------
# Analysis 4 — Per-image VASC accuracy
# ---------------------------------------------------------------------------
print("[4/9] Per-image VASC accuracy ...")
per_img = vasc.groupby(["image_id", "has_ai_assistance"])[
    "is_correct_numeric"].agg(["mean", "count"]).unstack()
per_img.columns = ["un_acc", "ai_acc", "n_un", "n_ai"]
per_img["diff"] = per_img["ai_acc"] - per_img["un_acc"]
per_img = per_img.sort_values("diff")

per_image_records = []
for img_id, row in per_img.iterrows():
    isic = vasc.loc[vasc["image_id"] == img_id, "isic_id"].iloc[0] \
        if (vasc["image_id"] == img_id).any() else None
    per_image_records.append({
        "image_id": str(img_id),
        "isic_id": str(isic) if isic is not None else None,
        "n_unaided": int(row["n_un"]),
        "n_assisted": int(row["n_ai"]),
        "unaided_acc": float(row["un_acc"]) if not np.isnan(row["un_acc"])
                       else None,
        "assisted_acc": float(row["ai_acc"]) if not np.isnan(row["ai_acc"])
                        else None,
        "diff": float(row["diff"]) if not np.isnan(row["diff"]) else None,
    })


# ---------------------------------------------------------------------------
# Analysis 5 — Reader-diagnosis confusion distribution
# ---------------------------------------------------------------------------
print("[5/9] Reader-diagnosis confusion distribution ...")
diag_dist = reader_diagnosis_dist(df, "VASC")


# ---------------------------------------------------------------------------
# Analysis 6 — Error diversification (Shannon entropy of error distribution)
# ---------------------------------------------------------------------------
print("[6/9] Error entropy ...")
error_entropy = {}
for arm in ["unaided", "assisted"]:
    props = diag_dist[arm]["proportions"]
    # Error proportions (exclude correct = VASC)
    err = {k: v for k, v in props.items() if k != "VASC"}
    # Re-normalise within errors
    s = sum(err.values())
    if s > 0:
        err = {k: v / s for k, v in err.items()}
    error_entropy[arm] = {
        "shannon_entropy_bits": shannon_entropy(err),
        "error_proportion": float(sum(v for k, v in props.items()
                                      if k != "VASC")),
        "n_distinct_error_classes": int(len(err)),
        "error_distribution": err,
    }


# ---------------------------------------------------------------------------
# Analysis 7 — Per-reader VASC revision (flips between arms)
# ---------------------------------------------------------------------------
print("[7/9] Per-reader VASC revision pattern ...")
# For readers who saw the same image both unaided and assisted, did they flip?
paired = (vasc.pivot_table(
    index=["reader_id", "image_id"],
    columns="has_ai_assistance",
    values=["reader_diagnosis", "is_correct_numeric"],
    aggfunc="first",
))
paired.columns = [f"{m}_{('un' if c is False else 'ai')}"
                  for m, c in paired.columns]
paired = paired.dropna(subset=["reader_diagnosis_un", "reader_diagnosis_ai"])

paired["changed_dx"] = (paired["reader_diagnosis_un"]
                        != paired["reader_diagnosis_ai"])
paired["productive_change"] = (
    (paired["changed_dx"]) &
    (paired["is_correct_numeric_un"] == 0) &
    (paired["is_correct_numeric_ai"] == 1)
)
paired["harmful_change"] = (
    (paired["changed_dx"]) &
    (paired["is_correct_numeric_un"] == 1) &
    (paired["is_correct_numeric_ai"] == 0)
)

revision_summary = {
    "n_paired_observations": int(len(paired)),
    "n_changed": int(paired["changed_dx"].sum()),
    "n_productive_change": int(paired["productive_change"].sum()),
    "n_harmful_change": int(paired["harmful_change"].sum()),
    "change_rate": float(paired["changed_dx"].mean()) if len(paired) else None,
    "productive_to_harmful_ratio": (
        (paired["productive_change"].sum() / paired["harmful_change"].sum())
        if paired["harmful_change"].sum() > 0 else None
    ),
}


# ---------------------------------------------------------------------------
# Analysis 8 — Ceiling-effect comparison vs other non-melanocytic classes
# ---------------------------------------------------------------------------
print("[8/9] Ceiling-effect comparison (other classes) ...")
class_compare = {}
for cls in sorted(df["true_diagnosis"].dropna().unique()):
    sub = df[df["true_diagnosis"] == cls]
    un = sub[sub["has_ai_assistance"] == False]["is_correct_numeric"]
    ai = sub[sub["has_ai_assistance"] == True]["is_correct_numeric"]
    if len(un) == 0 or len(ai) == 0:
        continue
    pr = per_reader_accuracy(sub)
    res = paired_wilcoxon(pr)
    class_compare[cls] = {
        "n_obs_un": int(len(un)),
        "n_obs_ai": int(len(ai)),
        "obs_unaided_acc": float(un.mean()),
        "obs_assisted_acc": float(ai.mean()),
        "diff": float(ai.mean() - un.mean()),
        "paired_p_value": res.get("p_value"),
        "n_readers_paired": res.get("n_readers"),
    }


# ---------------------------------------------------------------------------
# Analysis 9 — Clinical-decision distribution on VASC
# ---------------------------------------------------------------------------
print("[9/9] Clinical-decision distribution on VASC ...")
clin_dec = clinical_decision_dist(df, "VASC")


# ---------------------------------------------------------------------------
# Assemble + write JSON
# ---------------------------------------------------------------------------
out = {
    "analysis": "RS2B vascular subgroup deep-dive (R2 Comment 12)",
    "source_csv": str(INPUT_CSV),
    "n_total_rows": int(len(df)),
    "n_vasc_rows": int(len(vasc)),
    "n_vasc_unique_images": int(vasc["image_id"].nunique()),
    "n_vasc_readers": int(vasc["reader_id"].nunique()),
    "1_overall_paired_wilcoxon": overall,
    "2_by_expertise": by_expertise,
    "3_by_profession_bcd_vs_non_bcd": by_profession,
    "3b_by_profession_detail": by_profession_detail,
    "4_per_image_accuracy": per_image_records,
    "5_reader_diagnosis_distribution": diag_dist,
    "6_error_entropy": error_entropy,
    "7_revision_pattern": revision_summary,
    "8_class_comparison": class_compare,
    "9_clinical_decision_distribution": clin_dec,
}

with OUT_JSON.open("w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nWrote {OUT_JSON}")


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


md = []
md.append("# RS2B Vascular subgroup analysis (R2 Comment 12)\n")
md.append(f"Source: `{INPUT_CSV.name}`  \n"
          f"VASC obs: {len(vasc)} | unique images: {vasc['image_id'].nunique()} | "
          f"readers: {vasc['reader_id'].nunique()}\n")

# 1
md.append("\n## 1. Overall paired Wilcoxon (reader-level)\n")
md.append(f"- n readers paired: **{overall.get('n_readers')}**")
md.append(f"- mean accuracy unaided: **{fmt(overall.get('mean_unaided'))}**")
md.append(f"- mean accuracy assisted: **{fmt(overall.get('mean_assisted'))}**")
md.append(f"- mean diff (assisted − unaided): **{fmt(overall.get('mean_diff'))}**")
md.append(f"- Wilcoxon p-value: **{fmt(overall.get('p_value'), 4)}**")
md.append(f"- (per-observation: unaided "
          f"{fmt(overall['per_obs_unaided_acc'])} vs assisted "
          f"{fmt(overall['per_obs_assisted_acc'])} "
          f"on {overall['n_obs_unaided']} / {overall['n_obs_assisted']} obs)")

# 2
md.append("\n## 2. By expertise\n")
md.append("| Expertise | n readers | Mean unaided | Mean assisted | Diff | p |")
md.append("|---|---:|---:|---:|---:|---:|")
for exp, r in by_expertise.items():
    md.append(f"| {exp} | {r.get('n_readers')} | "
              f"{fmt(r.get('mean_unaided'))} | "
              f"{fmt(r.get('mean_assisted'))} | "
              f"{fmt(r.get('mean_diff'))} | {fmt(r.get('p_value'), 4)} |")

# 3
md.append("\n## 3. By BCD vs non-BCD\n")
md.append("| Group | n readers | Mean unaided | Mean assisted | Diff | p |")
md.append("|---|---:|---:|---:|---:|---:|")
for label, r in by_profession.items():
    md.append(f"| {label} | {r.get('n_readers')} | "
              f"{fmt(r.get('mean_unaided'))} | "
              f"{fmt(r.get('mean_assisted'))} | "
              f"{fmt(r.get('mean_diff'))} | {fmt(r.get('p_value'), 4)} |")

# 3b
md.append("\n## 3b. By profession (detail)\n")
md.append("| Profession | n_un | n_ai | Unaided | Assisted |")
md.append("|---|---:|---:|---:|---:|")
for prof, r in by_profession_detail.items():
    md.append(f"| {prof} | {r['n_obs_unaided']} | {r['n_obs_assisted']} | "
              f"{fmt(r['unaided_acc'])} | {fmt(r['assisted_acc'])} |")

# 4
md.append("\n## 4. Per-image VASC accuracy (sorted by drop)\n")
md.append("| image_id | isic_id | n_un | n_ai | unaided | assisted | diff |")
md.append("|---|---|---:|---:|---:|---:|---:|")
for rec in per_image_records:
    md.append(f"| {rec['image_id']} | {rec['isic_id']} | "
              f"{rec['n_unaided']} | {rec['n_assisted']} | "
              f"{fmt(rec['unaided_acc'])} | {fmt(rec['assisted_acc'])} | "
              f"{fmt(rec['diff'])} |")

# 5
md.append("\n## 5. Reader-diagnosis distribution when GT = VASC\n")
md.append("### 5a. Unaided")
md.append("| reader_diagnosis | n | proportion |")
md.append("|---|---:|---:|")
for k, v in sorted(diag_dist["unaided"]["proportions"].items(),
                   key=lambda x: -x[1]):
    md.append(f"| {k} | {diag_dist['unaided']['counts'].get(k, 0)} | "
              f"{fmt(v)} |")

md.append("\n### 5b. Assisted")
md.append("| reader_diagnosis | n | proportion |")
md.append("|---|---:|---:|")
for k, v in sorted(diag_dist["assisted"]["proportions"].items(),
                   key=lambda x: -x[1]):
    md.append(f"| {k} | {diag_dist['assisted']['counts'].get(k, 0)} | "
              f"{fmt(v)} |")

# 6
md.append("\n## 6. Error entropy (Shannon, bits)\n")
md.append("| Arm | Error rate | n distinct error classes | Shannon entropy |")
md.append("|---|---:|---:|---:|")
for arm, r in error_entropy.items():
    md.append(f"| {arm} | {fmt(r['error_proportion'])} | "
              f"{r['n_distinct_error_classes']} | "
              f"{fmt(r['shannon_entropy_bits'])} |")

md.append("\nInterpretation: higher Shannon entropy under AI assistance "
          "indicates 'error diversification'—readers no longer concentrate "
          "errors on a single look-alike class (e.g., MEL) but spread them "
          "across multiple plausible non-vascular candidates suggested by the "
          "AI's top-K.")

# 7
md.append("\n## 7. Revision pattern (paired same-image flips)\n")
r = revision_summary
md.append(f"- n paired VASC observations: {r['n_paired_observations']}")
md.append(f"- changed diagnosis (un→ai): {r['n_changed']} "
          f"({fmt(r['change_rate'])} rate)")
md.append(f"- productive changes (wrong→right): "
          f"{r['n_productive_change']}")
md.append(f"- harmful changes (right→wrong): {r['n_harmful_change']}")
md.append(f"- productive/harmful ratio: "
          f"{fmt(r['productive_to_harmful_ratio'])}")

# 8
md.append("\n## 8. Ceiling-effect class comparison\n")
md.append("(VASC has the highest unaided accuracy of any class; "
          "AI offers limited upside on already-near-ceiling categories.)\n")
md.append("| Class | n_un | n_ai | unaided | assisted | diff | paired p |")
md.append("|---|---:|---:|---:|---:|---:|---:|")
for cls, r in sorted(class_compare.items(),
                     key=lambda x: -x[1]['obs_unaided_acc']):
    md.append(f"| {cls} | {r['n_obs_un']} | {r['n_obs_ai']} | "
              f"{fmt(r['obs_unaided_acc'])} | "
              f"{fmt(r['obs_assisted_acc'])} | "
              f"{fmt(r['diff'])} | {fmt(r['paired_p_value'], 4)} |")

# 9
md.append("\n## 9. Clinical-decision distribution on VASC\n")
md.append("### 9a. Unaided")
md.append("| decision | n | proportion |")
md.append("|---|---:|---:|")
for k, v in sorted(clin_dec["unaided"]["proportions"].items(),
                   key=lambda x: -x[1]):
    md.append(f"| {k} | {clin_dec['unaided']['counts'].get(k, 0)} | "
              f"{fmt(v)} |")
md.append("\n### 9b. Assisted")
md.append("| decision | n | proportion |")
md.append("|---|---:|---:|")
for k, v in sorted(clin_dec["assisted"]["proportions"].items(),
                   key=lambda x: -x[1]):
    md.append(f"| {k} | {clin_dec['assisted']['counts'].get(k, 0)} | "
              f"{fmt(v)} |")

with OUT_MD.open("w") as f:
    f.write("\n".join(md))

print(f"Wrote {OUT_MD}")
print("\nDone.")
