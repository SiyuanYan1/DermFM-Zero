#!/usr/bin/env python3
"""Post-hoc power analysis for DermFM-Zero reader-study primary outcomes.

Outcomes
--------
1. RS1 diagnostic accuracy (paired, n=30 readers)
2. RS2A session accuracy (DermFM-Zero vs Humans-All, two-sample,
   n_h=1090 sessions, n_m=20000 iterations)
3. RS2B diagnostic accuracy (paired, 71-reader cohort)
4. RS2B management appropriateness (paired, 71-reader cohort)

Each outcome reports:
  * n
  * Cohen's d_z (paired) or Cohen's d (two-sample)
  * Achieved power at alpha=0.05 two-sided via statsmodels
  * Wilcoxon-ARE-adjusted power (d_z scaled by sqrt(0.864))

Outputs (real mode):
  * <rs?>/real_output/... not used; consolidated outputs land in
    reader_studies/real_output/post_hoc_power.json
    reader_studies/real_output/post_hoc_power_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_solve_power, tt_ind_solve_power


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RS1_STATS = BASE / "reader_study_rs1" / "real_output" / "rs1_statistics.csv"
RS2A_XLSX = BASE / "reader_study_rs2a" / "real_data" / "todiv_scores.xlsx"
RS2B_CSV = BASE / "reader_study_rs2b" / "real_output" / "panderm_cleaned_95pct.csv"

OUT_DIR = BASE / "real_output"

ALPHA = 0.05
WILCOXON_ARE = 0.864  # Pitman ARE for Wilcoxon signed-rank vs t-test under normality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def paired_power(d_z: float, n: int, alpha: float = ALPHA) -> float:
    """Achieved power for paired/one-sample t-test (two-sided)."""
    if not np.isfinite(d_z) or d_z == 0 or n < 2:
        return float("nan")
    return float(
        tt_solve_power(
            effect_size=abs(d_z),
            nobs=n,
            alpha=alpha,
            power=None,
            alternative="two-sided",
        )
    )


def two_sample_power(d: float, n1: int, n2: int, alpha: float = ALPHA) -> float:
    """Achieved power for two-sample (Welch-equivalent) t-test (two-sided)."""
    if not np.isfinite(d) or d == 0 or n1 < 2 or n2 < 2:
        return float("nan")
    ratio = n2 / n1
    return float(
        tt_ind_solve_power(
            effect_size=abs(d),
            nobs1=n1,
            alpha=alpha,
            power=None,
            ratio=ratio,
            alternative="two-sided",
        )
    )


def wilcoxon_adjusted(d_z: float) -> float:
    return d_z * math.sqrt(WILCOXON_ARE)


# ---------------------------------------------------------------------------
# Outcome 1: RS1 diagnostic accuracy
# ---------------------------------------------------------------------------
def analyse_rs1(stats_path: Path) -> dict:
    if not stats_path.exists():
        return {"outcome": "RS1 diagnostic accuracy", "status": "n/a",
                "reason": f"missing file: {stats_path}"}
    df = pd.read_csv(stats_path)
    row = df[df["Metric"].str.contains("Diagnostic Accuracy", case=False, na=False)]
    if row.empty:
        return {"outcome": "RS1 diagnostic accuracy", "status": "n/a",
                "reason": "Diagnostic Accuracy Score row not found"}
    row = row.iloc[0]
    n = int(row["n"])
    mean_diff = float(row["Mean_Difference"])
    sd_un = float(row["Unaided_SD"])
    sd_as = float(row["Assisted_SD"])
    r_eff = float(row["Effect_Size_r"])  # Wilcoxon effect size r = Z / sqrt(N)

    # Primary d_z: derived from Wilcoxon r via z = r * sqrt(N); convert to d_z
    # using the relationship d_z ~ z / sqrt(N) for paired Wilcoxon under
    # near-normal data -> d_z ~ r. This is the most defensible single-number
    # effect size when only summary stats are available.
    d_z_from_r = r_eff

    # Cross-check using a plausible Pearson correlation r=0.5 between
    # unaided/assisted (typical for repeated measures); SD_diff = sqrt(...)
    rho_assumed = 0.5
    sd_diff_assumed = math.sqrt(sd_un**2 + sd_as**2 - 2 * rho_assumed * sd_un * sd_as)
    d_z_assumed = mean_diff / sd_diff_assumed if sd_diff_assumed > 0 else float("nan")

    d_z = d_z_from_r  # primary
    pwr = paired_power(d_z, n)
    pwr_w = paired_power(wilcoxon_adjusted(d_z), n)

    return {
        "outcome": "RS1 diagnostic accuracy",
        "design": "paired t-test (Wilcoxon signed-rank in manuscript)",
        "n": n,
        "mean_diff": mean_diff,
        "unaided_sd": sd_un,
        "assisted_sd": sd_as,
        "effect_size_r_wilcoxon": r_eff,
        "d_z_primary": d_z,
        "d_z_method": "d_z ~ Wilcoxon r (Z/sqrt(N))",
        "d_z_sensitivity_rho0.5": d_z_assumed,
        "alpha": ALPHA,
        "power_t": pwr,
        "power_wilcoxon_ARE_adjusted": pwr_w,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Outcome 2: RS2A session accuracy (two-sample)
# ---------------------------------------------------------------------------
def analyse_rs2a(xlsx_path: Path) -> dict:
    if not xlsx_path.exists():
        return {"outcome": "RS2A session accuracy", "status": "n/a",
                "reason": f"missing file: {xlsx_path}"}

    xl = pd.ExcelFile(xlsx_path)
    humans = pd.read_excel(xl, sheet_name="Humans")
    # DermFM-Zero == "Milkk10 Zero Shot" (zero-shot DermFM run) in manuscript.
    derm = pd.read_excel(xl, sheet_name="Milkk10 Zero Shot")

    h_scores = pd.to_numeric(humans["score"], errors="coerce").dropna().to_numpy()
    m_scores = pd.to_numeric(derm["Tableau 1"], errors="coerce").dropna().to_numpy()

    n_h = int(len(h_scores))
    n_m = int(len(m_scores))
    mean_h = float(h_scores.mean())
    mean_m = float(m_scores.mean())
    sd_h = float(h_scores.std(ddof=1))
    sd_m = float(m_scores.std(ddof=1))

    # Cohen's d using pooled SD (independent-samples formula)
    pooled = math.sqrt(((n_h - 1) * sd_h**2 + (n_m - 1) * sd_m**2) / (n_h + n_m - 2))
    d = (mean_m - mean_h) / pooled if pooled > 0 else float("nan")

    # Effective sample size note: model "iterations" are not independent
    # human-equivalent sessions, so we also report power treating n_m=n_h
    # (a conservative 1090 vs 1090 calibration) for sensitivity.
    pwr_full = two_sample_power(d, n_h, n_m)
    pwr_conservative = two_sample_power(d, n_h, n_h)

    return {
        "outcome": "RS2A session accuracy (DermFM-Zero vs Humans-All)",
        "design": "two-sample Welch t-test approximation",
        "n_humans": n_h,
        "n_model_iterations": n_m,
        "mean_humans": mean_h,
        "sd_humans": sd_h,
        "mean_dermfm_zero": mean_m,
        "sd_dermfm_zero": sd_m,
        "pooled_sd": pooled,
        "cohen_d": d,
        "alpha": ALPHA,
        "power_t_full_n_m": pwr_full,
        "power_t_conservative_n_m_eq_n_h": pwr_conservative,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Outcomes 3 & 4: RS2B (paired by reader)
# ---------------------------------------------------------------------------
def _run_rs2b_filter() -> None:
    """Run 01_filter_reader.py --real to materialise panderm_cleaned_95pct.csv."""
    import subprocess
    cwd = BASE / "reader_study_rs2b"
    script = cwd / "01_filter_reader.py"
    if not script.exists():
        raise FileNotFoundError(f"missing {script}")
    subprocess.run(
        ["/usr/bin/python3", str(script), "--real"],
        cwd=str(cwd),
        check=True,
    )


def analyse_rs2b(csv_path: Path) -> tuple[dict, dict]:
    if not csv_path.exists():
        try:
            _run_rs2b_filter()
        except Exception as e:
            err = {"status": "n/a", "reason": f"could not produce {csv_path}: {e}"}
            return ({"outcome": "RS2B diagnostic accuracy", **err},
                    {"outcome": "RS2B management appropriateness", **err})

    df = pd.read_csv(csv_path)

    # ---- diagnostic accuracy -------------------------------------------------
    acc = (df.groupby(["reader_id", "test_mode"])["is_correct"]
             .mean()
             .unstack("test_mode"))
    if "without_ai" not in acc.columns or "with_ai" not in acc.columns:
        diag = {"outcome": "RS2B diagnostic accuracy", "status": "n/a",
                "reason": f"unexpected test_mode columns {list(acc.columns)}"}
    else:
        acc = acc.dropna(subset=["with_ai", "without_ai"])
        diffs = (acc["with_ai"] - acc["without_ai"]).to_numpy()
        n = int(len(diffs))
        mean_diff = float(diffs.mean())
        sd_diff = float(diffs.std(ddof=1))
        d_z = mean_diff / sd_diff if sd_diff > 0 else float("nan")
        diag = {
            "outcome": "RS2B diagnostic accuracy",
            "design": "paired t-test (within-reader, with_ai vs without_ai)",
            "n_pairs": n,
            "mean_without_ai": float(acc["without_ai"].mean()),
            "mean_with_ai": float(acc["with_ai"].mean()),
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "d_z": d_z,
            "alpha": ALPHA,
            "power_t": paired_power(d_z, n),
            "power_wilcoxon_ARE_adjusted": paired_power(wilcoxon_adjusted(d_z), n),
            "status": "ok",
        }

    # ---- management appropriateness -----------------------------------------
    # Encode the Fig. 5a Management Standards matrix (11 diagnoses x 4 actions).
    # Cells: 'Opt' = Optimal, 'App' = Appropriate, 'Inp' = Inappropriate.
    # An action is 'appropriate' (binary 1) iff cell in {Opt, App}; else 0.
    # Map clinical_decision -> matrix columns:
    #   dismiss -> Dismiss, monitor -> Monitor,
    #   local_therapy -> Treat, biopsy -> Excise.
    mgmt_matrix = {
        "AKIEC":           {"Dismiss": "Inp", "Monitor": "Inp", "Treat": "Opt", "Excise": "App"},
        "BCC":             {"Dismiss": "Inp", "Monitor": "Inp", "Treat": "App", "Excise": "Opt"},
        "OTHER_BENIGN":    {"Dismiss": "Opt", "Monitor": "App", "Treat": "Inp", "Excise": "Inp"},
        "BKL":             {"Dismiss": "Opt", "Monitor": "App", "Treat": "Inp", "Excise": "Inp"},
        "DF":              {"Dismiss": "Opt", "Monitor": "App", "Treat": "Inp", "Excise": "Inp"},
        "INF":             {"Dismiss": "App", "Monitor": "App", "Treat": "Opt", "Excise": "Inp"},
        "OTHER_MALIGNANT": {"Dismiss": "Inp", "Monitor": "Inp", "Treat": "Inp", "Excise": "Opt"},
        "MEL":             {"Dismiss": "Inp", "Monitor": "Inp", "Treat": "Inp", "Excise": "Opt"},
        "NV":              {"Dismiss": "Opt", "Monitor": "App", "Treat": "Inp", "Excise": "Inp"},
        "SCCKA":           {"Dismiss": "Inp", "Monitor": "Inp", "Treat": "Inp", "Excise": "Opt"},
        "VASC":            {"Dismiss": "Opt", "Monitor": "App", "Treat": "Inp", "Excise": "Inp"},
    }
    decision_to_action = {
        "dismiss": "Dismiss",
        "monitor": "Monitor",
        "local_therapy": "Treat",
        "biopsy": "Excise",
    }

    def _is_appropriate(row):
        dx = row.get("true_diagnosis")
        dec = row.get("clinical_decision")
        action = decision_to_action.get(dec)
        if dx not in mgmt_matrix or action is None:
            return np.nan
        cell = mgmt_matrix[dx][action]
        return 1 if cell in ("Opt", "App") else 0

    df_mgmt = df.copy()
    df_mgmt["is_appropriate"] = df_mgmt.apply(_is_appropriate, axis=1)
    n_unmatched = int(df_mgmt["is_appropriate"].isna().sum())
    df_mgmt = df_mgmt.dropna(subset=["is_appropriate"])

    appro = (df_mgmt.groupby(["reader_id", "test_mode"])["is_appropriate"]
                    .mean()
                    .unstack("test_mode"))
    if "without_ai" not in appro.columns or "with_ai" not in appro.columns:
        mgmt = {"outcome": "RS2B management appropriateness", "status": "n/a",
                "reason": f"unexpected test_mode columns {list(appro.columns)}"}
    else:
        appro = appro.dropna(subset=["with_ai", "without_ai"])
        m_diffs = (appro["with_ai"] - appro["without_ai"]).to_numpy()
        n_m_pairs = int(len(m_diffs))
        mean_diff_m = float(m_diffs.mean())
        sd_diff_m = float(m_diffs.std(ddof=1))
        d_z_m = mean_diff_m / sd_diff_m if sd_diff_m > 0 else float("nan")
        mgmt = {
            "outcome": "RS2B management appropriateness",
            "design": "paired t-test (within-reader, with_ai vs without_ai)",
            "ground_truth_source": "Fig. 5a Management Standards matrix (Opt/App=appropriate, Inp=inappropriate)",
            "decision_to_action_map": decision_to_action,
            "n_rows_unmatched_to_matrix": n_unmatched,
            "n_pairs": n_m_pairs,
            "mean_without_ai": float(appro["without_ai"].mean()),
            "mean_with_ai": float(appro["with_ai"].mean()),
            "per_reader_appropriateness_range": [
                float(np.minimum(appro["without_ai"].min(), appro["with_ai"].min())),
                float(np.maximum(appro["without_ai"].max(), appro["with_ai"].max())),
            ],
            "mean_diff": mean_diff_m,
            "sd_diff": sd_diff_m,
            "d_z": d_z_m,
            "alpha": ALPHA,
            "power_t": paired_power(d_z_m, n_m_pairs),
            "power_wilcoxon_ARE_adjusted": paired_power(wilcoxon_adjusted(d_z_m), n_m_pairs),
            "status": "ok",
        }

    return diag, mgmt


# ---------------------------------------------------------------------------
# Demo mode (synthetic)
# ---------------------------------------------------------------------------
def analyse_demo() -> list[dict]:
    rng = np.random.default_rng(42)
    out = []

    # RS1-like
    n = 30
    diff = rng.normal(0.5, 0.6, n)
    d_z = diff.mean() / diff.std(ddof=1)
    out.append({
        "outcome": "RS1 diagnostic accuracy (DEMO)",
        "design": "paired t-test",
        "n": n,
        "d_z": float(d_z),
        "power_t": paired_power(d_z, n),
        "power_wilcoxon_ARE_adjusted": paired_power(wilcoxon_adjusted(d_z), n),
        "status": "ok",
    })

    # RS2A-like
    h = rng.normal(65.9, 10.3, 1090)
    m = rng.normal(71.7, 3.8, 20000)
    pooled = math.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(m) - 1) * m.var(ddof=1)) / (len(h) + len(m) - 2))
    d = (m.mean() - h.mean()) / pooled
    out.append({
        "outcome": "RS2A session accuracy (DEMO)",
        "design": "two-sample",
        "n_humans": len(h),
        "n_model_iterations": len(m),
        "cohen_d": float(d),
        "power_t_full_n_m": two_sample_power(d, len(h), len(m)),
        "power_t_conservative_n_m_eq_n_h": two_sample_power(d, len(h), len(h)),
        "status": "ok",
    })

    # RS2B diag
    n = 71
    diffs = rng.normal(0.05, 0.1, n)
    d_z = diffs.mean() / diffs.std(ddof=1)
    out.append({
        "outcome": "RS2B diagnostic accuracy (DEMO)",
        "n_pairs": n,
        "d_z": float(d_z),
        "power_t": paired_power(d_z, n),
        "power_wilcoxon_ARE_adjusted": paired_power(wilcoxon_adjusted(d_z), n),
        "status": "ok",
    })

    # RS2B mgmt
    out.append({
        "outcome": "RS2B management appropriateness (DEMO)",
        "status": "n/a",
        "reason": "no management-appropriateness ground truth in demo data",
    })
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "post_hoc_power.json"
    md_path = out_dir / "post_hoc_power_summary.md"

    with json_path.open("w") as f:
        json.dump({"alpha": ALPHA, "wilcoxon_ARE": WILCOXON_ARE,
                   "results": results}, f, indent=2)

    # Markdown
    lines = []
    lines.append("# Post-hoc power analysis (R3 Major Comment 3)")
    lines.append("")
    lines.append(f"alpha = {ALPHA} (two-sided); Wilcoxon ARE adjustment = sqrt({WILCOXON_ARE}).")
    lines.append("")
    lines.append("| Outcome | n | Effect size | Power (t) | Power (Wilcoxon-adj.) | Status |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        name = r.get("outcome", "?")
        if r.get("status") != "ok":
            lines.append(f"| {name} | - | - | - | - | n/a: {r.get('reason','')} |")
            continue
        if "n_pairs" in r:
            n_str = str(r["n_pairs"])
            es_str = f"d_z={r.get('d_z',float('nan')):.3f}"
            pt = r.get("power_t", float("nan"))
            pw = r.get("power_wilcoxon_ARE_adjusted", float("nan"))
        elif "n_humans" in r:
            n_str = f"{r['n_humans']} vs {r['n_model_iterations']}"
            es_str = f"d={r.get('cohen_d',float('nan')):.3f}"
            pt = r.get("power_t_full_n_m", float("nan"))
            pw = r.get("power_t_conservative_n_m_eq_n_h", float("nan"))
        else:
            n_str = str(r.get("n", "?"))
            es_str = f"d_z={r.get('d_z_primary', r.get('d_z', float('nan'))):.3f}"
            pt = r.get("power_t", float("nan"))
            pw = r.get("power_wilcoxon_ARE_adjusted", float("nan"))
        lines.append(f"| {name} | {n_str} | {es_str} | {pt:.4f} | {pw:.4f} | ok |")

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- RS1 d_z is derived from the Wilcoxon effect size r (r = Z / sqrt(N)) "
                 "because individual-reader paired scores are not exposed in the summary CSV; "
                 "a sensitivity d_z assuming Pearson r=0.5 between unaided/assisted is also reported in the JSON.")
    lines.append("- RS2A is reported two ways: with the full n_m=20,000 model iterations "
                 "(which inflates power because iterations are not independent human-equivalent sessions), "
                 "and with a conservative calibration n_m=n_h=1,090.")
    lines.append("- RS2B management appropriateness is scored by mapping each (true_diagnosis, clinical_decision) "
                 "pair through the Fig. 5a Management Standards matrix (11 diagnoses x 4 actions: "
                 "dismiss/monitor/local_therapy[Treat]/biopsy[Excise]); cells marked Optimal or Appropriate "
                 "are counted as appropriate (1), Inappropriate as 0.")
    lines.append("- Wilcoxon-ARE-adjusted power = solve for power with d_z scaled by sqrt(0.864), "
                 "approximating the Pitman ARE of the Wilcoxon signed-rank test relative to the paired t-test under normality.")

    md_path.write_text("\n".join(lines))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run on synthetic demo data")
    ap.add_argument("--real", action="store_true", help="run on real data")
    args = ap.parse_args()
    real = args.real and not args.demo

    if real:
        # Real mode: require real inputs to be present. No demo fallback.
        for p in (RS1_STATS, RS2A_XLSX, RS2B_CSV):
            if not p.exists():
                print(
                    f"Input missing: {p}. "
                    ""
                )
                sys.exit(0)
        results = []
        results.append(analyse_rs1(RS1_STATS))
        results.append(analyse_rs2a(RS2A_XLSX))
        diag, mgmt = analyse_rs2b(RS2B_CSV)
        results.append(diag)
        results.append(mgmt)
        out_dir = OUT_DIR
    else:
        results = analyse_demo()
        out_dir = BASE / "demo_output"

    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(results, out_dir)

    # Console headline
    print("\n=== Headline power values ===")
    for r in results:
        name = r.get("outcome", "?")
        if r.get("status") != "ok":
            print(f"  {name}: n/a ({r.get('reason','')})")
            continue
        if "n_pairs" in r:
            print(f"  {name}: power_t={r['power_t']:.4f}  power_wilcoxon_adj={r['power_wilcoxon_ARE_adjusted']:.4f}")
        elif "n_humans" in r:
            print(f"  {name}: power_t(full)={r['power_t_full_n_m']:.4f}  power_t(cons.)={r['power_t_conservative_n_m_eq_n_h']:.4f}")
        else:
            print(f"  {name}: power_t={r['power_t']:.4f}  power_wilcoxon_adj={r['power_wilcoxon_ARE_adjusted']:.4f}")


if __name__ == "__main__":
    main()
