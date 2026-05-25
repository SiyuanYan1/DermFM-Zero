"""
02_fig2_table_clean.py
----------------------
Per-table summary statistics (5 tables) for the RS2B 6-panel figure,
computed from the ≥95% filtered cohort (71 readers).

Two changes vs the original v1 pipeline:
  1. Reads the filtered CSV produced by 01_filter_reader.py.
  2. Uses the data-driven `management_table.csv` instead of a hard-coded
     appropriateness dict (matches the clinician's new R script).

Usage
-----
    python 02_fig2_table_clean.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CLI: --demo (default) | --real
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="RS2B Step 2: summary tables.")
_parser.add_argument("--real", action="store_const", const="real", dest="mode",
                     help="Use real_data/ -> real_output/")
_parser.add_argument("--demo", action="store_const", const="demo", dest="mode",
                     help="Use demo_data/ -> demo_output/  (default)")
_parser.set_defaults(mode="demo")
_args = _parser.parse_args()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / ("real_data" if _args.mode == "real" else "demo_data")
OUT = ROOT / ("real_output" if _args.mode == "real" else "demo_output")
OUT.mkdir(parents=True, exist_ok=True)

input_csv = OUT / "panderm_cleaned_95pct.csv"
output_csv = OUT / "rs2b_reader_study_data_95pct.csv"

print("=" * 80)
print(f"EXPORTING RS2B FIGURE DATA  (>=95% per-test cohort)  [mode={_args.mode}]")
print("=" * 80)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(input_csv)
df = df[df["is_completed"]].copy()

diagnosis_map = {
    "AKIEC": "AKIEC", "BCC": "BCC", "BKL": "BKL", "DF": "DF", "INF": "INF",
    "MEL": "MEL", "NV": "NV", "SCCKA": "SCCKA", "VASC": "VASC",
    "OTHER_BENIGN": "OTHER_BEN", "OTHER_MALIGNANT": "OTHER_MAL",
}
df["Class"] = df["true_diagnosis"].map(diagnosis_map)
df["help"] = df["test_mode"].map({"without_ai": "Unaided", "with_ai": "AI-assisted"})
df["ability_group"] = df["reader_expertise"].map({"expert": "Expert", "non-expert": "Non-expert"})

decision_map = {"dismiss": "Dismiss", "monitor": "Monitor", "local_therapy": "Treat", "biopsy": "Excise"}
df["Action"] = df["clinical_decision"].map(decision_map)


# ---------------------------------------------------------------------------
# Data-driven management appropriateness (replaces hard-coded dict)
# ---------------------------------------------------------------------------
mgmt_tbl = pd.read_csv(DATA / "management_table.csv", sep=";")
mgmt_tbl.columns = [c.strip() for c in mgmt_tbl.columns]
mgmt_tbl = mgmt_tbl.set_index("diagnosis")
# Each cell is "optimal" / "appropriate" / "inappropriate" (whitespace possible)
mgmt_tbl = mgmt_tbl.apply(lambda col: col.astype(str).str.strip())

# Map our internal Action label -> column in management_table.csv
ACTION_TO_COL = {"Dismiss": "dismiss", "Monitor": "monitor", "Treat": "treat_locally", "Excise": "excise_biopsy"}


def classify_management_binary(row) -> int:
    """1 if action is optimal or appropriate per management_table.csv; else 0."""
    diag = row["true_diagnosis"]
    action = row["Action"]
    if pd.isna(action) or diag not in mgmt_tbl.index:
        return 0
    col = ACTION_TO_COL.get(action)
    if col is None:
        return 0
    return 1 if mgmt_tbl.loc[diag, col] in ("optimal", "appropriate") else 0


df["mgmt_correct"] = df.apply(classify_management_binary, axis=1)


# ---------------------------------------------------------------------------
# Table 1: Overall Accuracy
# ---------------------------------------------------------------------------
panderm_reader = (
    df.groupby(["reader_id", "help"]).agg(accuracy=("is_correct_numeric", "mean")).reset_index()
)

overall_stats = []
for help_val in ["Unaided", "AI-assisted"]:
    subset = panderm_reader[panderm_reader["help"] == help_val]["accuracy"]
    mean, std, n = subset.mean(), subset.std(ddof=1), len(subset)
    ci = 1.96 * std / np.sqrt(n)
    overall_stats.append(
        {
            "Condition": "Without AI assistance" if help_val == "Unaided" else "With AI assistance",
            "Mean": mean,
            "CI_Lower": mean - ci,
            "CI_Upper": mean + ci,
            "N": n,
        }
    )

pivot_acc = panderm_reader.pivot(index="reader_id", columns="help", values="accuracy").dropna()
_, p_val = stats.ttest_rel(pivot_acc["AI-assisted"], pivot_acc["Unaided"])

df_overall = pd.DataFrame(overall_stats)
df_overall["p_value"] = [np.nan, p_val]
df_overall["Table"] = "RS2_Overall_Accuracy"
print("✅ Prepared: RS2_Overall_Accuracy")


# ---------------------------------------------------------------------------
# Table 2: Class-specific Accuracy
# Two-sided paired t-test on per-reader proportions (matches paper Table 14).
# ---------------------------------------------------------------------------
class_stats = []
for cls in df["Class"].dropna().unique():
    sub = df[df["Class"] == cls]
    piv = (sub.pivot_table(index="reader_id", columns="help",
                           values="is_correct_numeric", aggfunc="mean")
              .dropna())
    if "Unaided" not in piv.columns or "AI-assisted" not in piv.columns:
        continue
    if len(piv) >= 10:
        u, a = piv["Unaided"], piv["AI-assisted"]
        m_u, s_u, n_u = u.mean(), u.std(ddof=1), len(u)
        m_a, s_a, n_a = a.mean(), a.std(ddof=1), len(a)
        _, p = stats.ttest_rel(a, u)
        class_stats.append(
            {
                "Class": cls,
                "Without_AI_Mean": m_u,
                "Without_AI_CI_Lower": m_u - 1.96 * s_u / np.sqrt(n_u),
                "Without_AI_CI_Upper": m_u + 1.96 * s_u / np.sqrt(n_u),
                "With_AI_Mean": m_a,
                "With_AI_CI_Lower": m_a - 1.96 * s_a / np.sqrt(n_a),
                "With_AI_CI_Upper": m_a + 1.96 * s_a / np.sqrt(n_a),
                "p_value": p,
                "N_readers": n_u,
            }
        )

df_class = pd.DataFrame(class_stats)
df_class["corrected_p_value"] = (df_class["p_value"] * len(df_class)).clip(upper=1.0)
df_class["Table"] = "RS2_Class_Specific_Accuracy"
print("✅ Prepared: RS2_Class_Specific_Accuracy")


# ---------------------------------------------------------------------------
# Table 3: Accuracy by Experience Level
# ---------------------------------------------------------------------------
ability_stats = []
for ability in ["Non-expert", "Expert"]:
    subset = df[df["ability_group"] == ability]
    unaided = subset[subset["help"] == "Unaided"].groupby("reader_id")["is_correct_numeric"].mean()
    assisted = subset[subset["help"] == "AI-assisted"].groupby("reader_id")["is_correct_numeric"].mean()
    if len(unaided) == 0 or len(assisted) == 0:
        continue
    m_u, s_u, n_u = unaided.mean(), unaided.std(ddof=1), len(unaided)
    m_a, s_a, n_a = assisted.mean(), assisted.std(ddof=1), len(assisted)
    paired = pd.DataFrame({"Unaided": unaided, "Assisted": assisted}).dropna()
    p = stats.ttest_rel(paired["Assisted"], paired["Unaided"]).pvalue if len(paired) > 1 else np.nan
    ability_stats.append(
        {
            "Experience_Group": ability,
            "Without_AI_Mean": m_u,
            "Without_AI_CI_Lower": m_u - 1.96 * s_u / np.sqrt(n_u),
            "Without_AI_CI_Upper": m_u + 1.96 * s_u / np.sqrt(n_u),
            "With_AI_Mean": m_a,
            "With_AI_CI_Lower": m_a - 1.96 * s_a / np.sqrt(n_a),
            "With_AI_CI_Upper": m_a + 1.96 * s_a / np.sqrt(n_a),
            "p_value": p,
            "N_readers": len(paired) if len(paired) > 1 else len(unaided),
        }
    )

df_ability = pd.DataFrame(ability_stats)
df_ability["corrected_p_value"] = (df_ability["p_value"] * len(df_ability)).clip(upper=1.0)
df_ability["Table"] = "RS2_Accuracy_by_Experience"
print("✅ Prepared: RS2_Accuracy_by_Experience")


# ---------------------------------------------------------------------------
# Table 4: Overall Management Appropriateness
# ---------------------------------------------------------------------------
mgmt_reader = (
    df.groupby(["reader_id", "help"]).agg(mgmt_appropriate=("mgmt_correct", "mean")).reset_index()
)

mgmt_overall = []
for help_val in ["Unaided", "AI-assisted"]:
    subset = mgmt_reader[mgmt_reader["help"] == help_val]["mgmt_appropriate"]
    mean, std, n = subset.mean(), subset.std(ddof=1), len(subset)
    ci = 1.96 * std / np.sqrt(n)
    mgmt_overall.append(
        {
            "Condition": "Without AI assistance" if help_val == "Unaided" else "With AI assistance",
            "Mean": mean,
            "CI_Lower": mean - ci,
            "CI_Upper": mean + ci,
            "N": n,
        }
    )

pivot_mgmt = mgmt_reader.pivot(index="reader_id", columns="help", values="mgmt_appropriate").dropna()
_, p_val = stats.ttest_rel(pivot_mgmt["AI-assisted"], pivot_mgmt["Unaided"])

df_mgmt_overall = pd.DataFrame(mgmt_overall)
df_mgmt_overall["p_value"] = [np.nan, p_val]
df_mgmt_overall["Table"] = "RS2_Overall_Management_Appropriate"
print("✅ Prepared: RS2_Overall_Management_Appropriate")


# ---------------------------------------------------------------------------
# Table 5: Management by Experience Level
# ---------------------------------------------------------------------------
mgmt_ability_stats = []
for ability in ["Non-expert", "Expert"]:
    subset = df[df["ability_group"] == ability]
    unaided = subset[subset["help"] == "Unaided"].groupby("reader_id")["mgmt_correct"].mean()
    assisted = subset[subset["help"] == "AI-assisted"].groupby("reader_id")["mgmt_correct"].mean()
    if len(unaided) == 0 or len(assisted) == 0:
        continue
    m_u, s_u, n_u = unaided.mean(), unaided.std(ddof=1), len(unaided)
    m_a, s_a, n_a = assisted.mean(), assisted.std(ddof=1), len(assisted)
    paired = pd.DataFrame({"Unaided": unaided, "Assisted": assisted}).dropna()
    p = stats.ttest_rel(paired["Assisted"], paired["Unaided"]).pvalue if len(paired) > 1 else np.nan
    mgmt_ability_stats.append(
        {
            "Experience_Group": ability,
            "Without_AI_Mean": m_u,
            "Without_AI_CI_Lower": m_u - 1.96 * s_u / np.sqrt(n_u),
            "Without_AI_CI_Upper": m_u + 1.96 * s_u / np.sqrt(n_u),
            "With_AI_Mean": m_a,
            "With_AI_CI_Lower": m_a - 1.96 * s_a / np.sqrt(n_a),
            "With_AI_CI_Upper": m_a + 1.96 * s_a / np.sqrt(n_a),
            "p_value": p,
            "N_readers": len(paired) if len(paired) > 1 else len(unaided),
        }
    )

df_mgmt_ability = pd.DataFrame(mgmt_ability_stats)
df_mgmt_ability["corrected_p_value"] = (df_mgmt_ability["p_value"] * len(df_mgmt_ability)).clip(upper=1.0)
df_mgmt_ability["Table"] = "RS2_Management_by_Experience"
print("✅ Prepared: RS2_Management_by_Experience")


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
all_dfs = []
for d, key in [
    (df_overall, "Condition"),
    (df_class, "Class"),
    (df_ability, "Experience_Group"),
    (df_mgmt_overall, "Condition"),
    (df_mgmt_ability, "Experience_Group"),
]:
    d2 = d.copy().rename(columns={key: "Category"})
    all_dfs.append(d2)

df_combined = pd.concat(all_dfs, ignore_index=True, sort=False)
cols = ["Table"] + [c for c in df_combined.columns if c != "Table"]
df_combined = df_combined[cols]
df_combined.to_csv(output_csv, index=False)

print(f"\n✅ Exported all data to: {output_csv.name}")
print("=" * 80)
