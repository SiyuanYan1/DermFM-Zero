"""
01_filter_reader.py
--------------------
Adapts the doctor-provided raw export (`data/panderm_reader_data.csv`,
semicolon-separated long format, 87 readers) into the standardized wide
format used by the original Nature pipeline (`fig2_table_clean.py` /
`fig2_plot.py`), and applies the ≥95% completion filter (matches the
clinician's new R script).

Inputs
------
data/panderm_reader_data.csv   Raw reader study export (semicolon, comma decimal).
                               One row per (reader x case x mode).

Outputs (in outputs/)
---------------------
panderm_standardized_v2.csv    Long-format export with old-style column names,
                               no filter applied (full 87 readers).
panderm_cleaned_95pct.csv      ≥95% answered items per test  →  71 readers,
                               165 tests, 4929 rows.  This is the canonical
                               cohort for all downstream analysis.
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CLI: --demo (default) | --real
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="RS2B Step 1: filter readers (>=95% completion).")
_parser.add_argument("--real", action="store_const", const="real", dest="mode",
                     help="Use real_data/ -> real_output/")
_parser.add_argument("--demo", action="store_const", const="demo", dest="mode",
                     help="Use demo_data/ -> demo_output/  (default)")
_parser.set_defaults(mode="real")
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / ("real_data" if _args.mode == "real" else "demo_data")
OUT = ROOT / ("real_output" if _args.mode == "real" else "demo_output")
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA / "panderm_reader_data.csv"
print(f"[mode={_args.mode}]  DATA={DATA.name}  OUT={OUT.name}")

# ---------------------------------------------------------------------------
# Column mapping: new (raw) -> old (standardized) schema
# ---------------------------------------------------------------------------
RENAME = {
    "tequ_user_id": "reader_id",
    "test_id": "test_session_id",
    "test_round": "round_number",
    "image_isic_id": "isic_id",
    "tequ_no": "question_number",
    "tequ_correct": "is_correct_numeric",
    "answer": "reader_diagnosis",
    "image_diag_id": "true_diagnosis",
}

# tequ_mode_id -> test_mode
MODE_MAP = {
    "radio_radio_management47": "without_ai",
    "radio_radio_management47_help": "with_ai",
}

# manage -> clinical_decision (matches old `panderm_cleaned_15cases.csv`)
DECISION_MAP = {
    "manage_dismiss": "dismiss",
    "manage_monitor": "monitor",
    "manage_biopsy": "biopsy",
    "manage_localtherapy": "local_therapy",
}

# ability_2pl -> reader_expertise (use hyphenated "non-expert" for compatibility
# with old fig2_*.py which has `{'expert': 'Expert', 'non-expert': 'Non-expert'}`)
ABILITY_MAP = {"expert": "expert", "non_expert": "non-expert"}


def standardize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw doctor export -> standardized old-style schema."""
    df = df_raw.copy()

    # `tequ_correct` may arrive as 0/1 ints or as strings; coerce
    df["tequ_correct"] = pd.to_numeric(df["tequ_correct"], errors="coerce")

    # Valid-answer flag (matches R script logic)
    ans = df["answer"].astype(str).str.strip()
    df["_valid_answer"] = (
        df["answer"].notna() & (ans != "") & (ans.str.lower() != "nan") & (df["answer"] != "timeout")
    )

    # Apply renames
    df = df.rename(columns=RENAME)
    df["test_mode"] = df["tequ_mode_id"].map(MODE_MAP)
    df["has_ai_assistance"] = df["test_mode"] == "with_ai"
    df["clinical_decision"] = df["manage"].map(DECISION_MAP)
    df["reader_expertise"] = df["ability_2pl"].map(ABILITY_MAP)
    df["is_completed"] = df["_valid_answer"]
    df["is_correct"] = df["is_correct_numeric"] == 1

    # Carry-over informative columns from the new export
    keep = [
        "reader_id",
        "reader_expertise",
        "test_session_id",
        "round_number",
        "question_number",
        "test_mode",
        "has_ai_assistance",
        "image_id",
        "isic_id",
        "true_diagnosis",
        "reader_diagnosis",
        "is_correct",
        "is_correct_numeric",
        "clinical_decision",
        "is_completed",
        # extras only in new data — useful for sensitivity analyses
        "expected_accuracy_2pl",
        "age",
        "yob",
        "profession",
        "gender",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def filter_95pct(df_std: pd.DataFrame) -> pd.DataFrame:
    """Keep tests with >=95% valid answers (>=29 of 30 rows)."""
    # `is_completed` already encodes valid_answer
    df_valid = df_std[df_std["is_completed"]].copy()
    counts = df_valid.groupby("test_session_id").size()
    good_tests = counts[counts >= 0.95 * 30].index
    return df_valid[df_valid["test_session_id"].isin(good_tests)].reset_index(drop=True)


def summarize(df: pd.DataFrame, label: str) -> None:
    n_rows = len(df)
    n_readers = df["reader_id"].nunique()
    by_exp = df.groupby("reader_expertise")["reader_id"].nunique().to_dict()
    n_tests = df["test_session_id"].nunique()
    print(
        f"  [{label}] rows={n_rows:>5}  readers={n_readers:>3}  "
        f"experts={by_exp.get('expert', 0):>3}  non-experts={by_exp.get('non-expert', 0):>3}  "
        f"tests={n_tests:>3}"
    )


def main() -> None:
    print(f"Reading {RAW_FILE} ...")
    # Doctor export: semicolon separator, European decimal comma
    df_raw = pd.read_csv(RAW_FILE, sep=";", decimal=",")
    print(f"  raw shape: {df_raw.shape}, raw readers: {df_raw['tequ_user_id'].nunique()}")

    df_std = standardize(df_raw)
    df_std.to_csv(OUT / "panderm_standardized_v2.csv", index=False)
    print(f"\nWrote outputs/panderm_standardized_v2.csv")
    summarize(df_std[df_std["is_completed"]], "standardized (any valid)")

    df_95 = filter_95pct(df_std)
    df_95.to_csv(OUT / "panderm_cleaned_95pct.csv", index=False)
    print(f"\nWrote outputs/panderm_cleaned_95pct.csv")
    summarize(df_95, "≥95% per test")

    print("\nDone.")


if __name__ == "__main__":
    main()
