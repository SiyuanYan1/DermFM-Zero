"""
generate_demo_data.py
---------------------
Generates a fully synthetic demo dataset matching the column schema and
qualitative shape of the real `panderm_reader_data.csv` (87-reader RS2B
raw export from DermaChallenge). The demo data lets the pipeline
(01 -> 02 -> 03 -> 04) be run end-to-end for code verification without
disclosing any participant-level real data.

Key properties of the generated data:
  - Same 17 columns, same semicolon + decimal-comma format as the real export.
  - 87 fully random reader IDs (40-char hex from a per-run RNG; NOT derived
    from any predictable seed pattern). By default seed=None, so each run
    produces a completely different CSV; pass `--seed N` only when
    reproducibility is needed.
  - Random per-run test_id (UUID-style hex), not sequential.
  - Image identifiers use the `DEMO_xxxxxxx` prefix (NOT `ISIC_xxxxxxx`) so
    the synthetic data cannot be confused with real ISIC archive references.
  - ~150-160 test sessions (each test = 15 cases x 2 modes = 30 reading rows).
  - 11 diagnosis classes with roughly the same class frequencies as real.
  - Reader expertise split ~ 44 expert / 43 non-expert (`ability_2pl` label).
  - Demographic distributions (profession, age, gender) are sampled from
    categorical buckets with +/-15-20% per-bucket jitter, so demo aggregate
    counts do not exactly mirror the real cohort breakdown.
  - `tequ_correct` sampled with a per-reader skill bias (experts ~70% correct,
    non-experts ~50% correct); AI-assisted readings carry a small positive
    treatment effect so the downstream pipeline produces plausible (not real)
    statistics.
  - Missingness pattern: ~18% of readers have ALL their tests marked as
    "incomplete" (timeouts/blanks); a further ~10% of remaining tests are
    incomplete. This reproduces the >=95% per-test filter behaviour of the
    real cohort (87 readers / ~194 tests raw -> ~71 readers / ~165 tests
    after filter).

Output:
  demo_data/panderm_reader_data.csv

Usage:
  python generate_demo_data.py            # writes a fresh, system-random CSV
  python generate_demo_data.py --seed 7   # reproducible run with explicit seed

The generated values are NOT real reader responses. Do not use this file for
any inference about DermFM-Zero's true clinical performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG (match qualitative shape of real export)
# ---------------------------------------------------------------------------
N_READERS = 87
CASES_PER_TEST = 15
MODES = ["radio_radio_management47", "radio_radio_management47_help"]  # unaided / AI

CLASSES = [
    "NV", "MEL", "BKL", "BCC", "AKIEC", "DF",
    "INF", "VASC", "SCCKA", "OTHER_BENIGN", "OTHER_MALIGNANT",
]
# Approximate class frequency from the real cohort
CLASS_PROBS = np.array([0.175, 0.118, 0.118, 0.142, 0.058, 0.057,
                        0.023, 0.057, 0.064, 0.028, 0.007])
CLASS_PROBS = CLASS_PROBS / CLASS_PROBS.sum()

MANAGE_CHOICES = ["manage_dismiss", "manage_monitor",
                  "manage_biopsy", "manage_localtherapy"]

# Demographic aggregate counts; jittered per-run so demo does not exactly
# mirror the real cohort breakdown (which is public via the manuscript).
PROFESSION_COUNTS = {
    "dermatologyResident": 28,
    "boardCertifiedDermatologist": 25,
    "generalPractitioner": 11,
    "medicalStudent": 7,
    "resident": 5,
    "medicalSpecialist": 4,
    "physicianAssistant": 3,
    "nonMedical": 2,
    "nursePractitioner": 2,
}  # sums to ~87
GENDER_COUNTS = {"Male": 44, "Female": 39, "Other": 1, None: 3}
AGE_COUNTS = {
    "20-30": 5, "31-40": 4, "41-50": 5, "51-60": 1, "61+": 1,
    None: 71,
}

# Missingness pattern -- reproduces the >=95% filter behaviour of the real export.
P_BAD_READER = 0.18      # fraction of readers whose ALL tests fail the filter
P_BAD_TEST = 0.10        # fraction of remaining (good-reader) tests that fail
BAD_MISSING_MIN = 3      # number of timeouts injected into a "bad" 30-reading test
BAD_MISSING_MAX = 8


# ---------------------------------------------------------------------------
# Helpers (anonymisation: random ids, DEMO_ prefix, no predictable seed)
# ---------------------------------------------------------------------------
def random_id(rng: np.random.Generator, length: int = 40) -> str:
    """Untraceable random hex id; NOT derived from any predictable seed pattern."""
    return rng.bytes(length // 2 + 1).hex()[:length]


def demo_image_id(rng: np.random.Generator) -> tuple[int, str]:
    """Synthetic image identifier in DEMO_xxxxxxx format (does NOT collide
    with real ISIC database IDs)."""
    img_int = int(rng.integers(1, 9_999_999))
    return img_int, f"DEMO_{img_int:07d}"


def jittered_counts(rng: np.random.Generator, counts: dict, jitter: float = 0.20) -> dict:
    """Return a copy of `counts` with +/- `jitter` random perturbation per entry,
    so demo aggregates do not exactly match the real cohort breakdown."""
    out = {}
    for k, v in counts.items():
        if v <= 0:
            out[k] = 0
            continue
        delta = int(round(v * jitter * (rng.random() * 2 - 1)))
        out[k] = max(0, v + delta)
    return out


def categorical_assign(rng: np.random.Generator, counts: dict, n: int) -> list:
    """Sample n labels from a counts dict, then shuffle."""
    labels = []
    for label, k in counts.items():
        labels.extend([label] * k)
    if len(labels) != n:
        if len(labels) < n:
            extras = rng.choice(list(counts.keys()), size=(n - len(labels)))
            labels.extend(extras.tolist())
        else:
            labels = labels[:n]
    rng.shuffle(labels)
    return labels


# ---------------------------------------------------------------------------
# Reader cohort
# ---------------------------------------------------------------------------
def build_reader_cohort(rng: np.random.Generator) -> pd.DataFrame:
    """One row per reader; defines ability, demographics, n_tests, bad_reader flag."""
    reader_ids = [random_id(rng) for _ in range(N_READERS)]

    abilities = rng.choice(["expert", "non_expert"],
                           size=N_READERS, p=[0.506, 0.494])

    exp_acc = np.where(
        abilities == "expert",
        rng.uniform(0.70, 0.87, N_READERS),
        rng.uniform(0.17, 0.69, N_READERS),
    )

    prof_jit = jittered_counts(rng, PROFESSION_COUNTS, jitter=0.20)
    gen_jit = jittered_counts(rng, GENDER_COUNTS, jitter=0.15)
    age_jit = jittered_counts(rng, AGE_COUNTS, jitter=0.15)
    professions = categorical_assign(rng, prof_jit, N_READERS)
    genders = categorical_assign(rng, gen_jit, N_READERS)
    ages = categorical_assign(rng, age_jit, N_READERS)

    n_tests_choices = rng.choice([1, 2, 3, 4, 5],
                                 size=N_READERS,
                                 p=[0.50, 0.27, 0.13, 0.07, 0.03])

    # Pre-mark which readers are "bad" (all their tests will fail the >=95% filter)
    bad_reader = rng.random(N_READERS) < P_BAD_READER

    return pd.DataFrame({
        "tequ_user_id": reader_ids,
        "ability_2pl": abilities,
        "expected_accuracy_2pl": exp_acc,
        "profession": professions,
        "gender": genders,
        "age": ages,
        "n_tests": n_tests_choices,
        "bad_reader": bad_reader,
    })


# ---------------------------------------------------------------------------
# Reading rows (with realistic missingness)
# ---------------------------------------------------------------------------
def pick_management(rng: np.random.Generator, true_dx: str, is_correct: bool) -> str:
    malignant = {"MEL", "BCC", "AKIEC", "SCCKA", "OTHER_MALIGNANT"}
    benign = {"NV", "BKL", "DF", "VASC", "OTHER_BENIGN"}
    inflam = {"INF"}

    if true_dx in malignant:
        weights = [0.05, 0.10, 0.55, 0.30]
    elif true_dx in benign:
        weights = [0.45, 0.40, 0.10, 0.05]
    elif true_dx in inflam:
        weights = [0.10, 0.25, 0.10, 0.55]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    if not is_correct:
        weights = [w * 0.7 + 0.075 for w in weights]

    weights = np.array(weights) / np.sum(weights)
    return rng.choice(MANAGE_CHOICES, p=weights)


def build_readings(rng: np.random.Generator, cohort: pd.DataFrame) -> pd.DataFrame:
    """Expand cohort into per-reading rows (one row per reader x case x mode).
    Some tests are marked as "bad": >=2 of their 30 readings become timeouts,
    so they fail the >=95% per-test filter downstream."""
    rows = []
    n_bad_tests_total = 0
    n_total_tests = 0

    for _, reader in cohort.iterrows():
        skill = reader["expected_accuracy_2pl"]
        for t in range(int(reader["n_tests"])):
            test_id = random_id(rng)
            round_no = t + 1
            n_total_tests += 1

            # Decide if this test is "bad" (will fail the >=95% filter)
            if reader["bad_reader"]:
                test_bad = True   # bad readers: ALL tests bad
            else:
                test_bad = rng.random() < P_BAD_TEST

            # 15 cases for this test
            case_classes = rng.choice(CLASSES, size=CASES_PER_TEST, p=CLASS_PROBS)
            case_image_data = [demo_image_id(rng) for _ in range(CASES_PER_TEST)]
            case_image_ids = [img_int for img_int, _ in case_image_data]
            case_isic_ids = [demo_str for _, demo_str in case_image_data]

            # If bad test, pre-select which reading rows (out of 30) become timeouts
            # 30 rows = 15 cases x 2 modes. We mark by flat index 0..29.
            if test_bad:
                n_bad_tests_total += 1
                n_missing = int(rng.integers(BAD_MISSING_MIN, BAD_MISSING_MAX + 1))
                miss_idx = set(rng.choice(30, size=n_missing, replace=False).tolist())
            else:
                miss_idx = set()

            flat_idx = 0
            for q_no, (cls, img_id, isic_id) in enumerate(
                zip(case_classes, case_image_ids, case_isic_ids), start=1
            ):
                for mode in MODES:
                    if flat_idx in miss_idx:
                        # Inject a "timeout" / blank reading
                        answer = "timeout"
                        manage = np.nan
                        is_correct = np.nan
                    else:
                        p_correct = skill + (0.06 if mode.endswith("_help") else 0.0)
                        p_correct = float(np.clip(p_correct, 0.05, 0.95))
                        is_correct = rng.random() < p_correct
                        if is_correct:
                            answer = cls
                        else:
                            wrong_pool = [c for c in CLASSES if c != cls]
                            answer = rng.choice(wrong_pool)
                        manage = pick_management(rng, cls, is_correct)

                    rows.append({
                        "tequ_user_id": reader["tequ_user_id"],
                        "test_id": test_id,
                        "test_round": round_no,
                        "image_id": img_id,
                        "image_isic_id": isic_id,
                        "tequ_no": q_no,
                        "tequ_correct": (np.nan if (isinstance(is_correct, float) and np.isnan(is_correct))
                                         else (1.0 if is_correct else 0.0)),
                        "answer": answer,
                        "manage": manage,
                        "image_diag_id": cls,
                        "tequ_mode_id": mode,
                        "expected_accuracy_2pl": skill,
                        "ability_2pl": reader["ability_2pl"],
                        "age": reader["age"] if reader["age"] is not None else np.nan,
                        "yob": np.nan,
                        "profession": reader["profession"],
                        "gender": reader["gender"] if reader["gender"] is not None else np.nan,
                    })
                    flat_idx += 1

    print(f"  {n_total_tests} total tests; {n_bad_tests_total} marked as 'bad' "
          f"(will fail >=95% filter)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RS2B demo data.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: None = system random, different each run). "
                             "Pass an explicit integer (e.g. --seed 42) to make it reproducible.")
    parser.add_argument("--out", type=str, default="demo_data/panderm_reader_data.csv",
                        help="Output CSV path (default: demo_data/panderm_reader_data.csv).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    seed_label = "system-random" if args.seed is None else str(args.seed)

    print(f"Generating synthetic RS2B data (seed={seed_label}) ...")
    cohort = build_reader_cohort(rng)
    n_bad_readers = int(cohort["bad_reader"].sum())
    print(f"  {len(cohort)} readers; "
          f"{(cohort['ability_2pl'] == 'expert').sum()} expert / "
          f"{(cohort['ability_2pl'] == 'non_expert').sum()} non_expert; "
          f"{int(cohort['n_tests'].sum())} tests total; "
          f"{n_bad_readers} pre-marked 'bad reader'")

    df = build_readings(rng, cohort)
    print(f"  {len(df)} reading rows generated")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"  wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print("\nDone. Demo data is purely synthetic -- do not use for inference.")


if __name__ == "__main__":
    main()
