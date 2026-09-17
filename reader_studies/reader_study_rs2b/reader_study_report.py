"""
04_reader_study_report.py
-------------------------
Produces a single PDF report `outputs/R2B_reader_study_report.pdf` summarising
the v2 reader study (≥95% filter, 71 readers):

  Page 1 — Title page + narrative description + headline results
  Page 2 — Reader cohort summary (v1 → v2 raw → v2 ≥95% kept)
  Page 3 — Detailed demographics of the included 71-reader cohort
           (profession, age band, gender, expertise)
  Page 4 — Reader inclusion / exclusion flow and filter rule
  Page 5 — Results summary tables (overall, by class, by experience)
  Page 6 — RS2B Figure 5 (6 panels) — merged from outputs/RS2B_figure.pdf

The figure PDF is merged in via pypdf so we keep the publication-quality
vector layout produced by 03_fig2_plot.py.

Run **after** 01/02/03 have been executed so all CSV/PDF inputs exist.

Usage
-----
    python 04_reader_study_report.py
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from pypdf import PdfReader, PdfWriter

# ---------------------------------------------------------------------------
# CLI: --demo (default) | --real
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="RS2B Step 4: combined report PDF.")
_parser.add_argument("--real", action="store_const", const="real", dest="mode",
                     help="Use real_data/ -> real_output/")
_parser.add_argument("--demo", action="store_const", const="demo", dest="mode",
                     help="Use demo_data/ -> demo_output/  (default) (demo data not shipped)")
_parser.set_defaults(mode="real")
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / ("real_data" if _args.mode == "real" else "demo_data")
OUT = ROOT / ("real_output" if _args.mode == "real" else "demo_output")
OUT.mkdir(parents=True, exist_ok=True)

# V1 cohort CSV (optional; only exists alongside the real-data legacy pipeline)
V1_CSV = ROOT.parent / "panderm_cleaned_15cases.csv"
RAW_V2 = DATA / "panderm_reader_data.csv"
CLEAN_V2 = OUT / "panderm_cleaned_95pct.csv"
RESULTS_CSV = OUT / "rs2b_reader_study_data_95pct.csv"
FIGURE_PDF = OUT / "RS2B_figure.pdf"

REPORT_PAGES_PDF = OUT / "_report_pages.pdf"      # tables + narrative (temp)
REPORT_PDF = OUT / "R2B_reader_study_report.pdf"  # final merged report
print(f"[mode={_args.mode}]  DATA={DATA.name}  OUT={OUT.name}")

# Clean-error pre-check on required inputs (real or demo).
for _p in (RAW_V2, CLEAN_V2, RESULTS_CSV):
    if not _p.exists():
        print(
            f"Input missing: {_p}. "
            ""
        )
        sys.exit(0)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

C_UNAIDED = "#4E79A7"
C_ASSISTED = "#E15759"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading inputs ...")
raw_v2 = pd.read_csv(RAW_V2, sep=";", decimal=",")
demo = (
    raw_v2.drop_duplicates("tequ_user_id")[
        ["tequ_user_id", "ability_2pl", "profession", "age", "gender",
         "expected_accuracy_2pl"]
    ]
    .rename(columns={"tequ_user_id": "reader_id", "ability_2pl": "expertise"})
    .replace({"expertise": {"non_expert": "non-expert"}})
)
demo["gender"] = (
    demo["gender"].astype(str).str.strip().str.capitalize().replace({"Nan": "Unknown"})
)
demo["age"] = demo["age"].fillna("Unknown").astype(str).str.strip()
# expected_accuracy_2pl is a continuous IRT ability score (0–1) — the
# closest proxy we have for "years of experience" / domain skill
demo["expected_accuracy_2pl"] = pd.to_numeric(
    demo["expected_accuracy_2pl"], errors="coerce"
)

if V1_CSV.exists():
    v1 = pd.read_csv(V1_CSV)
    v1_ids = set(v1["reader_id"].unique())
else:
    print(f"[note] V1 cohort CSV not found ({V1_CSV.name}); "
          f"v1-vs-v2 overlap stats will be skipped.")
    v1 = None
    v1_ids = set()
v2_all_ids = set(raw_v2["tequ_user_id"].unique())

clean_v2 = pd.read_csv(CLEAN_V2)
v2_95_ids = set(clean_v2["reader_id"].unique())
included_demo = demo[demo["reader_id"].isin(v2_95_ids)].copy()

results = pd.read_csv(RESULTS_CSV)
if "N_Unaided" not in results.columns:
    _n = results["N_readers"].where(results["N_readers"].notna(), results["N"])
    results["N_Unaided"] = _n
    results["N_Assisted"] = _n

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_table(ax, df: pd.DataFrame, title: str, font_size: int = 10,
                 col_widths=None):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.5)
    for j, _ in enumerate(df.columns):
        c = tbl[(0, j)]
        c.set_text_props(weight="bold", color="white")
        c.set_facecolor("#333333")
    for i in range(1, len(df) + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            tbl[(i, j)].set_facecolor(bg)
    last = df.iloc[-1].astype(str).iloc[0] if len(df) else ""
    if last == "TOTAL":
        for j in range(len(df.columns)):
            c = tbl[(len(df), j)]
            c.set_text_props(weight="bold")
            c.set_facecolor("#E8E8E8")


# Headline results extracted for the cover page
def _row(table_name: str, category: str):
    sub = results[(results["Table"] == table_name) & (results["Category"] == category)]
    return sub.iloc[0] if len(sub) else None


def _pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def _p(p):
    if pd.isna(p): return "—"
    if p < 0.001: return "<0.001"
    if p < 1e-10: return "<10⁻¹⁰"
    return f"{p:.3g}"


# ---------------------------------------------------------------------------
# Page 1 — title + narrative + headline numbers
# ---------------------------------------------------------------------------
def page_title(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    fig.text(0.06, 0.94, "RS2B Reader Study v2 — Cohort & Results Report",
             fontsize=18, fontweight="bold")
    fig.text(0.06, 0.91, f"Generated: {date.today().isoformat()}    "
             "Filter: ≥95% answered items per test    Cohort N = 71",
             fontsize=10, color="#555", style="italic")

    narrative = (
        "Study design.  RS2B is a prospective sequential multimodal reader "
        "study evaluating whether DermFM-Zero (zero-shot) assistance improves "
        "diagnostic accuracy and management appropriateness in skin cancer "
        "specialist care.  Each reader evaluated paired dermoscopic + clinical "
        "image cases spanning 11 lesion classes suspected of malignancy.  For "
        "every case a reader produced (i) an unaided differential diagnosis "
        "and clinical management decision, then (ii) repeated the assessment "
        "with DermFM-Zero's top-3 probabilities visible.  One test = 15 cases × "
        "2 modes = 30 readings.\n\n"
        "v2 changes vs the original Nature submission (34 readers):\n"
        "  • Expanded cohort — the clinician's updated export contains 87 "
        "registered participants (all 34 v1 readers retained, plus 53 new).\n"
        "  • New management appropriateness table — data-driven 11-class × "
        "4-action mapping (data/management_table.csv).  Replaces the hard-"
        "coded dict in v1 and now matches the clinician's lme4 GLMM script.\n"
        "  • New inclusion rule — keep a test only if ≥95% of its 30 readings "
        "are valid (i.e., ≥29 valid answers, no timeouts, no blanks).  Selects "
        "71 readers / 165 tests / 4,929 readings for the analytic cohort.\n\n"
        "Statistical analysis.  Reader-level proportions are compared with "
        "paired t-tests (this folder).  A separate folder (R2B_glmm_analysis/) "
        "fits a binomial generalised linear mixed model adjusting for class "
        "difficulty and within-reader / within-test correlation, used as the "
        "primary inference in the manuscript."
    )
    fig.text(0.06, 0.85, "Narrative", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.83, narrative, fontsize=10, va="top",
             wrap=True, linespacing=1.5)

    # Headline-results mini table
    overall_acc = results[results["Table"] == "RS2_Overall_Accuracy"]
    overall_mgmt = results[results["Table"] == "RS2_Overall_Management_Appropriate"]
    by_exp_acc = results[results["Table"] == "RS2_Accuracy_by_Experience"]
    by_exp_mgmt = results[results["Table"] == "RS2_Management_by_Experience"]

    headline = pd.DataFrame(
        [
            ["Overall diagnostic accuracy",
             _pct(overall_acc.iloc[0]["Mean"]), _pct(overall_acc.iloc[1]["Mean"]),
             _p(overall_acc.iloc[1]["p_value"])],
            ["Overall management appropriateness",
             _pct(overall_mgmt.iloc[0]["Mean"]), _pct(overall_mgmt.iloc[1]["Mean"]),
             _p(overall_mgmt.iloc[1]["p_value"])],
            ["Non-expert accuracy",
             _pct(by_exp_acc.set_index("Category").loc["Non-expert", "Without_AI_Mean"]),
             _pct(by_exp_acc.set_index("Category").loc["Non-expert", "With_AI_Mean"]),
             _p(by_exp_acc.set_index("Category").loc["Non-expert", "p_value"])],
            ["Expert accuracy",
             _pct(by_exp_acc.set_index("Category").loc["Expert", "Without_AI_Mean"]),
             _pct(by_exp_acc.set_index("Category").loc["Expert", "With_AI_Mean"]),
             _p(by_exp_acc.set_index("Category").loc["Expert", "p_value"])],
            ["Non-expert mgmt appropriateness",
             _pct(by_exp_mgmt.set_index("Category").loc["Non-expert", "Without_AI_Mean"]),
             _pct(by_exp_mgmt.set_index("Category").loc["Non-expert", "With_AI_Mean"]),
             _p(by_exp_mgmt.set_index("Category").loc["Non-expert", "p_value"])],
            ["Expert mgmt appropriateness",
             _pct(by_exp_mgmt.set_index("Category").loc["Expert", "Without_AI_Mean"]),
             _pct(by_exp_mgmt.set_index("Category").loc["Expert", "With_AI_Mean"]),
             _p(by_exp_mgmt.set_index("Category").loc["Expert", "p_value"])],
        ],
        columns=["Metric", "Unaided", "AI-assisted", "P (paired t)"]
    )
    ax_head = fig.add_axes([0.06, 0.18, 0.88, 0.20])
    render_table(ax_head, headline, "Headline results (paired t-test, 71 readers)",
                 font_size=10, col_widths=[0.45, 0.15, 0.18, 0.17])

    fig.text(0.06, 0.10,
             "Tip: For the primary inference in the manuscript, use the GLMM "
             "outputs from the sibling folder R2B_glmm_analysis/ — they adjust "
             "for image difficulty and within-reader correlation.",
             fontsize=9, style="italic", color="#666")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2 — cohort summary across v1 → v2 raw → v2 included
# ---------------------------------------------------------------------------
def page_cohort_summary(pdf):
    cohorts = [
        ("Original v1 (Nature submission)", v1_ids, "v1"),
        ("v2 raw export (any registered)",  v2_all_ids, "v2_raw"),
        ("v2 included (≥95% per test)",     v2_95_ids, "v2_inc"),
    ]
    # v1 expertise from its own column (only if v1 cohort CSV was found)
    if v1 is not None:
        v1_exp = v1.drop_duplicates("reader_id").set_index("reader_id")["reader_expertise"]
    else:
        v1_exp = pd.Series(dtype=object)  # empty; v1 cohort skipped

    rows = []
    for label, ids, _ in cohorts:
        d = demo[demo["reader_id"].isin(ids)].copy()
        if label.startswith("Original v1"):
            d["expertise"] = d["reader_id"].map(v1_exp).fillna(d["expertise"])
        n_exp = int((d["expertise"] == "expert").sum())
        n_non = int((d["expertise"] == "non-expert").sum())
        rows.append({
            "Cohort": label,
            "N readers": len(ids),
            "Experts": n_exp,
            "Non-experts": n_non,
            "Shared with v1": len(ids & v1_ids),
            "New vs v1": len(ids - v1_ids),
        })
    summary = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.94, "Page 2 · Reader cohort summary",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.91,
             "All 34 original readers were retained in v2; the ≥95% filter "
             "preserves 33 of them and adds 38 new participants for a final "
             "cohort of 71 (39 expert, 32 non-expert).",
             fontsize=10, color="#444", style="italic", wrap=True)

    ax_t = fig.add_axes([0.06, 0.68, 0.88, 0.20])
    render_table(ax_t, summary, "Table 1.  Cohort progression",
                 font_size=10, col_widths=[0.38, 0.13, 0.11, 0.14, 0.16, 0.12])

    # A small horizontal bar chart visualizing the cohort sizes
    ax_b = fig.add_axes([0.06, 0.18, 0.88, 0.40])
    cohort_names = [r["Cohort"] for r in rows]
    n_exp = [r["Experts"] for r in rows]
    n_non = [r["Non-experts"] for r in rows]
    y = np.arange(len(cohort_names))
    ax_b.barh(y, n_exp, label="Expert", color=C_UNAIDED, alpha=0.85,
              edgecolor="black", linewidth=0.6)
    ax_b.barh(y, n_non, left=n_exp, label="Non-expert", color=C_ASSISTED,
              alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, (e, n) in enumerate(zip(n_exp, n_non)):
        ax_b.text(e + n + 1.0, i, f"{e + n}", va="center", fontsize=10,
                  fontweight="bold")
        if e > 4:
            ax_b.text(e / 2, i, f"{e}", va="center", ha="center",
                      color="white", fontsize=9, fontweight="bold")
        if n > 4:
            ax_b.text(e + n / 2, i, f"{n}", va="center", ha="center",
                      color="white", fontsize=9, fontweight="bold")
    ax_b.set_yticks(y); ax_b.set_yticklabels(cohort_names, fontsize=10)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Readers", fontsize=11)
    ax_b.set_title("Figure 1.  Cohort size by expertise",
                   loc="left", fontsize=12, fontweight="bold")
    ax_b.legend(loc="lower right", fontsize=9, frameon=False)
    ax_b.set_xlim(0, max(e + n for e, n in zip(n_exp, n_non)) * 1.15)
    ax_b.grid(axis="x", alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3 — detailed demographics of the 71-reader cohort
# ---------------------------------------------------------------------------
def page_demographics(pdf):
    inc = included_demo

    # Profession table
    prof_counts = inc["profession"].value_counts(dropna=False)
    prof_pct = (prof_counts / prof_counts.sum() * 100).round(1)
    prof_tbl = pd.DataFrame({
        "Profession": prof_counts.index,
        "N": prof_counts.values,
        "%": [f"{p:.1f}" for p in prof_pct.values],
    })
    prof_tbl.loc[len(prof_tbl)] = ["TOTAL", int(prof_counts.sum()), "100.0"]

    # Expertise breakdown (with mean IRT ability)
    exp_grp = (
        inc.groupby("expertise")
           .agg(N=("reader_id", "count"),
                Mean_IRT=("expected_accuracy_2pl", "mean"),
                Median_IRT=("expected_accuracy_2pl", "median"),
                Min_IRT=("expected_accuracy_2pl", "min"),
                Max_IRT=("expected_accuracy_2pl", "max"))
           .reset_index()
    )
    exp_grp["Mean_IRT"] = exp_grp["Mean_IRT"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    exp_grp["Median_IRT"] = exp_grp["Median_IRT"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    exp_grp["Min_IRT"] = exp_grp["Min_IRT"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    exp_grp["Max_IRT"] = exp_grp["Max_IRT"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    exp_grp.columns = ["Expertise", "N", "Mean", "Median", "Min", "Max"]

    # Gender breakdown
    gender_counts = inc["gender"].value_counts(dropna=False)
    gender_tbl = pd.DataFrame({
        "Gender": gender_counts.index,
        "N": gender_counts.values,
        "%": [f"{(c/gender_counts.sum())*100:.1f}" for c in gender_counts.values],
    })
    gender_tbl.loc[len(gender_tbl)] = ["TOTAL", int(gender_counts.sum()), "100.0"]

    # Age breakdown
    age_order = ["20-30", "31-40", "41-50", "51-60", "61+", "Unknown"]
    age_counts = inc["age"].value_counts().reindex(
        [a for a in age_order if a in inc["age"].unique()]
        + [a for a in inc["age"].unique() if a not in age_order],
        fill_value=0,
    )
    age_tbl = pd.DataFrame({
        "Age band": age_counts.index,
        "N": age_counts.values,
        "%": [f"{(c/age_counts.sum())*100:.1f}" for c in age_counts.values],
    })
    age_tbl.loc[len(age_tbl)] = ["TOTAL", int(age_counts.sum()), "100.0"]

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.95, "Page 3 · Demographics of the included 71-reader cohort",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.92, "Profession is the most informative proxy for clinical "
             "experience; the IRT ability score (expected_accuracy_2pl, 0–1) is the "
             "platform-derived continuous skill estimate.",
             fontsize=9, color="#555", style="italic", wrap=True)

    ax1 = fig.add_axes([0.06, 0.55, 0.88, 0.32])
    render_table(ax1, prof_tbl, "Table 2.  Profession distribution",
                 font_size=10, col_widths=[0.55, 0.12, 0.13])

    ax2 = fig.add_axes([0.06, 0.35, 0.42, 0.16])
    render_table(ax2, gender_tbl, "Table 3.  Gender",
                 font_size=10, col_widths=[0.45, 0.25, 0.25])

    ax3 = fig.add_axes([0.52, 0.35, 0.42, 0.16])
    render_table(ax3, age_tbl, "Table 4.  Age band",
                 font_size=10, col_widths=[0.45, 0.25, 0.25])

    ax4 = fig.add_axes([0.06, 0.10, 0.88, 0.20])
    render_table(ax4, exp_grp, "Table 5.  Expertise group × IRT ability score",
                 font_size=10, col_widths=[0.30, 0.10, 0.15, 0.15, 0.15, 0.15])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4 — reader-flow / inclusion
# ---------------------------------------------------------------------------
def page_inclusion_flow(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.95, "Page 4 · Inclusion / exclusion flow",
             fontsize=15, fontweight="bold")

    # Compute flow stats
    n_registered = len(v2_all_ids)
    n_any_valid_test = raw_v2.copy()
    n_any_valid_test["valid"] = (
        n_any_valid_test["answer"].notna()
        & (n_any_valid_test["answer"].astype(str).str.strip() != "")
        & (n_any_valid_test["answer"] != "timeout")
    )
    n_tests_started = n_any_valid_test["test_id"].nunique()
    tests_valid_count = (
        n_any_valid_test[n_any_valid_test["valid"]].groupby("test_id").size()
    )
    n_tests_kept = (tests_valid_count >= 0.95 * 30).sum()
    n_tests_excluded = n_tests_started - n_tests_kept
    n_readers_kept = len(v2_95_ids)
    n_readers_excluded = n_registered - n_readers_kept

    flow = pd.DataFrame([
        ["Total readers in raw export", n_registered, "—"],
        ["Total tests in raw export (one test = 15 cases × 2 modes)", n_tests_started, "—"],
        ["Tests kept after ≥95% filter", n_tests_kept,
         f"({n_tests_kept / n_tests_started:.0%})"],
        ["Tests excluded (too many timeouts / blanks)", n_tests_excluded,
         f"({n_tests_excluded / n_tests_started:.0%})"],
        ["Readers retained (at least one valid test)", n_readers_kept,
         f"({n_readers_kept / n_registered:.0%})"],
        ["Readers excluded (no valid test)", n_readers_excluded,
         f"({n_readers_excluded / n_registered:.0%})"],
    ], columns=["Step", "N", "Share"])

    ax = fig.add_axes([0.06, 0.55, 0.88, 0.30])
    render_table(ax, flow, "Table 6.  Reader inclusion flow",
                 font_size=10, col_widths=[0.62, 0.18, 0.18])

    fig.text(0.06, 0.45, "Filter rule (matches new R script)", fontsize=12,
             fontweight="bold")
    fig.text(0.06, 0.27,
             "A test (one reader's 15-case round in both Unaided and AI modes) "
             "is included if at least 95% of its 30 expected answer slots are "
             "valid — i.e., ≥29 of 30 readings are non-blank, non-timeout. "
             "Tests below this threshold are dropped wholesale.  A reader is "
             "retained if at least one of their tests survives.  Readers who "
             "completed multiple test sessions contribute every kept session.\n\n"
             "Strict alternative (not used in this folder).  Earlier versions "
             "of the pipeline required all 30 slots filled (100%).  That rule "
             "kept 62 readers / 144 tests and produced quantitatively similar "
             "headline results.  The strict-15 version is still available in "
             "the GLMM sensitivity folder.",
             fontsize=10, va="top", wrap=True, linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5 — results tables
# ---------------------------------------------------------------------------
def page_results(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, "Page 5 · Results summary tables",
             fontsize=15, fontweight="bold")

    # Overall
    overall_acc = results[results["Table"] == "RS2_Overall_Accuracy"].copy()
    overall_mgmt = results[results["Table"] == "RS2_Overall_Management_Appropriate"].copy()
    overall = pd.concat([overall_acc.assign(Outcome="Accuracy"),
                          overall_mgmt.assign(Outcome="Mgmt appropriate")])
    overall["Mean"] = overall["Mean"].map(_pct)
    overall["CI"] = overall.apply(
        lambda r: f"[{r['CI_Lower']:.3f}, {r['CI_Upper']:.3f}]" if pd.notna(r["CI_Lower"]) else "—",
        axis=1,
    )
    overall["P"] = overall["p_value"].map(_p)
    overall_tbl = overall[["Outcome", "Category", "Mean", "CI", "P"]]
    overall_tbl.columns = ["Outcome", "Condition", "Mean", "95% CI", "P"]

    ax1 = fig.add_axes([0.06, 0.74, 0.88, 0.18])
    render_table(ax1, overall_tbl, "Table 7.  Overall results (Unaided vs AI)",
                 font_size=9.5, col_widths=[0.18, 0.32, 0.13, 0.22, 0.15])

    # By experience
    by_exp_acc = results[results["Table"] == "RS2_Accuracy_by_Experience"].copy()
    by_exp_mgmt = results[results["Table"] == "RS2_Management_by_Experience"].copy()
    by_exp = pd.concat([by_exp_acc.assign(Outcome="Accuracy"),
                         by_exp_mgmt.assign(Outcome="Mgmt appropriate")])
    by_exp_tbl = by_exp[[
        "Outcome", "Category", "Without_AI_Mean", "With_AI_Mean",
        "p_value", "corrected_p_value", "N_readers",
    ]].copy()
    by_exp_tbl["Without_AI_Mean"] = by_exp_tbl["Without_AI_Mean"].map(_pct)
    by_exp_tbl["With_AI_Mean"] = by_exp_tbl["With_AI_Mean"].map(_pct)
    by_exp_tbl["p_value"] = by_exp_tbl["p_value"].map(_p)
    by_exp_tbl["corrected_p_value"] = by_exp_tbl["corrected_p_value"].map(_p)
    by_exp_tbl["N_readers"] = by_exp_tbl["N_readers"].astype(int)
    by_exp_tbl.columns = ["Outcome", "Group", "Unaided", "AI", "P", "P (Bonf)", "N"]

    ax2 = fig.add_axes([0.06, 0.50, 0.88, 0.18])
    render_table(ax2, by_exp_tbl, "Table 8.  Stratified by experience",
                 font_size=9.5,
                 col_widths=[0.17, 0.17, 0.12, 0.12, 0.12, 0.15, 0.10])

    # Class-wise (top ranked by significance)
    cls = results[results["Table"] == "RS2_Class_Specific_Accuracy"].copy()
    cls_tbl = cls[[
        "Category", "Without_AI_Mean", "With_AI_Mean",
        "p_value", "corrected_p_value", "N_Unaided", "N_Assisted",
    ]].copy()
    cls_tbl["Without_AI_Mean"] = cls_tbl["Without_AI_Mean"].map(_pct)
    cls_tbl["With_AI_Mean"] = cls_tbl["With_AI_Mean"].map(_pct)
    cls_tbl["p_value"] = cls_tbl["p_value"].map(_p)
    cls_tbl["corrected_p_value"] = cls_tbl["corrected_p_value"].map(_p)
    cls_tbl["N_Unaided"] = cls_tbl["N_Unaided"].astype(int)
    cls_tbl["N_Assisted"] = cls_tbl["N_Assisted"].astype(int)
    cls_tbl.columns = ["Class", "Unaided", "AI", "P", "P (Bonf)",
                       "N (Una.)", "N (AI)"]

    ax3 = fig.add_axes([0.06, 0.06, 0.88, 0.38])
    render_table(ax3, cls_tbl, "Table 9.  Class-specific accuracy",
                 font_size=9, col_widths=[0.18, 0.13, 0.12, 0.12, 0.15, 0.15, 0.15])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------
print("Rendering report pages ...")
with PdfPages(REPORT_PAGES_PDF) as pdf:
    page_title(pdf)
    page_cohort_summary(pdf)
    page_demographics(pdf)
    page_inclusion_flow(pdf)
    page_results(pdf)

# Merge the figure PDF as the final page
print("Merging RS2B Figure 5 (6 panels) ...")
writer = PdfWriter()
for page in PdfReader(str(REPORT_PAGES_PDF)).pages:
    writer.add_page(page)
if FIGURE_PDF.exists():
    for page in PdfReader(str(FIGURE_PDF)).pages:
        writer.add_page(page)
else:
    print(f"  WARNING: {FIGURE_PDF.name} not found — run 03_fig2_plot.py first.")

with open(REPORT_PDF, "wb") as f:
    writer.write(f)

print(f"\n✅ Final report: {REPORT_PDF}")
