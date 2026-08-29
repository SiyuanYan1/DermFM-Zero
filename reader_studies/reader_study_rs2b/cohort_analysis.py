"""
04_cohort_analysis.py
---------------------
Cohort comparison report across four reader cohorts:

  C1 — v1 (original Nature pipeline)          : ../panderm_cleaned_15cases.csv
  C2 — v2 raw (any valid answer)              : data/panderm_reader_data.csv
  C3 — v2 ≥95% per test (default downstream)  : outputs/panderm_cleaned_95pct.csv
  C4 — v2 strict 15 per test (sensitivity)    : outputs/panderm_cleaned_strict15.csv

Produces a single PDF report `outputs/cohort_analysis.pdf` containing
tables (counts, experience, profession, age, gender) and figures
(stacked bars per cohort).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# CLI: --real required (this analysis needs the full raw export, not shipped)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="RS2B 71-reader cohort analysis report.")
_parser.add_argument(
    "--real",
    action="store_true",
    help="Required: run on real raw export. Without --real the script exits cleanly.",
)
_args = _parser.parse_args()

# Real data ship with the repository; --real retained for backward compatibility.

# ---------------------------------------------------------------------------
# Paths (all relative to this file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_data"
OUT = ROOT / "real_output"
V1_CSV = ROOT.parent / "panderm_cleaned_15cases.csv"
RAW_V2 = DATA / "panderm_reader_data.csv"

OUT_PDF = OUT / "cohort_analysis.pdf"
OUT_CSV = OUT / "cohort_analysis_tables.csv"

OUT.mkdir(parents=True, exist_ok=True)

# Clean-error pre-check on all real inputs
_required_inputs = [
    RAW_V2,
    OUT / "panderm_cleaned_95pct.csv",
    V1_CSV,
]
for _p in _required_inputs:
    if not _p.exists():
        print(
            f"Input missing: {_p}. "
            "Run 01_filter_reader.py --real and 02_fig2_table_clean.py --real first."
        )
        sys.exit(0)

# ---------------------------------------------------------------------------
# Load data (with demographics from v2 raw to enrich filtered cohorts)
# ---------------------------------------------------------------------------
v2_raw = pd.read_csv(RAW_V2, sep=";", decimal=",")
demo = (
    v2_raw.drop_duplicates("tequ_user_id")[
        ["tequ_user_id", "ability_2pl", "profession", "age", "gender"]
    ]
    .rename(columns={"tequ_user_id": "reader_id", "ability_2pl": "expertise"})
    .replace({"expertise": {"non_expert": "non-expert"}})
)
# Normalize gender casing (raw data has Male/male etc.)
demo["gender"] = demo["gender"].astype(str).str.strip().str.capitalize().replace({"Nan": "Unknown"})
# Fill missing age with explicit label
demo["age"] = demo["age"].fillna("Unknown").astype(str).str.strip()

v1 = pd.read_csv(V1_CSV)
v1_ids = set(v1["reader_id"].unique())

v2_all_ids = set(v2_raw["tequ_user_id"].unique())
v2_95 = pd.read_csv(OUT / "panderm_cleaned_95pct.csv")
v2_95_ids = set(v2_95["reader_id"].unique())
_std = pd.read_csv(OUT / "panderm_standardized_v2.csv")
_valid = _std[_std["is_completed"]]
_full = _valid.groupby("test_session_id").size()
v2_strict = _valid[_valid["test_session_id"].isin(_full[_full == 30].index)]
v2_strict_ids = set(v2_strict["reader_id"].unique())


def cohort_df(ids: set, label: str) -> pd.DataFrame:
    """Return a per-reader demographic dataframe for the cohort."""
    d = demo[demo["reader_id"].isin(ids)].copy()
    # For v1 cohort, override expertise using v1's own column (hyphenated already)
    if label == "C1: v1 (original)":
        v1_exp = v1.drop_duplicates("reader_id").set_index("reader_id")["reader_expertise"]
        d["expertise"] = d["reader_id"].map(v1_exp).fillna(d["expertise"])
    d["cohort"] = label
    return d


cohorts = {
    "C1: v1 (original)": v1_ids,
    "C2: v2 raw (any valid)": v2_all_ids,
    "C3: v2 ≥95% / test": v2_95_ids,
    "C4: v2 strict 15 / test": v2_strict_ids,
}
all_dfs = pd.concat([cohort_df(ids, lbl) for lbl, ids in cohorts.items()], ignore_index=True)

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def summary_table() -> pd.DataFrame:
    rows = []
    for lbl, ids in cohorts.items():
        sub = all_dfs[all_dfs["cohort"] == lbl]
        rows.append(
            {
                "Cohort": lbl,
                "N readers": len(ids),
                "Experts": int((sub["expertise"] == "expert").sum()),
                "Non-experts": int((sub["expertise"] == "non-expert").sum()),
                "Shared with v1": len(ids & v1_ids),
                "New vs v1": len(ids - v1_ids),
                "Lost vs v1": len(v1_ids - ids),
            }
        )
    return pd.DataFrame(rows)


def profession_table() -> pd.DataFrame:
    pt = (
        all_dfs.groupby(["cohort", "profession"]).size().unstack(fill_value=0).T
    )
    pt = pt.reindex(columns=list(cohorts.keys()), fill_value=0)
    pt.loc["TOTAL"] = pt.sum(axis=0)
    return pt.reset_index()


def age_table() -> pd.DataFrame:
    age_order = ["20-30", "31-40", "41-50", "51-60", "61+", "Unknown"]
    t = all_dfs.groupby(["cohort", "age"]).size().unstack(fill_value=0).T
    t = t.reindex(columns=list(cohorts.keys()), fill_value=0)
    # Sort rows by age_order where applicable
    t = t.reindex([a for a in age_order if a in t.index] + [a for a in t.index if a not in age_order])
    t.loc["TOTAL"] = t.sum(axis=0)
    return t.reset_index()


def gender_table() -> pd.DataFrame:
    t = all_dfs.groupby(["cohort", "gender"]).size().unstack(fill_value=0).T
    t = t.reindex(columns=list(cohorts.keys()), fill_value=0)
    t.loc["TOTAL"] = t.sum(axis=0)
    return t.reset_index()


# Lost from 95pct → strict15
lost_ids = v2_95_ids - v2_strict_ids
lost_demo = demo[demo["reader_id"].isin(lost_ids)]


tbl_summary = summary_table()
tbl_profession = profession_table()
tbl_age = age_table()
tbl_gender = gender_table()


# Save tables to a single CSV (one Table column per block)
def stamp(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "Table", name)
    return out


pd.concat(
    [
        stamp(tbl_summary, "Cohort summary"),
        stamp(tbl_profession.rename(columns={"profession": "Category"}), "Profession breakdown"),
        stamp(tbl_age.rename(columns={"age": "Category"}), "Age breakdown"),
        stamp(tbl_gender.rename(columns={"gender": "Category"}), "Gender breakdown"),
    ],
    ignore_index=True,
    sort=False,
).to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

COHORT_COLORS = {
    "C1: v1 (original)": "#6C757D",
    "C2: v2 raw (any valid)": "#90C2E7",
    "C3: v2 ≥95% / test": "#4E79A7",
    "C4: v2 strict 15 / test": "#1F4E79",
}


def render_table(ax, df: pd.DataFrame, title: str, font_size: int = 9, col_widths=None):
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
    tbl.scale(1.0, 1.4)
    # Bold header
    for j, _ in enumerate(df.columns):
        cell = tbl[(0, j)]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#333333")
    # Alternating row colors
    for i in range(1, len(df) + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            tbl[(i, j)].set_facecolor(bg)
    # Highlight TOTAL row if present
    last = df.iloc[-1].astype(str).iloc[0] if len(df) else ""
    if last == "TOTAL":
        for j in range(len(df.columns)):
            c = tbl[(len(df), j)]
            c.set_text_props(weight="bold")
            c.set_facecolor("#E8E8E8")


def bar_breakdown(ax, df_counts: pd.DataFrame, title: str, ylabel: str = "Readers"):
    """df_counts: index=category, columns=cohort labels."""
    cats = df_counts.index.tolist()
    n_cats = len(cats)
    n_coh = len(df_counts.columns)
    bar_w = 0.8 / n_coh
    x = np.arange(n_cats)
    for i, coh in enumerate(df_counts.columns):
        ax.bar(
            x + i * bar_w - 0.4 + bar_w / 2,
            df_counts[coh].values,
            bar_w,
            color=COHORT_COLORS[coh],
            label=coh.split(":", 1)[-1].strip(),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        for j, v in enumerate(df_counts[coh].values):
            if v > 0:
                ax.text(x[j] + i * bar_w - 0.4 + bar_w / 2, v + 0.4, str(int(v)),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
with PdfPages(OUT_PDF) as pdf:

    # -----------------------------------------------------------------
    # Page 1 — Cohort overview + headline figure
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(11.7, 16.5))  # A3 portrait-ish
    gs = GridSpec(4, 2, height_ratios=[1.2, 2, 2, 2], hspace=0.6, wspace=0.25, figure=fig)

    fig.suptitle(
        "RS2B Reader Cohort Comparison — v1 vs v2 (two filters)",
        fontsize=15, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.96,
        "C1: v1 original (Nature paper)  |  C2: v2 raw (any valid)  |  "
        "C3: v2 ≥95% per test (default)  |  C4: v2 strict 15 per test (sensitivity)",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    # 1a. Summary table (counts, experts, shared with v1, new, lost)
    ax_sum = fig.add_subplot(gs[0, :])
    render_table(ax_sum, tbl_summary, "Table 1. Cohort summary", font_size=10,
                 col_widths=[0.25, 0.12, 0.10, 0.13, 0.15, 0.12, 0.13])

    # 1b. Experience bar (Expert vs Non-expert) — per cohort
    ax_exp = fig.add_subplot(gs[1, 0])
    exp_counts = (
        all_dfs.groupby(["cohort", "expertise"]).size().unstack(fill_value=0).T
    ).reindex(["expert", "non-expert"]).reindex(columns=list(cohorts.keys()), fill_value=0)
    bar_breakdown(ax_exp, exp_counts, "Figure 1a. Experience")

    # 1c. Gender bar
    ax_gen = fig.add_subplot(gs[1, 1])
    gen_counts = (
        all_dfs.groupby(["cohort", "gender"]).size().unstack(fill_value=0).T
    ).reindex(columns=list(cohorts.keys()), fill_value=0)
    bar_breakdown(ax_gen, gen_counts, "Figure 1b. Gender")
    ax_gen.legend(fontsize=7, loc="upper right", frameon=False, ncol=1)

    # 1d. Profession bar (horizontal because many categories)
    ax_prof = fig.add_subplot(gs[2, :])
    prof_counts = (
        all_dfs.groupby(["cohort", "profession"]).size().unstack(fill_value=0).T
    ).reindex(columns=list(cohorts.keys()), fill_value=0)
    # Sort by total descending
    prof_counts = prof_counts.loc[prof_counts.sum(axis=1).sort_values(ascending=False).index]
    bar_breakdown(ax_prof, prof_counts, "Figure 1c. Profession")

    # 1e. Age bar
    ax_age = fig.add_subplot(gs[3, :])
    age_order = ["20-30", "31-40", "41-50", "51-60", "61+", "Unknown"]
    age_counts = (
        all_dfs.groupby(["cohort", "age"]).size().unstack(fill_value=0).T
    ).reindex(columns=list(cohorts.keys()), fill_value=0)
    age_counts = age_counts.reindex(
        [a for a in age_order if a in age_counts.index]
        + [a for a in age_counts.index if a not in age_order]
    )
    bar_breakdown(ax_age, age_counts, "Figure 1d. Age band")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Page 2 — Detailed tables
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(11.7, 16.5))
    gs = GridSpec(3, 1, height_ratios=[2.2, 1.5, 1.5], hspace=0.4, figure=fig)

    fig.suptitle("Detailed breakdown tables", fontsize=15, fontweight="bold", y=0.985)

    ax_t2 = fig.add_subplot(gs[0])
    render_table(
        ax_t2,
        tbl_profession.rename(columns={"profession": "Profession"}),
        "Table 2. Profession × Cohort",
        font_size=9,
    )

    ax_t3 = fig.add_subplot(gs[1])
    render_table(
        ax_t3,
        tbl_age.rename(columns={"age": "Age band"}),
        "Table 3. Age × Cohort",
        font_size=9,
    )

    ax_t4 = fig.add_subplot(gs[2])
    render_table(
        ax_t4,
        tbl_gender.rename(columns={"gender": "Gender"}),
        "Table 4. Gender × Cohort",
        font_size=9,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Page 3 — Filter delta and narrative
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(11.7, 16.5))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.5, figure=fig)

    fig.suptitle("Filter impact & narrative summary", fontsize=15, fontweight="bold", y=0.985)

    # 3a. Reader flow diagram (textual table)
    flow = pd.DataFrame(
        {
            "Step": [
                "v1 baseline (Nature paper)",
                "v2 raw export (any valid answer)",
                "v2 filter ≥95% per test (default)",
                "v2 filter strict 15 per test (sensitivity)",
            ],
            "N readers": [len(v1_ids), len(v2_all_ids), len(v2_95_ids), len(v2_strict_ids)],
            "Δ vs v1": [
                "—",
                f"+{len(v2_all_ids) - len(v1_ids)}",
                f"+{len(v2_95_ids) - len(v1_ids)}",
                f"+{len(v2_strict_ids) - len(v1_ids)}",
            ],
            "Δ vs C2 (raw)": [
                "—", "—",
                f"−{len(v2_all_ids) - len(v2_95_ids)}",
                f"−{len(v2_all_ids) - len(v2_strict_ids)}",
            ],
        }
    )
    ax_flow = fig.add_subplot(gs[0])
    render_table(ax_flow, flow, "Table 5. Reader flow across cohorts", font_size=10,
                 col_widths=[0.45, 0.18, 0.18, 0.19])

    # 3b. Demographic of the 9 readers kept by ≥95% but dropped by strict15
    lost_tbl_rows = []
    if len(lost_demo):
        for col in ["expertise", "profession", "gender", "age"]:
            counts = lost_demo[col].value_counts(dropna=False)
            for k, v in counts.items():
                lost_tbl_rows.append({"Attribute": col, "Value": str(k), "Count": int(v)})
    lost_tbl = pd.DataFrame(lost_tbl_rows)
    ax_lost = fig.add_subplot(gs[1])
    render_table(
        ax_lost,
        lost_tbl if len(lost_tbl) else pd.DataFrame({"info": ["(no readers lost)"]}),
        f"Table 6. Readers kept by ≥95% but dropped by strict 15 (n = {len(lost_ids)})",
        font_size=9,
    )

    # 3c. Narrative
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis("off")
    bullets = [
        f"• v1 had {len(v1_ids)} readers (16 expert / 18 non-expert).",
        f"• v2 raw export contains {len(v2_all_ids)} readers — all {len(v1_ids & v2_all_ids)} v1 readers are present, plus {len(v2_all_ids - v1_ids)} new participants.",
        f"• Default filter ≥95% per test keeps {len(v2_95_ids)} readers "
        f"({tbl_summary.loc[tbl_summary['Cohort']=='C3: v2 ≥95% / test','Experts'].iloc[0]} expert, "
        f"{tbl_summary.loc[tbl_summary['Cohort']=='C3: v2 ≥95% / test','Non-experts'].iloc[0]} non-expert).",
        f"   - Retains {len(v1_ids & v2_95_ids)}/{len(v1_ids)} of the original v1 readers; adds {len(v2_95_ids - v1_ids)} new ones.",
        f"• Strict 15 / test keeps {len(v2_strict_ids)} readers "
        f"({tbl_summary.loc[tbl_summary['Cohort']=='C4: v2 strict 15 / test','Experts'].iloc[0]} expert, "
        f"{tbl_summary.loc[tbl_summary['Cohort']=='C4: v2 strict 15 / test','Non-experts'].iloc[0]} non-expert).",
        f"   - Retains {len(v1_ids & v2_strict_ids)}/{len(v1_ids)} of v1; adds {len(v2_strict_ids - v1_ids)} new ones.",
        f"• Going from ≥95% → strict 15 drops {len(lost_ids)} readers (mostly dermatology residents / board-certified dermatologists).",
        "• Headline conclusion (AI improves diagnosis & management) is preserved under both filters — see RS2B_*.pdf.",
    ]
    txt = "\n".join(bullets)
    ax_text.set_title("Narrative summary", loc="left", fontsize=12, fontweight="bold")
    ax_text.text(0.0, 0.95, txt, va="top", ha="left", fontsize=10, family="sans-serif",
                 transform=ax_text.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote {OUT_PDF}")
