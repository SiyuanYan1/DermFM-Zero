"""
03_fig2_plot.py
---------------
Generates the RS2B 6-panel figure (Figure 5 in the manuscript) from the
≥95%-filtered reader cohort (71 readers).

Two changes vs the original v1 pipeline:
  1. The simulated-data RS2A section (`generate_scores` boxplots) is removed.
  2. The hard-coded management appropriateness dict is replaced with a
     `management_table.csv` lookup (matches the clinician's new R script).

Usage
-----
    python 03_fig2_plot.py
"""

from pathlib import Path

from matplotlib import rcParams

rcParams["svg.fonttype"] = "none"
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["font.size"] = 24

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# --- Style ---
sns.set_style("white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 11

C_UNAIDED = "#4E79A7"  # Blue
C_ASSISTED = "#E15759"  # Red
PALETTE = [C_UNAIDED, C_ASSISTED]
C_POINTS = "#808080"

CAT_INAPPR = "Inappropriate"
CAT_APPR = "Appropriate"
CAT_OPT = "Optimal"
COLORS_MGMT = {CAT_INAPPR: "#E8B4B8", CAT_APPR: "#B8D4E8", CAT_OPT: "#2E5A87"}


# ---------------------------------------------------------------------------
# CLI: --demo (default) | --real
# ---------------------------------------------------------------------------
import argparse

_parser = argparse.ArgumentParser(description="RS2B Step 3: 6-panel figure.")
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

input_csv = OUT / "panderm_cleaned_95pct.csv"
output_pdf = OUT / "RS2B_figure.pdf"
output_svg = OUT / "RS2B_figure.svg"
print(f"[mode={_args.mode}]  DATA={DATA.name}  OUT={OUT.name}")

print("=" * 80)
print("GENERATING RS2B FIGURE  (≥95% per-test cohort, 71 readers)")
print("=" * 80)

# ---------------------------------------------------------------------------
# Load + map
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
# Data-driven management appropriateness
# ---------------------------------------------------------------------------
mgmt_tbl = pd.read_csv(DATA / "management_table.csv", sep=";")
mgmt_tbl.columns = [c.strip() for c in mgmt_tbl.columns]
mgmt_tbl = mgmt_tbl.set_index("diagnosis")
mgmt_tbl = mgmt_tbl.apply(lambda col: col.astype(str).str.strip())

ACTION_TO_COL = {"Dismiss": "dismiss", "Monitor": "monitor", "Treat": "treat_locally", "Excise": "excise_biopsy"}


def classify_management_binary(row) -> int:
    diag = row["true_diagnosis"]
    action = row["Action"]
    if pd.isna(action) or diag not in mgmt_tbl.index:
        return 0
    col = ACTION_TO_COL.get(action)
    if col is None:
        return 0
    return 1 if mgmt_tbl.loc[diag, col] in ("optimal", "appropriate") else 0


def classify_management_optimal(row) -> int:
    diag = row["true_diagnosis"]
    action = row["Action"]
    if pd.isna(action) or diag not in mgmt_tbl.index:
        return 0
    col = ACTION_TO_COL.get(action)
    if col is None:
        return 0
    return 1 if mgmt_tbl.loc[diag, col] == "optimal" else 0


df["mgmt_correct"] = df.apply(classify_management_binary, axis=1)
df["mgmt_optimal"] = df.apply(classify_management_optimal, axis=1)


# ---------------------------------------------------------------------------
# Reader-level aggregations
# ---------------------------------------------------------------------------
panderm_reader = (
    df.groupby(["reader_id", "help", "ability_group"])
    .agg(prop_correct=("is_correct_numeric", "mean"))
    .reset_index()
)

mgmt_reader_appr = (
    df.groupby(["reader_id", "help", "ability_group"])
    .agg(prop_mgmt_correct=("mgmt_correct", "mean"))
    .reset_index()
)


# Class-wise stats
class_stats_list = []
for cls in df["Class"].dropna().unique():
    for help_val in ["Unaided", "AI-assisted"]:
        subset = df[(df["Class"] == cls) & (df["help"] == help_val)]
        if len(subset) >= 10:
            class_stats_list.append(
                {
                    "Class": cls,
                    "help": help_val,
                    "Accuracy": subset["is_correct_numeric"].mean(),
                    "Accuracy_SE": subset["is_correct_numeric"].sem(),
                    "Mgmt_Appropriate": subset["mgmt_correct"].mean(),
                    "Mgmt_Appropriate_SE": subset["mgmt_correct"].sem(),
                    "Count": len(subset),
                }
            )
class_stats = pd.DataFrame(class_stats_list)


# Overall stats
diff_overall = panderm_reader.pivot(index="reader_id", columns="help", values="prop_correct").dropna()
_, p_val_acc = stats.ttest_rel(diff_overall["AI-assisted"], diff_overall["Unaided"])

diff_mgmt_appr = mgmt_reader_appr.pivot(index="reader_id", columns="help", values="prop_mgmt_correct").dropna()
_, p_val_mgmt_appr = stats.ttest_rel(diff_mgmt_appr["AI-assisted"], diff_mgmt_appr["Unaided"])


# ---------------------------------------------------------------------------
# Plotting helpers (verbatim from original fig2_plot.py)
# ---------------------------------------------------------------------------
def plot_mgmt_matrix(ax):
    """Panel a: Management matrix (built from data/management_table.csv)."""
    classes = ["AKIEC", "BCC", "OTHER_BEN", "BKL", "DF", "INF", "OTHER_MAL", "MEL", "NV", "SCCKA", "VASC"]
    actions = ["Dismiss", "Monitor", "Treat", "Excise"]
    # Map back from short form to long form used in mgmt_tbl
    class_to_diag = {"OTHER_BEN": "OTHER_BENIGN", "OTHER_MAL": "OTHER_MALIGNANT"}
    sev_map = {"inappropriate": 0, "appropriate": 1, "optimal": 2}

    data = []
    for cls in classes:
        diag = class_to_diag.get(cls, cls)
        row = []
        for act in actions:
            col = ACTION_TO_COL[act]
            val = mgmt_tbl.loc[diag, col] if diag in mgmt_tbl.index else "inappropriate"
            row.append(sev_map.get(val, 0))
        data.append(row)

    df_mat = pd.DataFrame(data, index=classes, columns=actions)
    cmap = LinearSegmentedColormap.from_list(
        "Mgmt", [COLORS_MGMT[CAT_INAPPR], COLORS_MGMT[CAT_APPR], COLORS_MGMT[CAT_OPT]], N=3
    )
    sns.heatmap(df_mat, cmap=cmap, annot=False, linewidths=1.5, linecolor="white", cbar=False, ax=ax)
    for y in range(len(classes)):
        for x in range(len(actions)):
            val = df_mat.iloc[y, x]
            txt = {2: "Opt", 1: "App", 0: "Inp"}[val]
            col = "white" if val in [0, 2] else "black"
            ax.text(x + 0.5, y + 0.5, txt, ha="center", va="center", color=col, fontsize=9)
    ax.set_title("a. Management Standards", loc="left", fontsize=14)
    ax.tick_params(labelsize=11)


def plot_barplot_overall(ax, reader_df, metric, ylabel, title, ylim_min=0, p_val=None, show_legend=False):
    reader_df = reader_df.copy()
    reader_df["help"] = pd.Categorical(reader_df["help"], categories=["Unaided", "AI-assisted"], ordered=True)
    means = reader_df.groupby("help", observed=True)[metric].mean()
    sems = reader_df.groupby("help", observed=True)[metric].sem()
    x_positions = [0, 1]

    ax.bar(
        x_positions,
        [means["Unaided"], means["AI-assisted"]],
        width=0.6,
        color=PALETTE,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.errorbar(
        x_positions,
        [means["Unaided"], means["AI-assisted"]],
        yerr=[sems["Unaided"], sems["AI-assisted"]],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.2,
        zorder=5,
    )

    for i, help_val in enumerate(["Unaided", "AI-assisted"]):
        data = reader_df[reader_df["help"] == help_val][metric].values
        jitter = np.random.normal(i, 0.05, len(data))
        ax.scatter(jitter, data, color=C_POINTS, alpha=0.3, s=30, zorder=3, edgecolors="none")

    y_range = 1.0 - ylim_min
    for i, help_val in enumerate(["Unaided", "AI-assisted"]):
        mean_val = means[help_val]
        sem_val = sems[help_val]
        ax.text(i, mean_val + sem_val + 0.015, f"{mean_val:.1%}", ha="center", va="bottom", fontsize=12)

    if p_val is not None:
        p_text = f"P={p_val:.3f}" if p_val >= 0.001 else "P<0.001"
        ax.text(0.5, ylim_min + y_range * 0.62, p_text, ha="center", fontsize=12, style="italic")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Unaided", "AI-assisted"], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(ylim_min, 1.0)
    ax.set_title(title, loc="left", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    if show_legend:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=c,
                markersize=12,
                label=l,
                alpha=0.8,
                markeredgecolor="black",
                markeredgewidth=1.2,
            )
            for c, l in zip(PALETTE, ["Unaided", "AI-assisted"])
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11, frameon=False)


def plot_class_bars_with_errorbar(ax, stats_df, metric, title):
    pivot_mean = stats_df.pivot(index="Class", columns="help", values=metric).dropna()
    pivot_se = stats_df.pivot(index="Class", columns="help", values=f"{metric}_SE").dropna()

    desired_order = ["NV", "BKL", "DF", "OTHER_BEN", "INF", "VASC", "AKIEC", "SCCKA", "BCC", "OTHER_MAL", "MEL"]
    classes_order = [c for c in desired_order if c in pivot_mean.index]
    pivot_mean = pivot_mean.reindex(classes_order)
    pivot_se = pivot_se.reindex(classes_order)

    classes = pivot_mean.index.tolist()
    y = np.arange(len(classes))
    height = 0.35

    ax.barh(
        y - height / 2,
        pivot_mean["Unaided"].values,
        height,
        xerr=pivot_se["Unaided"].values,
        label="Unaided",
        color=C_UNAIDED,
        alpha=0.8,
        error_kw={"linewidth": 1.2},
    )
    ax.barh(
        y + height / 2,
        pivot_mean["AI-assisted"].values,
        height,
        xerr=pivot_se["AI-assisted"].values,
        label="AI-assisted",
        color=C_ASSISTED,
        alpha=0.8,
        error_kw={"linewidth": 1.2},
    )

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Accuracy", fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.set_title(title, loc="left", fontsize=14)
    ax.grid(axis="x", alpha=0.3)


def plot_ability_barplots(ax, reader_df, metric, ylabel, title, ylim_min=0):
    reader_df = reader_df.copy()
    ability_order = ["Non-expert", "Expert"]
    reader_df["ability_group"] = pd.Categorical(reader_df["ability_group"], categories=ability_order, ordered=True)

    x_positions, bar_data, colors_list, error_data = [], [], [], []

    for i, ability in enumerate(ability_order):
        subset = reader_df[reader_df["ability_group"] == ability]
        if len(subset) == 0:
            continue
        x_base = i * 1.5
        for j, help_val in enumerate(["Unaided", "AI-assisted"]):
            help_data = subset[subset["help"] == help_val][metric].values
            if len(help_data) == 0:
                continue
            pos = x_base + j * 0.5
            x_positions.append(pos)
            bar_data.append(help_data.mean())
            error_data.append(help_data.std() / np.sqrt(len(help_data)))
            colors_list.append(PALETTE[j])
            jitter = np.random.normal(pos, 0.05, len(help_data))
            ax.scatter(jitter, help_data, color=C_POINTS, alpha=0.3, s=25, zorder=3, edgecolors="none")

    ax.bar(x_positions, bar_data, width=0.35, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.2, zorder=2)
    ax.errorbar(x_positions, bar_data, yerr=error_data, fmt="none", ecolor="black", capsize=4, linewidth=1.2, zorder=5)

    y_range = 1.0 - ylim_min
    for i, (pos, val) in enumerate(zip(x_positions, bar_data)):
        ax.text(pos, val + error_data[i] + 0.015, f"{val:.1%}", ha="center", va="bottom", fontsize=11)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(ylim_min, 1.0)
    ax.set_xticks([0.25, 1.75])
    ax.set_xticklabels(ability_order, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, loc="left", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    for i, ability in enumerate(ability_order):
        subset = reader_df[reader_df["ability_group"] == ability]
        if len(subset) == 0:
            continue
        x_base = i * 1.5
        pivot = subset.pivot(index="reader_id", columns="help", values=metric).dropna()
        if len(pivot) > 1:
            _, p_v = stats.ttest_rel(pivot["AI-assisted"], pivot["Unaided"])
            p_text = f"P={p_v:.3f}" if p_v >= 0.001 else "P<0.001"
            ax.text(x_base + 0.25, ylim_min + y_range * 0.92, p_text, ha="center", fontsize=12, style="italic")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
np.random.seed(42)
fig2 = plt.figure(figsize=(20, 12))
gs = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.35, height_ratios=[1, 1], width_ratios=[1.3, 1.0, 1.0])

ax_a = fig2.add_subplot(gs[0, 0])
plot_mgmt_matrix(ax_a)

ax_b = fig2.add_subplot(gs[0, 1])
plot_barplot_overall(ax_b, panderm_reader, "prop_correct", "Accuracy", "b. Overall Accuracy",
                     ylim_min=0.30, p_val=p_val_acc, show_legend=True)

ax_c = fig2.add_subplot(gs[0, 2])
plot_barplot_overall(ax_c, mgmt_reader_appr, "prop_mgmt_correct", "Appropriate Mgmt", "c. Overall Mgmt Appropriate",
                     ylim_min=0.50, p_val=p_val_mgmt_appr, show_legend=False)

ax_d = fig2.add_subplot(gs[1, 0])
plot_class_bars_with_errorbar(ax_d, class_stats, "Accuracy", "d. Class-wise Accuracy")

ax_e = fig2.add_subplot(gs[1, 1])
plot_ability_barplots(ax_e, panderm_reader, "prop_correct", "Accuracy", "e. Accuracy by Ability", ylim_min=0.30)

ax_f = fig2.add_subplot(gs[1, 2])
plot_ability_barplots(ax_f, mgmt_reader_appr, "prop_mgmt_correct", "Appropriate Mgmt",
                      "f. Mgmt Appropriate by Ability", ylim_min=0.35)

plt.savefig(output_pdf, dpi=600, transparent=True, bbox_inches="tight")
plt.savefig(output_svg, dpi=600, transparent=True, bbox_inches="tight")
print(f"✅ Saved figure to: {output_pdf.name}")
print(f"✅ Saved figure to: {output_svg.name}  (editable text, transparent)")
plt.close()

print("=" * 80)
