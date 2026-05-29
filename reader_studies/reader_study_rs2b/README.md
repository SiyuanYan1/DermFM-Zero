# Reader Study 2B: Multimodal AI-Assisted Specialist Diagnosis

Within-subject study evaluating AI assistance on multimodal skin-cancer diagnosis and management on the DermaChallenge platform: 71 readers (≥95% per-test completion filter; from 87 raw), 165 tests, 4,929 readings across 11 lesion classes.

## 📋 Scripts

| Script | Function |
|--------|----------|
| `01_filter_reader.py` | Long→wide reshape + ≥95% per-test completion filter |
| `02_fig2_table_clean.py` | 5 summary tables (overall / by experience / by class); two-sided paired t-tests on per-reader proportions, Bonferroni-corrected class-specific tests |
| `03_fig2_plot.py` | 6-panel Figure 5 (publication PDF + editable SVG) |
| `cohort_analysis.py` | 71-reader analytic-cohort progression from the 87-reader raw export; BCD vs non-BCD subgroup analysis. |
| `reader_study_report.py` | Revision-behaviour report under AI assistance (unaided correctness × expertise factors). |
| `vascular_subgroup_analysis.py` | Vascular-lesion subgroup deep-dive (R2 Comment 12): paired Wilcoxon, by-expertise / by-profession stratification, per-image accuracy, reader-diagnosis confusion, error entropy, revision pattern, ceiling-effect comparison vs other classes, clinical-decision distribution. |
| `generate_demo_data.py` | Generate synthetic demo data for code verification |

> `cohort_analysis.py`, `reader_study_report.py`, and `vascular_subgroup_analysis.py` are reviewer-requested R2-revision analyses and ship as code only (no demo data; pass `--real` to run on the requested data corpus).

## 📂 Data & Output

| Folder | Purpose |
|--------|---------|
| `demo_data/` | Synthetic 87-reader export for code verification |
| `demo_output/` | Results from `--demo` |

Each data folder expects two CSVs: `panderm_reader_data.csv` (raw export, semicolon-separated, European decimal comma) and `management_table.csv` (rubric).

Reader-level real data are not included to protect participant privacy and are available upon reasonable request to the corresponding author, subject to institutional data sharing agreements.

## 🚀 Run

```bash
pip install pandas numpy scipy matplotlib seaborn

python generate_demo_data.py                     # one-time setup
python 01_filter_reader.py        --demo
python 02_fig2_table_clean.py     --demo
python 03_fig2_plot.py            --demo

# Reviewer-requested analyses (code only; require --real and 02_fig2_table_clean output):
# python cohort_analysis.py --real
# python reader_study_report.py --real
# python vascular_subgroup_analysis.py --real
# (cohort_analysis.py and reader_study_report.py also depend on the cleaned
#  CSVs produced by 02_fig2_table_clean.py --real.)
```
