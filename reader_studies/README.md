# Reader Studies

Three multinational reader studies evaluating DermFM-Zero in collaborative clinical workflows: primary care (RS1), specialist benchmarking (RS2A), and specialist collaborative diagnosis (RS2B). Each subfolder is self-contained with code, a small synthetic demo dataset, and pre-computed demo outputs.

## 📋 Studies at a glance

| Study | Setting | Design | Readers (n) | Cases | Primary tests | Real data |
|-------|---------|--------|-------------|-------|---------------|-----------|
| [RS1](reader_study_rs1/)   | Primary care (CN + AU/EN)            | Within-subject, paired   | 30 PCPs (20 GPs + 10 NPs) | 150   | Wilcoxon signed-rank (reader-level); McNemar (case-level)          | Not included          |
| [RS2A](reader_study_rs2a/) | Specialist benchmark (TODIV)         | Independent cohort       | 1,090 sessions (652 unique clinicians) | 1,117 | Welch's two-sided t-test                                             | **Aggregated anonymized scores included** |
| [RS2B](reader_study_rs2b/) | Specialist collab. (DermaChallenge)  | Within-subject, paired   | 71 (from 87 raw, ≥95% per-test completion) | 1,048 | Paired t-test on per-reader proportions; Bonferroni for class-specific | Not included |

*R2-revision additions implement post-hoc power (`post_hoc_power.py`), BCD–agent inter-rater reliability (`reader_study_rs1/bcd_agent_irr.py`), RS2B 71-reader cohort + revision-behaviour + vascular-subgroup analyses (`reader_study_rs2b/cohort_analysis.py`, `reader_study_report.py`, `vascular_subgroup_analysis.py`). All ship as code only; real data on request.*

## 📂 Repository Structure

```
reader_studies/
├── reader_study_rs1/
│   ├── rs1_statistical_analysis.py
│   ├── agent_grader.py
│   ├── bcd_agent_irr.py        # NEW (R3 Minor 3)
│   ├── generate_demo_data.py
│   ├── demo_data/        demo_output/
│   └── README.md
├── reader_study_rs2a/
│   ├── todiv_statistical_analysis.py
│   ├── demo_data/        demo_output/
│   ├── real_data/        real_output/      # aggregated anonymized scores
│   └── README.md
├── reader_study_rs2b/
│   ├── 01_filter_reader.py
│   ├── 02_fig2_table_clean.py
│   ├── 03_fig2_plot.py
│   ├── cohort_analysis.py      # NEW (R1 C20/C21)
│   ├── reader_study_report.py  # NEW (R2 C2)
│   ├── vascular_subgroup_analysis.py        # NEW (R2 C12)
│   ├── generate_demo_data.py
│   ├── demo_data/        demo_output/
│   └── README.md
├── post_hoc_power.py           # NEW (R3 Major 3)
└── README.md             # this file
```

## 🚀 Quick Start

```bash
pip install pandas numpy scipy matplotlib seaborn openpyxl

# RS1
cd reader_study_rs1
python generate_demo_data.py
python rs1_statistical_analysis.py --demo

# RS2A
cd ../reader_study_rs2a
python todiv_statistical_analysis.py --demo   # or --real (aggregated data included)

# RS2B
cd ../reader_study_rs2b
python generate_demo_data.py
python 01_filter_reader.py    --demo
python 02_fig2_table_clean.py --demo
python 03_fig2_plot.py        --demo

# Reviewer-requested analyses (R2 revision — code only; require real data on request)
python post_hoc_power.py --real
cd reader_study_rs1   && python bcd_agent_irr.py --real
cd ../reader_study_rs2b && python cohort_analysis.py --real
cd ../reader_study_rs2b && python reader_study_report.py --real
cd ../reader_study_rs2b && python vascular_subgroup_analysis.py --real
```

Each script accepts `--demo` (uses `demo_data/`, writes to `demo_output/`) or `--real` (uses `real_data/`, writes to `real_output/`). Only RS2A ships with real data (aggregated, anonymized). For RS1 and RS2B, `--real` requires obtaining the underlying data from the corresponding author.

### 📑 Reviewer-requested analyses (R2 revision)

These five additional scripts (`post_hoc_power.py`, `reader_study_rs1/bcd_agent_irr.py`, and three under `reader_study_rs2b/`) implement specific analyses requested during peer review. Unlike the main RS1/RS2A/RS2B pipeline scripts, **they ship as code only**: no synthetic demo data is provided, because the analyses (post-hoc power, weighted-κ inter-rater reliability, subgroup statistics) would yield statistically meaningless outputs on synthetic data. To reproduce the published numbers, request the real data corpus from the corresponding author and pass `--real` to each script.

## 🔬 Data Sharing

- **RS2A** ships with aggregated, anonymized session-level scores (`todiv_scores.xlsx`). Raw clinician-level responses are held by the data provider.
- **RS1** and **RS2B** include synthetic demo data only. Reader-level real data are not included to protect participant privacy and are available upon reasonable request to the corresponding author, subject to institutional data sharing agreements.

## 📄 Citation

If you use this code, please cite the main DermFM-Zero paper (see the top-level repository [README](../README.md#citation)).
