# Reader Studies

Three multinational reader studies evaluating DermFM-Zero in collaborative clinical workflows: primary care (RS1), specialist benchmarking (RS2A), and specialist collaborative diagnosis (RS2B). Each subfolder is self-contained with code and real data: RS1 and RS2B ship full de-identified reader-level datasets, and RS2A ships its complete session-level analysis dataset.

## 📋 Studies at a glance

| Study | Setting | Design | Readers (n) | Cases | Primary tests | Real data |
|-------|---------|--------|-------------|-------|---------------|-----------|
| [RS1](reader_study_rs1/)   | Primary care (CN + AU/EN)            | Within-subject, paired   | 38 PCPs (28 GPs + 10 NPs) | 146   | Wilcoxon signed-rank (reader-level); McNemar (case-level)          | **De-identified reader-level data included** |
| [RS2A](reader_study_rs2a/) | Specialist benchmark (TODIV)         | Independent cohort       | 1,090 sessions (652 unique clinicians) | 1,117 | Welch's two-sided t-test                                             | **Complete session-level analysis dataset included** |
| [RS2B](reader_study_rs2b/) | Specialist collab. (DermaChallenge)  | Within-subject, paired   | 71 (from 87 raw, ≥95% per-test completion) | 1,048 | Paired t-test on per-reader proportions; Bonferroni for experience groups | **De-identified reader-level data included** |

*R2-revision additions implement post-hoc power (`post_hoc_power.py`), BCD–agent inter-rater reliability (`reader_study_rs1/bcd_agent_irr.py`), RS2B 71-reader cohort + revision-behaviour + vascular-subgroup analyses (`reader_study_rs2b/cohort_analysis.py`, `reader_study_report.py`, `vascular_subgroup_analysis.py`). All run on the included datasets via `--real`, except `bcd_agent_irr.py`, which requires the BCD-graded grading bundle (available on request).*

## 📂 Repository Structure

```
reader_studies/
├── reader_study_rs1/
│   ├── rs1_statistical_analysis.py
│   ├── agent_grader.py
│   ├── bcd_agent_irr.py        # NEW (R3 Minor 3)
│   ├── real_data/        real_output/      # de-identified reader-level data
│   └── README.md
├── reader_study_rs2a/
│   ├── todiv_statistical_analysis.py
│   ├── real_data/        real_output/      # complete session-level analysis dataset
│   └── README.md
├── reader_study_rs2b/
│   ├── 01_filter_reader.py
│   ├── 02_fig2_table_clean.py
│   ├── 03_fig2_plot.py
│   ├── cohort_analysis.py      # NEW (R1 C20/C21)
│   ├── reader_study_report.py  # NEW (R2 C2)
│   ├── vascular_subgroup_analysis.py        # NEW (R2 C12)
│   ├── real_data/        real_output/       # de-identified reader-level data
│   └── README.md
├── post_hoc_power.py           # NEW (R3 Major 3)
└── README.md             # this file
```

## 🚀 Quick Start

```bash
pip install pandas numpy scipy matplotlib seaborn openpyxl

# RS1 (de-identified reader-level data included)
cd reader_study_rs1
python rs1_statistical_analysis.py --real

# RS2A (session-level analysis dataset included)
cd ../reader_study_rs2a
python todiv_statistical_analysis.py --real

# RS2B (de-identified reader-level data included)
cd ../reader_study_rs2b
python 01_filter_reader.py    --real
python 02_fig2_table_clean.py --real
python 03_fig2_plot.py        --real

# Reviewer-requested analyses (R2 revision)
cd ../reader_study_rs2b && python cohort_analysis.py --real
python reader_study_report.py --real
python vascular_subgroup_analysis.py --real
cd .. && python post_hoc_power.py --real
cd reader_study_rs1 && python bcd_agent_irr.py --real   # requires BCD grading bundle (on request)
```

Every study ships its real dataset and runs with `--real` (reads `real_data/`, writes `real_output/`).

### 📑 Reviewer-requested analyses (R2 revision)

These five additional scripts (`post_hoc_power.py`, `reader_study_rs1/bcd_agent_irr.py`, and three under `reader_study_rs2b/`) implement specific analyses requested during peer review. The RS2B scripts and `post_hoc_power.py` reproduce the published numbers directly from the included datasets (`--real`); `bcd_agent_irr.py` additionally requires the BCD-graded XLSX bundle, available from the corresponding author.

## 🔬 Data Sharing

- **RS1** ships with the full de-identified reader-level dataset (`reader_study_rs1/real_data/rs1_reader_data.csv`), released under an approved MUHREC amendment (Project 49479): pseudonymous platform study IDs; age, gender, country, experience, timestamps, and contact fields removed. Source images are from public atlases (SNU-134, PASSION, SKINTONE), referenced by case ID and source dataset.
- **RS2A** ships with aggregated, anonymized session-level scores (`todiv_scores.xlsx`), the unit of analysis for all reported RS2A comparisons. Raw clinician-level responses are held by the data provider.
- **RS2B** ships with the full de-identified reader-level dataset (`reader_study_rs2b/real_data/`), released with the agreement of the hosting platform (Department of Dermatology, Medical University of Vienna): pseudonymous reader IDs; age, year of birth, and gender removed; no timestamps or country data. Profession, expertise, votes, and ground truth are retained.

## 📄 Citation

If you use this code, please cite the main DermFM-Zero paper (see the top-level repository [README](../README.md#citation)).
