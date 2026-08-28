# Reader Study 1: AI-Assisted GP Diagnostic Accuracy

Within-subject study evaluating AI assistance on clinician diagnostic accuracy and management across two sites (CN and AU/EN): 38 readers (28 GPs, 10 NPs), 146 cases, 861 paired unaided + AI-assisted observations.

## 📋 Scripts

| Script | Function |
|--------|----------|
| `rs1_statistical_analysis.py` | Two-sided Wilcoxon signed-rank tests (reader-level) and two-sided McNemar's tests (case-level); produces 6-panel publication figure |
| `agent_grader.py` | GPT-4o-mini grading agent (diagnosis 1–5 rubric + management decision matrix). Requires `OPENAI_API_KEY` env var. |
| `bcd_agent_irr.py` | BCD–agent inter-rater reliability (weighted κ, exact-match, correction proportion) for the GPT-4o-mini grading pipeline. Requires the BCD-graded XLSX bundle (not shipped; on request). |

> `bcd_agent_irr.py` is a reviewer-requested R2-revision analysis and ships as code only (pass `--real` plus `--grading_dir` to run on the requested data corpus).

## 📂 Data

| Folder | Purpose |
|--------|---------|
| `real_data/` | **De-identified reader-level study data (included)**: `rs1_reader_data.csv` |
| `real_output/` | Results written by `rs1_statistical_analysis.py --real` |

`real_data/rs1_reader_data.csv` — one row per case read (both phases), 861 rows from 38 readers over the 146-case bank.

### Column dictionary

| Column | Description |
|--------|-------------|
| `Cohort` | Reader cohort: `CN` (China) or `EN` (Australia/Italy/Austria) |
| `Responder_ID` | Pseudonymous reader ID (platform study ID; not linkable to identity) |
| `Clinician_Role` | `GP` or `NP` |
| `Experience_Band` | Years of dermatology experience, banded (`0-1`, `2-5`, `6-10`, `>10`) |
| `Case ID` / `Source_Dataset` | Case identifier and its source atlas (`SNU-134`, `PASSION`, `SKINTONE`) |
| `GT` | Ground-truth diagnosis |
| `Unaided_Dx_Text`, `PRE_Differential_1/2` | Reader's unaided primary diagnosis and differentials (English; CN cohort responses translated) |
| `Unaided_Dx_Score` | Agent-graded diagnostic accuracy (1–5 rubric) |
| `Unaided_Top3_SpotOn` / `Unaided_Top3_Generic` | Exact / generic correct diagnosis present in the reader's top-3 |
| `Unaided_Mgmt_Grade` | Management grade (Perfect / Adequate / Inadequate but harmless / Inadequate and dangerous) |
| `Unaided_Dx_Confidence`, `Unaided_Mgmt_Confidence` | Reader-entered confidence (1–5) |
| `AI_Top1..3_Prediction/Confidence` | DermFM-Zero's zero-shot top-3 shown to the reader |
| `Assisted_*` | Post-AI mirrors of the fields above |
| `Changed_Diagnosis`, `Changed_Management` | Whether the reader revised after AI assistance |

### De-identification

Released under an approved MUHREC amendment (Project 49479): platform study IDs only (anonymised at collection and not linkable to individuals); age, gender, country, timestamps, and all contact fields removed; dermatology experience reported as bands.

## 🚀 Run

```bash
pip install pandas numpy scipy matplotlib

python rs1_statistical_analysis.py --real
```
