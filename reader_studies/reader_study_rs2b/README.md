# Reader Study 2B: Multimodal AI-Assisted Specialist Diagnosis

Within-subject study evaluating AI assistance on multimodal skin-cancer diagnosis and management on the DermaChallenge platform: 71 readers (≥95% per-test completion filter; from 87 raw), 165 tests, 4,929 readings (2,454 paired unaided + AI-assisted observations) across 11 lesion classes.

## 📋 Scripts

| Script | Function |
|--------|----------|
| `01_filter_reader.py` | Long→wide reshape + ≥95% per-test completion filter |
| `02_fig2_table_clean.py` | 5 summary tables (overall / by experience / by class); two-sided paired t-tests on per-reader proportions, Bonferroni-corrected experience-group tests |
| `03_fig2_plot.py` | 6-panel Figure 5 (publication PDF + editable SVG) |
| `cohort_analysis.py` | 71-reader analytic-cohort progression from the 87-reader raw export; BCD vs non-BCD subgroup analysis. |
| `reader_study_report.py` | Revision-behaviour report under AI assistance (unaided correctness × expertise factors). |
| `vascular_subgroup_analysis.py` | Vascular-lesion subgroup deep-dive: paired Wilcoxon, by-expertise / by-profession stratification, per-image accuracy, reader-diagnosis confusion, error entropy, revision pattern, ceiling-effect comparison vs other classes, clinical-decision distribution. |

> `cohort_analysis.py` and `reader_study_report.py` also depend on the cleaned CSVs produced by `02_fig2_table_clean.py --real`.

## 📂 Data

| Folder | Purpose |
|--------|---------|
| `real_data/` | **De-identified reader-level study data (included)** |
| `real_output/` | Results written by the scripts when run with `--real` |

`real_data/` contains two CSVs:

- `panderm_reader_data.csv` — the full reader-level export (semicolon-separated, European decimal comma): 5,820 readings from 87 recruited readers across 194 test sessions (the ≥95% completion filter applied by `01_filter_reader.py` yields the 71-reader / 165-test analytic cohort reported in the paper).
- `management_table.csv` — the management-appropriateness rubric (11 diagnoses × 4 clinical actions → optimal / appropriate / inappropriate), developed by consensus of four senior board-certified dermatologists.
> To view the CSVs in Excel, use Data → Get Data (Text/CSV) and set the delimiter to semicolon; the files are kept semicolon-separated because the analysis scripts read them in this format.

### Column dictionary (`panderm_reader_data.csv`)

| Column | Description |
|--------|-------------|
| `tequ_user_id` | Pseudonymous reader ID (platform hash; not linkable to identity) |
| `test_id` | Test-session ID (each session = 15 cases × 2 modes = 30 reading slots) |
| `test_round` | The reader's session number (readers could complete multiple sessions) |
| `image_id` / `image_isic_id` | Case image identifiers (ISIC archive IDs; MILK10K held-out partition) |
| `tequ_no` | Reading-slot number within the session (1–30) |
| `tequ_correct` | 1 if `answer` matches ground truth, else 0 |
| `answer` | Reader's diagnosis (11-class ISIC-DX taxonomy; empty or `timeout` = invalid slot) |
| `manage` | Reader's management decision (`manage_dismiss` / `manage_monitor` / `manage_biopsy` / `manage_localtherapy`) |
| `image_diag_id` | Ground-truth diagnosis |
| `tequ_mode_id` | Reading mode: `radio_radio_management47` = without AI; `radio_radio_management47_help` = with AI |
| `expected_accuracy_2pl` | Reader's IRT-derived expected accuracy (Bayesian 2PL on the platform population) |
| `ability_2pl` | Expertise group derived from `expected_accuracy_2pl` (`expert` ≥ 0.70 / `non_expert`) |
| `age`, `yob`, `gender` | **Blanked for de-identification** (columns retained for schema compatibility) |
| `profession` | Reader profession (e.g., boardCertifiedDermatologist) |

### De-identification

The released data are fully anonymized under the conditions agreed with the hosting platform (Department of Dermatology, Medical University of Vienna): reader IDs are pseudonymous platform hashes; age, year of birth, and gender values have been removed; the export contains no timestamps and no country information. Retained fields are profession, expertise, votes (diagnosis and management), and ground truth.

## 🚀 Run

```bash
pip install pandas numpy scipy matplotlib seaborn

python 01_filter_reader.py        --real
python 02_fig2_table_clean.py     --real
python 03_fig2_plot.py            --real

# Reviewer-requested analyses (require 02_fig2_table_clean --real output):
python cohort_analysis.py             --real
python reader_study_report.py         --real
python vascular_subgroup_analysis.py  --real
```

Note: the demographic tables in `cohort_analysis.py` report age and gender as "Unknown", reflecting the de-identification above; all diagnostic-accuracy, management, subgroup, and revision-behaviour results in the paper are fully reproducible from the released data.
