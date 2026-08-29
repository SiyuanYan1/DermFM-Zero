# Reader Study 2A: TODIV Platform Evaluation

Independent-cohort benchmark of DermFM-Zero zero-shot performance on the TODIV dermoscopy platform (1,117 cases, 9 classes): 1,090 clinician sessions contributed by 652 unique clinicians.

## 📋 Script

`todiv_statistical_analysis.py` — Descriptive stats (mean, 95% CI), Welch's two-sided independent t-tests, boxplot figure.

## 📂 Data & Output

| Folder | Purpose |
|--------|---------|
| `real_data/` | **The complete RS2A analysis dataset (included)**: `todiv_scores.xlsx` |
| `real_output/` | Results written by `todiv_statistical_analysis.py --real` |

The unit of analysis in RS2A is the clinician session: each of the 1,090 sessions yields an overall diagnostic score, and all reported RS2A comparisons, including the experience-stratified analyses, are computed from these session-level scores. `real_data/todiv_scores.xlsx` contains this complete analysis dataset — the anonymized session-level scores of all clinicians with profession, dermoscopy experience, and platform-use frequency, together with the bootstrapped score distributions of each evaluated model — so the reported RS2A results are reproducible directly from this repository.

The raw data were collected on the TODIV platform by the TODIV study investigators and are held by them; requests for read-level data should be directed to the TODIV study team. The 652-clinician and 1,117-case counts are properties of the TODIV platform cohort and are not recoverable from the released session-level file.

## 🚀 Run

```bash
pip install pandas numpy scipy matplotlib openpyxl

python todiv_statistical_analysis.py --real
```
