# Reader Study 2A: TODIV Platform Evaluation

Independent-cohort benchmark of DermFM-Zero zero-shot performance against 1,090+ clinicians on the TODIV dermoscopy platform (1,117 cases, 9 classes).

## 📋 Script

`todiv_statistical_analysis.py` — Descriptive stats (mean, 95% CI), Welch's two-sided independent t-tests, boxplot figure.

## 📂 Data & Output

| Folder | Purpose |
|--------|---------|
| `demo_data/` | Synthetic TODIV scores for code verification |
| `real_data/` | Aggregated, anonymized TODIV evaluation scores (`todiv_scores.xlsx`), included in this repository |
| `demo_output/` | Results from `--demo` |
| `real_output/` | Results from `--real` |

Raw clinician-level response data are not included to protect participant privacy and are available upon reasonable request to the corresponding author, subject to institutional data sharing agreements.

## 🚀 Run

```bash
pip install pandas numpy scipy matplotlib openpyxl

python todiv_statistical_analysis.py --demo   # demo data
python todiv_statistical_analysis.py --real   # real data (aggregated)
```
