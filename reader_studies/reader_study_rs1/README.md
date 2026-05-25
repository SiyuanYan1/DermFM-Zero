# Reader Study 1: AI-Assisted GP Diagnostic Accuracy

Within-subject study evaluating AI assistance on clinician diagnostic accuracy and management across two sites (CN and AU/EN), 30 readers, 701 cases.

## 📋 Scripts

| Script | Function |
|--------|----------|
| `rs1_statistical_analysis.py` | Two-sided Wilcoxon signed-rank tests (reader-level) and two-sided McNemar's tests (case-level); produces 6-panel publication figure |
| `agent_grader.py` | GPT-4o-mini grading agent (diagnosis 1–5 rubric + management decision matrix). Requires `OPENAI_API_KEY` env var. |
| `generate_demo_data.py` | Generate synthetic demo data for code verification |

## 📂 Data & Output

| Folder | Purpose |
|--------|---------|
| `demo_data/` | Synthetic scores for code verification (`cn_graded.csv`, `en_graded.csv`) |
| `demo_output/` | Results from `--demo` |

Reader-level real data are not included to protect participant privacy and are available upon reasonable request to the corresponding author, subject to institutional data sharing agreements.

## 🚀 Run

```bash
pip install pandas numpy scipy matplotlib

python generate_demo_data.py               # one-time setup
python rs1_statistical_analysis.py --demo  # demo data
```
