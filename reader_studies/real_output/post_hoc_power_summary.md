# Post-hoc power analysis (R3 Major Comment 3)

alpha = 0.05 (two-sided); Wilcoxon ARE adjustment = sqrt(0.864).

| Outcome | n | Effect size | Power (t) | Power (Wilcoxon-adj.) | Status |
|---|---|---|---|---|---|
| RS1 diagnostic accuracy | 38 | d_z=0.747 | 0.9942 | 0.9864 | ok |
| RS2A session accuracy (DermFM-Zero vs Humans-All) | 1090 vs 20000 | d=1.323 | 1.0000 | 1.0000 | ok |
| RS2B diagnostic accuracy | 71 | d_z=0.642 | 0.9996 | 0.9986 | ok |
| RS2B management appropriateness | 71 | d_z=0.321 | 0.7606 | 0.6986 | ok |

## Caveats

- RS1 d_z is derived from the Wilcoxon effect size r (r = Z / sqrt(N)) because individual-reader paired scores are not exposed in the summary CSV; a sensitivity d_z assuming Pearson r=0.5 between unaided/assisted is also reported in the JSON.
- RS2A is reported two ways: with the full n_m=20,000 model iterations (which inflates power because iterations are not independent human-equivalent sessions), and with a conservative calibration n_m=n_h=1,090.
- RS2B management appropriateness is scored by mapping each (true_diagnosis, clinical_decision) pair through the Fig. 5a Management Standards matrix (11 diagnoses x 4 actions: dismiss/monitor/local_therapy[Treat]/biopsy[Excise]); cells marked Optimal or Appropriate are counted as appropriate (1), Inappropriate as 0.
- Wilcoxon-ARE-adjusted power = solve for power with d_z scaled by sqrt(0.864), approximating the Pitman ARE of the Wilcoxon signed-rank test relative to the paired t-test under normality.