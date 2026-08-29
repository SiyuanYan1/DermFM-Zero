# RS2B Vascular subgroup analysis (R2 Comment 12)

Source: `panderm_cleaned_95pct.csv`  
VASC obs: 333 | unique images: 8 | readers: 71


## 1. Overall paired Wilcoxon (reader-level)

- n readers paired: **71**
- mean accuracy unaided: **0.750**
- mean accuracy assisted: **0.716**
- mean diff (assisted − unaided): **-0.035**
- Wilcoxon p-value: **0.4696**
- (per-observation: unaided 0.723 vs assisted 0.683 on 166 / 167 obs)

## 2. By expertise

| Expertise | n readers | Mean unaided | Mean assisted | Diff | p |
|---|---:|---:|---:|---:|---:|
| expert | 39 | 0.699 | 0.792 | 0.092 | 0.2288 |
| non-expert | 32 | 0.812 | 0.622 | -0.190 | 0.0259 |

## 3. By BCD vs non-BCD

| Group | n readers | Mean unaided | Mean assisted | Diff | p |
|---|---:|---:|---:|---:|---:|
| BCD | 21 | 0.768 | 0.850 | 0.082 | 0.5230 |
| non-BCD | 50 | 0.743 | 0.659 | -0.084 | 0.1946 |

## 3b. By profession (detail)

| Profession | n_un | n_ai | Unaided | Assisted |
|---|---:|---:|---:|---:|
| boardCertifiedDermatologist | 62 | 63 | 0.774 | 0.810 |
| dermatologyResident | 42 | 42 | 0.714 | 0.714 |
| generalPractitioner | 22 | 22 | 0.727 | 0.545 |
| medicalSpecialist | 15 | 15 | 0.533 | 0.533 |
| medicalStudent | 9 | 9 | 0.778 | 0.444 |
| nonMedical | 7 | 7 | 0.429 | 0.143 |
| resident | 6 | 6 | 0.833 | 0.833 |

## 4. Per-image VASC accuracy (sorted by drop)

| image_id | isic_id | n_un | n_ai | unaided | assisted | diff |
|---|---|---:|---:|---:|---:|---:|
| 242762 | ISIC_8791057 | 21 | 21 | 0.905 | 0.524 | -0.381 |
| 241004 | ISIC_1708907 | 21 | 21 | 0.952 | 0.667 | -0.286 |
| 241596 | ISIC_1996485 | 20 | 21 | 0.550 | 0.333 | -0.217 |
| 241124 | ISIC_0417486 | 20 | 20 | 0.600 | 0.400 | -0.200 |
| 241542 | ISIC_8614567 | 18 | 18 | 0.889 | 0.944 | 0.056 |
| 244306 | ISIC_3570541 | 24 | 24 | 0.917 | 1.000 | 0.083 |
| 240191 | ISIC_8462138 | 22 | 22 | 0.636 | 0.864 | 0.227 |
| 241622 | ISIC_4821214 | 20 | 20 | 0.300 | 0.700 | 0.400 |

## 5. Reader-diagnosis distribution when GT = VASC

### 5a. Unaided
| reader_diagnosis | n | proportion |
|---|---:|---:|
| VASC | 120 | 0.723 |
| MEL | 26 | 0.157 |
| NV | 6 | 0.036 |
| BCC | 6 | 0.036 |
| OTHER_MALIGNANT | 4 | 0.024 |
| OTHER_BENIGN | 3 | 0.018 |
| INF | 1 | 0.006 |

### 5b. Assisted
| reader_diagnosis | n | proportion |
|---|---:|---:|
| VASC | 114 | 0.683 |
| MEL | 17 | 0.102 |
| BCC | 15 | 0.090 |
| NV | 11 | 0.066 |
| OTHER_BENIGN | 8 | 0.048 |
| AKIEC | 1 | 0.006 |
| OTHER_MALIGNANT | 1 | 0.006 |

## 6. Error entropy (Shannon, bits)

| Arm | Error rate | n distinct error classes | Shannon entropy |
|---|---:|---:|---:|
| unaided | 0.277 | 6 | 1.915 |
| assisted | 0.317 | 6 | 2.140 |

Interpretation: higher Shannon entropy under AI assistance indicates 'error diversification'—readers no longer concentrate errors on a single look-alike class (e.g., MEL) but spread them across multiple plausible non-vascular candidates suggested by the AI's top-K.

## 7. Revision pattern (paired same-image flips)

- n paired VASC observations: 148
- changed diagnosis (un→ai): 41 (0.277 rate)
- productive changes (wrong→right): 14
- harmful changes (right→wrong): 21
- productive/harmful ratio: 0.667

## 8. Ceiling-effect class comparison

(VASC has the highest unaided accuracy of any class; AI offers limited upside on already-near-ceiling categories.)

| Class | n_un | n_ai | unaided | assisted | diff | paired p |
|---|---:|---:|---:|---:|---:|---:|
| VASC | 166 | 167 | 0.723 | 0.683 | -0.040 | 0.4696 |
| MEL | 345 | 344 | 0.675 | 0.709 | 0.034 | 0.6545 |
| BCC | 414 | 412 | 0.618 | 0.711 | 0.093 | 0.0028 |
| NV | 511 | 511 | 0.566 | 0.658 | 0.092 | 0.2295 |
| AKIEC | 168 | 170 | 0.542 | 0.624 | 0.082 | 0.0452 |
| DF | 163 | 165 | 0.472 | 0.642 | 0.170 | 0.0001 |
| SCCKA | 186 | 187 | 0.446 | 0.481 | 0.035 | 0.0819 |
| BKL | 342 | 344 | 0.380 | 0.494 | 0.114 | 0.0003 |
| OTHER_BENIGN | 81 | 81 | 0.210 | 0.296 | 0.086 | 0.1544 |
| INF | 66 | 66 | 0.167 | 0.227 | 0.061 | 0.1241 |
| OTHER_MALIGNANT | 20 | 20 | 0.000 | 0.000 | 0.000 | — |

## 9. Clinical-decision distribution on VASC

### 9a. Unaided
| decision | n | proportion |
|---|---:|---:|
| dismiss | 71 | 0.428 |
| biopsy | 51 | 0.307 |
| monitor | 42 | 0.253 |
| local_therapy | 2 | 0.012 |

### 9b. Assisted
| decision | n | proportion |
|---|---:|---:|
| dismiss | 70 | 0.419 |
| biopsy | 61 | 0.365 |
| monitor | 34 | 0.204 |
| local_therapy | 2 | 0.012 |