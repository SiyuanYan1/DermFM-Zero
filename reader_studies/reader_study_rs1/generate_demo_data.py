"""
Generate synthetic demo data for RS1 (2 CSVs: cn_graded.csv, en_graded.csv).
Reader IDs and demographics are randomized; statistical patterns preserved approximately.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

OUTPUT_DIR = 'demo_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MGMT_GRADES = ['Inadequate and dangerous', 'Inadequate but harmless', 'Adequate', 'Perfect']
DX_SCORE_PROBS_UNAIDED = [0.15, 0.15, 0.25, 0.25, 0.20]
DX_SCORE_PROBS_ASSISTED = [0.08, 0.10, 0.18, 0.28, 0.36]
MGMT_PROBS_UNAIDED = [0.20, 0.15, 0.40, 0.25]
MGMT_PROBS_ASSISTED = [0.10, 0.10, 0.35, 0.45]


def generate_cohort(n_readers, cases_options, roles):
    rows = []
    case_counter = 1
    for r in range(n_readers):
        reader_id = r + 1
        n_cases = np.random.choice(cases_options)
        role = roles[r % len(roles)]
        for _ in range(n_cases):
            unaided_dx = np.random.choice([1, 2, 3, 4, 5], p=DX_SCORE_PROBS_UNAIDED)
            assisted_dx = min(5, unaided_dx + np.random.choice([0, 0, 1, 1, 2], p=[0.3, 0.2, 0.25, 0.15, 0.1]))
            unaided_top3 = 1 if unaided_dx >= 4 or np.random.random() < 0.3 else 0
            assisted_top3 = 1 if assisted_dx >= 4 or np.random.random() < 0.4 else 0

            rows.append({
                'Case ID': case_counter,
                'GT': f'Disease_{np.random.randint(1, 20)}',
                'Responder_ID': reader_id,
                'User Role': role,
                'Years_Derm_Experience': np.random.choice(['0', '1-2', '3-5', '6-10', '>10']),
                'Age_Bracket': np.random.choice(['25-34', '35-44', '45-54', '55-64']),
                'Gender': np.random.choice(['Male', 'Female']),
                'Country': 'DEMO',
                'Unaided_Dx_Score': unaided_dx,
                'Unaided_Dx_Reason': 'Demo',
                'Unaided_Dx_Confidence': np.random.choice([1, 2, 3, 4]),
                'Unaided_Top3_SpotOn': bool(unaided_top3),
                'Unaided_Top3_Generic': bool(np.random.random() < 0.3),
                'Unaided_Mgmt_Grade': np.random.choice(MGMT_GRADES, p=MGMT_PROBS_UNAIDED),
                'Unaided_Mgmt_Reason': 'Demo',
                'Unaided_Mgmt_Confidence': np.random.choice([1, 2, 3, 4]),
                'Assisted_Dx_Score': assisted_dx,
                'Assisted_Dx_Reason': 'Demo',
                'Assisted_Dx_Confidence': np.random.choice([1, 2, 3, 4]),
                'Assisted_Top3_SpotOn': bool(assisted_top3),
                'Assisted_Top3_Generic': bool(np.random.random() < 0.4),
                'Assisted_Mgmt_Grade': np.random.choice(MGMT_GRADES, p=MGMT_PROBS_ASSISTED),
                'Assisted_Mgmt_Reason': 'Demo',
                'Assisted_Mgmt_Confidence': np.random.choice([1, 2, 3, 4]),
                'AI_Trust_Score': np.random.choice([1, 2, 3, 4, 5]),
                'Changed_Diagnosis': bool(np.random.random() < 0.3),
                'Changed_Management': bool(np.random.random() < 0.25),
            })
            case_counter += 1
    return pd.DataFrame(rows)


print("Generating demo data...")

cn_demo = generate_cohort(15, [15, 17, 20, 25, 29, 30], ['GP'])
en_demo = generate_cohort(15, [15, 20, 25, 27, 30, 58],
                          ['GP'] * 5 + ['NP'] * 10)

cn_demo.to_csv(f'{OUTPUT_DIR}/cn_graded.csv', index=False)
en_demo.to_csv(f'{OUTPUT_DIR}/en_graded.csv', index=False)

print(f"  cn_graded.csv: {cn_demo['Responder_ID'].nunique()} readers, {len(cn_demo)} cases")
print(f"  en_graded.csv: {en_demo['Responder_ID'].nunique()} readers, {len(en_demo)} cases")
print(f"  Total: {cn_demo['Responder_ID'].nunique() + en_demo['Responder_ID'].nunique()} readers")
print("Done.")
