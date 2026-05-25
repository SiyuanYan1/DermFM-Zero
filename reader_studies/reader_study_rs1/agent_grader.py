"""
Reader Study 1: GPT-4o-mini Grading Agent
Automated diagnostic and management grading for RS1 reader responses.

Uses GPT-4o-mini to evaluate each case against ground truth, producing:
  - Diagnosis score (1-5 rubric)
  - Top-3 spot-on / generic flags
  - Management grade (clinical decision matrix)

Input:  Raw reader response CSV (exported from survey platform)
Output: Graded Excel file with 46 standardized columns

Usage:
    pip install pandas openai tqdm
    export OPENAI_API_KEY="sk-..."
    python agent_grader.py --input raw_responses.csv --output graded.xlsx
    python agent_grader.py --input raw_responses.csv --output graded.xlsx --demo  # first 10 cases
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
# API key must be set via environment variable (never hardcode)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it before running: export OPENAI_API_KEY='sk-...'"
    )

DEFAULT_INPUT_FILE = "raw_responses.csv"
DEFAULT_OUTPUT_FILE = "graded_output.xlsx"

# Concurrency settings
MAX_WORKERS = 10   # Parallel API requests (adjust per rate limit, recommend 5-20)
BATCH_SIZE = 50    # Cases per progress batch

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# ============================================================================
# SYSTEM PROMPT (Diagnosis + Management Grading Rubric)
# ============================================================================
SYSTEM_PROMPT = """
You are an expert Dermatology Evaluation Agent. Your task is to grade a doctor's diagnosis AND management plan against the Ground Truth (GT).

### PART 1: DIAGNOSIS GRADING RULES (1-5)

- **5 (Spot On):** The diagnosis is exactly correct (e.g., GT: Blue Nevus -> Doc: Blue Nevus).
- **4 (Generic):** Correct but too generic (e.g., GT: Blue Nevus -> Doc: Nevus). The category is right, but specificity is missing.
- **3 (Same Class):** Incorrect diagnosis, but belongs to the same class (e.g., benign proliferation) and has the same consequences. (e.g., GT: Blue Nevus -> Doc: Apocrine Cystadenoma).
- **2 (Different Class, Same Consequences):** Wrong class, but the outcome/management is effectively the same (e.g., benign). (e.g., GT: Blue Nevus -> Doc: Tattoo).
- **1 (Harmful/Different Class):** Wrong class with potential harm. (e.g., GT: Blue Nevus -> Doc: BCC). Or missing a malignancy (GT: Melanoma -> Doc: Nevus).

### PART 2: TOP-3 ANALYSIS

- **Spot On in Top 3:** Is the exact diagnosis (Score 5 level) present in the doctor's Top 3 list? (true/false)
- **Generic/Correct in Top 3:** Is a correct but generic diagnosis (Score 4 level) present in the Top 3? (true/false)

### PART 3: MANAGEMENT GRADING (Clinical Decision Matrix)

**STEP 1: Determine GT Category**
- **Malignant / Pre-malignant (High Risk):** Melanoma, BCC, SCC, AK, IEC (Bowen's), etc.
- **Benign (Low Risk):** Nevus, Seborrheic Keratosis (SK), Dermatofibroma, Vascular, etc.

**STEP 2: Apply Matrix Rules**

#### SCENARIO A: GT is MALIGNANT (Goal: Safety / Treatment)
| Action Combination | Grade | Reasoning |
| :--- | :--- | :--- |
| **Any + Refer** | **Adequate** | **Safe.** Referral guarantees specialist review (Safety Net). |
| **Biopsy + Manage Myself** | **Adequate** | **Safe.** Doctor identified risk and is taking action (e.g., excision). |
| **None + Manage Myself** | **Inadequate and dangerous** | **Risky.** Treating cancer (e.g., with cryo) without pathology is blind and dangerous. |
| **None + Reassure** | **Inadequate and dangerous** | **Critical Error.** Missed diagnosis leading to fatal delay. |
| **Biopsy + Reassure** | **Inadequate and dangerous** | **Contradictory.** Logic failure (biopsy then ignore?). |

#### SCENARIO B: GT is BENIGN (Goal: Efficiency / Avoid Overtreatment)
| Action Combination | Grade | Reasoning |
| :--- | :--- | :--- |
| **None + Reassure** | **Perfect** | **Gold Standard.** Correctly identified as harmless. |
| **None + Manage Myself** | **Perfect** | **Appropriate.** Cosmetic removal or symptomatic treatment. |
| **Any + Refer** | **Adequate** | **Inefficient.** Patient is safe, but specialist resources are wasted. |
| **Biopsy + Any** | **Inadequate but harmless** | **Over-treatment.** Unnecessary procedure/scarring, but no serious harm. |

---

### IN-CONTEXT LEARNING EXAMPLES

**Example 1 (Benign, Perfect Diagnosis & Management):**
Input: {"GT": "Blue Nevus", "Doc_Dx": "Blue Nevus", "Doc_Top3": ["Blue Nevus", "Melanoma", "Tattoo"], "Doc_Mgmt": "None and Reassure"}
Output: {
    "dx_score": 5,
    "dx_reason": "Diagnosis is spot on.",
    "spot_on_in_top3": true,
    "generic_in_top3": false,
    "mgmt_grade": "Perfect",
    "mgmt_reason": "Gold Standard. Correctly identified as harmless benign lesion."
}

**Example 2 (Malignant, Missed but Safe by Referral):**
Input: {"GT": "Melanoma", "Doc_Dx": "Nevus", "Doc_Top3": ["Nevus", "SK", "Blue Nevus"], "Doc_Mgmt": "Biopsy and Refer"}
Output: {
    "dx_score": 1,
    "dx_reason": "Missed malignancy (Harmful).",
    "spot_on_in_top3": false,
    "generic_in_top3": false,
    "mgmt_grade": "Adequate",
    "mgmt_reason": "Safe. Referral guarantees specialist review (Safety Net principle). The referral will catch the diagnostic error."
}

**Example 3 (Malignant, Dangerous Management):**
Input: {"GT": "BCC", "Doc_Dx": "Seborrheic Keratosis", "Doc_Top3": ["SK", "Wart", "Nevus"], "Doc_Mgmt": "None and Reassure"}
Output: {
    "dx_score": 1,
    "dx_reason": "Missed malignancy (Different class with potential harm).",
    "spot_on_in_top3": false,
    "generic_in_top3": false,
    "mgmt_grade": "Inadequate and dangerous",
    "mgmt_reason": "Critical Error. Reassuring patient with BCC leads to treatment delay and potential progression."
}

**Example 4 (Benign, Over-treatment):**
Input: {"GT": "Seborrheic Keratosis", "Doc_Dx": "BCC", "Doc_Top3": ["BCC", "Melanoma", "AK"], "Doc_Mgmt": "Biopsy and Refer"}
Output: {
    "dx_score": 1,
    "dx_reason": "Wrong class (Benign vs Malignant) with unnecessary concern.",
    "spot_on_in_top3": false,
    "generic_in_top3": false,
    "mgmt_grade": "Inadequate but harmless",
    "mgmt_reason": "Over-treatment. Biopsy is unnecessary for benign SK, causing unnecessary procedure and patient anxiety, but no serious harm."
}

---

### OUTPUT FORMAT:
Return strictly valid JSON with these keys:
{
    "dx_score": <1-5>,
    "dx_reason": "<brief explanation>",
    "spot_on_in_top3": <true/false>,
    "generic_in_top3": <true/false>,
    "mgmt_grade": "Perfect" | "Adequate" | "Inadequate but harmless" | "Inadequate and dangerous",
    "mgmt_reason": "<clinical explanation>"
}
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def safe_str(value) -> str:
    """Safely convert a value to a stripped string; return '' for NaN."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def convert_management_to_text(investigation: str, next_step: str) -> str:
    """
    Convert survey investigation_action and next_step_action fields
    into standardized management text.

    investigation_action values: NONE, BIOPSY, OTHER
    next_step_action values:     REASSURE, MANAGE_MYSELF, REFER
    """
    investigation = safe_str(investigation).upper()
    next_step = safe_str(next_step).upper()

    # Map investigation action
    if investigation == "BIOPSY":
        inv_text = "Biopsy"
    elif investigation == "OTHER":
        inv_text = "Other Investigation"
    else:  # NONE or empty
        inv_text = "None"

    # Map next-step action
    if next_step == "REASSURE":
        step_text = "Reassure"
    elif next_step == "MANAGE_MYSELF":
        step_text = "Manage Myself"
    elif next_step == "REFER":
        step_text = "Refer"
    else:
        step_text = "Reassure"  # default

    return f"{inv_text} and {step_text}"


# ============================================================================
# GPT-4o-mini EVALUATION
# ============================================================================
def get_ai_evaluation_single(
    gt: str, doc_dx: str, doc_top3: List[str], doc_mgmt: str,
    max_retries: int = 3
) -> Optional[Dict]:
    """Call GPT-4o-mini to grade a single case against ground truth."""
    user_input = {
        "GT": gt,
        "Doc_Dx": doc_dx,
        "Doc_Top3": doc_top3,
        "Doc_Mgmt": doc_mgmt
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
            return json.loads(result_text)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                print(f"  API Error after {max_retries} retries: {str(e)[:100]}")
                return None

    return None


def process_single_case(index: int, row: pd.Series) -> Optional[Dict]:
    """Process one case: grade both unaided (PRE) and assisted (POST) phases."""
    gt = row.get('Ground truth diagnosis')
    if pd.isna(gt):
        return None

    # --- Demographics (9 columns) ---
    case_info = {
        'Case ID': row.get('Case id'),
        'GT': gt,
        'Responder_ID': row.get('Responder id'),
        'User Role': row.get('User role'),
        'Years_Derm_Experience': row.get('Years derm experience', 0),
        'Age_Bracket': row.get('Age bracket', ''),
        'Gender': row.get('Gender', ''),
        'Country': row.get('Country code', ''),
        'Experience': row.get('Years experience', ''),
    }

    # --- Phase 1: Unaided (PRE-AI) ---
    unaided_dx = safe_str(row.get('Preferred diagnosis'))
    unaided_top3 = [
        safe_str(row.get('Preferred diagnosis')),
        safe_str(row.get('Differential diagnosis', '')),
        safe_str(row.get('Secondary differential diagnosis', ''))
    ]
    unaided_mgmt_text = convert_management_to_text(
        row.get('PRE investigation_action'),
        row.get('PRE next_step_action')
    )

    res_unaided = get_ai_evaluation_single(gt, unaided_dx, unaided_top3, unaided_mgmt_text)
    if not res_unaided:
        return None

    # Unaided diagnosis columns (8)
    case_info['Unaided_Dx_Text'] = unaided_dx
    case_info['Unaided_Dx_Score'] = res_unaided.get('dx_score')
    case_info['Unaided_Dx_Reason'] = res_unaided.get('dx_reason')
    case_info['Unaided_Dx_Confidence'] = row.get('Preferred diagnosis confidence (1-5)', '')
    case_info['PRE_Differential_1'] = safe_str(row.get('Differential diagnosis', ''))
    case_info['PRE_Differential_2'] = safe_str(row.get('Secondary differential diagnosis', ''))
    case_info['Unaided_Top3_SpotOn'] = res_unaided.get('spot_on_in_top3')
    case_info['Unaided_Top3_Generic'] = res_unaided.get('generic_in_top3')

    # Unaided management columns (4)
    case_info['Unaided_Mgmt_Text'] = unaided_mgmt_text
    case_info['Unaided_Mgmt_Grade'] = res_unaided.get('mgmt_grade')
    case_info['Unaided_Mgmt_Reason'] = res_unaided.get('mgmt_reason')
    case_info['Unaided_Mgmt_Confidence'] = row.get('Recommended management confidence (1-5)', '')

    # --- AI Predictions (6 columns) ---
    case_info['AI_Top1_Prediction'] = row.get('AI top1 prediction', '')
    case_info['AI_Top1_Confidence'] = row.get('AI top1 confidence', '')
    case_info['AI_Top2_Prediction'] = row.get('AI top2 prediction', '')
    case_info['AI_Top2_Confidence'] = row.get('AI top2 confidence', '')
    case_info['AI_Top3_Prediction'] = row.get('AI top3 prediction', '')
    case_info['AI_Top3_Confidence'] = row.get('AI top3 confidence', '')

    # --- Phase 2: Assisted (POST-AI) ---
    assisted_dx = safe_str(row.get('Preferred diagnosis after AI'))
    assisted_top3 = [
        safe_str(row.get('Preferred diagnosis after AI')),
        safe_str(row.get('Differential diagnosis after AI', '')),
        safe_str(row.get('Secondary differential diagnosis after AI', ''))
    ]
    assisted_mgmt_text = convert_management_to_text(
        row.get('POST investigation_action'),
        row.get('POST next_step_action')
    )

    res_assisted = get_ai_evaluation_single(gt, assisted_dx, assisted_top3, assisted_mgmt_text)
    if not res_assisted:
        return None

    # Assisted diagnosis columns (8)
    case_info['Assisted_Dx_Text'] = assisted_dx
    case_info['Assisted_Dx_Score'] = res_assisted.get('dx_score')
    case_info['Assisted_Dx_Reason'] = res_assisted.get('dx_reason')
    case_info['Assisted_Dx_Confidence'] = row.get('Preferred diagnosis confidence after AI (1-5)', '')
    case_info['POST_Differential_1'] = safe_str(row.get('Differential diagnosis after AI', ''))
    case_info['POST_Differential_2'] = safe_str(row.get('Secondary differential diagnosis after AI', ''))
    case_info['Assisted_Top3_SpotOn'] = res_assisted.get('spot_on_in_top3')
    case_info['Assisted_Top3_Generic'] = res_assisted.get('generic_in_top3')

    # Assisted management columns (4)
    case_info['Assisted_Mgmt_Text'] = assisted_mgmt_text
    case_info['Assisted_Mgmt_Grade'] = res_assisted.get('mgmt_grade')
    case_info['Assisted_Mgmt_Reason'] = res_assisted.get('mgmt_reason')
    case_info['Assisted_Mgmt_Confidence'] = row.get('Recommended management confidence after AI (1-5)', '')

    # --- Feedback & Interaction (4 columns) ---
    case_info['AI_Usefulness'] = row.get('AI usefulness in this case', '')
    case_info['AI_Trust_Score'] = row.get('AI trust score', '')
    case_info['Changed_Diagnosis'] = row.get('Change diagnosis after AI', False)
    case_info['Changed_Management'] = row.get('Change management after AI', False)

    # --- Timestamps (3 columns) ---
    case_info['Started_At'] = row.get('Started at', '')
    case_info['Completed_PRE_At'] = row.get('Completed PRE at', '')
    case_info['Completed_POST_At'] = row.get('Completed POST at', '')

    return case_info


# ============================================================================
# BATCH PROCESSING
# ============================================================================
def process_batch_parallel(df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Dict]:
    """Process a batch of cases in parallel using ThreadPoolExecutor."""
    batch_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_single_case, idx, row): idx
            for idx, row in df.iloc[start_idx:end_idx].iterrows()
        }

        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc=f"Batch {start_idx}-{end_idx}"):
            try:
                result = future.result()
                if result:
                    batch_results.append(result)
            except Exception as e:
                print(f"  Error processing case: {str(e)[:100]}")

    return batch_results


# ============================================================================
# MAIN
# ============================================================================
def main():
    global MAX_WORKERS

    parser = argparse.ArgumentParser(
        description='RS1 GPT-4o-mini Grading Agent: grade reader diagnoses and management plans'
    )
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help='Input CSV file path (raw reader responses)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                        help='Output Excel file path (graded results)')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: process only the first 10 cases')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help='Number of concurrent API workers (default: 10)')
    args = parser.parse_args()

    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    DEMO_MODE = args.demo
    MAX_WORKERS = args.workers

    print("=" * 80)
    print("RS1 GPT-4o-mini GRADING AGENT")
    print("=" * 80)
    print(f"  Input:    {INPUT_FILE}")
    print(f"  Output:   {OUTPUT_FILE}")
    print(f"  Workers:  {MAX_WORKERS}")
    print(f"  Demo:     {'ON (first 10 cases)' if DEMO_MODE else 'OFF (all cases)'}")
    print("=" * 80)

    # 1. Load data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} cases from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if DEMO_MODE:
        df = df.head(10)
        print(f"DEMO MODE: Processing only first 10 cases")

    # 2. Parallel batch processing
    all_results = []
    total_cases = len(df)
    start_time = time.time()

    print(f"\nStarting evaluation with {MAX_WORKERS} parallel workers...")

    for batch_start in range(0, total_cases, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_cases)
        print(f"\nProcessing batch {batch_start + 1}-{batch_end}/{total_cases}")

        batch_results = process_batch_parallel(df, batch_start, batch_end)
        all_results.extend(batch_results)

        elapsed = time.time() - start_time
        cases_done = len(all_results)
        if cases_done > 0:
            avg_time = elapsed / cases_done
            remaining = (total_cases - cases_done) * avg_time
            print(f"  Progress: {cases_done}/{total_cases} | "
                  f"Avg: {avg_time:.2f}s/case | "
                  f"ETA: {remaining / 60:.1f} min")

    elapsed_time = time.time() - start_time

    # 3. Save results
    if all_results:
        final_df = pd.DataFrame(all_results)

        # Standardized 46-column output order
        standard_columns = [
            # Demographics (9)
            'Case ID', 'GT', 'Responder_ID', 'User Role', 'Years_Derm_Experience',
            'Age_Bracket', 'Gender', 'Country', 'Experience',
            # Unaided diagnosis (8)
            'Unaided_Dx_Text', 'Unaided_Dx_Score', 'Unaided_Dx_Reason',
            'Unaided_Dx_Confidence',
            'PRE_Differential_1', 'PRE_Differential_2',
            'Unaided_Top3_SpotOn', 'Unaided_Top3_Generic',
            # Unaided management (4)
            'Unaided_Mgmt_Text', 'Unaided_Mgmt_Grade', 'Unaided_Mgmt_Reason',
            'Unaided_Mgmt_Confidence',
            # AI predictions (6)
            'AI_Top1_Prediction', 'AI_Top1_Confidence',
            'AI_Top2_Prediction', 'AI_Top2_Confidence',
            'AI_Top3_Prediction', 'AI_Top3_Confidence',
            # Assisted diagnosis (8)
            'Assisted_Dx_Text', 'Assisted_Dx_Score', 'Assisted_Dx_Reason',
            'Assisted_Dx_Confidence',
            'POST_Differential_1', 'POST_Differential_2',
            'Assisted_Top3_SpotOn', 'Assisted_Top3_Generic',
            # Assisted management (4)
            'Assisted_Mgmt_Text', 'Assisted_Mgmt_Grade', 'Assisted_Mgmt_Reason',
            'Assisted_Mgmt_Confidence',
            # Feedback & interaction (4)
            'AI_Usefulness', 'AI_Trust_Score', 'Changed_Diagnosis', 'Changed_Management',
            # Timestamps (3)
            'Started_At', 'Completed_PRE_At', 'Completed_POST_At'
        ]

        final_df = final_df[standard_columns]
        final_df.to_excel(OUTPUT_FILE, index=False)

        print(f"\n{'=' * 80}")
        print("Evaluation Complete")
        print(f"{'=' * 80}")
        print(f"  Processed: {len(all_results)}/{total_cases} valid cases")
        print(f"  Output:    {OUTPUT_FILE} ({len(standard_columns)} columns)")
        print(f"  Time:      {elapsed_time / 60:.2f} minutes")
        print(f"  Speed:     {elapsed_time / len(all_results):.2f} seconds/case")

        # Distribution summary
        print(f"\n{'=' * 80}")
        print("Diagnosis Score Distribution")
        print(f"{'=' * 80}")
        print("\nUnaided:")
        print(final_df['Unaided_Dx_Score'].value_counts().sort_index())
        print("\nAssisted:")
        print(final_df['Assisted_Dx_Score'].value_counts().sort_index())

        print(f"\n{'=' * 80}")
        print("Management Grade Distribution")
        print(f"{'=' * 80}")
        print("\nUnaided:")
        print(final_df['Unaided_Mgmt_Grade'].value_counts())
        print("\nAssisted:")
        print(final_df['Assisted_Mgmt_Grade'].value_counts())

        print(f"\n{'=' * 80}")
        print("Data Quality")
        print(f"{'=' * 80}")
        print(f"  Unique readers: {sorted(final_df['Responder_ID'].unique())}")
        print(f"  Unique cases:   {final_df['Case ID'].nunique()}")
        print(f"  Output columns: {len(final_df.columns)}")
    else:
        print("No results generated. Check input data and API connection.")


if __name__ == "__main__":
    main()
