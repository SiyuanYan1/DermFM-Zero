"""Build the Derm7pt-VQA train/val/test CSVs.

Outputs the artefacts under
    data/VQA/derm7pt-VQA/meta/{train,val,test}.csv

What the pipeline does
----------------------
1. Load the source `meta.csv` of derm7pt (one row per case).
2. Generate VQA samples by sampling natural-language question templates for
   each of 7 question types (concept_existence, concept_identification,
   concept_criteria, texture, management, diagnosis, modality). Each case
   contributes both a clinical-photo and a dermoscopy view.
3. Filter answers with < 5 occurrences; build a 4k balanced subset stratified
   by (question_type × modality × answer) with the 40/35/15/10 % mixture
   across question groups.
4. Apply a *case-level* (image_id) train/val/test partition so no image
   crosses splits. By default the partition is a seeded random 60/20/20
   shuffle of the cases that survived steps 1–3. Pass
   `--case_split data/.../case_split.csv` to instead consume the published
   case-level manifest (recommended for reproducing the paper).
5. Build `answer_id` from `train.answer.unique()` and broadcast to val/test.
6. Write CSVs with absolute `image_path` values.

Usage
-----
    cd multimodal_finetune/preprocessing
    # 1) reproduce the paper splits exactly (default --case_split points to
    #    the published case-split manifest)
    python build_derm7pt_vqa.py

    # 2) generate a fresh seeded build from raw meta only
    python build_derm7pt_vqa.py --case_split "" --seed 42
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

SEVEN_POINT_CONCEPTS = [
    'pigment_network', 'streaks', 'pigmentation', 'regression_structures',
    'dots_and_globules', 'blue_whitish_veil', 'vascular_structures',
]

TEMPLATES = {
    'concept_existence': {
        'derm': [
            "Is {concept} present in this dermoscopic image?",
            "Does this dermoscopic view show {concept}?",
            "Can you identify {concept} in this dermoscopic image?",
            "Is there evidence of {concept} in this dermoscopic examination?",
            "Does this lesion exhibit {concept} under dermoscopy?",
            "Is {concept} visible in this dermoscopic view?",
            "Can {concept} be observed in this dermoscopic image?",
            "Does this dermoscopic image demonstrate {concept}?",
            "Is {concept} detectable in this dermoscopic examination?",
            "Can you see {concept} in this dermoscopic picture?",
            "Is {concept} apparent in this dermoscopic view?",
            "Does this dermoscopic image contain {concept}?",
            "Is {concept} discernible in this dermoscopic examination?",
            "Can {concept} be found in this dermoscopic image?",
            "Does this dermoscopic view reveal {concept}?",
        ],
        'clinic': [
            "Is {concept} visible in this clinical photograph?",
            "Does this clinical image show {concept}?",
            "Can you observe {concept} in this clinical photograph?",
            "Is there evidence of {concept} in this clinical view?",
            "Can you see {concept} in this clinical image?",
            "Is {concept} present in this clinical photograph?",
            "Does this clinical view demonstrate {concept}?",
            "Can {concept} be identified in this clinical image?",
            "Is {concept} apparent in this clinical photograph?",
            "Does this clinical picture show {concept}?",
            "Can {concept} be observed in this clinical view?",
            "Is {concept} detectable in this clinical image?",
            "Does this clinical photograph contain {concept}?",
            "Can you identify {concept} in this clinical picture?",
            "Is {concept} discernible in this clinical view?",
        ],
    },
    'concept_identification': {
        'derm': [
            "What dermatological features are visible in this dermoscopic image?",
            "Which skin concepts can you identify in this dermoscopic view?",
            "What are the key dermoscopic findings in this image?",
            "Identify the prominent dermoscopic features in this image.",
            "What dermoscopic patterns do you observe?",
            "What dermoscopic characteristics are present in this image?",
            "Which dermoscopic features can you detect?",
            "What skin patterns are visible under dermoscopy?",
            "Identify the dermoscopic structures in this image.",
            "What dermoscopic elements can you recognize?",
            "Which dermatological patterns are evident in this dermoscopic view?",
            "What features can you identify in this dermoscopic examination?",
            "Describe the dermoscopic findings you can observe.",
            "What dermoscopic criteria are visible in this image?",
            "Which skin features are apparent under dermoscopic examination?",
        ],
        'clinic': [
            "What clinical features are visible in this photograph?",
            "Which skin characteristics can you identify in this clinical image?",
            "What are the key clinical findings in this image?",
            "Identify the prominent clinical features in this photograph.",
            "What clinical patterns do you observe?",
            "What skin characteristics are visible in this clinical view?",
            "Which clinical features can you detect in this image?",
            "What dermatological findings are apparent in this photograph?",
            "Identify the clinical structures visible in this image.",
            "What clinical elements can you recognize?",
            "Which skin patterns are evident in this clinical photograph?",
            "What features can you identify in this clinical examination?",
            "Describe the clinical findings you can observe.",
            "What clinical criteria are visible in this image?",
            "Which dermatological features are apparent in this photograph?",
        ],
    },
    'concept_criteria': {
        'derm': [
            "Describe the {concept} characteristics in this dermoscopic image.",
            "What type of {concept} is present in this dermoscopic view?",
            "Evaluate the {concept} pattern in this dermoscopic examination.",
            "Assess the {concept} features visible in this dermoscopic image.",
            "Characterize the {concept} in this dermoscopic view.",
            "How would you describe the {concept} in this dermoscopic image?",
            "What is the nature of {concept} shown in this dermoscopic view?",
            "Analyze the {concept} pattern visible in this dermoscopic examination.",
            "What classification of {concept} is demonstrated in this image?",
            "Describe the morphology of {concept} in this dermoscopic view.",
            "What variant of {concept} is visible in this dermoscopic image?",
            "How would you categorize the {concept} seen in this examination?",
            "What subtype of {concept} is present in this dermoscopic view?",
            "Evaluate the distribution of {concept} in this dermoscopic image.",
            "What grade or severity of {concept} is shown in this view?",
        ],
        'clinic': [
            "Describe the {concept} characteristics visible in this clinical photograph.",
            "What type of {concept} can be observed in this clinical image?",
            "Evaluate the {concept} pattern in this clinical view.",
            "Assess the {concept} features visible in this clinical photograph.",
            "Characterize the {concept} in this clinical image.",
            "How would you describe the {concept} in this clinical photograph?",
            "What is the nature of {concept} shown in this clinical view?",
            "Analyze the {concept} pattern visible in this clinical examination.",
            "What classification of {concept} is demonstrated in this photograph?",
            "Describe the morphology of {concept} in this clinical image.",
            "What variant of {concept} is visible in this clinical photograph?",
            "How would you categorize the {concept} seen in this clinical view?",
            "What subtype of {concept} is present in this clinical image?",
            "Evaluate the distribution of {concept} in this clinical photograph.",
            "What grade or severity of {concept} is shown in this clinical view?",
        ],
    },
    'texture': {
        'derm': [
            "What is the elevation of this skin lesion?",
            "Describe the texture of this lesion.",
            "What is the surface characteristic of this lesion?",
            "How would you describe the physical texture of this lesion?",
            "What is the tactile quality of this skin lesion?",
            "What is the morphological elevation of this lesion?",
            "Describe the three-dimensional characteristics of this lesion.",
            "What is the topography of this skin lesion?",
            "How would you characterize the surface relief of this lesion?",
            "What is the structural elevation of this skin lesion?",
            "Describe the physical prominence of this lesion.",
            "What is the dimensional characteristic of this lesion?",
            "How would you describe the raised nature of this lesion?",
            "What is the vertical profile of this skin lesion?",
            "Describe the surface architecture of this lesion.",
        ],
    },
    'management': {
        'derm': [
            "Based on this dermoscopic image of a {age} {sex} patient's {location} lesion, what is your clinical recommendation?",
            "What management approach would you suggest for this {sex} patient based on the dermoscopic findings?",
            "What is the recommended next step for this {location} lesion seen in dermoscopy?",
            "Given the dermoscopic features of this {age} {sex} patient's lesion, what clinical action should be taken?",
            "What is your management suggestion for this {location} lesion based on dermoscopic examination?",
            "What treatment plan would you recommend for this {age} {sex} patient's {location} lesion?",
            "Based on the dermoscopic findings, what follow-up is indicated for this {sex} patient?",
            "What clinical decision would you make regarding this {location} lesion in a {age} {sex} patient?",
            "Given this dermoscopic presentation, what management strategy is appropriate for this patient?",
            "What therapeutic approach would you suggest for this {location} lesion?",
            "Based on dermoscopic examination, what intervention is recommended for this {sex} patient?",
            "What is the most appropriate management for this {age} {sex} patient's lesion?",
            "Given the dermoscopic characteristics, what action should be taken for this {location} lesion?",
            "What clinical pathway would you recommend for this patient based on dermoscopic findings?",
            "Based on this dermoscopic image, what management protocol is indicated?",
        ],
        'clinic': [
            "Based on this clinical photograph of a {age} {sex} patient's {location} lesion, what is your clinical recommendation?",
            "What management approach would you suggest for this {sex} patient based on the clinical appearance?",
            "What is the recommended next step for this {location} lesion seen in the clinical image?",
            "Given the clinical features of this {age} {sex} patient's lesion, what action should be taken?",
            "What is your management suggestion for this {location} lesion based on clinical examination?",
            "What treatment plan would you recommend for this {age} {sex} patient's {location} lesion?",
            "Based on the clinical findings, what follow-up is indicated for this {sex} patient?",
            "What clinical decision would you make regarding this {location} lesion in a {age} {sex} patient?",
            "Given this clinical presentation, what management strategy is appropriate for this patient?",
            "What therapeutic approach would you suggest for this {location} lesion?",
            "Based on clinical examination, what intervention is recommended for this {sex} patient?",
            "What is the most appropriate management for this {age} {sex} patient's lesion?",
            "Given the clinical characteristics, what action should be taken for this {location} lesion?",
            "What clinical pathway would you recommend for this patient based on clinical findings?",
            "Based on this clinical image, what management protocol is indicated?",
        ],
    },
    'diagnosis': {
        'derm': [
            "Based on this dermoscopic image of a {age} {sex} patient's {location} lesion, what is your diagnosis?",
            "What condition does this {location} lesion represent in the dermoscopic view?",
            "What is the most likely diagnosis for this {sex} patient based on dermoscopic findings?",
            "Identify the skin condition shown in this dermoscopic examination of a {age} {sex} patient.",
            "What dermatological diagnosis would you make from this dermoscopic image?",
            "Given the dermoscopic features of this {location} lesion in a {age} {sex} patient, what is your diagnosis?",
            "What skin condition is demonstrated in this dermoscopic view of a {sex} patient's lesion?",
            "Based on dermoscopic examination, what is the most probable diagnosis for this {location} lesion?",
            "What dermatological entity does this {age} {sex} patient's lesion represent?",
            "Given the dermoscopic characteristics, what condition would you diagnose?",
            "What is your differential diagnosis for this {location} lesion in a {sex} patient?",
            "Based on this dermoscopic presentation, what skin disease is most likely?",
            "What pathological condition is suggested by this dermoscopic image of a {age} {sex} patient?",
            "Given the dermoscopic morphology, what is your clinical diagnosis?",
            "What dermatological disorder is evident in this {location} lesion?",
        ],
        'clinic': [
            "Based on this clinical photograph of a {age} {sex} patient's {location} lesion, what is your diagnosis?",
            "What condition does this {location} lesion represent in the clinical image?",
            "What is the most likely diagnosis for this {sex} patient based on clinical appearance?",
            "Identify the skin condition shown in this clinical photograph of a {age} {sex} patient.",
            "What dermatological diagnosis would you make from this clinical image?",
            "Given the clinical features of this {location} lesion in a {age} {sex} patient, what is your diagnosis?",
            "What skin condition is demonstrated in this clinical view of a {sex} patient's lesion?",
            "Based on clinical examination, what is the most probable diagnosis for this {location} lesion?",
            "What dermatological entity does this {age} {sex} patient's lesion represent?",
            "Given the clinical characteristics, what condition would you diagnose?",
            "What is your differential diagnosis for this {location} lesion in a {sex} patient?",
            "Based on this clinical presentation, what skin disease is most likely?",
            "What pathological condition is suggested by this clinical image of a {age} {sex} patient?",
            "Given the clinical morphology, what is your clinical diagnosis?",
            "What dermatological disorder is evident in this {location} lesion?",
        ],
    },
    'modality': {
        'derm': [
            "What is the modality of this skin image?",
            "What imaging technique was used to capture this lesion?",
            "What type of dermatological imaging is shown in this image?",
            "Identify the imaging modality used for this examination.",
            "What kind of medical image is this?",
            "What photographic technique was employed for this image?",
            "What type of dermatological photography is this?",
            "What imaging method was used to document this lesion?",
            "What kind of dermatological examination technique is shown?",
            "What type of clinical imaging modality is demonstrated?",
            "What photographic modality was used for this skin lesion?",
            "What imaging approach was taken for this dermatological examination?",
            "What type of visual documentation method is this?",
            "What kind of dermatological imaging technique is employed?",
            "What photographic methodology was used for this lesion?",
        ],
    },
}
# texture / modality templates are modality-agnostic — alias 'clinic' to 'derm'.
TEMPLATES['texture']['clinic']  = TEMPLATES['texture']['derm']
TEMPLATES['modality']['clinic'] = TEMPLATES['modality']['derm']


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def generate_vqa(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Produce one VQA row per (case, modality, question_type[, concept])."""
    rows = []
    for idx, row in meta_df.iterrows():
        for modality in ('derm', 'clinic'):
            if pd.isna(row.get(modality)):
                continue
            patient = {'age': 'adult', 'sex': row['sex'], 'location': row['location']}

            for c in SEVEN_POINT_CONCEPTS:
                q = random.choice(TEMPLATES['concept_existence'][modality]).format(
                    concept=c.replace('_', ' '), **patient)
                rows.append(dict(image_id=idx, modality=modality, concept=c,
                                 question_type='concept_existence', question=q,
                                 answer='Yes' if row[c] != 'absent' else 'No'))

            present = [c for c in SEVEN_POINT_CONCEPTS if row[c] != 'absent']
            if len(present) == 1:
                q = random.choice(TEMPLATES['concept_identification'][modality]).format(**patient)
                rows.append(dict(image_id=idx, modality=modality, concept=np.nan,
                                 question_type='concept_identification', question=q,
                                 answer=present[0].replace('_', ' ')))

            for c in SEVEN_POINT_CONCEPTS:
                if row[c] == 'absent':
                    continue
                q = random.choice(TEMPLATES['concept_criteria'][modality]).format(
                    concept=c.replace('_', ' '), **patient)
                rows.append(dict(image_id=idx, modality=modality, concept=c,
                                 question_type='concept_criteria', question=q, answer=row[c]))

            for kind, ans_key in (('texture', 'elevation'),
                                  ('management', 'management')):
                q = random.choice(TEMPLATES[kind][modality]).format(**patient)
                rows.append(dict(image_id=idx, modality=modality, concept=np.nan,
                                 question_type=kind, question=q, answer=row[ans_key]))

            q = random.choice(TEMPLATES['modality'][modality]).format(**patient)
            rows.append(dict(image_id=idx, modality=modality, concept=np.nan,
                             question_type='modality', question=q,
                             answer='dermoscopic' if modality == 'derm' else 'clinical'))

            if 'diagnosis' in meta_df.columns:
                q = random.choice(TEMPLATES['diagnosis'][modality]).format(**patient)
                rows.append(dict(image_id=idx, modality=modality, concept=np.nan,
                                 question_type='diagnosis', question=q, answer=row['diagnosis']))

    return pd.DataFrame(rows)[
        ['image_id', 'question', 'answer', 'question_type', 'concept', 'modality']
    ]


def attach_image_path(vqa_df, meta_df, image_root):
    path_map = {}
    for _, row in meta_df.iterrows():
        for modality in ('derm', 'clinic'):
            rel = row.get(modality)
            if pd.notna(rel):
                # case_num is 1-indexed; image_id (= original df index) is 0-indexed.
                path_map[(row['case_num'] - 1, modality)] = os.path.join(image_root, rel)
    out = vqa_df.copy()
    out['image_path'] = out.apply(lambda r: path_map.get((r['image_id'], r['modality'])), axis=1)
    return out.dropna(subset=['image_path']).reset_index(drop=True)


def filter_by_answer_count(df, min_count=5):
    keep = df['answer'].value_counts().pipe(lambda s: s[s >= min_count]).index
    return df[df['answer'].isin(keep)].copy()


def stratified_sample(df, target, min_count=5, seed=42):
    df = filter_by_answer_count(df, min_count)
    if df.empty:
        return df
    groups = df.groupby(['question_type', 'modality', 'answer'])
    n = len(groups)
    per_group, remainder = divmod(target, n)
    parts = []
    for i, (_, g) in enumerate(groups):
        k = per_group + (1 if i < remainder else 0)
        parts.append(g if len(g) <= k else g.sample(n=k, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(n=target, random_state=seed) if len(out) > target else out


def build_balanced_subset(df, target=4000, min_count=5, seed=42):
    """4k samples ≈ 40% diagnosis+management, 35% concept core, 15% existence, 10% other."""
    df = filter_by_answer_count(df, min_count)
    groups = {
        'diagnosis_management': (['diagnosis', 'management'], 0.40),
        'concept_core':         (['concept_identification', 'concept_criteria'], 0.35),
        'existence':            (['concept_existence'], 0.15),
        'others':                (['texture', 'modality'], 0.10),
    }
    parts = []
    for qtypes, pct in groups.values():
        sub = df[df['question_type'].isin(qtypes)]
        k = int(target * pct)
        parts.append(sub if len(sub) <= k else stratified_sample(sub, k, min_count, seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < target:
        leftover = df.drop(out.index, errors='ignore')
        if len(leftover):
            out = pd.concat(
                [out, leftover.sample(n=min(target - len(out), len(leftover)), random_state=seed)],
                ignore_index=True,
            )
    keep = out['answer'].value_counts().pipe(lambda s: s[s >= min_count]).index
    out = out[out['answer'].isin(keep)]
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def random_case_split(df, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Case-level random shuffle → no image leakage across splits."""
    ids = np.array(sorted(df['image_id'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    return {
        'train': set(ids[:n_train].tolist()),
        'val':   set(ids[n_train:n_train + n_val].tolist()),
        'test':  set(ids[n_train + n_val:].tolist()),
    }


def load_case_split(path):
    """Load a precomputed case-level partition manifest (CSV with image_id,split)."""
    cs = pd.read_csv(path)
    return {s: set(cs.loc[cs['split'] == s, 'image_id'].astype(int)) for s in cs['split'].unique()}


def apply_case_split(df, case_split):
    """Drop rows whose image_id isn't in any split, then label by split."""
    image_to_split = {i: s for s, ids in case_split.items() for i in ids}
    out = df.copy()
    out['split'] = out['image_id'].map(image_to_split)
    return out.dropna(subset=['split'])


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # multimodal_finetune/preprocessing → DermFM-Zero/

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--meta_csv',
                    default=str(repo_root / 'data' / 'VQA' /
                                'preprocessing_inputs' / 'derm7pt-VQA' / 'derm7pt-meta.csv'),
                    help='Source derm7pt meta CSV.')
    ap.add_argument('--image_root',
                    default='../data/VQA/derm7pt-VQA/images',
                    help='Image-root prefix written into each row\'s image_path. '
                         'Default resolves from multimodal_finetune/ (the CWD of train.py).')
    ap.add_argument('--output_dir',
                    default=str(repo_root / 'data' / 'VQA' /
                                'derm7pt-VQA' / 'meta'))
    ap.add_argument('--case_split',
                    default=str(repo_root / 'data' / 'VQA' /
                                'preprocessing_inputs' / 'derm7pt-VQA' / 'case_split.csv'),
                    help='Optional CSV with columns (image_id, split). When provided '
                         'the case-level partition is read from this manifest instead '
                         'of being generated by `--seed`.')
    ap.add_argument('--target_size',  type=int, default=4000)
    ap.add_argument('--min_answer_count', type=int, default=5)
    ap.add_argument('--seed',         type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    meta = pd.read_csv(args.meta_csv)
    print(f"meta.csv rows: {len(meta)}")

    vqa = generate_vqa(meta)
    print(f"raw VQA samples: {len(vqa)}")

    vqa = attach_image_path(vqa, meta, args.image_root)
    print(f"after image_path attach: {len(vqa)}")

    vqa = build_balanced_subset(vqa, target=args.target_size,
                                min_count=args.min_answer_count, seed=args.seed)
    print(f"balanced subset: {len(vqa)}")

    if args.case_split:
        split = load_case_split(args.case_split)
        print(f"using case_split manifest: {args.case_split}")
    else:
        split = random_case_split(vqa, seed=args.seed)
        print(f"seeded random case split (seed={args.seed})")

    vqa = apply_case_split(vqa, split)
    print(vqa['split'].value_counts())

    train = vqa[vqa['split'] == 'train']
    answer2id = {a: i for i, a in enumerate(train['answer'].unique())}
    vqa['answer_id'] = vqa['answer'].map(answer2id)

    cols = ['image_id', 'question', 'answer', 'question_type', 'concept',
            'modality', 'image_path', 'split', 'answer_id']
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in ('train', 'val', 'test'):
        sub = vqa[vqa['split'] == s][cols]
        path = out_dir / f"{s}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(sub)}  unique_image_ids={sub['image_id'].nunique()}")


if __name__ == '__main__':
    main()
