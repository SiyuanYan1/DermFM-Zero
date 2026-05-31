"""Build the SkinCap-VQA train/val/test CSVs.

Outputs the artefacts under
    data/multimodal_finetune_VQA/SkinCap-VQA/meta/{train,val,test}.csv

What the pipeline does
----------------------
1. Parse the public MCQA JSONs (one entry per case, each holding a chain of
   human/gpt conversation turns) into (image_path, question, answer) rows.
2. Keep only the SkinCap subset (rows whose image id contains "skincap").
3. Drop answers occurring <= 5 times across the whole pool.
4. Image-level random 60/20/20 split (np.random.seed=42).
5. Restrict to the intersection of answers present in train, val and test —
   ensures every answer class is represented in every split.
6. Rewrite image_path to point at the shipped SkinCap image directory.
7. Build `answer_id` from the unique answers (declaration order matches the
   shipped CSVs).

Usage
-----
    cd multimodal_finetune/preprocessing
    python build_skincap_vqa.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------

_ANSWER_LABEL_RX = re.compile(r'^[A-Z]\)\s*', flags=re.IGNORECASE)
_OPTIONS_BLOCK_RX = re.compile(r'\s*Options:\s*\n.*', flags=re.DOTALL)
_TRAILING_CHOICE_RX = re.compile(r'\n[A-Z]\)\s*.*', flags=re.MULTILINE)


def _clean_question(q):
    q = _OPTIONS_BLOCK_RX.sub('', q)
    q = _TRAILING_CHOICE_RX.sub('', q)
    return q.strip()


def conversations_to_rows(json_path):
    """Flatten the conversation JSON into (image_path, question, answer) rows.

    The JSON is a list of items; each item has an `image` field and a
    `conversations` list alternating between human and gpt turns. The human
    turn carries the question (with an "<image>\\n" prefix on the first one
    and an `Options:\\nA) ... B) ...` block — both stripped here) and the gpt
    turn carries the answer (prefixed with "A) " / "B) " etc.).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    out = []
    for item in items:
        img = item['image']
        conv = item['conversations']
        for i in range(0, len(conv), 2):
            if i + 1 >= len(conv):
                break
            q = _clean_question(conv[i]['value'].replace('<image>\n', '').strip())
            a = _ANSWER_LABEL_RX.sub('', conv[i + 1]['value']).strip()
            out.append({'image_path': img, 'question': q, 'answer': a})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Filter / split pipeline
# ---------------------------------------------------------------------------

def filter_by_answer_count(df, min_count=5):
    """Drop rows whose answer occurs less than `min_count` times."""
    keep = df['answer'].value_counts().pipe(lambda s: s[s > min_count]).index
    return df[df['answer'].isin(keep)].copy()


def image_level_split(df, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Random shuffle of unique image_paths → 60/20/20 with no leakage."""
    images = df['image_path'].unique()
    np.random.seed(seed)
    np.random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * (train_ratio + val_ratio))
    image_to_split = {}
    image_to_split.update({im: 'train' for im in images[:n_train]})
    image_to_split.update({im: 'val'   for im in images[n_train:n_val]})
    image_to_split.update({im: 'test'  for im in images[n_val:]})
    out = df.copy()
    out['split'] = out['image_path'].map(image_to_split)
    return out


def restrict_to_common_answers(df):
    """Keep only answers that appear in all three splits."""
    sets = {s: set(df.loc[df['split'] == s, 'answer'].unique()) for s in ('train', 'val', 'test')}
    common = sets['train'] & sets['val'] & sets['test']
    return df[df['answer'].isin(common)].copy()


def rewrite_image_path(df, image_root):
    """Strip the source-dataset prefix and emit `<image_root>/<file>.png`."""
    out = df.copy()
    out['image_path'] = out['image_path'].astype(str).map(
        lambda p: f"{image_root}/{p.split('skincap_')[-1]}"
    )
    return out


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # multimodal_finetune/preprocessing → DermFM-Zero/

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs_dir = (repo_root / 'data' / 'multimodal_finetune_VQA'
                  / 'preprocessing_inputs' / 'SkinCap-VQA')
    default_train_json = str(inputs_dir / 'train_public_MCQA.json')
    default_test_json  = str(inputs_dir / 'test_public_MCQA.json')
    ap.add_argument('--train_json', default=default_train_json,
                    help='Public-MCQA train JSON (DermVQA4 layout).')
    ap.add_argument('--test_json',  default=default_test_json,
                    help='Public-MCQA test JSON (DermVQA4 layout).')
    ap.add_argument('--output_dir',
                    default=str(repo_root / 'data' / 'multimodal_finetune_VQA'
                                / 'SkinCap-VQA' / 'meta'))
    ap.add_argument('--image_root',
                    default='../data/multimodal_finetune_VQA/SkinCap-VQA/images',
                    help='Image-root prefix written into each row\'s image_path. '
                         'Default resolves from multimodal_finetune/ (the CWD of train.py).')
    ap.add_argument('--min_answer_count', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # 1. Parse the conversation JSONs.
    train_df = conversations_to_rows(args.train_json)
    test_df  = conversations_to_rows(args.test_json)
    print(f"train_json raw rows: {len(train_df)}")
    print(f"test_json  raw rows: {len(test_df)}")

    # 2. Keep only the SkinCap rows.
    train_df = train_df[train_df['image_path'].str.contains('skincap', case=False)]
    test_df  = test_df [test_df ['image_path'].str.contains('skincap', case=False)]
    print(f"after skincap-only filter: train={len(train_df)}, test={len(test_df)}")

    vqa = pd.concat([train_df, test_df], ignore_index=True)

    # 3. Drop rare answers.
    vqa = filter_by_answer_count(vqa, args.min_answer_count)
    print(f"after rare-answer filter (>{args.min_answer_count}): {len(vqa)}")

    # 4. Image-level random split.
    vqa = image_level_split(vqa, seed=args.seed)
    print("post-split distribution:")
    print(vqa['split'].value_counts())

    # 5. Restrict to answers present in all three splits.
    vqa = restrict_to_common_answers(vqa)
    print(f"after common-answer filter: {len(vqa)} (answers={vqa['answer'].nunique()})")

    # 6. Rewrite image_path.
    vqa = rewrite_image_path(vqa, args.image_root)

    # 7. Reorder rows train → val → test so `unique()` yields the canonical
    #    answer_id mapping (declaration order in the published CSVs).
    split_order = pd.CategoricalDtype(['train', 'val', 'test'], ordered=True)
    vqa = vqa.assign(split=vqa['split'].astype(split_order)).sort_values(
        'split', kind='stable',
    )
    answer2id = {a: i for i, a in enumerate(vqa['answer'].unique())}
    vqa['answer_id'] = vqa['answer'].map(answer2id)
    vqa = vqa.assign(split=vqa['split'].astype(str))

    # 8. Write per-split CSVs.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ['image_path', 'question', 'answer', 'split', 'answer_id']
    for s in ('train', 'val', 'test'):
        sub = vqa[vqa['split'] == s][cols]
        path = out_dir / f"{s}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(sub)}  unique_images={sub['image_path'].nunique()}")


if __name__ == '__main__':
    main()
