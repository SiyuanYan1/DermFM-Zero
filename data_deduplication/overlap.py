"""
Detect overlap (data leakage) between a pretrain dataset and downstream
evaluation sets using cosine similarity over SSCD embeddings.

Two modes:
  downstream     queries are zero-shot CSVs at
                 {downstream_root}/data/zero-shot-classification/{ds}/meta.csv
                 with an `image_path` column.
  reader_study   queries are flat image folders under
                 {reader_study_root}/{RSx_images}/, embedded in folder mode.

Per-dataset outputs (under --output_dir):
  {ds}_overlaps.csv       overlap pairs with cosine_similarity >= threshold
  {ds}_overlap_viz.png    top-K side-by-side visualisation of matched pairs
  overlap_summary.csv     per-dataset overlap totals + rates
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must come before pyplot import)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DOWNSTREAM_DATASETS = [
    "HAM-official-7-zero-shot",
    "ph2-2-zero-shot",
    "isic2020-2-zero-shot",
    "snu-134-zero-shot",
    "daffodil-5-zero-shot",
    "pad-zero-shot",
    "sd-128-zero-shot",
]

READER_STUDY_DATASETS = ["RS1_images", "RS2_images", "RS3_images"]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def top1_overlaps(query_emb, ref_emb, threshold, batch_size=512):
    """For each query, find the top-1 reference whose cosine sim >= threshold.

    Returns list of (query_idx, ref_idx, similarity).
    """
    matches = []
    for s in range(0, len(query_emb), batch_size):
        e = min(s + batch_size, len(query_emb))
        sims = query_emb[s:e].astype(np.float32) @ ref_emb.astype(np.float32).T
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(best)), best]
        for i, (j, sim) in enumerate(zip(best, best_sim)):
            if sim >= threshold:
                matches.append((s + i, int(j), float(sim)))
    return matches


def visualize(
    matches, query_label_fn, query_path_fn,
    pretrain_csv, name, output_dir, max_pairs, threshold,
    viz_exclude_sources=None,
):
    """Side-by-side query vs. matched pretrain image for the top-K pairs.

    `viz_exclude_sources` drops pretrain rows whose `source` column value is
    in the given list from the visualisation only — the overlap CSV is
    written from the unfiltered `matches` list.
    """
    if not matches:
        print(f"  No overlaps to visualize for {name}")
        return

    if viz_exclude_sources:
        excl = set(viz_exclude_sources)
        before = len(matches)
        matches = [
            m for m in matches
            if str(pretrain_csv.iloc[m[1]].get("source", "")) not in excl
        ]
        dropped = before - len(matches)
        if dropped:
            print(f"  Viz filter dropped {dropped} pair(s) with source in {sorted(excl)}")
        if not matches:
            print(f"  All matches excluded from viz for {name}")
            return

    matches_sorted = sorted(matches, key=lambda x: x[2], reverse=True)[:max_pairs]
    n_show = len(matches_sorted)
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 3.5))
    fig.suptitle(
        f"{name}: Top {n_show} Overlapping Pairs (threshold >= {threshold})",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(n_rows, n_cols * 2, hspace=0.4, wspace=0.15)

    for idx, (q_idx, p_idx, sim) in enumerate(matches_sorted):
        row, col = idx // n_cols, (idx % n_cols) * 2
        q_path = query_path_fn(q_idx)
        p_path = str(pretrain_csv.iloc[p_idx]["image_path"])

        ax_q = fig.add_subplot(gs[row, col])
        try:
            ax_q.imshow(Image.open(q_path).convert("RGB"))
        except Exception:
            ax_q.text(0.5, 0.5, "Load Error", ha="center", va="center",
                      transform=ax_q.transAxes)
        ax_q.set_title(
            query_label_fn(q_idx, sim),
            fontsize=8, color="red", fontweight="bold",
        )
        ax_q.axis("off")

        ax_p = fig.add_subplot(gs[row, col + 1])
        try:
            ax_p.imshow(Image.open(p_path).convert("RGB"))
        except Exception:
            ax_p.text(0.5, 0.5, "Load Error", ha="center", va="center",
                      transform=ax_p.transAxes)
        ax_p.set_title(f"Pretrain\n{os.path.basename(p_path)[:30]}", fontsize=8)
        ax_p.axis("off")

    out_path = os.path.join(output_dir, "overlap_viz.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Visualization saved: {out_path}")


def run_downstream(args, pretrain_emb, pretrain_csv):
    """Pretrain vs. zero-shot CSV datasets under {downstream_root}/data/zero-shot-classification/."""
    summary = []
    for ds in DOWNSTREAM_DATASETS:
        print(f"\n{'=' * 60}\nProcessing: {ds}\n{'=' * 60}")
        emb_path = os.path.join(args.embedding_dir, f"{ds}_embeddings.npy")
        if not os.path.exists(emb_path):
            print(f"  SKIP: embeddings not found at {emb_path}")
            continue
        ds_csv_path = os.path.join(
            args.downstream_root, "data", "zero-shot-classification", ds, "meta.csv"
        )
        ds_csv = pd.read_csv(ds_csv_path)
        ds_emb = l2_normalize(np.load(emb_path))
        total = len(ds_emb)
        print(f"  Downstream: {total} images")

        matches = top1_overlaps(ds_emb, pretrain_emb, threshold=args.threshold)
        n = len(matches)
        rate = 100.0 * n / total if total else 0.0
        print(f"  Overlapping: {n}/{total} ({rate:.2f}%)")

        if matches:
            records = []
            for q_idx, p_idx, sim in matches:
                records.append({
                    "downstream_image": str(ds_csv.iloc[q_idx]["image_path"]),
                    "pretrain_image"  : str(pretrain_csv.iloc[p_idx]["image_path"]),
                    "cosine_similarity": round(sim, 4),
                    "downstream_label" : str(
                        ds_csv.iloc[q_idx].get(
                            "diag", ds_csv.iloc[q_idx].get("label", "")
                        )
                    ),
                    "downstream_index" : q_idx,
                    "pretrain_index"   : p_idx,
                })
            df = pd.DataFrame(records).sort_values("cosine_similarity", ascending=False)
            ds_dir = os.path.join(args.output_dir, ds)
            os.makedirs(ds_dir, exist_ok=True)
            csv_out = os.path.join(ds_dir, "overlaps.csv")
            df.to_csv(csv_out, index=False)
            print(f"  Overlap CSV saved: {csv_out}")

            def q_path_fn(i, ds_csv=ds_csv):
                p = str(ds_csv.iloc[i]["image_path"])
                if args.downstream_root and not os.path.isabs(p):
                    p = os.path.join(args.downstream_root, p)
                return p

            def q_label_fn(_i, sim):
                return f"Downstream\nsim={sim:.3f}"

            visualize(
                matches, q_label_fn, q_path_fn, pretrain_csv, ds,
                ds_dir, args.max_viz_pairs, args.threshold,
                viz_exclude_sources=args.viz_exclude_source,
            )

        summary.append({
            "dataset"           : ds,
            "total_images"      : total,
            "overlapping_images": n,
            "overlap_rate"      : f"{rate:.2f}%",
        })
    return summary, "overlap_summary.csv"


def run_reader_study(args, pretrain_emb, pretrain_csv):
    """Pretrain vs. flat reader-study image folders."""
    summary = []
    for rs in READER_STUDY_DATASETS:
        rs_folder = os.path.join(args.reader_study_root, rs)
        if not os.path.exists(rs_folder):
            print(f"  SKIP: folder not found: {rs_folder}")
            continue
        emb_path  = os.path.join(args.embedding_dir, f"{rs}_embeddings.npy")
        fn_path   = os.path.join(args.embedding_dir, f"{rs}_filenames.npy")
        if not os.path.exists(emb_path):
            print(f"  SKIP: embeddings not found at {emb_path}")
            continue

        print(f"\n{'=' * 60}\nProcessing: {rs}\n{'=' * 60}")
        rs_emb = l2_normalize(np.load(emb_path))
        rs_filenames = np.load(fn_path, allow_pickle=True)
        total = len(rs_emb)
        print(f"  Reader study: {total} images")

        matches = top1_overlaps(rs_emb, pretrain_emb, threshold=args.threshold)
        n = len(matches)
        rate = 100.0 * n / total if total else 0.0
        print(f"  Overlapping: {n}/{total} ({rate:.2f}%)")

        if matches:
            records = []
            for q_idx, p_idx, sim in matches:
                records.append({
                    "reader_study_image": rs_filenames[q_idx],
                    "pretrain_image"    : str(pretrain_csv.iloc[p_idx]["image_path"]),
                    "cosine_similarity" : round(sim, 4),
                    "reader_study_index": q_idx,
                    "pretrain_index"    : p_idx,
                })
            df = pd.DataFrame(records).sort_values("cosine_similarity", ascending=False)
            rs_dir = os.path.join(args.output_dir, rs)
            os.makedirs(rs_dir, exist_ok=True)
            csv_out = os.path.join(rs_dir, "overlaps.csv")
            df.to_csv(csv_out, index=False)
            print(f"  Overlap CSV saved: {csv_out}")

            def q_path_fn(i, rs_filenames=rs_filenames, rs_folder=rs_folder):
                return os.path.join(rs_folder, rs_filenames[i])

            def q_label_fn(i, sim, rs_filenames=rs_filenames):
                return f"Reader Study\nsim={sim:.3f}\n{str(rs_filenames[i])[:25]}"

            visualize(
                matches, q_label_fn, q_path_fn, pretrain_csv, rs,
                rs_dir, args.max_viz_pairs, args.threshold,
                viz_exclude_sources=args.viz_exclude_source,
            )

        summary.append({
            "dataset"           : rs,
            "total_images"      : total,
            "overlapping_images": n,
            "overlap_rate"      : f"{rate:.2f}%",
        })
    return summary, "reader_study_overlap_summary.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["downstream", "reader_study"])
    ap.add_argument("--pretrain_emb", required=True)
    ap.add_argument("--pretrain_csv", required=True)
    ap.add_argument("--embedding_dir", default="embeddings")
    ap.add_argument("--output_dir",    required=True)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--max_viz_pairs", type=int, default=20)
    ap.add_argument(
        "--viz_exclude_source", action="append", default=[],
        help=("Repeatable: drop pairs from viz PNGs when the pretrain "
              "`source` column equals this value. Overlap CSVs are NOT "
              "affected. Example: --viz_exclude_source <source_name>"),
    )
    # downstream-only
    ap.add_argument("--downstream_root", default="./downstream_root")
    # reader-study-only
    ap.add_argument("--reader_study_root", default="reader-study-meta")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading pretrain embeddings...")
    pretrain_emb = l2_normalize(np.load(args.pretrain_emb))
    print(f"  Pretrain: {pretrain_emb.shape[0]} images, {pretrain_emb.shape[1]}d")

    print("Loading pretrain CSV...")
    pretrain_csv = pd.read_csv(args.pretrain_csv, low_memory=False)
    print(f"  Pretrain CSV rows: {len(pretrain_csv)}")

    if args.mode == "downstream":
        summary, _ = run_downstream(args, pretrain_emb, pretrain_csv)
        title = "OVERLAP SUMMARY"
    else:
        summary, _ = run_reader_study(args, pretrain_emb, pretrain_csv)
        title = "READER STUDY OVERLAP SUMMARY"
    summary_name = "overlap_summary.csv"

    print(f"\n{'=' * 60}\n{title} (threshold={args.threshold})\n{'=' * 60}")
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    out = os.path.join(args.output_dir, summary_name)
    df.to_csv(out, index=False)
    print(f"\nSummary saved: {out}")


if __name__ == "__main__":
    main()
