"""
Compute SSCD copy-detection embeddings.

Two modes:
  csv     read image paths from a CSV column
  folder  walk a flat folder of images

Outputs (under --output_dir):
  {name}_embeddings.npy   float32 (N, 512), one row per successfully loaded image
  {name}_indices.npy      int64  (N,), maps embedding row -> original dataset row
  {name}_filenames.npy    str    (M,), folder mode only — sorted file list
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SSCD_INPUT_SIZE = 320


def build_transform():
    return transforms.Compose([
        transforms.Resize([SSCD_INPUT_SIZE, SSCD_INPUT_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CSVDataset(Dataset):
    def __init__(self, csv_path, image_column, image_root):
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.image_column = image_column
        self.image_root = image_root
        print(f"Loaded CSV with {len(self.df)} rows")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = str(self.df.iloc[idx][self.image_column])
        if self.image_root:
            p = os.path.join(self.image_root, p.lstrip("/"))
        return idx, p


class FolderDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        )
        print(f"Found {len(self.files)} images in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return idx, os.path.join(self.folder, self.files[idx])


def make_collate(transform):
    def collate(batch):
        imgs, idxs = [], []
        for i, path in batch:
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(transform(img))
                idxs.append(i)
            except Exception:
                # silently skip broken images; counted at the end
                pass
        if not imgs:
            return None, []
        return torch.stack(imgs), idxs
    return collate


def compute(dataset, model, device, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_collate(build_transform()),
        pin_memory=True,
    )

    chunks, indices = [], []
    t0 = time.time()
    with torch.no_grad():
        for tensor, idxs in tqdm(loader, desc="embedding"):
            if tensor is None:
                continue
            tensor = tensor.to(device, non_blocking=True)
            emb = model(tensor)
            chunks.append(emb.cpu().numpy())
            indices.extend(idxs)

    embeddings = np.vstack(chunks) if chunks else np.zeros((0, 512), dtype=np.float32)
    elapsed = time.time() - t0
    print(
        f"Processed {len(indices)}/{len(dataset)} in {elapsed:.1f}s "
        f"({len(indices) / max(elapsed, 1e-6):.1f} img/s)"
    )
    return embeddings, np.array(indices, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["csv", "folder"])
    ap.add_argument("--csv")
    ap.add_argument("--image_column", default="image_path")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--folder")
    ap.add_argument("--model_path", default="models/sscd_disc_mixup.torchscript.pt")
    ap.add_argument("--output_dir", default="embeddings")
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    if args.mode == "csv" and not args.csv:
        ap.error("--csv is required in csv mode")
    if args.mode == "folder" and not args.folder:
        ap.error("--folder is required in folder mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading SSCD model: {args.model_path}")
    model = torch.jit.load(args.model_path).to(device).eval()

    if args.mode == "csv":
        ds = CSVDataset(args.csv, args.image_column, args.image_root)
    else:
        ds = FolderDataset(args.folder)

    embeddings, indices = compute(ds, model, device, args.batch_size, args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, args.output_name)
    np.save(base + "_embeddings.npy", embeddings)
    np.save(base + "_indices.npy", indices)
    if args.mode == "folder":
        np.save(base + "_filenames.npy", np.array(ds.files))
        print(f"Saved: {base}_filenames.npy")
    print(f"Saved: {base}_embeddings.npy  shape={embeddings.shape}")
    print(f"Saved: {base}_indices.npy")


if __name__ == "__main__":
    main()
