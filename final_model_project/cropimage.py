#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def progress_iter(iterable, total: int, desc: str):
    return iterable


MIN_BRIGHT = 40
MAX_BRIGHT = 220
MIN_SHARP = 15.0
BORDER_W = 8
MAX_DARK_BORDER = 0.20


def merge_split_indices(base_dir: Path, meta_dir: Path, splits: list[str]) -> Path:
    all_dfs: list[pd.DataFrame] = []
    for split in splits:
        split_csv = base_dir / split / "dataset_index.csv"
        if split_csv.exists():
            all_dfs.append(pd.read_csv(split_csv))

    if not all_dfs:
        raise FileNotFoundError("No split dataset_index.csv files were found.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    out_csv = meta_dir / "dataset_index_all.csv"
    df_all.to_csv(out_csv, index=False)

    n_real = int((df_all["label"] == "real").sum())
    n_fake = int((df_all["label"] == "fake").sum())
    print(f"Saved: {out_csv}")
    print(f"Real={n_real:,} / Fake={n_fake:,} / Total={len(df_all):,}")

    if max(n_real, n_fake) > 0 and abs(n_real - n_fake) > 0.05 * max(n_real, n_fake):
        m = min(n_real, n_fake)
        df_bal = (
            pd.concat(
                [
                    df_all[df_all.label == "real"].sample(m, random_state=42),
                    df_all[df_all.label == "fake"].sample(m, random_state=42),
                ]
            )
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        balanced_csv = meta_dir / "dataset_index_balanced.csv"
        df_bal.to_csv(balanced_csv, index=False)
        print(f"Balanced saved: {balanced_csv} (each class={m:,})")

    return out_csv


def quality(path: Path) -> str:
    try:
        image = cv2.imread(str(path))
        if image is None:
            return "invalid"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < MIN_BRIGHT or brightness > MAX_BRIGHT:
            return "bad_brightness"

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < MIN_SHARP:
            return "blurry"

        border_w = BORDER_W
        mask = np.zeros_like(gray, dtype=bool)
        mask[:border_w, :] = True
        mask[-border_w:, :] = True
        mask[:, :border_w] = True
        mask[:, -border_w:] = True
        if np.mean(gray[mask] < 8) > MAX_DARK_BORDER:
            return "bad_border"

        return "good"
    except Exception:
        return "error"


def run_quality_check(input_csv: Path, meta_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(input_csv)
    df["status"] = "unchecked"

    for i, row in progress_iter(df.iterrows(), total=len(df), desc="quality"):
        image_path = Path(str(row["path"]))
        df.at[i, "status"] = (
            "missing" if not image_path.exists() else quality(image_path)
        )

    checked_csv = meta_dir / "dataset_quality_checked.csv"
    good_csv = meta_dir / "dataset_filtered_good.csv"

    df.to_csv(checked_csv, index=False)
    good = df[df.status == "good"].copy()
    good.to_csv(good_csv, index=False)
    ratio = (len(good) / max(1, len(df))) * 100
    print(f"Quality checked: {checked_csv}")
    print(f"Filtered good: {good_csv} ({len(good):,}/{len(df):,}, {ratio:.2f}%)")

    return checked_csv, good_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split indices and run quality checks."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val_internal", "val_external"],
        help="Split directories expected under --base-dir",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional prebuilt CSV. If omitted, merged CSV is created from splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.meta_dir.mkdir(parents=True, exist_ok=True)

    input_csv = args.input_csv
    if input_csv is None:
        input_csv = merge_split_indices(args.base_dir, args.meta_dir, args.splits)

    run_quality_check(input_csv, args.meta_dir)


if __name__ == "__main__":
    main()
