"""
split_dataset.py — Assign train/val/test splits to manifest.csv.

Splits by transcript group (not by clip) to prevent leakage.
T1, T2, T3 -> train
T4         -> val
T5, T6     -> test

Usage:
    python scripts/split_dataset.py --manifest data/manifests/manifest.csv
"""

import argparse
import csv
from pathlib import Path


SPLIT_MAP = {
    "T1": "train",
    "T2": "train",
    "T3": "train",
    "T4": "val",
    "T5": "test",
    "T6": "test",
}


def main():
    parser = argparse.ArgumentParser(description="Assign splits to manifest.csv")
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    # Read
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Assign splits
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        text_id = row["text_id"].upper()
        split = SPLIT_MAP.get(text_id, "train")
        row["split"] = split
        counts[split] += 1

    # Write back
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Splits assigned: {counts}")
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
