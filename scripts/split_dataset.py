#!/usr/bin/env python3
"""
Train/val/test split.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def set_seed(seed: int):
    random.seed(seed)

def write_run_manifest(name: str, args: dict, results: dict):
    runs_dir = Path("results/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / f"run_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"script": name, "args": args, "results": results}, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    results = {"train_count": 0, "val_count": 0, "test_count": 0}
        
    write_run_manifest("split_dataset", vars(args), results)

if __name__ == "__main__":
    main()
