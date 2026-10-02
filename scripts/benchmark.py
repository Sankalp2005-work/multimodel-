#!/usr/bin/env python3
"""
Run detector on a split and compute metrics.
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
    parser = argparse.ArgumentParser(description="Run benchmark and compute metrics.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--split", type=str, choices=["val", "test"], required=True, help="Dataset split")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {"metrics": {"accuracy": 0.0, "f1": 0.0}}
        
    write_run_manifest("benchmark", vars(args), results)

if __name__ == "__main__":
    main()
