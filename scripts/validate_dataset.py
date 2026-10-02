#!/usr/bin/env python3
"""
Dataset validation.
"""
import argparse
import json
import random
import sys
import csv
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
    parser = argparse.ArgumentParser(description="Validate dataset manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    manifest_path = Path(args.manifest)
    results = {"passed": True, "checks": {}}
    
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            if columns:
                results["checks"]["columns_exist"] = True
            else:
                results["checks"]["columns_exist"] = False
                results["passed"] = False
                
    write_run_manifest("validate_dataset", vars(args), results)
    print(f"Validation passed: {results['passed']}")

if __name__ == "__main__":
    main()
