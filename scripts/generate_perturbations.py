#!/usr/bin/env python3
"""
Generate synthetic perturbations from ideal recordings.
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
    parser = argparse.ArgumentParser(description="Generate synthetic perturbations.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/perturb.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for perturbed clips")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dummy perturbation loop
    results = {"generated_clips": 0}
        
    write_run_manifest("generate_perturbations", vars(args), results)

if __name__ == "__main__":
    main()
