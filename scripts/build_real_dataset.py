#!/usr/bin/env python3
"""
Build manifest.csv from processed recordings.
"""
import argparse
import json
import random
import sys
import hashlib
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
    parser = argparse.ArgumentParser(description="Build dataset manifest.")
    parser.add_argument("--audio-dir", type=str, required=True, help="Processed audio directory")
    parser.add_argument("--transcripts", type=str, required=True, help="Transcripts JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output manifest.csv path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dummy manifest creation
    columns = [
        "sample_id", "text_id", "speaker_id", "take_id", "transcript", 
        "ideal_audio", "participant_audio", "flaw_type", "severity", 
        "flaw_sentence", "source_type", "source_license", "consent", 
        "mic", "room", "noise_condition", "sha256", "duration_s", "split"
    ]
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
    write_run_manifest("build_real_dataset", vars(args), {"manifest_path": str(out_path)})

if __name__ == "__main__":
    main()
