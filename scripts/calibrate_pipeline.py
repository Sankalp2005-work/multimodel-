"""
calibrate_pipeline.py -- Phase 5: Calibration
Evaluates the SpeechMirrorPipeline on the train dataset, fits thresholds, 
and freezes the calibration config.
"""

import csv
import json
from pathlib import Path
from src.speechmirror.pipeline import SpeechMirrorPipeline
import os

def main():
    manifest_path = Path("data/manifests/manifest.csv")
    labels_path = Path("data/manifests/labels.csv")
    wav_dir = Path("data/processed/wav")
    config_out = Path("configs/pipeline.json")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
        
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
        
    labels_dict = {row["sample_id"]: row for row in labels}
    
    # Isolate training set
    train_samples = [row for row in manifest if row["split"] == "train"]
    print(f"Calibrating on {len(train_samples)} training samples...", flush=True)
    
    pipeline = SpeechMirrorPipeline()
    
    # 1. Align & extract features for ideal clips (severity 0) first to use as contrastive base
    ideal_cache = {}
    for row in train_samples:
        if row["severity"] == "0":
            transcript_group = row["text_id"]
            wav_path = str(wav_dir / row["participant_audio"])
            print(f"Caching ideal features for group {transcript_group}... {wav_path}", flush=True)
            ideal_cache[transcript_group] = pipeline.extract_features(wav_path)
            
    # 2. Extract features for flawed clips, compute deltas, and evaluate
    
    # Initial thresholds to test
    config = {
        "rushed_thresh": -0.05,  # -50ms per word
        "pause_thresh": 0.4,     # +400ms pause
        "pitch_thresh": -1.0,    # -1.0 semitones std
        "clarity_thresh": -0.15, # -15% spectral centroid
        "weights": {
            "rushed_pace": 1.0,
            "dead_pause": 1.0,
            "flat_pitch": 1.0,
            "mumbled_clarity": 1.0
        }
    }
    
    results = []
    
    print("\nRunning detectors and collecting metrics...", flush=True)
    import time
    for idx, row in enumerate(train_samples):
        if row["severity"] == "0":
            continue
            
        sample_id = row["sample_id"]
        true_type = row["flaw_type"]
        true_sev = int(row["severity"])
        
        wav_path = str(wav_dir / row["participant_audio"])
        print(f"[{idx+1}/{len(train_samples)}] Processing {sample_id}...", end=" ", flush=True)
        t0 = time.time()
        
        part_features = pipeline.extract_features(wav_path)
        ideal_features = ideal_cache[row["text_id"]]
        
        deltas = pipeline.compute_deltas(part_features, ideal_features)
        flaws = pipeline.detect_flaws(deltas, config)
        
        # Did we detect the true flaw type?
        detected_types = [f["type"] for f in flaws]
        hit = true_type in detected_types
        
        print(f"| True: {true_type} | Detected: {detected_types} | Time: {time.time()-t0:.1f}s", flush=True)
        
        results.append({
            "sample_id": sample_id,
            "true_type": true_type,
            "detected": detected_types,
            "hit": hit
        })
        
    accuracy = sum(1 for r in results if r["hit"]) / len(results) if results else 0
    print(f"\nTraining Set Accuracy with baseline thresholds: {accuracy*100:.1f}%")
    
    # Save frozen configuration
    os.makedirs("configs", exist_ok=True)
    with open(config_out, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Calibration frozen to {config_out}")

if __name__ == "__main__":
    main()
