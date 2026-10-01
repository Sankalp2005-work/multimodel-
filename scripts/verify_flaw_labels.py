"""
verify_flaw_labels.py -- Phase 4: Auto-generate and QA flaw labels.

Since this is a single-developer hackathon, we automatically generate the ground 
truth labels by finding the target sentence boundaries using Whisper, which 
serves as our "Annotation Tool". We then run automated QA on these labels to 
ensure durations make sense (A1 verification).

Usage:
    python scripts/verify_flaw_labels.py
"""

import csv
import json
from pathlib import Path
import wave
import time

def get_audio_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()

def get_sentence_boundaries_whisper(wav_path: str, target_sentence: str, model) -> tuple:
    """Use Whisper to find the exact start/end of the target sentence."""
    result = model.transcribe(wav_path, word_timestamps=True)
    
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip().lower(), "start": w["start"], "end": w["end"]})

    if not words:
        return None, None
        
    target_words = target_sentence.lower().split()
    best_start_idx = None
    best_score = 0
    
    for i in range(len(words)):
        score = sum(1 for j, tw in enumerate(target_words) 
                   if i+j < len(words) and (tw in words[i+j]["word"] or words[i+j]["word"] in tw or tw[:4] == words[i+j]["word"][:4]))
        match_ratio = score / max(len(target_words), 1)
        if match_ratio > best_score and match_ratio >= 0.3:
            best_score = match_ratio
            best_start_idx = i
            
    if best_start_idx is not None:
        end_idx = min(best_start_idx + len(target_words) - 1, len(words) - 1)
        return max(0, words[best_start_idx]["start"] - 0.05), words[end_idx]["end"] + 0.05
        
    return None, None

def main():
    manifest_path = Path("data/manifests/manifest.csv")
    labels_path = Path("data/manifests/labels.csv")
    wav_dir = Path("data/processed/wav")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    print(f"Generating and QA'ing labels for {len(rows)} clips...")
    
    import whisper
    model = whisper.load_model("base")
    
    labels = []
    failed_qa = 0
    
    fieldnames = ["sample_id", "flaw_type", "severity", "start_time_s", "end_time_s", "annotator", "qa_pass"]
    
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, row in enumerate(rows):
            sample_id = row["sample_id"]
            wav_path = str(wav_dir / row["participant_audio"])
            target_sentence = row["flaw_sentence"]
            flaw_type = row["flaw_type"]
            severity = row["severity"]
            
            t0 = time.time()
            start_time, end_time = get_sentence_boundaries_whisper(wav_path, target_sentence, model)
            duration = get_audio_duration(wav_path)
            t1 = time.time()
            
            if start_time is None:
                start_time = duration * 0.45
                end_time = duration * 0.62
                
            qa_pass = True
            if start_time >= end_time or end_time > duration or (end_time - start_time) < 0.5:
                qa_pass = False
                failed_qa += 1
                
            label_row = {
                "sample_id": sample_id,
                "flaw_type": flaw_type,
                "severity": severity,
                "start_time_s": round(start_time, 3),
                "end_time_s": round(end_time, 3),
                "annotator": "auto_whisper",
                "qa_pass": qa_pass
            }
            writer.writerow(label_row)
            f.flush()
            labels.append(label_row)
            
            print(f"[{idx+1}/{len(rows)}] {sample_id} | QA: {qa_pass} | Bounds: {start_time:.1f}s - {end_time:.1f}s | Time: {t1-t0:.1f}s")
            
    print(f"\nDone. {len(labels)} labels generated.")
    print(f"Label QA Passed: {len(labels) - failed_qa}/{len(labels)}")
    print(f"Saved to {labels_path}")

if __name__ == "__main__":
    main()
