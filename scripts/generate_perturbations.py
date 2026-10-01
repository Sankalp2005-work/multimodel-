"""
generate_perturbations.py -- Phase 3: Generate synthetic perturbation clips.

For each ideal take of each transcript, generates 4 flaw types x 4 severity
levels = 16 synthetic clips. Uses Whisper to locate the target sentence,
then applies deterministic perturbations from src/speechmirror/perturb.py.

Usage:
    python scripts/generate_perturbations.py --config configs/perturb.json --seed 7
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speechmirror.perturb import apply_perturbation


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def read_wav(path: str):
    """Read WAV file, return (samples_float64, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return samples, sr


def write_wav(path: str, samples: np.ndarray, sr: int):
    """Write float64 samples to 16-bit PCM WAV."""
    # Clip to [-1, 1] and convert
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Target sentence detection via Whisper
# ---------------------------------------------------------------------------

def find_target_sentence_boundaries(
    wav_path: str,
    target_sentence: str,
    sr: int = 16000,
) -> tuple:
    """Use Whisper to find word-level timestamps, then locate the target sentence.

    Returns (start_sample, end_sample) of the target sentence.
    Falls back to heuristic (sentence 4 of 6 by duration ratio) if matching fails.
    """
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(wav_path, word_timestamps=True)

    # Collect all word-level timestamps
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip().lower(),
                "start": w["start"],
                "end": w["end"],
            })

    if not words:
        print("    [WARN] No word timestamps from Whisper, using heuristic")
        return _heuristic_boundaries(wav_path, sr)

    # Try to match the target sentence by finding a contiguous subsequence
    target_words = target_sentence.lower().split()

    best_start_idx = None
    best_score = 0

    for i in range(len(words)):
        # Try matching starting from word i
        score = 0
        for j, tw in enumerate(target_words):
            if i + j >= len(words):
                break
            # Fuzzy match: check if the target word is contained in the whisper word
            ww = words[i + j]["word"]
            if tw in ww or ww in tw or tw[:4] == ww[:4]:
                score += 1

        match_ratio = score / max(len(target_words), 1)
        if match_ratio > best_score and match_ratio >= 0.3:
            best_score = match_ratio
            best_start_idx = i

    if best_start_idx is not None:
        # Determine end index
        end_idx = min(best_start_idx + len(target_words) - 1, len(words) - 1)
        start_sec = words[best_start_idx]["start"]
        end_sec = words[end_idx]["end"]

        # Add small buffer
        start_sec = max(0, start_sec - 0.05)
        end_sec = end_sec + 0.05

        print(f"    Target sentence found: {start_sec:.2f}s - {end_sec:.2f}s (match={best_score:.0%})")
        return int(start_sec * sr), int(end_sec * sr)
    else:
        print("    [WARN] Could not match target sentence, using heuristic")
        return _heuristic_boundaries(wav_path, sr)


def _heuristic_boundaries(wav_path: str, sr: int) -> tuple:
    """Fallback: assume target is sentence 4 of 6, roughly at 40%-60% of duration."""
    with wave.open(wav_path, "rb") as wf:
        total = wf.getnframes()
    # Sentence 4 of 6 starts at ~50% and ends at ~67%
    start = int(total * 0.45)
    end = int(total * 0.62)
    print(f"    Heuristic boundaries: {start/sr:.2f}s - {end/sr:.2f}s")
    return start, end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic perturbation clips.")
    parser.add_argument("--config", default="configs/perturb.json")
    parser.add_argument("--wav-dir", default="data/processed/wav")
    parser.add_argument("--out-dir", default="data/processed/wav")
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument("--transcripts", default="data/processed/transcripts.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speaker-id", default="SPK01")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Load transcripts
    with open(args.transcripts, "r", encoding="utf-8") as f:
        transcripts_data = json.load(f)

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    severities = config.get("generate_severities", [1, 2, 3, 4])
    flaw_types = list(config["perturbations"].keys())
    source_takes = config.get("source_takes", ["ideal_1"])

    # Load existing manifest
    manifest_path = Path(args.manifest)
    existing_rows = []
    fieldnames = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_rows = list(reader)

    # Determine next sample_id counter
    max_id = 0
    for row in existing_rows:
        try:
            num = int(row["sample_id"].replace("S", ""))
            max_id = max(max_id, num)
        except (ValueError, KeyError):
            pass
    sample_counter = max_id

    new_rows = []
    generated_count = 0

    # Split map for synthetic clips: match the transcript's split
    SPLIT_MAP = {"T1": "train", "T2": "train", "T3": "train",
                 "T4": "val", "T5": "test", "T6": "test"}

    print(f"Generating perturbations from {len(source_takes)} source take(s)")
    print(f"Flaw types: {flaw_types}")
    print(f"Severities: {severities}")
    print("=" * 70)

    for t_key, t_data in transcripts_data.items():
        text_id = t_key.upper()
        target_sentence = t_data["sentences"][t_data["flaw_sentence_index"]]
        full_transcript = " ".join(t_data["sentences"])

        for take in source_takes:
            ideal_wav = wav_dir / f"{text_id}_{take}.wav"
            if not ideal_wav.exists():
                print(f"\n[SKIP] {ideal_wav} not found")
                continue

            print(f"\n--- {text_id} / {take} ---")
            audio, sr = read_wav(str(ideal_wav))

            # Find target sentence boundaries
            start_sample, end_sample = find_target_sentence_boundaries(
                str(ideal_wav), target_sentence, sr
            )

            for flaw_type in flaw_types:
                for sev in severities:
                    sample_counter += 1
                    sample_id = f"S{sample_counter:03d}"
                    out_filename = f"{text_id}_synth_{flaw_type}_sev{sev}.wav"
                    out_path = out_dir / out_filename

                    print(f"  [{sample_id}] {flaw_type} sev={sev} -> {out_filename}")

                    try:
                        perturbed, new_start, new_end = apply_perturbation(
                            audio=audio.copy(),
                            sr=sr,
                            flaw_type=flaw_type,
                            severity=sev,
                            target_start_sample=start_sample,
                            target_end_sample=end_sample,
                            config=config,
                            crossfade_ms=config.get("crossfade_ms", 30),
                            seed=args.seed + sev,  # vary seed slightly by severity
                        )

                        write_wav(str(out_path), perturbed, sr)
                        sha = compute_sha256(str(out_path))
                        duration = len(perturbed) / sr

                        row = {
                            "sample_id": sample_id,
                            "text_id": text_id,
                            "speaker_id": args.speaker_id,
                            "take_id": f"synth_{flaw_type}_sev{sev}",
                            "transcript": full_transcript,
                            "ideal_audio": f"{text_id}_{take}.wav",
                            "participant_audio": out_filename,
                            "flaw_type": flaw_type,
                            "severity": sev,
                            "flaw_sentence": target_sentence,
                            "source_type": "synthetic_perturbation",
                            "source_license": "CC-BY-4.0",
                            "consent": "yes",
                            "mic": "phone",
                            "room": "indoor_quiet",
                            "noise_condition": "clean",
                            "sha256": sha,
                            "duration_s": round(duration, 3),
                            "split": SPLIT_MAP.get(text_id, "train"),
                            "peak_dbfs": "",
                            "snr_db": "",
                            "qc_status": "SYNTH",
                        }
                        new_rows.append(row)
                        generated_count += 1

                    except Exception as e:
                        print(f"    [ERROR] {e}")

    # Append to manifest
    all_rows = existing_rows + new_rows
    if fieldnames is None and new_rows:
        fieldnames = list(new_rows[0].keys())

    if fieldnames:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n{'=' * 70}")
    print(f"Generated {generated_count} synthetic clips.")
    print(f"Manifest updated: {manifest_path} (total: {len(all_rows)} rows)")

    # Write run manifest
    run_manifest = {
        "script": "generate_perturbations.py",
        "config": args.config,
        "seed": args.seed,
        "generated_count": generated_count,
        "total_manifest_rows": len(all_rows),
    }
    run_path = Path("results/runs")
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "perturbation_run.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
