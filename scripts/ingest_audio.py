"""
ingest_audio.py — Phase 2: Convert, trim, QC, hash all raw audio files.

Converts .m4a files to 16 kHz mono 16-bit PCM WAV, runs quality checks
(clipping, SNR, duration), computes SHA-256 hashes, and builds manifest.csv.

Usage:
    python scripts/ingest_audio.py --help
    python scripts/ingest_audio.py --raw-dir data/raw --out-dir data/processed --manifest data/manifests/manifest.csv
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio file to 16 kHz mono 16-bit PCM WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",      # 16 kHz sample rate
        "-ac", "1",          # mono
        "-sample_fmt", "s16",  # 16-bit signed int
        "-f", "wav",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] ffmpeg failed for {input_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------

def read_wav_samples(wav_path: str) -> tuple:
    """Read a WAV file and return (samples_float, sample_rate)."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype = np.int16 if sample_width == 2 else np.int32
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_channels > 1:
        samples = samples[::n_channels]  # take first channel
    # Normalize to [-1, 1]
    max_val = float(2 ** (8 * sample_width - 1))
    samples_float = samples / max_val
    return samples_float, sample_rate


def check_clipping(samples: np.ndarray, threshold: float = 0.99) -> tuple:
    """Check for clipping. Returns (has_clipping, peak_dBFS)."""
    peak = np.max(np.abs(samples))
    peak_dbfs = 20 * np.log10(peak + 1e-12)
    has_clipping = peak >= threshold
    return has_clipping, round(peak_dbfs, 2)


def estimate_snr(samples: np.ndarray, sample_rate: int, frame_ms: int = 25) -> float:
    """Estimate SNR by comparing top-energy frames (speech) to bottom-energy frames (noise)."""
    frame_len = int(sample_rate * frame_ms / 1000)
    n_frames = len(samples) // frame_len
    if n_frames < 10:
        return 0.0

    frames = samples[:n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)

    # Sort energies
    sorted_e = np.sort(energies)
    # Bottom 10% as noise estimate, top 50% as signal estimate
    noise_count = max(1, n_frames // 10)
    signal_count = max(1, n_frames // 2)

    noise_power = np.mean(sorted_e[:noise_count]) + 1e-12
    signal_power = np.mean(sorted_e[-signal_count:]) + 1e-12

    snr_db = 10 * np.log10(signal_power / noise_power)
    return round(snr_db, 2)


def get_duration(wav_path: str) -> float:
    """Get duration in seconds."""
    with wave.open(wav_path, "rb") as wf:
        return round(wf.getnframes() / wf.getframerate(), 3)


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

FLAW_MAP = {
    "normal1": ("none", 0),
    "normal2": ("none", 0),
    "audio2": ("none", 0),       # alternate ideal take name
    "rushed": ("rushed_pace", 3),
    "deadpause": ("dead_pause", 3),
    "flaw": ("flat_pitch", 3),   # generic "flaw" label → treat as flat_pitch
    "flatpitch": ("flat_pitch", 3),
    "mumbeled": ("mumbled_clarity", 3),
    "mumbled": ("mumbled_clarity", 3),
}


def parse_filename(filename: str) -> dict:
    """Parse a filename like 'T1-rushed.m4a' into metadata."""
    stem = Path(filename).stem.strip()
    # Normalize: remove leading/trailing spaces, replace spaces around dash
    stem = stem.replace(" ", "")

    parts = stem.split("-", 1)
    if len(parts) != 2:
        return None

    text_id = parts[0].upper()   # e.g. "T1"
    flaw_label = parts[1].lower()  # e.g. "rushed"

    if flaw_label not in FLAW_MAP:
        print(f"  [WARN] Unknown flaw label '{flaw_label}' in {filename}")
        return None

    flaw_type, severity = FLAW_MAP[flaw_label]

    # Determine take_id
    if "normal1" in flaw_label or "audio2" in flaw_label:
        take_id = "ideal_1"
    elif "normal2" in flaw_label:
        take_id = "ideal_2"
    else:
        take_id = flaw_label

    return {
        "text_id": text_id,
        "flaw_type": flaw_type,
        "severity": severity,
        "take_id": take_id,
        "flaw_label": flaw_label,
    }


# ---------------------------------------------------------------------------
# Transcript mapping
# ---------------------------------------------------------------------------

def load_transcripts(json_path: str) -> dict:
    """Load transcript text from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for key, val in data.items():
        full_text = " ".join(val["sentences"])
        flaw_sentence = val["sentences"][val["flaw_sentence_index"]]
        result[key.upper()] = {
            "transcript": full_text,
            "flaw_sentence": flaw_sentence,
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest and QC raw audio files for SpeechMirror.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw .m4a files")
    parser.add_argument("--out-dir", default="data/processed/wav", help="Output directory for WAV files")
    parser.add_argument("--manifest", default="data/manifests/manifest.csv", help="Output manifest CSV path")
    parser.add_argument("--transcripts", default="data/processed/transcripts.json", help="Transcripts JSON path")
    parser.add_argument("--speaker-id", default="SPK01", help="Speaker ID for all files")
    parser.add_argument("--mic", default="phone", help="Microphone used")
    parser.add_argument("--room", default="indoor_quiet", help="Room condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (unused currently)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    # Load transcripts
    transcripts = load_transcripts(args.transcripts)

    # Collect all audio files
    audio_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".m4a", ".mp4", ".wav", ".mp3"))
    ])

    print(f"Found {len(audio_files)} audio files in {raw_dir}")
    print("=" * 70)

    rows = []
    qc_pass = 0
    qc_fail = 0
    sample_counter = 0

    for filename in audio_files:
        input_path = str(raw_dir / filename)
        meta = parse_filename(filename)
        if meta is None:
            print(f"  [SKIP] Could not parse filename: {filename}")
            qc_fail += 1
            continue

        sample_counter += 1
        sample_id = f"S{sample_counter:03d}"
        wav_filename = f"{meta['text_id']}_{meta['take_id']}.wav"
        wav_path = str(out_dir / wav_filename)

        print(f"\n[{sample_id}] {filename} -> {wav_filename}")

        # 1. Convert to WAV
        if not convert_to_wav(input_path, wav_path):
            print(f"  [FAIL] Conversion failed")
            qc_fail += 1
            continue

        # 2. QC checks
        try:
            samples, sr = read_wav_samples(wav_path)
        except Exception as e:
            print(f"  [FAIL] Cannot read WAV: {e}")
            qc_fail += 1
            continue

        has_clipping, peak_dbfs = check_clipping(samples)
        snr = estimate_snr(samples, sr)
        duration = get_duration(wav_path)
        sha = compute_sha256(wav_path)

        status = "PASS"
        issues = []
        if has_clipping:
            issues.append(f"clipping (peak={peak_dbfs} dBFS)")
            status = "WARN"
        if meta["flaw_type"] == "none" and snr < 25:
            issues.append(f"low SNR ({snr} dB < 25 dB)")
            status = "WARN"
        if duration < 30 or duration > 120:
            issues.append(f"unusual duration ({duration}s)")
            status = "WARN"

        if status == "PASS":
            qc_pass += 1
        else:
            qc_fail += 1

        print(f"  Duration: {duration}s | Peak: {peak_dbfs} dBFS | SNR: {snr} dB | Status: {status}")
        if issues:
            print(f"  Issues: {', '.join(issues)}")

        # 3. Build manifest row
        t_info = transcripts.get(meta["text_id"], {})

        # Determine ideal audio reference (always points to ideal_1 of same transcript)
        ideal_wav = f"{meta['text_id']}_ideal_1.wav"

        row = {
            "sample_id": sample_id,
            "text_id": meta["text_id"],
            "speaker_id": args.speaker_id,
            "take_id": meta["take_id"],
            "transcript": t_info.get("transcript", ""),
            "ideal_audio": ideal_wav,
            "participant_audio": wav_filename,
            "flaw_type": meta["flaw_type"],
            "severity": meta["severity"],
            "flaw_sentence": t_info.get("flaw_sentence", ""),
            "source_type": "real",
            "source_license": "CC-BY-4.0",
            "consent": "yes",
            "mic": args.mic,
            "room": args.room,
            "noise_condition": "clean",
            "sha256": sha,
            "duration_s": duration,
            "split": "",  # will be assigned later by split_dataset.py
            "peak_dbfs": peak_dbfs,
            "snr_db": snr,
            "qc_status": status,
        }
        rows.append(row)

    # Write manifest
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{'=' * 70}")
        print(f"Manifest written to {args.manifest}")
    
    print(f"\nSummary: {qc_pass} passed, {qc_fail} warnings/failures out of {len(audio_files)} files.")
    print("Done.")


if __name__ == "__main__":
    main()
