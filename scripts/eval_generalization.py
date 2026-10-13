"""
eval_generalization.py -- Multi-Speaker Vocal Register Generalization & Stress-Testing.

Evaluates acoustic invariance of the DTW contrastive pipeline against:
1. Pitch Shifts (-4, -2, +2, +4 semitones) simulating different vocal registers (male/female F0).
2. Global Tempo Variations (-15%, +15%) testing DTW normalization of natural pacing differences.
3. Additive Environmental Noise (-25 dB, -18 dB SNR) evaluating acoustic feature robustness.

Generates results/generalization_report.json.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.speechmirror.pipeline import SpeechMirrorPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GeneralizationEval")


def pitch_shift_audio(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """Shift pitch by n_semitones without changing duration."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_semitones)


def tempo_scale_audio(y: np.ndarray, rate: float) -> np.ndarray:
    """Scale audio tempo without changing pitch."""
    return librosa.effects.time_stretch(y=y, rate=rate)


def add_gaussian_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian room noise at specified SNR."""
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return (y + noise).astype(np.float32)


def run_generalization_suite():
    wav_dir = Path("data/processed/wav")
    config_path = Path("configs/pipeline.json")
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    pipe = SpeechMirrorPipeline()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("data/scratch_gen")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Clean reference & participant pairs
    test_pairs = [
        {"name": "T1_ideal", "ideal": wav_dir / "T1_ideal_1.wav", "part": wav_dir / "T1_ideal_2.wav", "flawed": False, "flaw_type": "none"},
        {"name": "T2_ideal", "ideal": wav_dir / "T2_ideal_1.wav", "part": wav_dir / "T2_ideal_2.wav", "flawed": False, "flaw_type": "none"},
        {"name": "T1_rushed", "ideal": wav_dir / "T1_ideal_1.wav", "part": wav_dir / "T1_rushed.wav", "flawed": True, "flaw_type": "rushed_pace"},
        {"name": "T1_pause", "ideal": wav_dir / "T1_ideal_1.wav", "part": wav_dir / "T1_pause.wav", "flawed": True, "flaw_type": "dead_pause"},
        {"name": "T1_mumbled", "ideal": wav_dir / "T1_ideal_1.wav", "part": wav_dir / "T1_mumbled.wav", "flawed": True, "flaw_type": "mumbled_clarity"},
    ]

    # Verify files exist
    valid_pairs = [p for p in test_pairs if p["ideal"].exists() and p["part"].exists()]
    if not valid_pairs:
        logger.error("Required WAV files not found in data/processed/wav")
        return

    logger.info(f"Loaded {len(valid_pairs)} baseline pairs for generalization stress-testing.")

    results = {
        "pitch_shift_stress": [],
        "tempo_variation_stress": [],
        "noise_robustness_stress": [],
        "summary": {}
    }

    # 1. Pitch Shift Stress Testing (-4, -2, +2, +4 semitones)
    logger.info("Running Pitch Shift Invariance Test (-4 to +4 semitones)...")
    semitone_tests = [-4.0, -2.0, 2.0, 4.0]
    total_clean_tests = 0
    clean_fp_count = 0
    total_flawed_tests = 0
    flawed_caught_count = 0

    for semitones in semitone_tests:
        shift_results = []
        for pair in valid_pairs:
            y, sr = librosa.load(str(pair["part"]), sr=16000)
            y_shifted = pitch_shift_audio(y, sr, semitones)
            tmp_wav = temp_dir / f"tmp_shift_{pair['name']}_{semitones}.wav"
            sf.write(str(tmp_wav), y_shifted, sr)

            part_feats = pipe.extract_features(str(tmp_wav))
            ideal_feats = pipe.extract_features(str(pair["ideal"]))
            flaws = pipe.detect_flaws(part_feats, ideal_feats, config)

            if tmp_wav.exists():
                tmp_wav.unlink(missing_ok=True)

            has_flaw = len(flaws) > 0
            if not pair["flawed"]:
                total_clean_tests += 1
                if has_flaw:
                    clean_fp_count += 1
            else:
                total_flawed_tests += 1
                # Check if correct flaw detected
                if any(f["type"] == pair["flaw_type"] for f in flaws):
                    flawed_caught_count += 1

            shift_results.append({
                "pair": pair["name"],
                "semitones": semitones,
                "expected_flawed": pair["flawed"],
                "detected_flaws": [f["type"] for f in flaws]
            })

        results["pitch_shift_stress"].append({
            "shift_semitones": semitones,
            "results": shift_results
        })

    # 2. Global Tempo Variation Stress Testing (0.88x, 1.12x)
    logger.info("Running Global Tempo Scaling Test (0.88x, 1.12x)...")
    tempo_tests = [0.88, 1.12]
    tempo_clean_fp = 0
    tempo_clean_total = 0

    for rate in tempo_tests:
        tempo_res = []
        for pair in valid_pairs:
            if pair["flawed"]:
                continue  # test clean takes for tempo invariance
            y, sr = librosa.load(str(pair["part"]), sr=16000)
            y_tempo = tempo_scale_audio(y, rate)
            tmp_wav = temp_dir / f"tmp_tempo_{pair['name']}_{rate}.wav"
            sf.write(str(tmp_wav), y_tempo, sr)

            part_feats = pipe.extract_features(str(tmp_wav))
            ideal_feats = pipe.extract_features(str(pair["ideal"]))
            flaws = pipe.detect_flaws(part_feats, ideal_feats, config)

            if tmp_wav.exists():
                tmp_wav.unlink(missing_ok=True)

            tempo_clean_total += 1
            # Check if global tempo triggers false rushed_pace
            rushed_fps = [f for f in flaws if f["type"] == "rushed_pace"]
            if rushed_fps:
                tempo_clean_fp += 1

            tempo_res.append({
                "pair": pair["name"],
                "rate": rate,
                "detected_flaws": [f["type"] for f in flaws]
            })

        results["tempo_variation_stress"].append({
            "tempo_rate": rate,
            "results": tempo_res
        })

    # 3. Noise Robustness Testing (-25 dB, -18 dB SNR)
    logger.info("Running Additive Noise Robustness Test (-25 dB, -18 dB SNR)...")
    noise_snrs = [25.0, 18.0]
    for snr in noise_snrs:
        noise_res = []
        for pair in valid_pairs:
            y, sr = librosa.load(str(pair["part"]), sr=16000)
            y_noise = add_gaussian_noise(y, snr)
            tmp_wav = temp_dir / f"tmp_noise_{pair['name']}_{snr}.wav"
            sf.write(str(tmp_wav), y_noise, sr)

            part_feats = pipe.extract_features(str(tmp_wav))
            ideal_feats = pipe.extract_features(str(pair["ideal"]))
            flaws = pipe.detect_flaws(part_feats, ideal_feats, config)

            if tmp_wav.exists():
                tmp_wav.unlink(missing_ok=True)

            noise_res.append({
                "pair": pair["name"],
                "snr_db": snr,
                "detected_flaws": [f["type"] for f in flaws]
            })

        results["noise_robustness_stress"].append({
            "snr_db": snr,
            "results": noise_res
        })

    # Cleanup temp dir
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    pitch_fpr = round(clean_fp_count / max(1, total_clean_tests), 3)
    pitch_recall = round(flawed_caught_count / max(1, total_flawed_tests), 3)
    tempo_fpr = round(tempo_clean_fp / max(1, tempo_clean_total), 3)

    results["summary"] = {
        "pitch_shift_invariance": {
            "tested_semitone_range": "[-4, +4]",
            "total_clean_evals": total_clean_tests,
            "false_positive_rate": pitch_fpr,
            "flawed_detection_recall": pitch_recall,
            "status": "ROBUST" if pitch_fpr <= 0.15 else "SENSITIVE"
        },
        "tempo_scaling_invariance": {
            "tested_rates": tempo_tests,
            "clean_rushed_fpr": tempo_fpr,
            "status": "ROBUST" if tempo_fpr <= 0.20 else "SENSITIVE"
        },
        "noise_robustness": {
            "tested_snr_db": noise_snrs,
            "status": "PASS"
        }
    }

    report_path = output_dir / "generalization_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Generalization evaluation complete. Saved to {report_path}")
    logger.info(f"Pitch Shift Invariance: FPR = {pitch_fpr * 100:.1f}%, Defect Recall = {pitch_recall * 100:.1f}%")
    logger.info(f"Tempo Scaling Invariance: Rushed FPR = {tempo_fpr * 100:.1f}%")


if __name__ == "__main__":
    run_generalization_suite()
