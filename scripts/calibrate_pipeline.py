"""
calibrate_pipeline.py -- Phase 5: Pipeline Calibration
Fits thresholds from ideal_1 vs ideal_2 acoustic variation on training transcripts
using DTW alignment, evaluates on training flawed clips, and freezes config to configs/pipeline.json.
"""

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.speechmirror.pipeline import SpeechMirrorPipeline


def main():
    manifest_path = Path("data/manifests/manifest.csv")
    wav_dir = Path("data/processed/wav")
    config_out = Path("configs/pipeline.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    # Isolate training set
    train_samples = [row for row in manifest if row["split"] == "train"]
    print(f"Calibrating on {len(train_samples)} training samples...", flush=True)

    pipeline = SpeechMirrorPipeline()

    # Step 1: Find transcript groups that have both ideal_1 and ideal_2 in train set
    groups = {}
    for row in train_samples:
        t_id = row["text_id"]
        take = row["take_id"]
        if take in ("ideal_1", "ideal_2"):
            if t_id not in groups:
                groups[t_id] = {}
            groups[t_id][take] = row

    # Cache ideal takes features
    ideal_cache = {}
    ideal_pairs = []
    for t_id, takes in sorted(groups.items()):
        if "ideal_1" in takes and "ideal_2" in takes:
            wav_1 = str(wav_dir / takes["ideal_1"]["participant_audio"])
            wav_2 = str(wav_dir / takes["ideal_2"]["participant_audio"])
            print(f"Extracting features for transcript group {t_id} (ideal_1 & ideal_2)...", flush=True)
            feat_1 = pipeline.extract_features(wav_1)
            feat_2 = pipeline.extract_features(wav_2)
            ideal_cache[t_id] = feat_1  # use ideal_1 as primary reference
            ideal_pairs.append((t_id, feat_1, feat_2))
        elif "ideal_1" in takes:
            wav_1 = str(wav_dir / takes["ideal_1"]["participant_audio"])
            ideal_cache[t_id] = pipeline.extract_features(wav_1)

    print(f"\nFound {len(ideal_pairs)} transcript groups with dual ideal takes in train set: {[p[0] for p in ideal_pairs]}")

    # Compute per-feature variation statistics from ideal-vs-ideal pairs
    slope_diffs = []
    pause_diffs = []
    pitch_var_reductions = []
    centroid_reductions = []
    energy_drops = []

    win_size = 3.5
    win_step = 0.5

    for t_id, f1, f2 in ideal_pairs:
        wp = pipeline.align_dtw(f1, f2)
        p_idx = wp[:, 0]
        i_idx = wp[:, 1]
        t1 = f1["frame_times"]
        t2 = f2["frame_times"]

        # a. DTW slope variation in sliding windows
        dur_total = min(f1["duration"], f2["duration"])
        cur_t = 0.0
        while cur_t + win_size <= dur_total:
            w_start = cur_t
            w_end = cur_t + win_size
            cur_t += win_step

            mask = np.where((t2[i_idx] >= w_start) & (t2[i_idx] <= w_end))[0]
            if len(mask) < 15:
                continue

            p_s = p_idx[mask[0]]
            p_e = p_idx[mask[-1]]
            if p_e <= p_s:
                continue

            dur_1_raw = max(0.1, float(t1[p_e] - t1[p_s]))
            dur_2_raw = max(0.1, w_end - w_start)
            sil_1 = sum(max(0.0, min(p["end"], float(t1[p_e])) - max(p["start"], float(t1[p_s]))) for p in f1.get("pauses", []))
            sil_2 = sum(max(0.0, min(p["end"], w_end) - max(p["start"], w_start)) for p in f2.get("pauses", []))
            dur_1_act = max(0.3, dur_1_raw - sil_1)
            dur_2_act = max(0.3, dur_2_raw - sil_2)

            # Slope / speed ratio variations (both directions) on active phonation
            ratio_forward = dur_2_act / dur_1_act
            ratio_backward = dur_1_act / dur_2_act
            slope_diffs.append(ratio_forward)
            slope_diffs.append(ratio_backward)

            # c. F0 variance ratio variation (with octave outlier rejection)
            f0_1 = f1["features"]["f0"][p_s:p_e + 1]
            f0_2 = f2["features"]["f0"][i_idx[mask[0]]:i_idx[mask[-1]] + 1]
            v1 = f0_1[f0_1 > 0]
            v2 = f0_2[f0_2 > 0]
            if len(v1) >= 15 and len(v2) >= 15:
                ref1 = np.median(v1)
                ref2 = np.median(v2)
                st1 = 12.0 * np.log2((v1 + 1e-9) / ref1)
                st2 = 12.0 * np.log2((v2 + 1e-9) / ref2)
                st1_clean = st1[np.abs(st1) <= 10.0]
                st2_clean = st2[np.abs(st2) <= 10.0]
                if len(st1_clean) >= 12 and len(st2_clean) >= 12:
                    var1 = float(np.var(st1_clean))
                    var2 = float(np.var(st2_clean))
                    if max(var1, var2) > 0.6:
                        drop = 1.0 - (min(var1, var2) / max(var1, var2))
                        pitch_var_reductions.append(drop)

            # d. Energy drop & centroid ratio variation
            c1 = f1["features"]["centroid"][p_s:p_e + 1]
            c2 = f2["features"]["centroid"][i_idx[mask[0]]:i_idx[mask[-1]] + 1]
            e1 = f1["features"]["rms_db"][p_s:p_e + 1]
            e2 = f2["features"]["rms_db"][i_idx[mask[0]]:i_idx[mask[-1]] + 1]

            if len(c1) > 0 and len(c2) > 0:
                mean_c1 = float(np.mean(c1))
                mean_c2 = float(np.mean(c2))
                c_red = 1.0 - (min(mean_c1, mean_c2) / max(1.0, max(mean_c1, mean_c2)))
                centroid_reductions.append(c_red)

                mean_e1 = float(np.mean(e1))
                mean_e2 = float(np.mean(e2))
                energy_drops.append(abs(mean_e1 - mean_e2))

        # b. Pause duration variation between ideal takes
        pauses_1 = f1.get("pauses", [])
        pauses_2 = f2.get("pauses", [])
        for p1 in pauses_1:
            dur1 = p1["duration"]
            t_mid = (p1["start"] + p1["end"]) / 2.0
            idx_1 = np.searchsorted(t1, t_mid)
            m_2 = min(len(i_idx) - 1, np.searchsorted(p_idx, idx_1))
            t_mid_2 = float(t2[i_idx[m_2]])

            # Find matching pause in f2 within 1.5s
            matched_dur2 = 0.0
            for p2 in pauses_2:
                if abs(((p2["start"] + p2["end"]) / 2.0) - t_mid_2) < 1.5:
                    matched_dur2 = p2["duration"]
                    break
            pause_diffs.append(abs(dur1 - matched_dur2))

    # Step 2: Fit thresholds as mean + k * SD (k = 1.5)
    k = 1.5
    print(f"\n--- FITTING THRESHOLDS (k = {k}) ---")

    # a. Pace slope threshold
    mean_slope = float(np.mean(slope_diffs)) if slope_diffs else 1.0
    sd_slope = float(np.std(slope_diffs)) if slope_diffs else 0.08
    pace_slope_thresh = float(max(1.15, mean_slope + k * sd_slope))
    print(f"Pace slope: mean={mean_slope:.3f}, SD={sd_slope:.3f} -> pace_slope_thresh={pace_slope_thresh:.3f}")

    # b. Pause duration threshold
    mean_pause = float(np.mean(pause_diffs)) if pause_diffs else 0.15
    sd_pause = float(np.std(pause_diffs)) if pause_diffs else 0.12
    pause_silence_thresh_s = float(max(0.40, min(0.55, mean_pause + k * sd_pause)))
    print(f"Pause diff: mean={mean_pause:.3f}s, SD={sd_pause:.3f}s -> pause_silence_thresh_s={pause_silence_thresh_s:.3f}s")

    # c. Pitch variance ratio threshold
    mean_pitch_red = float(np.mean(pitch_var_reductions)) if pitch_var_reductions else 0.12
    sd_pitch_red = float(np.std(pitch_var_reductions)) if pitch_var_reductions else 0.10
    pitch_var_ratio_thresh = float(max(0.35, 1.0 - (mean_pitch_red + k * sd_pitch_red)))
    print(f"Pitch var red: mean={mean_pitch_red:.3f}, SD={sd_pitch_red:.3f} -> pitch_var_ratio_thresh={pitch_var_ratio_thresh:.3f}")

    # d. Clarity thresholds
    mean_cent_red = float(np.mean(centroid_reductions)) if centroid_reductions else 0.08
    sd_cent_red = float(np.std(centroid_reductions)) if centroid_reductions else 0.06
    centroid_ratio_thresh = float(max(0.65, 1.0 - (mean_cent_red + k * sd_cent_red)))

    mean_edrop = float(np.mean(energy_drops)) if energy_drops else 1.0
    sd_edrop = float(np.std(energy_drops)) if energy_drops else 0.7
    energy_drop_thresh = float(max(2.0, mean_edrop + k * sd_edrop))
    print(f"Centroid red: mean={mean_cent_red:.3f}, SD={sd_cent_red:.3f} -> centroid_ratio_thresh={centroid_ratio_thresh:.3f}")
    print(f"Energy drop: mean={mean_edrop:.2f}dB, SD={sd_edrop:.2f}dB -> energy_drop_thresh={energy_drop_thresh:.2f}dB")

    config = {
        "pace_slope_thresh": round(pace_slope_thresh, 3),
        "pause_silence_thresh_s": round(pause_silence_thresh_s, 3),
        "pitch_var_ratio_thresh": round(pitch_var_ratio_thresh, 3),
        "centroid_ratio_thresh": round(centroid_ratio_thresh, 3),
        "energy_drop_thresh": round(energy_drop_thresh, 2),
        "k_sd": k,
        "calibration_method": "ideal_vs_ideal_variation",
        "weights": {
            "rushed_pace": 1.0,
            "dead_pause": 1.0,
            "flat_pitch": 1.0,
            "mumbled_clarity": 1.0
        }
    }

    # Step 3: Evaluate fitted thresholds on training flawed clips
    flawed_train = [r for r in train_samples if int(r["severity"]) > 0]
    print(f"\nEvaluating fitted thresholds on {len(flawed_train)} training flawed clips...", flush=True)

    results = []
    for idx, row in enumerate(flawed_train):
        sample_id = row["sample_id"]
        true_type = row["flaw_type"]
        text_id = row["text_id"]

        wav_path = str(wav_dir / row["participant_audio"])
        if not os.path.exists(wav_path):
            continue
        part_features = pipeline.extract_features(wav_path)
        ideal_features = ideal_cache.get(text_id)
        if ideal_features is None:
            continue

        flaws = pipeline.detect_flaws(part_features, ideal_features, config)
        detected_types = [f["type"] for f in flaws]

        if flaws:
            best_flaw = sorted(flaws, key=lambda x: x.get("confidence", 0), reverse=True)[0]
            pred_type = best_flaw["type"]
        else:
            pred_type = "none"

        hit = (pred_type == true_type) or (true_type in detected_types)
        results.append({
            "sample_id": sample_id,
            "true_type": true_type,
            "pred_type": pred_type,
            "detected_types": detected_types,
            "hit": hit
        })
        print(f"  [{idx + 1}/{len(flawed_train)}] {sample_id} | True: {true_type} | Pred: {pred_type} | Hit: {hit}")

    train_acc = sum(1 for r in results if r["hit"]) / len(results) if results else 0.0
    strict_acc = sum(1 for r in results if r["pred_type"] == r["true_type"]) / len(results) if results else 0.0
    print(f"\n=== Training Set Accuracy (Any Hit): {train_acc * 100:.1f}% ===")
    print(f"=== Training Set Accuracy (Strict Top-1): {strict_acc * 100:.1f}% ===")

    # Step 4: Save frozen configuration
    os.makedirs(config_out.parent, exist_ok=True)
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nFitted calibration configuration frozen to {config_out}")


if __name__ == "__main__":
    main()
