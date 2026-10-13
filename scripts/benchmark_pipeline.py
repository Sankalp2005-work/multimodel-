"""
benchmark_pipeline.py -- Phase 6: Benchmark & Stress Tests
Evaluates SpeechMirrorPipeline on the sealed TEST dataset without label peeking.
Includes ideal clips to measure false-positive rate, reports metrics separately
for all clips, real-only clips, and synthetic-only clips, computes Wilson 95% CI,
and saves results to results/test_report.json.
"""

import csv
import json
import os
import time
from datetime import datetime
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.speechmirror.pipeline import SpeechMirrorPipeline

try:
    from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
    import pandas as pd
except ImportError:
    import sys
    print("Installing required metric libraries...")
    os.system(f"{sys.executable} -m pip install scikit-learn pandas")
    from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
    import pandas as pd


def compute_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Compute Intersection over Union for two 1D intervals."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = max(0.0, union_end - union_start)

    if union == 0:
        return 0.0
    return float(intersection / union)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate Wilson 95% score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / den
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def evaluate_subset(results: list[dict], name: str = "Subset") -> dict:
    """Evaluate accuracy, classification report, confusion matrix, and regression metrics for a subset."""
    total = len(results)
    if total == 0:
        return {
            "count": 0,
            "accuracy": None,
            "wilson_ci_95": None,
            "false_positive_rate": None
        }

    y_true = [r["true_type"] for r in results]
    y_pred = [r["pred_type"] for r in results]

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total
    ci_low, ci_high = wilson_ci(correct, total)

    all_labels = sorted(list(set(y_true + y_pred)))
    cls_report = classification_report(y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Ideal clips FP evaluation
    ideal_clips = [r for r in results if r["true_sev"] == 0]
    ideal_total = len(ideal_clips)
    ideal_fp = sum(1 for r in ideal_clips if r["has_flaw_detected"])
    fp_rate = (ideal_fp / ideal_total) if ideal_total > 0 else 0.0

    # Flawed clips regression metrics
    flawed_clips = [r for r in results if r["true_sev"] > 0]
    ious = [r["iou"] for r in flawed_clips if r["iou"] is not None]
    ts_errors = [r["err_ms"] for r in flawed_clips if r["err_ms"] is not None]
    sev_true = [r["true_sev"] for r in flawed_clips]
    sev_pred = [r["pred_sev"] for r in flawed_clips]

    mean_iou = float(np.mean(ious)) if ious else 0.0
    median_ts_err = float(np.median(ts_errors)) if ts_errors else 0.0
    sev_mae = float(mean_absolute_error(sev_true, sev_pred)) if flawed_clips else 0.0

    print(f"\n--- {name} (N = {total}) ---")
    print(f"Accuracy: {accuracy * 100:.1f}% ({correct}/{total})")
    print(f"Wilson 95% CI: [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]")
    if ideal_total > 0:
        print(f"False Positive Rate on Ideal: {fp_rate * 100:.1f}% ({ideal_fp}/{ideal_total})")
    if flawed_clips:
        print(f"Severity MAE: {sev_mae:.2f}")
        print(f"Mean Temporal IoU: {mean_iou:.2f}")
        print(f"Median Timestamp Error: {median_ts_err:.1f} ms")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    df_cm = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print("Confusion Matrix:")
    print(df_cm)

    return {
        "count": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "wilson_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "ideal_samples": ideal_total,
        "ideal_false_positives": ideal_fp,
        "false_positive_rate": round(fp_rate, 4),
        "flawed_samples": len(flawed_clips),
        "mean_iou": round(mean_iou, 4),
        "median_timestamp_error_ms": round(median_ts_err, 2),
        "severity_mae": round(sev_mae, 4),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "cm_labels": all_labels
    }


def main():
    manifest_path = Path("data/manifests/manifest.csv")
    labels_path = Path("data/manifests/labels.csv")
    wav_dir = Path("data/processed/wav")
    config_in = Path("configs/pipeline.json")
    report_out = Path("results/test_report.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    with open(config_in, "r", encoding="utf-8") as f:
        config = json.load(f)

    labels_dict = {row["sample_id"]: row for row in labels}

    # Isolate sealed test set
    test_samples = [row for row in manifest if row["split"] == "test"]
    print(f"Benchmarking on {len(test_samples)} sealed TEST samples (including ideal clips)...", flush=True)

    pipeline = SpeechMirrorPipeline()

    # Pre-cache ideal audio features
    ideal_cache = {}
    for row in test_samples:
        ideal_wav = row["ideal_audio"]
        if ideal_wav not in ideal_cache:
            wav_path = str(wav_dir / ideal_wav)
            if not os.path.exists(wav_path):
                continue
            print(f"Caching ideal reference features for {ideal_wav}...", flush=True)
            ideal_cache[ideal_wav] = pipeline.extract_features(wav_path)

    all_results = []

    print("\nRunning detectors on TEST set (no answer peeking)...", flush=True)
    for idx, row in enumerate(test_samples):
        sample_id = row["sample_id"]
        true_type = row["flaw_type"]
        true_sev = int(row["severity"])
        source_type = row["source_type"]
        wav_path = str(wav_dir / row["participant_audio"])
        if not os.path.exists(wav_path):
            continue

        gt = labels_dict.get(sample_id)
        gt_start = float(gt["start_time_s"]) if gt else 0.0
        gt_end = float(gt["end_time_s"]) if gt else 0.0

        t0 = time.time()
        part_features = pipeline.extract_features(wav_path)
        ideal_features = ideal_cache[row["ideal_audio"]]

        # Contrastive flaw detection
        flaws = pipeline.detect_flaws(part_features, ideal_features, config)

        # STRICT UNBIASED SELECTION:
        # Pick ONE prediction with highest confidence, with NO knowledge of ground truth!
        if flaws:
            best_flaw = sorted(flaws, key=lambda x: x.get("confidence", 0), reverse=True)[0]
            pred_type = best_flaw["type"]
            pred_sev = int(best_flaw["severity"])
            pred_start = float(best_flaw["start"])
            pred_end = float(best_flaw["end"])
            pred_conf = float(best_flaw["confidence"])
            has_flaw_detected = True
        else:
            pred_type = "none"
            pred_sev = 0
            pred_start = 0.0
            pred_end = 0.0
            pred_conf = 0.0
            has_flaw_detected = False

        # Compute IoU and timestamp error for flawed clips
        iou = None
        err_ms = None
        sev_err = None
        if true_sev > 0 and gt is not None:
            iou = compute_iou(gt_start, gt_end, pred_start, pred_end)
            if pred_type != "none":
                gt_center = (gt_start + gt_end) / 2.0
                pred_center = (pred_start + pred_end) / 2.0
                err_ms = abs(gt_center - pred_center) * 1000.0
            sev_err = abs(true_sev - pred_sev)

        sample_res = {
            "sample_id": sample_id,
            "source_type": source_type,
            "true_type": true_type,
            "pred_type": pred_type,
            "true_sev": true_sev,
            "pred_sev": pred_sev,
            "pred_conf": pred_conf,
            "has_flaw_detected": has_flaw_detected,
            "iou": iou,
            "err_ms": err_ms,
            "sev_err": sev_err,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "pred_start": pred_start,
            "pred_end": pred_end
        }
        all_results.append(sample_res)

        elapsed = time.time() - t0
        print(f"[{idx + 1:02d}/{len(test_samples):02d}] {sample_id} ({source_type[:4]}) | True: {true_type:<15} | Pred: {pred_type:<15} | Conf: {pred_conf:.2f} | Time: {elapsed:.2f}s")

    # Evaluate Subsets
    print("\n================== BENCHMARK RESULTS ==================")
    overall_metrics = evaluate_subset(all_results, "ALL TEST CLIPS (COMBINED)")

    real_results = [r for r in all_results if r["source_type"] == "real"]
    real_metrics = evaluate_subset(real_results, "REAL-ONLY TEST CLIPS")

    synth_results = [r for r in all_results if r["source_type"] != "real"]
    synth_metrics = evaluate_subset(synth_results, "SYNTHETIC-ONLY TEST CLIPS")

    # Save comprehensive report
    os.makedirs(report_out.parent, exist_ok=True)
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_test_samples": len(all_results),
        "real_count": len(real_results),
        "synthetic_count": len(synth_results),
        "overall": overall_metrics,
        "real_only": real_metrics,
        "synthetic_only": synth_metrics,
        "clip_predictions": all_results
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"\nComprehensive test report successfully saved to {report_out}")


if __name__ == "__main__":
    main()
