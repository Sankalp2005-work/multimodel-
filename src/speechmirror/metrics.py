"""
Evaluation metrics module for SpeechMirror.
Computes all required metrics for benchmarking from scratch.
"""
import math
import numpy as np
from typing import Any

def temporal_iou(pred_start: float, pred_end: float, true_start: float, true_end: float) -> float:
    """Compute temporal Intersection over Union for two regions."""
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    intersection = max(0.0, intersection_end - intersection_start)
    
    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = max(0.0, union_end - union_start)
    
    if union == 0.0:
        return 0.0
    return intersection / union

def match_regions(pred_regions: list[dict], true_regions: list[dict], iou_threshold: float = 0.3) -> tuple[list, list, list]:
    """Match predicted regions to ground truth regions based on IoU and type.
    Returns (TP, FP, FN) as lists of indices or region pairs.
    Assumes regions are dicts with 'start', 'end', and 'type'.
    """
    tp = []
    fp = []
    
    matched_true_idx = set()
    
    for p_idx, pred in enumerate(pred_regions):
        best_iou = 0.0
        best_t_idx = -1
        
        for t_idx, true in enumerate(true_regions):
            if t_idx in matched_true_idx:
                continue
            if pred.get('type') != true.get('type'):
                continue
                
            iou = temporal_iou(pred['start'], pred['end'], true['start'], true['end'])
            if iou > best_iou:
                best_iou = iou
                best_t_idx = t_idx
                
        if best_iou >= iou_threshold:
            tp.append((pred, true_regions[best_t_idx], best_iou))
            matched_true_idx.add(best_t_idx)
        else:
            fp.append(pred)
            
    fn = [true for t_idx, true in enumerate(true_regions) if t_idx not in matched_true_idx]
    
    return tp, fp, fn

def compute_detection_metrics(pred_regions: list[dict], true_regions: list[dict], iou_threshold: float = 0.3) -> dict:
    """Compute precision, recall, and F1 for region detection."""
    tp, fp, fn = match_regions(pred_regions, true_regions, iou_threshold)
    
    tp_count = len(tp)
    fp_count = len(fp)
    fn_count = len(fn)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count
    }

def clip_level_accuracy(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute clip-level flaw-type accuracy."""
    correct = 0
    total = len(predictions)
    
    for pred, true in zip(predictions, ground_truth):
        p_type = pred.get('dominant_type', 'none')
        t_type = true.get('dominant_type', 'none')
        if p_type == t_type:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def severity_mae(matched_pairs: list[tuple]) -> float:
    """Compute Mean Absolute Error for severity over True Positive regions."""
    if not matched_pairs:
        return 0.0
        
    total_error = 0.0
    for pred, true, _ in matched_pairs:
        p_sev = pred.get('severity', 0)
        t_sev = true.get('severity', 0)
        total_error += abs(p_sev - t_sev)
        
    return total_error / len(matched_pairs)

def fp_rate_on_ideal(predictions: list[dict], ideal_sample_ids: list[str]) -> float:
    """Compute fraction of ideal clips with >=1 flagged region."""
    if not ideal_sample_ids:
        return 0.0
        
    fp_count = 0
    for pred in predictions:
        if pred.get('clip_id') in ideal_sample_ids:
            if len(pred.get('regions', [])) > 0:
                fp_count += 1
                
    return fp_count / len(ideal_sample_ids)

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Compute Wilson 95% Confidence Interval. Returns (center, lower, upper)."""
    if n == 0:
        return 0.0, 0.0, 0.0
        
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denominator
    
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    
    return center, lower, upper

def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa for annotator agreement."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
        
    total = len(labels_a)
    classes = set(labels_a + labels_b)
    
    # Observed agreement
    observed_matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = observed_matches / total
    
    # Expected agreement
    pe = 0.0
    for c in classes:
        p_a = sum(1 for x in labels_a if x == c) / total
        p_b = sum(1 for x in labels_b if x == c) / total
        pe += p_a * p_b
        
    if pe == 1.0:
        return 1.0
        
    return (po - pe) / (1 - pe)

def boundary_agreement(annotations_a: list[dict], annotations_b: list[dict], tolerance_s: float = 0.15) -> float:
    """Compute share of annotator pairs within tolerance_s."""
    if not annotations_a:
        return 1.0 if not annotations_b else 0.0
        
    matches = 0
    total = len(annotations_a) * 2 # start and end boundaries
    
    for a in annotations_a:
        a_start = a.get('start', 0.0)
        a_end = a.get('end', 0.0)
        
        start_matched = any(abs(a_start - b.get('start', 0.0)) <= tolerance_s for b in annotations_b)
        end_matched = any(abs(a_end - b.get('end', 0.0)) <= tolerance_s for b in annotations_b)
        
        if start_matched:
            matches += 1
        if end_matched:
            matches += 1
            
    return matches / total if total > 0 else 0.0

def bootstrap_iou_ci(ious: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap CI for mean IoU."""
    if not ious:
        return 0.0, 0.0, 0.0
        
    ious_arr = np.array(ious)
    means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(ious_arr, size=len(ious_arr), replace=True)
        means.append(np.mean(sample))
        
    means = np.sort(means)
    lower_percentile = (1.0 - ci) / 2.0
    upper_percentile = 1.0 - lower_percentile
    
    lower_idx = int(n_bootstrap * lower_percentile)
    upper_idx = int(n_bootstrap * upper_percentile)
    
    center = float(np.mean(ious_arr))
    lower = float(means[lower_idx])
    upper = float(means[min(upper_idx, n_bootstrap - 1)])
    
    return center, lower, upper

def confusion_matrix(predictions: list[str], truths: list[str], labels: list[str] = None) -> dict:
    """Compute confusion matrix as a nested dict."""
    if labels is None:
        labels = list(set(predictions + truths))
        
    matrix = {t: {p: 0 for p in labels} for t in labels}
    
    for p, t in zip(predictions, truths):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
            
    return matrix

def full_benchmark(predictions: list[dict], ground_truth: list[dict], ideal_ids: list[str] = None) -> dict:
    """Run full evaluation suite."""
    all_pred_regions = []
    all_true_regions = []
    
    pred_dominant_types = []
    true_dominant_types = []
    
    clip_ids = []
    
    for p, t in zip(predictions, ground_truth):
        p_regions = p.get('regions', [])
        t_regions = t.get('regions', [])
        all_pred_regions.extend(p_regions)
        all_true_regions.extend(t_regions)
        
        pred_dominant_types.append(p.get('dominant_type', 'none'))
        true_dominant_types.append(t.get('dominant_type', 'none'))
        clip_ids.append(p.get('clip_id'))
        
    metrics_03 = compute_detection_metrics(all_pred_regions, all_true_regions, iou_threshold=0.3)
    metrics_05 = compute_detection_metrics(all_pred_regions, all_true_regions, iou_threshold=0.5)
    
    tp_03, fp_03, fn_03 = match_regions(all_pred_regions, all_true_regions, iou_threshold=0.3)
    
    mae_sev = severity_mae(tp_03)
    
    clip_acc = clip_level_accuracy(predictions, ground_truth)
    
    fp_ideal = 0.0
    if ideal_ids:
        fp_ideal = fp_rate_on_ideal(predictions, ideal_ids)
        
    cm = confusion_matrix(pred_dominant_types, true_dominant_types)
    
    # Calculate timestamp errors
    start_errors = []
    end_errors = []
    ious = []
    for p, t, iou in tp_03:
        start_errors.append(abs(p.get('start', 0.0) - t.get('start', 0.0)))
        end_errors.append(abs(p.get('end', 0.0) - t.get('end', 0.0)))
        ious.append(iou)
        
    ts_errors = {}
    if start_errors:
        ts_errors = {
            "start_median": float(np.median(start_errors)),
            "start_p90": float(np.percentile(start_errors, 90)),
            "end_median": float(np.median(end_errors)),
            "end_p90": float(np.percentile(end_errors, 90))
        }
        
    iou_ci = bootstrap_iou_ci(ious) if ious else (0.0, 0.0, 0.0)

    return {
        "detection_iou_0.3": metrics_03,
        "detection_iou_0.5": metrics_05,
        "severity_mae": mae_sev,
        "clip_accuracy": clip_acc,
        "fp_rate_ideal": fp_ideal,
        "confusion_matrix": cm,
        "timestamp_errors": ts_errors,
        "iou_ci": {
            "mean": iou_ci[0],
            "lower_95": iou_ci[1],
            "upper_95": iou_ci[2]
        }
    }
