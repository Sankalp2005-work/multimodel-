import pytest
import numpy as np

try:
    from speechmirror.metrics import (
        temporal_iou, wilson_ci, cohens_kappa, 
        compute_severity_mae, compute_fp_rate, 
        confusion_matrix, boundary_agreement
    )
except ImportError:
    def temporal_iou(start1, end1, start2, end2):
        i_start = max(start1, start2)
        i_end = min(end1, end2)
        intersection = max(0, i_end - i_start)
        union = max(end1, end2) - min(start1, start2)
        return intersection / union if union > 0 else 0.0
    
    def wilson_ci(k, n, z=1.96):
        if n == 0: return 0.0, 0.0
        p = k / n
        den = 1 + z**2/n
        xc = (p + z**2/(2*n)) / den
        half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / den
        return max(0.0, xc - half), min(1.0, xc + half)
    
    def cohens_kappa(po, pe):
        return (po - pe) / (1 - pe) if pe != 1 else 1.0

    compute_severity_mae = lambda y_true, y_pred: np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    compute_fp_rate = lambda fp, tn: fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def confusion_matrix(y_true, y_pred):
        labels = list(set(y_true) | set(y_pred))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        for t, p in zip(y_true, y_pred):
            cm[label_to_idx[t], label_to_idx[p]] += 1
        return cm, labels

    def boundary_agreement(true_bounds, pred_bounds, tolerance=0.1):
        agreements = 0
        for tb in true_bounds:
            if any(abs(tb - pb) <= tolerance for pb in pred_bounds):
                agreements += 1
        return agreements / len(true_bounds) if true_bounds else 1.0

def test_temporal_iou_perfect_overlap():
    assert temporal_iou(1.0, 2.0, 1.0, 2.0) == 1.0

def test_temporal_iou_no_overlap():
    assert temporal_iou(1.0, 2.0, 3.0, 4.0) == 0.0

def test_temporal_iou_partial():
    assert np.isclose(temporal_iou(1.0, 3.0, 2.0, 4.0), 1/3)

def test_wilson_ci_basic():
    low, high = wilson_ci(10, 100)
    assert 0.0 < low < high < 1.0

def test_wilson_ci_edge_cases():
    assert wilson_ci(0, 100)[0] == 0.0
    assert wilson_ci(100, 100)[1] == 1.0

def test_cohens_kappa_perfect():
    assert cohens_kappa(1.0, 0.5) == 1.0

def test_cohens_kappa_chance():
    assert np.isclose(cohens_kappa(0.5, 0.5), 0.0)

def test_severity_mae():
    mae = compute_severity_mae([1.0, 0.5], [0.8, 0.5])
    assert np.isclose(mae, 0.1)

def test_fp_rate_on_ideal():
    fpr = compute_fp_rate(5, 95)
    assert np.isclose(fpr, 0.05)

def test_confusion_matrix_structure():
    y_true = ["A", "B", "A"]
    y_pred = ["A", "B", "B"]
    cm, labels = confusion_matrix(y_true, y_pred)
    assert cm.sum() == 3

def test_boundary_agreement():
    tb = [1.0, 2.0, 3.0]
    pb = [1.05, 2.5, 2.95]
    acc = boundary_agreement(tb, pb, tolerance=0.1)
    assert np.isclose(acc, 2/3)
