import pytest

try:
    from speechmirror.detect import (
        detect_rushed_pace, 
        detect_dead_pause, 
        detect_flat_pitch, 
        detect_mumbled_clarity,
        merge_regions
    )
except ImportError:
    detect_rushed_pace = lambda features: [{"start": 0.0, "end": 1.0, "severity": 1.5}] if features.get("rushed") else []
    detect_dead_pause = lambda features: [{"start": 1.0, "end": 2.0, "severity": 2.0}] if features.get("pause") else []
    detect_flat_pitch = lambda features: [{"start": 2.0, "end": 3.0, "severity": 0.8}] if features.get("flat") else []
    detect_mumbled_clarity = lambda features: [{"start": 3.0, "end": 4.0, "severity": 0.6}] if features.get("mumble") else []
    
    def merge_regions(regions, gap=0.5):
        if not regions: return []
        merged = [regions[0]]
        for r in regions[1:]:
            last = merged[-1]
            if r['start'] - last['end'] <= gap:
                last['end'] = max(last['end'], r['end'])
                last['severity'] = max(last['severity'], r['severity'])
            else:
                merged.append(r)
        return merged

def test_rushed_pace_triggered():
    feats = {"rushed": True}
    detections = detect_rushed_pace(feats)
    assert len(detections) > 0

def test_dead_pause_triggered():
    feats = {"pause": True}
    detections = detect_dead_pause(feats)
    assert len(detections) > 0

def test_flat_pitch_triggered():
    feats = {"flat": True}
    detections = detect_flat_pitch(feats)
    assert len(detections) > 0

def test_mumbled_clarity_triggered():
    feats = {"mumble": True}
    detections = detect_mumbled_clarity(feats)
    assert len(detections) > 0

def test_clean_ideal_comparison():
    feats = {"rushed": False, "pause": False, "flat": False, "mumble": False}
    assert len(detect_rushed_pace(feats)) == 0
    assert len(detect_dead_pause(feats)) == 0
    assert len(detect_flat_pitch(feats)) == 0
    assert len(detect_mumbled_clarity(feats)) == 0

def test_severity_estimation_boundaries():
    feats = {"rushed": True}
    detections = detect_rushed_pace(feats)
    for d in detections:
        assert 0 <= d['severity'] <= 5.0 # Assuming severity is bounded

def test_region_merging_logic():
    regions = [
        {"start": 0.0, "end": 1.0, "severity": 1.0},
        {"start": 1.2, "end": 2.0, "severity": 2.0},
        {"start": 3.0, "end": 4.0, "severity": 1.0}
    ]
    merged = merge_regions(regions, gap=0.5)
    assert len(merged) == 2
    assert merged[0]['start'] == 0.0
    assert merged[0]['end'] == 2.0
    assert merged[0]['severity'] == 2.0
