"""
SpeechMirror Flaw Detection Module

Detects flaw regions (rushed_pace, dead_pause, flat_pitch, mumbled_clarity)
based on contrastive speech features and word alignments.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import structures from adjacent modules
from .compare import WordComparison, SentenceComparison
from .align import AlignmentResult


@dataclass
class DetectedRegion:
    flaw_type: str
    start_sec: float
    end_sec: float
    severity: int
    confidence: float
    evidence: list[dict]
    word_indices: list[int]


@dataclass
class DetectionResult:
    sample_id: str
    regions: list[DetectedRegion]
    scores: dict
    alignment_method: str
    alignment_wer: float
    alignment_confidence: float


class FlawDetector:
    def __init__(self, config_path: str = 'configs/pipeline.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self.thresholds = self.config.get('detection_thresholds', {
            'rushed_pace': {'rate_ratio_min': 1.25, 'gap_compression_max': 0.8},
            'dead_pause': {'excess_pause_min': 0.4},
            'flat_pitch': {'f0_var_ratio_max': 0.6, 'f0_range_ratio_max': 0.7},
            'mumbled_clarity': {'energy_drop_max': -2.5, 'centroid_ratio_max': 0.8}
        })
        self.weights = self.config.get('scoring_weights', {
            'pace': 0.25,
            'pause': 0.25,
            'pitch': 0.25,
            'clarity': 0.25
        })

    def _load_config(self) -> dict[str, Any]:
        path = Path(self.config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def detect(
        self,
        comparisons: list[SentenceComparison],
        ideal_alignment: AlignmentResult,
        participant_alignment: AlignmentResult,
        sample_id: str = ''
    ) -> DetectionResult:
        all_regions = []
        for sentence in comparisons:
            # Flatten words from sentence for simplicity; assuming sentence has words
            words = sentence.words
            
            all_regions.extend(self._detect_rushed_pace(words))
            all_regions.extend(self._detect_dead_pause(words))
            all_regions.extend(self._detect_flat_pitch(words))
            all_regions.extend(self._detect_mumbled_clarity(words))
            
        merged_regions = self._merge_adjacent_regions(all_regions)
        
        scores = self._compute_scores(comparisons, merged_regions)
        
        return DetectionResult(
            sample_id=sample_id,
            regions=merged_regions,
            scores=scores,
            alignment_method=participant_alignment.method,
            alignment_wer=participant_alignment.wer,
            alignment_confidence=participant_alignment.confidence
        )

    def _detect_rushed_pace(self, words: list[WordComparison]) -> list[DetectedRegion]:
        regions = []
        th = self.thresholds['rushed_pace']
        
        for idx, w in enumerate(words):
            if w.ref_duration > 0:
                rate_ratio = (w.ref_duration / w.obs_duration) if hasattr(w, 'obs_duration') and w.obs_duration > 0 else 0
                if rate_ratio > th['rate_ratio_min']:
                    mag = rate_ratio - th['rate_ratio_min']
                    sev = self._estimate_severity('rushed_pace', mag)
                    conf = min(1.0, mag / 2.0)
                    evidence = [{
                        'feature': 'speech_rate_ratio',
                        'ref_value': 1.0,
                        'obs_value': rate_ratio,
                        'delta_pct': (rate_ratio - 1) * 100,
                        'z_score': None
                    }]
                    regions.append(DetectedRegion(
                        flaw_type='rushed_pace',
                        start_sec=w.obs_start,
                        end_sec=w.obs_end,
                        severity=sev,
                        confidence=conf,
                        evidence=evidence,
                        word_indices=[idx]
                    ))
        return regions

    def _detect_dead_pause(self, words: list[WordComparison]) -> list[DetectedRegion]:
        regions = []
        th = self.thresholds['dead_pause']
        
        for idx in range(len(words) - 1):
            w1 = words[idx]
            w2 = words[idx + 1]
            
            obs_gap = w2.obs_start - w1.obs_end
            ref_gap = w2.ref_start - w1.ref_end
            excess_pause = obs_gap - ref_gap
            
            if excess_pause > th['excess_pause_min']:
                sev = self._estimate_severity('dead_pause', excess_pause)
                conf = min(1.0, excess_pause / 2.0)
                evidence = [{
                    'feature': 'excess_pause',
                    'ref_value': ref_gap,
                    'obs_value': obs_gap,
                    'delta_pct': None,
                    'z_score': None
                }]
                regions.append(DetectedRegion(
                    flaw_type='dead_pause',
                    start_sec=w1.obs_end,
                    end_sec=w2.obs_start,
                    severity=sev,
                    confidence=conf,
                    evidence=evidence,
                    word_indices=[idx, idx + 1]
                ))
        return regions

    def _detect_flat_pitch(self, words: list[WordComparison]) -> list[DetectedRegion]:
        regions = []
        th = self.thresholds['flat_pitch']
        
        for idx, w in enumerate(words):
            if not hasattr(w, 'obs_f0_var') or not hasattr(w, 'ref_f0_var'):
                continue
                
            f0_var_ratio = w.obs_f0_var / w.ref_f0_var if w.ref_f0_var > 0 else 1.0
            
            if f0_var_ratio < th['f0_var_ratio_max']:
                mag = th['f0_var_ratio_max'] - f0_var_ratio
                sev = self._estimate_severity('flat_pitch', mag)
                conf = min(1.0, mag * 2.0)
                evidence = [{
                    'feature': 'f0_variance_ratio',
                    'ref_value': 1.0,
                    'obs_value': f0_var_ratio,
                    'delta_pct': (f0_var_ratio - 1) * 100,
                    'z_score': None
                }]
                regions.append(DetectedRegion(
                    flaw_type='flat_pitch',
                    start_sec=w.obs_start,
                    end_sec=w.obs_end,
                    severity=sev,
                    confidence=conf,
                    evidence=evidence,
                    word_indices=[idx]
                ))
        return regions

    def _detect_mumbled_clarity(self, words: list[WordComparison]) -> list[DetectedRegion]:
        regions = []
        th = self.thresholds['mumbled_clarity']
        
        for idx, w in enumerate(words):
            if not hasattr(w, 'obs_energy') or not hasattr(w, 'ref_energy'):
                continue
                
            energy_drop = w.obs_energy - w.ref_energy
            centroid_ratio = w.obs_spectral_centroid / w.ref_spectral_centroid if w.ref_spectral_centroid > 0 else 1.0
            
            if energy_drop < th['energy_drop_max'] and centroid_ratio < th['centroid_ratio_max']:
                mag = (th['energy_drop_max'] - energy_drop) + (th['centroid_ratio_max'] - centroid_ratio)
                sev = self._estimate_severity('mumbled_clarity', mag)
                conf = min(1.0, mag / 10.0)
                evidence = [
                    {
                        'feature': 'energy_drop',
                        'ref_value': 0.0,
                        'obs_value': energy_drop,
                        'delta_pct': None,
                        'z_score': None
                    },
                    {
                        'feature': 'spectral_centroid_ratio',
                        'ref_value': 1.0,
                        'obs_value': centroid_ratio,
                        'delta_pct': (centroid_ratio - 1) * 100,
                        'z_score': None
                    }
                ]
                regions.append(DetectedRegion(
                    flaw_type='mumbled_clarity',
                    start_sec=w.obs_start,
                    end_sec=w.obs_end,
                    severity=sev,
                    confidence=conf,
                    evidence=evidence,
                    word_indices=[idx]
                ))
        return regions

    def _merge_adjacent_regions(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        if not regions:
            return []
            
        merged = []
        # Group by flaw_type
        by_type = {}
        for r in regions:
            by_type.setdefault(r.flaw_type, []).append(r)
            
        for flaw_type, type_regions in by_type.items():
            sorted_r = sorted(type_regions, key=lambda x: x.start_sec)
            current = sorted_r[0]
            for next_r in sorted_r[1:]:
                # If they overlap or are very close (e.g. adjacent words)
                if next_r.start_sec <= current.end_sec + 0.1:  # 100ms tolerance
                    current.end_sec = max(current.end_sec, next_r.end_sec)
                    current.severity = max(current.severity, next_r.severity)
                    current.confidence = max(current.confidence, next_r.confidence)
                    current.evidence.extend(next_r.evidence)
                    # deduplicate indices
                    current.word_indices = sorted(list(set(current.word_indices + next_r.word_indices)))
                else:
                    merged.append(current)
                    current = next_r
            merged.append(current)
            
        return sorted(merged, key=lambda x: x.start_sec)

    def _estimate_severity(self, flaw_type: str, magnitude: float) -> int:
        if flaw_type == 'dead_pause':
            if magnitude >= 2.0: return 4
            if magnitude >= 1.4: return 3
            if magnitude >= 0.8: return 2
            if magnitude >= 0.4: return 1
            return 0
        elif flaw_type == 'rushed_pace':
            if magnitude >= 1.0: return 4
            if magnitude >= 0.75: return 3
            if magnitude >= 0.5: return 2
            if magnitude >= 0.25: return 1
            return 0
        elif flaw_type == 'flat_pitch':
            if magnitude >= 0.5: return 4
            if magnitude >= 0.4: return 3
            if magnitude >= 0.3: return 2
            if magnitude >= 0.1: return 1
            return 0
        elif flaw_type == 'mumbled_clarity':
            if magnitude >= 5.0: return 4
            if magnitude >= 3.0: return 3
            if magnitude >= 1.5: return 2
            if magnitude >= 0.5: return 1
            return 0
        return 1

    def _compute_scores(self, comparisons: list[SentenceComparison], regions: list[DetectedRegion]) -> dict:
        # Base score is 100, deduct based on regions severity
        scores = {
            'pace': 100.0,
            'pause': 100.0,
            'pitch': 100.0,
            'clarity': 100.0,
            'overall': 100.0
        }
        
        deductions = {'rushed_pace': 0, 'dead_pause': 0, 'flat_pitch': 0, 'mumbled_clarity': 0}
        
        for r in regions:
            deductions[r.flaw_type] += r.severity * 5
            
        scores['pace'] = max(0.0, scores['pace'] - deductions['rushed_pace'])
        scores['pause'] = max(0.0, scores['pause'] - deductions['dead_pause'])
        scores['pitch'] = max(0.0, scores['pitch'] - deductions['flat_pitch'])
        scores['clarity'] = max(0.0, scores['clarity'] - deductions['mumbled_clarity'])
        
        scores['overall'] = (
            scores['pace'] * self.weights['pace'] +
            scores['pause'] * self.weights['pause'] +
            scores['pitch'] * self.weights['pitch'] +
            scores['clarity'] * self.weights['clarity']
        )
        
        return scores
