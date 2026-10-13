"""Causal explanation generator for SpeechMirror.

Produces human-readable, evidence-bearing explanations for each detected flaw region
based on DTW contrastive alignment acoustic features and threshold comparisons.
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ExplanationEvidence:
    """Quantitative acoustic evidence comparing observed vs reference audio."""
    feature: str
    ref_value: float
    obs_value: float
    delta_pct: float
    z_score: float
    direction: str  # 'increase' | 'decrease'


@dataclass
class FlawExplanation:
    """Human-readable explanation of a detected flaw region."""
    flaw_type: str
    region_start: float
    region_end: float
    severity: int
    summary: str  # Human-readable one-liner
    details: str  # Multi-line explanation
    evidence: list[ExplanationEvidence]


def format_time_range(start: float, end: float) -> str:
    """Format time range as 'M:SS.s-M:SS.s'.
    
    Args:
        start: Start time in seconds.
        end: End time in seconds.
        
    Returns:
        Formatted string, e.g. '0:02.4-0:03.8'.
    """
    def _format(seconds: float) -> str:
        sec = max(0.0, float(seconds))
        m = int(sec // 60)
        s = sec % 60
        return f"{m}:{s:04.1f}"

    return f"{_format(start)}-{_format(end)}"


def explain_region(region: Any, comparisons: Any = None) -> FlawExplanation:
    """Generate causal explanation for a detected flaw region.

    Handles both flaw dictionaries (from SpeechMirrorPipeline.detect_flaws)
    and DetectedRegion dataclass objects.

    Args:
        region: Flaw dictionary or DetectedRegion object.
        comparisons: Optional list of comparisons (retained for backward compatibility).

    Returns:
        FlawExplanation containing human-readable summaries and structured evidence.
    """
    if isinstance(region, dict):
        flaw_type = str(region.get('type', region.get('flaw_type', 'unknown')))
        start = float(region.get('start', region.get('start_sec', region.get('region_start', 0.0))))
        end = float(region.get('end', region.get('end_sec', region.get('region_end', 0.0))))
        severity = int(region.get('severity', 1))
        confidence = float(region.get('confidence', 1.0))
        raw_explanation = str(region.get('explanation', ''))
        raw_evidence = region.get('evidence', [])
    else:
        flaw_type = str(getattr(region, 'type', getattr(region, 'flaw_type', 'unknown')))
        start = float(getattr(region, 'start', getattr(region, 'start_sec', getattr(region, 'region_start', 0.0))))
        end = float(getattr(region, 'end', getattr(region, 'end_sec', getattr(region, 'region_end', 0.0))))
        severity = int(getattr(region, 'severity', 1))
        confidence = float(getattr(region, 'confidence', 1.0))
        raw_explanation = str(getattr(region, 'explanation', ''))
        raw_evidence = getattr(region, 'evidence', [])

    # Format time window as M:SS.s
    tr = format_time_range(start, end)
    dur = max(0.0, end - start)

    # Parse and structure numeric evidence
    evidence_list: List[ExplanationEvidence] = []
    if isinstance(raw_evidence, list):
        for ev in raw_evidence:
            if isinstance(ev, ExplanationEvidence):
                evidence_list.append(ev)
            elif isinstance(ev, dict):
                feature = str(ev.get('feature', 'acoustic_metric'))
                ref_val = float(ev.get('ref', ev.get('ref_value', 0.0)))
                obs_val = float(ev.get('obs', ev.get('obs_value', 0.0)))
                d_pct = ev.get('delta_pct')
                if d_pct is not None:
                    delta_pct = float(d_pct)
                else:
                    delta_pct = round((obs_val - ref_val) / ref_val * 100.0, 1) if ref_val != 0 else 0.0

                z_val = ev.get('z', ev.get('z_score', 0.0))
                z_score = float(z_val) if z_val is not None else 0.0

                direction = str(ev.get('direction', 'increase' if obs_val >= ref_val else 'decrease'))

                evidence_list.append(ExplanationEvidence(
                    feature=feature,
                    ref_value=ref_val,
                    obs_value=obs_val,
                    delta_pct=delta_pct,
                    z_score=z_score,
                    direction=direction
                ))

    # Construct human-readable summary
    if raw_explanation:
        summary = f"{raw_explanation} at {tr} (Sev {severity})"
    else:
        type_labels = {
            'rushed_pace': 'Pace accelerated vs ideal',
            'dead_pause': 'Unnatural pause detected',
            'flat_pitch': 'Monotone pitch variance detected',
            'mumbled_clarity': 'Muffled speech clarity detected'
        }
        label = type_labels.get(flaw_type, f"{flaw_type.replace('_', ' ').title()} detected")
        summary = f"{label} at {tr} (Sev {severity})"

    # Construct detailed multi-line explanation with acoustic evidence
    detail_lines = [
        f"Region: {tr} ({dur:.2f}s duration on participant audio timeline)",
        f"Severity: Level {severity}/4 (confidence: {confidence:.0%})",
    ]
    if raw_explanation:
        detail_lines.append(f"Observation: {raw_explanation}")

    if evidence_list:
        detail_lines.append("Acoustic Evidence:")
        for ev in evidence_list:
            sign = "+" if ev.delta_pct > 0 else ""
            detail_lines.append(
                f"  - {ev.feature}: observed={ev.obs_value:.2f}, reference={ev.ref_value:.2f} "
                f"({ev.direction}, {sign}{ev.delta_pct:.1f}%)"
            )
    else:
        detail_lines.append(f"Observation: Acoustic deviation in {flaw_type.replace('_', ' ')} detected against reference audio.")

    # Actionable coaching guidance
    coaching_tips = {
        "rushed_pace": f"Coaching Tip: You spoke at accelerated speed ({dur:.1f}s window). Slow down by ~20-30%, insert natural pauses at commas, and articulate key syllables to improve audience comprehension.",
        "dead_pause": f"Coaching Tip: Unnatural silence of {dur:.1f}s detected. Aim for a natural 0.5s-1.0s breath pause between sections to keep the audience engaged.",
        "flat_pitch": "Coaching Tip: Monotone cadence detected. Modulate your pitch by raising intonation on introductory clauses and key emphasis terms.",
        "mumbled_clarity": "Coaching Tip: Muffled articulation with lowered high-frequency energy. Project your voice forward and enunciate consonants crisply."
    }
    tip = coaching_tips.get(flaw_type, "Coaching Tip: Align closer to the reference cadence and articulation.")
    detail_lines.append(f"\nActionable Guidance:\n  {tip}")

    details = "\n".join(detail_lines)

    return FlawExplanation(
        flaw_type=flaw_type,
        region_start=start,
        region_end=end,
        severity=severity,
        summary=summary,
        details=details,
        evidence=evidence_list
    )


def explain_all(detection_result: Any, comparisons: Any = None) -> list[FlawExplanation]:
    """Generate explanations for all detected flaw regions.

    Supports:
      - List of flaw dicts returned by SpeechMirrorPipeline.detect_flaws
      - DetectionResult object with a .regions attribute
      - Dict containing a 'regions' key

    Args:
        detection_result: Flaw detection output (list or container object).
        comparisons: Optional list of comparisons (for backward compatibility).

    Returns:
        List of FlawExplanation objects.
    """
    if isinstance(detection_result, list):
        regions = detection_result
    elif hasattr(detection_result, 'regions'):
        regions = detection_result.regions
    elif isinstance(detection_result, dict) and 'regions' in detection_result:
        regions = detection_result['regions']
    else:
        regions = []

    return [explain_region(r, comparisons) for r in regions]
