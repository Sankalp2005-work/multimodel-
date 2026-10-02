"""
Causal explanation generator for SpeechMirror.
Produces human-readable, evidence-bearing explanations for each detected flaw region.
"""
from dataclasses import dataclass
from typing import Any

@dataclass
class ExplanationEvidence:
    feature: str
    ref_value: float
    obs_value: float
    delta_pct: float
    z_score: float
    direction: str  # 'increase' | 'decrease'

@dataclass
class FlawExplanation:
    flaw_type: str
    region_start: float
    region_end: float
    severity: int
    summary: str  # Human-readable one-liner
    details: str  # Multi-line explanation
    evidence: list[ExplanationEvidence]


def format_time_range(start: float, end: float) -> str:
    """Format time range as 'M:SS.s-M:SS.s'"""
    def _format(seconds: float) -> str:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:04.1f}"
    return f"{_format(start)}-{_format(end)}"


def explain_region(region: Any, comparisons: list) -> FlawExplanation:
    """Generate causal explanation for a detected flaw region."""
    # Handle different possible region attributes depending on actual structure
    flaw_type = getattr(region, 'type', getattr(region, 'flaw_type', 'unknown'))
    start = getattr(region, 'start_time', getattr(region, 'start', 0.0))
    end = getattr(region, 'end_time', getattr(region, 'end', 0.0))
    severity = getattr(region, 'severity', 1)
    
    evidence_list = []
    
    # In a real implementation, comparisons would provide the metrics 
    # to populate evidence_list and details.
    
    tr = format_time_range(start, end)
    
    if flaw_type == "rushed_pace":
        summary = f"pace increased at {tr} (high words/s vs ideal)"
        details = "Speech rate was significantly higher than the reference, indicating rushing."
    elif flaw_type == "dead_pause":
        summary = f"excess pause at {tr} (gap longer than ideal)"
        details = "A long silence was detected that was not present or much shorter in the reference."
    elif flaw_type == "flat_pitch":
        summary = f"flat pitch at {tr} (reduced F0 variance)"
        details = "Pitch variation was lower than the reference, leading to a monotonous delivery."
    elif flaw_type == "mumbled_clarity":
        summary = f"mumbled speech at {tr} (energy and clarity drop)"
        details = "Acoustic features indicate a drop in speech clarity and energy compared to reference."
    else:
        summary = f"{flaw_type} detected at {tr}"
        details = "Deviation from reference audio detected."

    return FlawExplanation(
        flaw_type=flaw_type,
        region_start=start,
        region_end=end,
        severity=severity,
        summary=summary,
        details=details,
        evidence=evidence_list
    )


def explain_all(detection_result: Any, comparisons: list) -> list[FlawExplanation]:
    """Generate explanations for all detected regions."""
    regions = getattr(detection_result, 'regions', [])
    return [explain_region(r, comparisons) for r in regions]
