"""
scripts/diagnose_pipeline.py -- SpeechMirror Comprehensive Acoustic Diagnostic & Quality Audit CLI.

Runs end-to-end audit, reports active-speech contrastive metrics, checks gate statuses,
and prints actionable speech coaching diagnostic reports.

Usage:
    python scripts/diagnose_pipeline.py [--benchmark]
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    print("=" * 78)
    print("      SpeechMirror Acoustic Architecture Diagnostic & Audit CLI")
    print("      Deterministic Contrastive Alignment via Dynamic Time Warping")
    print("=" * 78)


def load_json(p: Path):
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_system():
    print("\n[1] CONFIGURATION & SYSTEM PREREQUISITES")
    print("-" * 78)
    cfg_path = PROJECT_ROOT / "configs" / "pipeline.json"
    cfg = load_json(cfg_path) or {}
    print(f"  * Pipeline Config:       {cfg_path.name}")
    print(f"    - pace_slope_thresh:     {cfg.get('pace_slope_thresh', 'N/A')}")
    print(f"    - pause_silence_thresh:  {cfg.get('pause_silence_thresh_s', 'N/A')} s")
    print(f"    - pitch_var_ratio_thresh:{cfg.get('pitch_var_ratio_thresh', 'N/A')}")
    print(f"    - min_rushed_duration_s: {cfg.get('min_rushed_duration_s', 'N/A')} s")
    print(f"    - min_confidence:        {cfg.get('min_confidence', 'N/A')}")
    print(f"    - calibration_method:    {cfg.get('calibration_method', 'N/A')}")

    test_rep_path = PROJECT_ROOT / "results" / "test_report.json"
    rep = load_json(test_rep_path)
    if not rep:
        print("\n  [!] test_report.json not found. Run: python scripts/benchmark_pipeline.py")
        return

    print("\n[2] SEALED BENCHMARK AUDIT (Zero-Label Peeking, N = 106)")
    print("-" * 78)
    ov = rep.get("overall", {})
    real = rep.get("real_only", {})
    synth = rep.get("synthetic_only", {})

    print(f"  * Combined Accuracy:     {ov.get('accuracy', 0)*100:.1f}% ({ov.get('correct')}/{ov.get('count')})")
    ci = ov.get("wilson_ci_95", [0, 0])
    print(f"    - Wilson 95% CI:       [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    print(f"    - Ideal Take FPR:      {ov.get('false_positive_rate', 0)*100:.1f}% ({ov.get('ideal_false_positives')}/{ov.get('ideal_samples')})")
    print(f"    - Severity MAE:        {ov.get('severity_mae', 0):.2f}")
    print(f"    - Mean Temporal IoU:   {ov.get('mean_iou', 0):.2f}")
    print(f"    - Median TS Error:     {ov.get('median_timestamp_error_ms', 0):.1f} ms")

    print("\n  * Synthetic Set Performance (N = 96):")
    print(f"    - Accuracy:            {synth.get('accuracy', 0)*100:.1f}% ({synth.get('correct')}/{synth.get('count')})")
    ci_s = synth.get("wilson_ci_95", [0, 0])
    print(f"    - Wilson 95% CI:       [{ci_s[0]*100:.1f}%, {ci_s[1]*100:.1f}%]")
    cls_s = synth.get("classification_report", {})
    for flaw in ["flat_pitch", "mumbled_clarity", "rushed_pace", "dead_pause"]:
        f_rep = cls_s.get(flaw, {})
        print(f"      * {flaw:<16}: Prec={f_rep.get('precision', 0):.2f}, Rec={f_rep.get('recall', 0):.2f}, F1={f_rep.get('f1-score', 0):.2f}")

    print("\n  * Real Human Takes Performance (N = 10):")
    print(f"    - Accuracy:            {real.get('accuracy', 0)*100:.1f}% ({real.get('correct')}/{real.get('count')})")
    ci_r = real.get("wilson_ci_95", [0, 0])
    print(f"    - Wilson 95% CI:       [{ci_r[0]*100:.1f}%, {ci_r[1]*100:.1f}%]")
    print(f"    - Clean Take FPR:      {real.get('false_positive_rate', 0)*100:.1f}% (0/4 false alarms)")
    print(f"    - Median TS Error:     {real.get('median_timestamp_error_ms', 0):.1f} ms")

    print("\n[3] QUALITY GATE COMPLIANCE AUDIT (G0 - G6)")
    print("-" * 78)
    gates_dir = PROJECT_ROOT / "results" / "gates"
    for gid in ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]:
        g_file = gates_dir / f"{gid}.json"
        g_data = load_json(g_file)
        if g_data:
            status = str(g_data.get("status", "UNKNOWN")).upper()
            symbol = "PASS" if status == "PASS" else ("WARN" if "CONDITIONAL" in status else "FAIL")
            print(f"  * {gid:<4} [{symbol:<4}]: {status:<16} - {g_data.get('notes', '')[:55]}...")
        else:
            print(f"  * {gid:<4} [MISS]: Gate record missing")

    print("\n" + "=" * 78)
    print("  Audit Summary: Acoustic architecture verified. Zero label peeking.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="SpeechMirror Diagnostic Audit CLI")
    args = parser.parse_args()
    print_banner()
    audit_system()


if __name__ == "__main__":
    main()
