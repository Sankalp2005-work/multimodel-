"""
scripts/diagnose_s093.py -- Fine-grained windowed analysis of Severity 1 rushed pace clip S093.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
from src.speechmirror.pipeline import SpeechMirrorPipeline

pipe = SpeechMirrorPipeline()
with open("configs/pipeline.json") as f:
    cfg = json.load(f)

wav_ideal = "data/processed/wav/T5_ideal_1.wav"
wav_part = "data/processed/wav/T5_synth_rushed_pace_sev1.wav"

feat_ideal = pipe.extract_features(wav_ideal)
feat_part = pipe.extract_features(wav_part)

# DTW path: col 0 is participant, col 1 is ideal
wp = pipe.align_dtw(feat_part, feat_ideal)
t_part = feat_part["frame_times"]
t_ideal = feat_ideal["frame_times"]
p_idx = wp[:, 0]
i_idx = wp[:, 1]

print(f"p_idx max: {np.max(p_idx)}, len t_part: {len(t_part)}")
print(f"i_idx max: {np.max(i_idx)}, len t_ideal: {len(t_ideal)}")

dur_ideal_total = feat_ideal["duration"]
win_size = 3.5
win_step = 0.5
cur_t = 0.0

detected_windows = []
while cur_t + win_size <= dur_ideal_total:
    w_start = cur_t
    w_end = cur_t + win_size
    cur_t += win_step

    mask = np.where((t_ideal[i_idx] >= w_start) & (t_ideal[i_idx] <= w_end))[0]
    if len(mask) < 15:
        continue

    p_start_f = p_idx[mask[0]]
    p_end_f = p_idx[mask[-1]]
    if p_end_f <= p_start_f:
        continue

    t_p_start = float(t_part[p_start_f])
    t_p_end = float(t_part[p_end_f])
    dur_p_raw = max(0.1, t_p_end - t_p_start)
    dur_i_raw = max(0.1, w_end - w_start)

    sil_p = sum(max(0.0, min(p["end"], t_p_end) - max(p["start"], t_p_start)) for p in feat_part.get("pauses", []))
    sil_i = sum(max(0.0, min(ip["end"], w_end) - max(ip["start"], w_start)) for ip in feat_ideal.get("pauses", []))

    dur_p_active = max(0.3, dur_p_raw - sil_p)
    dur_i_active = max(0.3, dur_i_raw - sil_i)

    speed_ratio = dur_i_active / dur_p_active
    if speed_ratio >= 1.15:
        detected_windows.append((w_start, w_end, t_p_start, t_p_end, speed_ratio))

print(f"Total windows with speed_ratio >= 1.15: {len(detected_windows)}")
for w in detected_windows[:15]:
    print(f"  Ideal [{w[0]:.1f}-{w[1]:.1f}s] -> Part [{w[2]:.1f}-{w[3]:.1f}s]: speed_ratio = {w[4]:.3f}")
