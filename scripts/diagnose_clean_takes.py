"""
scripts/diagnose_clean_takes.py -- Empirical variance analysis of ideal takes.
Measures active-speech speed ratio distribution on clean takes (ideal_2 vs ideal_1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.speechmirror.pipeline import SpeechMirrorPipeline

pipe = SpeechMirrorPipeline()

clean_pairs = [
    ("data/processed/wav/T1_ideal_2.wav", "data/processed/wav/T1_ideal_1.wav"),
    ("data/processed/wav/T2_ideal_2.wav", "data/processed/wav/T2_ideal_1.wav"),
    ("data/processed/wav/T3_ideal_2.wav", "data/processed/wav/T3_ideal_1.wav"),
    ("data/processed/wav/T4_ideal_2.wav", "data/processed/wav/T4_ideal_1.wav"),
    ("data/processed/wav/T5_ideal_2.wav", "data/processed/wav/T5_ideal_1.wav"),
    ("data/processed/wav/T6_ideal_2.wav", "data/processed/wav/T6_ideal_1.wav"),
]

win_size = 3.5
win_step = 0.5

print("Measuring Clean Takes (Ideal_2 vs Ideal_1):")
for p_wav, i_wav in clean_pairs:
    feat_part = pipe.extract_features(p_wav)
    feat_ideal = pipe.extract_features(i_wav)
    wp = pipe.align_dtw(feat_part, feat_ideal)
    t_part = feat_part["frame_times"]
    t_ideal = feat_ideal["frame_times"]
    p_idx = wp[:, 0]
    i_idx = wp[:, 1]
    
    dur_ideal = feat_ideal["duration"]
    cur_t = 0.0
    ratios = []
    
    while cur_t + win_size <= dur_ideal:
        w_start = cur_t
        w_end = cur_t + win_size
        cur_t += win_step
        
        mask = np.where((t_ideal[i_idx] >= w_start) & (t_ideal[i_idx] <= w_end))[0]
        if len(mask) < 15:
            continue
        p_sf = p_idx[mask[0]]
        p_ef = p_idx[mask[-1]]
        if p_ef <= p_sf:
            continue
        t_ps = float(t_part[p_sf])
        t_pe = float(t_part[p_ef])
        dur_p_raw = max(0.1, t_pe - t_ps)
        dur_i_raw = max(0.1, w_end - w_start)
        
        sil_p = sum(max(0.0, min(p["end"], t_pe) - max(p["start"], t_ps)) for p in feat_part.get("pauses", []))
        sil_i = sum(max(0.0, min(ip["end"], w_end) - max(ip["start"], w_start)) for ip in feat_ideal.get("pauses", []))
        
        dur_p_active = max(0.3, dur_p_raw - sil_p)
        dur_i_active = max(0.3, dur_i_raw - sil_i)
        
        in_pause = (sil_p / dur_p_raw > 0.40)
        if not in_pause:
            ratios.append(dur_i_active / dur_p_active)
            
    ratios = np.array(ratios)
    t_name = Path(p_wav).stem.split("_")[0]
    print(f"  {t_name}: Mean Ratio={np.mean(ratios):.3f}, 95th-pct={np.percentile(ratios, 95):.3f}, Max Ratio={np.max(ratios):.3f}")
