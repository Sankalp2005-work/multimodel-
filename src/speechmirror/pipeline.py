"""
pipeline.py -- SpeechMirror DTW-based contrastive evaluation pipeline.
Uses DTW-aligned acoustic feature comparison (no ASR / Whisper required).
"""

import warnings
import numpy as np
import librosa
from pathlib import Path
from scipy.spatial.distance import cdist

try:
    import pyworld as pw
except ImportError:
    pw = None

warnings.filterwarnings("ignore")


class SpeechMirrorPipeline:
    """
    SpeechMirror Contrastive Pipeline.
    Extracts acoustic features from audio and detects flaw regions
    via Dynamic Time Warping (DTW) contrastive comparison against an ideal take.
    """

    def __init__(self, hop_length: int = 512, sr: int = 16000):
        self.sr = sr
        self.hop_length = hop_length

    def extract_features(self, wav_path: str) -> dict:
        """
        Extract acoustic features from audio file without ASR.
        Features extracted:
          - RMS energy (frame-level in dB)
          - F0 / pitch contour (DIO + Stonemask or YIN)
          - MFCCs (13 coefficients for alignment)
          - Spectral centroid, flatness, bandwidth, ZCR
          - Pauses / silences (intervals where RMS < -32 dB)
        """
        y, sr = librosa.load(str(wav_path), sr=self.sr)
        duration = float(len(y) / sr)

        # 1. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # 2. MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)

        # 3. F0 / Pitch (Fast robust autocorrelation on 4kHz audio, 0.4s runtime)
        n_frames = len(rms)
        step_down = max(1, sr // 4000)
        y_4k = y[::step_down]
        hop_4k = max(1, self.hop_length // step_down)
        frame_len_4k = hop_4k * 4
        min_lag = int(4000 / 400)  # 10 samples (400 Hz)
        max_lag = int(4000 / 60)   # 66 samples (60 Hz)

        f0 = np.zeros(n_frames, dtype=np.float64)
        n_avail = len(y_4k) // hop_4k
        for i in range(min(n_frames, n_avail)):
            start_s = i * hop_4k
            frame_s = y_4k[start_s:start_s + frame_len_4k]
            if len(frame_s) >= max_lag and np.std(frame_s) >= 1e-4:
                frame_s_zero = frame_s - np.mean(frame_s)
                corr = np.correlate(frame_s_zero, frame_s_zero, mode='full')
                corr = corr[len(corr) // 2:]
                lags = corr[min_lag:max_lag]
                if len(lags) > 0 and np.max(lags) > 0.28 * corr[0]:
                    best_lag = min_lag + np.argmax(lags)
                    f0[i] = 4000.0 / best_lag

        # Ensure f0 length matches rms
        n_frames = len(rms)
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        elif len(f0) > n_frames:
            f0 = f0[:n_frames]

        # Voiced semitones relative to median
        voiced = f0 > 0
        valid_f0 = f0[voiced]
        f0_semitones = np.full_like(f0, -100.0)
        if len(valid_f0) > 0:
            ref_f0 = float(np.median(valid_f0))
            if ref_f0 > 0:
                f0_semitones[voiced] = 12.0 * np.log2((f0[voiced] + 1e-9) / ref_f0)

        # 4. Spectral Features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]

        # 5. Frame Times
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=self.hop_length)

        # 6. Silence / Pauses (segments with rms_db < -32 dB lasting >= 0.25s)
        silent_frames = rms_db < -32.0
        pauses = []
        in_silence = False
        silence_start = 0.0

        for idx, is_sil in enumerate(silent_frames):
            t_curr = frame_times[idx]
            if is_sil and not in_silence:
                in_silence = True
                silence_start = t_curr
            elif not is_sil and in_silence:
                in_silence = False
                sil_dur = t_curr - silence_start
                if sil_dur >= 0.25:
                    pauses.append({
                        "start": round(float(silence_start), 3),
                        "end": round(float(t_curr), 3),
                        "duration": round(float(sil_dur), 3)
                    })
        if in_silence:
            sil_dur = duration - silence_start
            if sil_dur >= 0.25:
                pauses.append({
                    "start": round(float(silence_start), 3),
                    "end": round(float(duration), 3),
                    "duration": round(float(sil_dur), 3)
                })

        return {
            "duration": duration,
            "sr": sr,
            "frame_times": frame_times,
            "pauses": pauses,
            "features": {
                "rms_db": rms_db,
                "f0": f0,
                "f0_semitones": f0_semitones,
                "mfcc": mfcc,
                "centroid": centroid,
                "flatness": flatness,
                "bandwidth": bandwidth,
                "zcr": zcr
            }
        }

    def align_dtw(self, feat_part: dict, feat_ideal: dict) -> np.ndarray:
        """
        Align two recordings via DTW on normalized MFCC frames.
        Returns array of shape (L, 2) where col 0 is participant frame idx
        and col 1 is ideal frame idx, monotonically increasing.
        """
        X = feat_part["features"]["mfcc"]
        Y = feat_ideal["features"]["mfcc"]

        X_std = np.std(X, axis=1, keepdims=True) + 1e-8
        Y_std = np.std(Y, axis=1, keepdims=True) + 1e-8
        X_norm = (X - np.mean(X, axis=1, keepdims=True)) / X_std
        Y_norm = (Y - np.mean(Y, axis=1, keepdims=True)) / Y_std

        # Try librosa DTW
        try:
            D, wp = librosa.sequence.dtw(X=X_norm, Y=Y_norm, metric='cosine')
            return wp[::-1]
        except Exception:
            pass

        # Robust fast DP fallback
        n_part = X_norm.shape[1]
        n_ideal = Y_norm.shape[1]
        step = 1
        if max(n_part, n_ideal) > 2500:
            step = int(np.ceil(max(n_part, n_ideal) / 1500))
            X_sub = X_norm[:, ::step]
            Y_sub = Y_norm[:, ::step]
        else:
            X_sub = X_norm
            Y_sub = Y_norm

        cost = cdist(X_sub.T, Y_sub.T, metric='cosine')
        N, M = cost.shape
        D = np.full((N, M), np.inf, dtype=np.float32)
        D[0, 0] = cost[0, 0]
        for i in range(1, N):
            D[i, 0] = D[i - 1, 0] + cost[i, 0]
        for j in range(1, M):
            D[0, j] = D[0, j - 1] + cost[0, j]

        for i in range(1, N):
            for j in range(1, M):
                D[i, j] = cost[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        i, j = N - 1, M - 1
        path = [(i * step, j * step)]
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
                best = np.argmin(candidates)
                if best == 0:
                    i, j = i - 1, j - 1
                elif best == 1:
                    i -= 1
                else:
                    j -= 1
            path.append((min(n_part - 1, i * step), min(n_ideal - 1, j * step)))
        path.reverse()
        return np.array(path, dtype=np.int32)

    dtw_align = align_dtw
    compute_deltas = align_dtw

    def detect_flaws(self, part_features: dict, ideal_features: dict, config: dict = None) -> list[dict]:
        """
        Contrastive flaw detection between participant and ideal audio.
        Compares aligned acoustics using DTW and thresholds from config:
          - rushed_pace: DTW slope / speed ratio > pace_slope_thresh
          - dead_pause: excess silence duration > pause_silence_thresh_s
          - flat_pitch: F0 variance ratio < pitch_var_ratio_thresh
          - mumbled_clarity: centroid ratio < centroid_ratio_thresh
        """
        if config is None:
            config = {}

        pace_slope_thresh = float(config.get("pace_slope_thresh", 1.20))
        pause_silence_thresh_s = float(config.get("pause_silence_thresh_s", config.get("pause_thresh", 0.50)))
        pitch_var_ratio_thresh = float(config.get("pitch_var_ratio_thresh", 0.65))
        centroid_ratio_thresh = float(config.get("centroid_ratio_thresh", 0.82))
        energy_drop_thresh = float(config.get("energy_drop_thresh", 2.5))

        wp = self.align_dtw(part_features, ideal_features)
        p_idx = wp[:, 0]
        i_idx = wp[:, 1]
        t_part = part_features["frame_times"]
        t_ideal = ideal_features["frame_times"]

        raw_flaws = []
        detected_pause_intervals = []

        # 1. Dead pause detection (exclude leading lead-in and trailing lead-out)
        dur_total_part = float(part_features["duration"])
        for p_pause in part_features.get("pauses", []):
            dur_p = p_pause["duration"]
            if dur_p < 0.35:
                continue
            p_start = p_pause["start"]
            p_end = p_pause["end"]
            if p_start < 0.8 or p_end > dur_total_part - 1.2:
                continue

            idx_s = np.searchsorted(t_part, p_start)
            idx_e = np.searchsorted(t_part, p_end)
            m_s = min(len(i_idx) - 1, np.searchsorted(p_idx, idx_s))
            m_e = min(len(i_idx) - 1, np.searchsorted(p_idx, idx_e))
            i_start = float(t_ideal[i_idx[m_s]])
            i_end = float(t_ideal[i_idx[m_e]])

            dur_i = 0.0
            for ip in ideal_features.get("pauses", []):
                overlap = max(0.0, min(ip["end"], i_end) - max(ip["start"], i_start))
                if overlap > 0:
                    dur_i += overlap

            excess = dur_p - dur_i
            if excess >= pause_silence_thresh_s:
                if excess >= 2.0:
                    sev = 4
                elif excess >= 1.3:
                    sev = 3
                elif excess >= 0.75:
                    sev = 2
                else:
                    sev = 1
                conf = min(1.0, excess / 2.0)
                raw_flaws.append({
                    "type": "dead_pause",
                    "severity": sev,
                    "confidence": round(float(conf), 3),
                    "start": round(float(p_start), 2),
                    "end": round(float(p_end), 2),
                    "explanation": f"Unnatural dead pause: {excess:.2f}s excess silence vs ideal take.",
                    "evidence": [{
                        "feature": "pause_duration_s",
                        "ref": round(float(dur_i), 2),
                        "obs": round(float(dur_p), 2),
                        "delta_pct": round(float(excess) * 100, 1),
                        "z": round(float(excess) / 0.5, 2)
                    }]
                })
                detected_pause_intervals.append((float(p_start), float(p_end)))

        # 2. Sliding window contrastive analysis for pace, pitch, and clarity
        dur_ideal_total = ideal_features["duration"]
        win_size = 3.5  # seconds
        win_step = 0.5  # seconds

        cur_t = 0.0
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

            # Active phonation durations (subtract unshared pauses)
            sil_p = sum(max(0.0, min(p["end"], t_p_end) - max(p["start"], t_p_start)) for p in part_features.get("pauses", []))
            sil_i = sum(max(0.0, min(ip["end"], w_end) - max(ip["start"], w_start)) for ip in ideal_features.get("pauses", []))
            dur_p_active = max(0.3, dur_p_raw - sil_p)
            dur_i_active = max(0.3, dur_i_raw - sil_i)

            # Check if current window overlaps a detected dead pause or is mostly silent
            in_pause = (sil_p / dur_p_raw > 0.40) or any(max(0.0, min(pe, t_p_end) - max(ps, t_p_start)) > 0.6 for ps, pe in detected_pause_intervals)

            # Rushed pace on active speech
            speed_ratio = dur_i_active / dur_p_active
            if not in_pause and speed_ratio >= pace_slope_thresh:
                if speed_ratio >= 1.65:
                    sev = 4
                elif speed_ratio >= 1.45:
                    sev = 3
                elif speed_ratio >= 1.25:
                    sev = 2
                else:
                    sev = 1
                conf = min(1.0, (speed_ratio - pace_slope_thresh) / 0.5)
                raw_flaws.append({
                    "type": "rushed_pace",
                    "severity": sev,
                    "confidence": round(float(conf), 3),
                    "start": round(t_p_start, 2),
                    "end": round(t_p_end, 2),
                    "explanation": f"Rushed pace: speed is {speed_ratio:.2f}x faster than ideal take.",
                    "evidence": [{
                        "feature": "speed_ratio",
                        "ref": 1.0,
                        "obs": round(float(speed_ratio), 2),
                        "delta_pct": round((float(speed_ratio) - 1.0) * 100, 1),
                        "z": round((float(speed_ratio) - 1.0) / 0.2, 2)
                    }]
                })

            # Flat pitch
            f0_p = part_features["features"]["f0"][p_start_f:p_end_f + 1]
            i_start_f = i_idx[mask[0]]
            i_end_f = i_idx[mask[-1]]
            f0_i = ideal_features["features"]["f0"][i_start_f:i_end_f + 1]
            vp = f0_p[f0_p > 0]
            vi = f0_i[f0_i > 0]
            if len(vp) >= 15 and len(vi) >= 15:
                ref_p = np.median(vp)
                ref_i = np.median(vi)
                if ref_p > 0 and ref_i > 0:
                    st_p = 12.0 * np.log2((vp + 1e-9) / ref_p)
                    st_i = 12.0 * np.log2((vi + 1e-9) / ref_i)
                    # Reject octave tracking outliers (|st| > 10 semitones)
                    st_p_clean = st_p[np.abs(st_p) <= 10.0]
                    st_i_clean = st_i[np.abs(st_i) <= 10.0]
                    if len(st_p_clean) >= 12 and len(st_i_clean) >= 12:
                        var_p = float(np.var(st_p_clean))
                        var_i = float(np.var(st_i_clean))
                        if var_i > 0.6 and var_p <= 1.2:
                            p_ratio = var_p / var_i
                            if p_ratio <= pitch_var_ratio_thresh:
                                if p_ratio <= 0.15:
                                    sev = 4
                                elif p_ratio <= 0.35:
                                    sev = 3
                                elif p_ratio <= 0.50:
                                    sev = 2
                                else:
                                    sev = 1
                                conf = min(1.0, (pitch_var_ratio_thresh - p_ratio) / pitch_var_ratio_thresh)
                                raw_flaws.append({
                                    "type": "flat_pitch",
                                    "severity": sev,
                                    "confidence": round(float(conf), 3),
                                    "start": round(t_p_start, 2),
                                    "end": round(t_p_end, 2),
                                    "explanation": f"Monotone delivery: pitch variance dropped by {(1.0 - p_ratio)*100:.0f}% vs ideal.",
                                    "evidence": [{
                                        "feature": "pitch_variance_ratio",
                                        "ref": 1.0,
                                        "obs": round(float(p_ratio), 2),
                                        "delta_pct": round((float(p_ratio) - 1.0) * 100, 1),
                                        "z": round((1.0 - float(p_ratio)) / 0.2, 2)
                                    }]
                                })

            # Mumbled clarity
            cent_p = part_features["features"]["centroid"][p_start_f:p_end_f + 1]
            cent_i = ideal_features["features"]["centroid"][i_start_f:i_end_f + 1]
            rms_p = part_features["features"]["rms_db"][p_start_f:p_end_f + 1]
            rms_i = ideal_features["features"]["rms_db"][i_start_f:i_end_f + 1]

            if len(cent_p) > 0 and len(cent_i) > 0:
                mean_cp = float(np.mean(cent_p))
                mean_ci = float(np.mean(cent_i))
                c_ratio = mean_cp / max(1.0, mean_ci)
                e_drop = float(np.mean(rms_i) - np.mean(rms_p))

                if c_ratio <= centroid_ratio_thresh or (e_drop >= energy_drop_thresh and c_ratio <= 0.90):
                    if c_ratio <= 0.60 or e_drop >= 8.0:
                        sev = 4
                    elif c_ratio <= 0.70 or e_drop >= 5.5:
                        sev = 3
                    elif c_ratio <= 0.78 or e_drop >= 3.5:
                        sev = 2
                    else:
                        sev = 1
                    conf = min(1.0, (centroid_ratio_thresh - c_ratio) / centroid_ratio_thresh + max(0.0, e_drop) / 10.0)
                    raw_flaws.append({
                        "type": "mumbled_clarity",
                        "severity": sev,
                        "confidence": round(float(conf), 3),
                        "start": round(t_p_start, 2),
                        "end": round(t_p_end, 2),
                        "explanation": f"Muffled clarity: spectral centroid ratio {c_ratio:.2f}, energy delta {e_drop:+.1f}dB.",
                        "evidence": [
                            {
                                "feature": "spectral_centroid_ratio",
                                "ref": round(mean_ci, 1),
                                "obs": round(mean_cp, 1),
                                "delta_pct": round((c_ratio - 1.0) * 100, 1),
                                "z": round((1.0 - c_ratio) / 0.15, 2)
                            },
                            {
                                "feature": "energy_drop_db",
                                "ref": 0.0,
                                "obs": round(-e_drop, 1),
                                "delta_pct": round(-e_drop, 1),
                                "z": round(e_drop / 2.0, 2)
                            }
                        ]
                    })

        # 3. Merge overlapping regions per flaw type
        min_rushed_dur = float(config.get("min_rushed_duration_s", 4.0)) if config else 4.0
        min_flat_dur = float(config.get("min_flat_pitch_duration_s", 4.0)) if config else 4.0
        merged = []
        def _validate_flaw(c):
            dur_c = c["end"] - c["start"]
            if c["type"] == "rushed_pace":
                if dur_c < min_rushed_dur:
                    return False
                idx_s = np.searchsorted(t_part, c["start"])
                idx_e = np.searchsorted(t_part, c["end"])
                m_s = min(len(i_idx) - 1, np.searchsorted(p_idx, idx_s))
                m_e = min(len(i_idx) - 1, np.searchsorted(p_idx, idx_e))
                i_s_t = float(t_ideal[i_idx[m_s]])
                i_e_t = float(t_ideal[i_idx[m_e]])
                dur_i_span = i_e_t - i_s_t
                sil_p_span = sum(max(0.0, min(p["end"], c["end"]) - max(p["start"], c["start"])) for p in part_features.get("pauses", []))
                sil_i_span = sum(max(0.0, min(ip["end"], i_e_t) - max(ip["start"], i_s_t)) for ip in ideal_features.get("pauses", []))
                act_p = max(0.3, dur_c - sil_p_span)
                act_i = max(0.3, dur_i_span - sil_i_span)
                if (act_i / act_p) < pace_slope_thresh:
                    return False
            elif c["type"] == "flat_pitch":
                if dur_c < min_flat_dur:
                    return False
            elif c["type"] == "dead_pause":
                if dur_c < pause_silence_thresh_s:
                    return False
            return True

        for ftype in ["dead_pause", "rushed_pace", "flat_pitch", "mumbled_clarity"]:
            type_flaws = [f for f in raw_flaws if f["type"] == ftype]
            if not type_flaws:
                continue

            type_flaws.sort(key=lambda x: x["start"])
            cur = dict(type_flaws[0])
            for f in type_flaws[1:]:
                if f["start"] <= cur["end"] + 1.2:
                    cur["end"] = max(cur["end"], f["end"])
                    cur["severity"] = max(cur["severity"], f["severity"])
                    cur["confidence"] = max(cur["confidence"], f["confidence"])
                    if "evidence" in f and "evidence" in cur:
                        cur["evidence"] = cur["evidence"]
                else:
                    if _validate_flaw(cur):
                        merged.append(cur)
                    cur = dict(f)
            if _validate_flaw(cur):
                merged.append(cur)

        min_conf = float(config.get("min_confidence", 0.20)) if config else 0.20
        merged = [f for f in merged if f["confidence"] >= min_conf]
        return merged

    def compute_deltas(self, part_features: dict, ideal_features: dict) -> list[dict]:
        """Backward compatibility wrapper returning contrastive windows."""
        flaws = self.detect_flaws(part_features, ideal_features)
        return flaws
