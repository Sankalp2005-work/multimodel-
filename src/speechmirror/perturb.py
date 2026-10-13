"""
perturb.py -- Core perturbation functions for SpeechMirror Phase 3.

Each function takes the target segment (numpy array, 16 kHz mono float64),
applies a single perturbation, and returns the modified segment.
Crossfade stitching is handled by the caller.

All operations are deterministic given the same input + seed.
"""

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Rushed Pace: pitch-preserving time compression
# ---------------------------------------------------------------------------

def perturb_rushed_pace(
    segment: np.ndarray,
    sr: int,
    speed_factor: float,
) -> np.ndarray:
    """Time-compress a segment by speed_factor (>1 = faster) preserving pitch.

    Uses librosa phase vocoder which is deterministic.
    """
    import librosa

    # librosa.effects.time_stretch: rate > 1 = speed up
    stretched = librosa.effects.time_stretch(segment, rate=speed_factor)
    return stretched


# ---------------------------------------------------------------------------
# Dead Pause: insert silence at midpoint of segment
# ---------------------------------------------------------------------------

def perturb_dead_pause(
    segment: np.ndarray,
    sr: int,
    pause_duration_s: float,
    seed: int = 7,
) -> np.ndarray:
    """Insert a silence gap at the midpoint of the segment.

    Uses a tiny amount of room-tone noise to sound natural.
    """
    rng = np.random.RandomState(seed)

    pause_samples = int(pause_duration_s * sr)
    midpoint = len(segment) // 2

    # Estimate room-tone (low level dither ~ -50 dB)
    room_tone_len = min(int(0.05 * sr), len(segment))
    if room_tone_len > 0:
        room_tone_rms = float(np.sqrt(np.mean(segment[:room_tone_len] ** 2)))
        noise_level = min(5e-4, max(1e-5, room_tone_rms * 0.005))
    else:
        noise_level = 1e-4
    noise = rng.randn(pause_samples) * noise_level

    # Stitch: [first half] + [pause] + [second half]
    result = np.concatenate([
        segment[:midpoint],
        noise.astype(segment.dtype),
        segment[midpoint:],
    ])
    return result


# ---------------------------------------------------------------------------
# Flat Pitch: compress F0 contour toward sentence mean
# ---------------------------------------------------------------------------

def perturb_flat_pitch(
    segment: np.ndarray,
    sr: int,
    retained_variance_fraction: float,
) -> np.ndarray:
    """Flatten F0 contour by compressing toward the mean.

    retained_variance_fraction: 1.0 = no change, 0.0 = perfectly flat.
    Uses Praat Parselmouth (with pyworld fallback) for F0 analysis and synthesis.
    """
    if len(segment) < int(sr * 0.05):
        return segment.astype(np.float64)

    # 1. Primary: Praat Parselmouth (pre-compiled binary wheel, robust on all platforms)
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(segment.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch()
        med_f0 = call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        if np.isnan(med_f0) or med_f0 <= 50.0:
            med_f0 = 135.0

        dur = snd.get_total_duration()
        manip = call(snd, "To Manipulation", 0.01, 75, 600)
        pt = call(manip, "Extract pitch tier")

        # Flat line or compressed pitch tier
        call(pt, "Remove points between", 0.0, dur)
        if retained_variance_fraction <= 0.15:
            call(pt, "Add point", 0.0, float(med_f0))
            call(pt, "Add point", float(dur), float(med_f0))
        else:
            # Subtle intonation compression
            call(pt, "Add point", 0.0, float(med_f0 * (1.0 - 0.05 * retained_variance_fraction)))
            call(pt, "Add point", float(dur * 0.5), float(med_f0 * (1.0 + 0.05 * retained_variance_fraction)))
            call(pt, "Add point", float(dur), float(med_f0 * (1.0 - 0.05 * retained_variance_fraction)))

        call([pt, manip], "Replace pitch tier")
        resynth = call(manip, "Get resynthesis (overlap-add)")
        result = resynth.values[0]

        if len(result) > len(segment):
            result = result[:len(segment)]
        elif len(result) < len(segment):
            result = np.pad(result, (0, len(segment) - len(result)))
        return result.astype(np.float64)

    except Exception:
        pass

    # 2. Secondary fallback: pyworld
    try:
        import pyworld as pw
        segment_f64 = segment.astype(np.float64)
        f0, t = pw.harvest(segment_f64, sr)
        sp = pw.cheaptrick(segment_f64, f0, t, sr)
        ap = pw.d4c(segment_f64, f0, t, sr)
        voiced_mask = f0 > 0
        if voiced_mask.any():
            mean_f0 = np.mean(f0[voiced_mask])
            f0_new = f0.copy()
            f0_new[voiced_mask] = mean_f0 + retained_variance_fraction * (f0[voiced_mask] - mean_f0)
            f0 = f0_new
        result = pw.synthesize(f0, sp, ap, sr)
        if len(result) > len(segment):
            result = result[:len(segment)]
        elif len(result) < len(segment):
            result = np.pad(result, (0, len(segment) - len(result)))
        return result.astype(np.float64)
    except Exception:
        return (segment * (0.95 + 0.05 * retained_variance_fraction)).astype(np.float64)


# ---------------------------------------------------------------------------
# Mumbled Clarity: gain reduction + low-pass filter
# ---------------------------------------------------------------------------

def perturb_mumbled_clarity(
    segment: np.ndarray,
    sr: int,
    gain_reduction_db: float,
    lowpass_cutoff_hz: float,
) -> np.ndarray:
    """Reduce gain and apply low-pass filter to simulate mumbling.

    30ms crossfade at boundaries is handled by the caller.
    """
    # Apply gain reduction
    gain_linear = 10 ** (gain_reduction_db / 20.0)
    result = segment * gain_linear

    # Low-pass filter (4th order Butterworth)
    nyquist = sr / 2.0
    if lowpass_cutoff_hz < nyquist:
        normalized_cutoff = lowpass_cutoff_hz / nyquist
        sos = butter(4, normalized_cutoff, btype='low', output='sos')
        result = sosfilt(sos, result)

    return result


# ---------------------------------------------------------------------------
# Crossfade utility
# ---------------------------------------------------------------------------

def crossfade_stitch(
    before: np.ndarray,
    modified: np.ndarray,
    after: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Stitch three segments with smooth crossfade at boundaries."""
    if crossfade_samples <= 0 or crossfade_samples > len(before) or crossfade_samples > len(modified):
        return np.concatenate([before, modified, after])

    fade_in = np.linspace(0, 1, crossfade_samples)
    fade_out = np.linspace(1, 0, crossfade_samples)

    # Crossfade at before->modified boundary
    before_end = before[:-crossfade_samples]
    overlap_1 = before[-crossfade_samples:] * fade_out + modified[:crossfade_samples] * fade_in
    modified_mid = modified[crossfade_samples:]

    if crossfade_samples <= len(modified_mid) and crossfade_samples <= len(after):
        # Crossfade at modified->after boundary
        modified_body = modified_mid[:-crossfade_samples]
        overlap_2 = modified_mid[-crossfade_samples:] * fade_out + after[:crossfade_samples] * fade_in
        after_rest = after[crossfade_samples:]
        return np.concatenate([before_end, overlap_1, modified_body, overlap_2, after_rest])
    else:
        return np.concatenate([before_end, overlap_1, modified_mid, after])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PERTURBATION_FNS = {
    "rushed_pace": perturb_rushed_pace,
    "dead_pause": perturb_dead_pause,
    "flat_pitch": perturb_flat_pitch,
    "mumbled_clarity": perturb_mumbled_clarity,
}


def apply_perturbation(
    audio: np.ndarray,
    sr: int,
    flaw_type: str,
    severity: int,
    target_start_sample: int,
    target_end_sample: int,
    config: dict,
    crossfade_ms: int = 30,
    seed: int = 7,
) -> Tuple[np.ndarray, int, int]:
    """Apply a perturbation to a target region of a full audio clip.

    Returns: (perturbed_audio, new_target_start, new_target_end)
    """
    before = audio[:target_start_sample]
    segment = audio[target_start_sample:target_end_sample]
    after = audio[target_end_sample:]

    params = config["perturbations"][flaw_type]["severities"][str(severity)]
    crossfade_samples = int(crossfade_ms * sr / 1000)

    if flaw_type == "rushed_pace":
        modified = perturb_rushed_pace(segment, sr, speed_factor=params)
    elif flaw_type == "dead_pause":
        modified = perturb_dead_pause(segment, sr, pause_duration_s=params, seed=seed)
    elif flaw_type == "flat_pitch":
        modified = perturb_flat_pitch(segment, sr, retained_variance_fraction=params)
    elif flaw_type == "mumbled_clarity":
        gain_db, lp_cutoff = params
        modified = perturb_mumbled_clarity(segment, sr, gain_reduction_db=gain_db, lowpass_cutoff_hz=lp_cutoff)
    else:
        raise ValueError(f"Unknown flaw type: {flaw_type}")

    if len(modified) == len(segment):
        result = audio.copy()
        if crossfade_samples > 0 and crossfade_samples <= len(modified) // 2:
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            mod_blended = modified.copy()
            if target_start_sample > 0:
                orig_start = audio[target_start_sample:target_start_sample + crossfade_samples]
                mod_blended[:crossfade_samples] = orig_start * fade_out + modified[:crossfade_samples] * fade_in
            if target_end_sample < len(audio):
                orig_end = audio[target_end_sample - crossfade_samples:target_end_sample]
                mod_blended[-crossfade_samples:] = modified[-crossfade_samples:] * fade_out + orig_end * fade_in
            result[target_start_sample:target_end_sample] = mod_blended
        else:
            result[target_start_sample:target_end_sample] = modified
        return result, target_start_sample, target_end_sample

    result = crossfade_stitch(before, modified, after, crossfade_samples)

    new_start = target_start_sample
    new_end = target_start_sample + len(modified)

    return result, new_start, new_end
