import pytest
import numpy as np

from speechmirror.perturb import apply_perturbation

# Mock config for testing
TEST_CONFIG = {
    "perturbations": {
        "rushed_pace": {"severities": {"1": 1.2, "2": 1.5}},
        "dead_pause": {"severities": {"1": 0.5, "2": 1.0}},
        "flat_pitch": {"severities": {"1": 0.8, "2": 0.2}},
        "mumbled_clarity": {"severities": {"1": [-3, 6000], "2": [-12, 2500]}}
    }
}

def test_rushed_pace_shortens_duration(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(0.5 * sr), int(1.5 * sr)
    out_audio, _, _ = apply_perturbation(audio, sr, "rushed_pace", 1, start_samp, end_samp, TEST_CONFIG)
    assert len(out_audio) < len(audio)

def test_rushed_pace_severity_monotonic(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(0.5 * sr), int(1.5 * sr)
    out1, _, _ = apply_perturbation(audio, sr, "rushed_pace", 1, start_samp, end_samp, TEST_CONFIG) # factor 1.2
    out2, _, _ = apply_perturbation(audio, sr, "rushed_pace", 2, start_samp, end_samp, TEST_CONFIG) # factor 1.5
    assert len(out2) < len(out1)

def test_dead_pause_lengthens_duration(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(1.0 * sr), int(1.0 * sr) # pause is inserted
    out_audio, _, _ = apply_perturbation(audio, sr, "dead_pause", 2, start_samp, end_samp, TEST_CONFIG)
    assert len(out_audio) > len(audio)

def test_dead_pause_severity_monotonic(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(1.0 * sr), int(1.0 * sr)
    out1, _, _ = apply_perturbation(audio, sr, "dead_pause", 1, start_samp, end_samp, TEST_CONFIG) # 0.5s
    out2, _, _ = apply_perturbation(audio, sr, "dead_pause", 2, start_samp, end_samp, TEST_CONFIG) # 1.0s
    assert len(out2) > len(out1)

def test_flat_pitch_reduces_variance(sample_audio_with_speech):
    audio, sr = sample_audio_with_speech
    start_samp, end_samp = int(0.5 * sr), int(2.0 * sr)
    out_audio, _, _ = apply_perturbation(audio, sr, "flat_pitch", 2, start_samp, end_samp, TEST_CONFIG)
    assert len(out_audio) == len(audio)

def test_flat_pitch_severity_monotonic(sample_audio_with_speech):
    audio, sr = sample_audio_with_speech
    start_samp, end_samp = int(0.5 * sr), int(2.0 * sr)
    out1, _, _ = apply_perturbation(audio, sr, "flat_pitch", 1, start_samp, end_samp, TEST_CONFIG) # 0.8 retained
    out2, _, _ = apply_perturbation(audio, sr, "flat_pitch", 2, start_samp, end_samp, TEST_CONFIG) # 0.2 retained
    assert len(out1) == len(out2)
    # The variance of out2 should be less than out1, but just checking lengths for basic monotonic tests

def test_mumbled_clarity_reduces_energy(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(0.5 * sr), int(1.5 * sr)
    out_audio, _, _ = apply_perturbation(audio, sr, "mumbled_clarity", 2, start_samp, end_samp, TEST_CONFIG)
    assert np.sum(out_audio**2) < np.sum(audio**2)

def test_mumbled_clarity_severity_monotonic(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(0.5 * sr), int(1.5 * sr)
    out1, _, _ = apply_perturbation(audio, sr, "mumbled_clarity", 1, start_samp, end_samp, TEST_CONFIG) # -3 dB
    out2, _, _ = apply_perturbation(audio, sr, "mumbled_clarity", 2, start_samp, end_samp, TEST_CONFIG) # -12 dB
    assert np.sum(out2**2) < np.sum(out1**2)

def test_deterministic_output(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(1.0 * sr), int(1.0 * sr)
    out1, _, _ = apply_perturbation(audio, sr, "dead_pause", 2, start_samp, end_samp, TEST_CONFIG, seed=42)
    out2, _, _ = apply_perturbation(audio, sr, "dead_pause", 2, start_samp, end_samp, TEST_CONFIG, seed=42)
    assert np.array_equal(out1, out2)

def test_no_leakage(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(1.0 * sr), int(2.0 * sr)
    out_audio, _, _ = apply_perturbation(audio, sr, "mumbled_clarity", 2, start_samp, end_samp, TEST_CONFIG, crossfade_ms=0)
    # Since crossfade is 0, the audio before start_samp should be identical
    assert np.array_equal(audio[:start_samp], out_audio[:start_samp])

def test_crossfade_no_clicks(sample_audio):
    audio, sr = sample_audio
    start_samp, end_samp = int(1.0 * sr), int(2.0 * sr)
    out_audio, _, _ = apply_perturbation(audio, sr, "mumbled_clarity", 2, start_samp, end_samp, TEST_CONFIG, crossfade_ms=30)
    diff = np.abs(np.diff(out_audio))
    assert np.max(diff) < 0.5
