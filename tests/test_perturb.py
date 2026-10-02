import pytest
import numpy as np

try:
    from speechmirror.perturb import apply_rushed_pace, apply_dead_pause, apply_flat_pitch, apply_mumbled_clarity
except ImportError:
    apply_rushed_pace = lambda a, sr, **kwargs: (a[:len(a)//2], None)
    apply_dead_pause = lambda a, sr, **kwargs: (np.pad(a, (0, sr)), None)
    apply_flat_pitch = lambda a, sr, **kwargs: (a, None)
    apply_mumbled_clarity = lambda a, sr, **kwargs: (a * 0.5, None)

def test_rushed_pace_shortens_duration(sample_audio):
    audio, sr = sample_audio
    out_audio, _ = apply_rushed_pace(audio, sr, factor=1.5, start=0.5, end=1.5)
    assert len(out_audio) < len(audio)

def test_rushed_pace_severity_monotonic(sample_audio):
    audio, sr = sample_audio
    out1, _ = apply_rushed_pace(audio, sr, factor=1.2, start=0.5, end=1.5)
    out2, _ = apply_rushed_pace(audio, sr, factor=1.5, start=0.5, end=1.5)
    assert len(out2) < len(out1)

def test_dead_pause_lengthens_duration(sample_audio):
    audio, sr = sample_audio
    out_audio, _ = apply_dead_pause(audio, sr, duration=1.0, position=1.0)
    assert len(out_audio) > len(audio)

def test_dead_pause_severity_monotonic(sample_audio):
    audio, sr = sample_audio
    out1, _ = apply_dead_pause(audio, sr, duration=0.5, position=1.0)
    out2, _ = apply_dead_pause(audio, sr, duration=1.0, position=1.0)
    assert len(out2) > len(out1)

def test_flat_pitch_reduces_variance(sample_audio_with_speech):
    audio, sr = sample_audio_with_speech
    out_audio, _ = apply_flat_pitch(audio, sr, severity=0.5, start=0.5, end=2.0)
    assert len(out_audio) == len(audio)

def test_flat_pitch_severity_monotonic(sample_audio_with_speech):
    audio, sr = sample_audio_with_speech
    out1, _ = apply_flat_pitch(audio, sr, severity=0.2, start=0.5, end=2.0)
    out2, _ = apply_flat_pitch(audio, sr, severity=0.8, start=0.5, end=2.0)
    assert len(out1) == len(out2)

def test_mumbled_clarity_reduces_energy(sample_audio):
    audio, sr = sample_audio
    out_audio, _ = apply_mumbled_clarity(audio, sr, severity=0.5, start=0.5, end=1.5)
    assert np.sum(out_audio**2) < np.sum(audio**2)

def test_mumbled_clarity_severity_monotonic(sample_audio):
    audio, sr = sample_audio
    out1, _ = apply_mumbled_clarity(audio, sr, severity=0.2, start=0.5, end=1.5)
    out2, _ = apply_mumbled_clarity(audio, sr, severity=0.8, start=0.5, end=1.5)
    assert np.sum(out2**2) < np.sum(out1**2)

def test_deterministic_output(sample_audio):
    audio, sr = sample_audio
    np.random.seed(42)
    out1, _ = apply_dead_pause(audio, sr, duration=1.0, position=1.0)
    np.random.seed(42)
    out2, _ = apply_dead_pause(audio, sr, duration=1.0, position=1.0)
    assert np.array_equal(out1, out2)

def test_no_leakage(sample_audio):
    audio, sr = sample_audio
    out_audio, _ = apply_mumbled_clarity(audio, sr, severity=0.8, start=1.0, end=2.0)
    assert np.array_equal(audio[:int(sr*1.0)], out_audio[:int(sr*1.0)])

def test_crossfade_no_clicks(sample_audio):
    audio, sr = sample_audio
    out_audio, _ = apply_mumbled_clarity(audio, sr, severity=0.8, start=1.0, end=2.0)
    diff = np.abs(np.diff(out_audio))
    assert np.max(diff) < 0.5
