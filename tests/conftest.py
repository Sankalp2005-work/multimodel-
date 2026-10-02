import pytest
import numpy as np
import tempfile
import soundfile as sf
import json
import os

@pytest.fixture
def sample_audio():
    """Generate a synthetic 3-second audio signal (440Hz sine + harmonics) at 16kHz."""
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    # 440Hz sine + 880Hz harmonic
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    return audio, sr

@pytest.fixture
def sample_audio_with_speech():
    """Approximate speech-like signal with varying F0 and energy."""
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    # Varying F0 and energy
    f0 = 200 + 50 * np.sin(2 * np.pi * 1 * t)
    audio = 0.5 * (1 + 0.5 * np.sin(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * f0 * t)
    return audio, sr

@pytest.fixture
def sample_word_timestamps():
    """Mock word timestamps for 5 words."""
    return [
        {"word": "hello", "start": 0.1, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
        {"word": "this", "start": 1.0, "end": 1.3},
        {"word": "is", "start": 1.3, "end": 1.5},
        {"word": "test", "start": 1.6, "end": 2.1},
    ]

@pytest.fixture
def sample_config():
    """Load or create a mock pipeline.json config."""
    return {
        "audio": {"sr": 16000, "mono": True},
        "features": {"hop_length": 160, "win_length": 400},
        "detectors": {
            "rushed_pace": {"threshold": 1.5},
            "dead_pause": {"threshold": 1.0}
        }
    }

@pytest.fixture
def perturb_config():
    """Load or create a mock perturb.json config."""
    return {
        "rushed_pace": {"factors": [0.5, 0.7, 0.9]},
        "dead_pause": {"durations": [0.5, 1.0, 2.0]},
        "flat_pitch": {"variances": [0.1, 0.5]},
        "mumbled_clarity": {"energy_drops": [0.5, 0.2]}
    }

@pytest.fixture
def tmp_audio_file(sample_audio):
    """Write sample audio to temp WAV file."""
    audio, sr = sample_audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        path = f.name
    yield path
    os.remove(path)
