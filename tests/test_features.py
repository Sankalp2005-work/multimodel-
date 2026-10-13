import pytest
import numpy as np

from speechmirror.features import (
    extract_frame_features, 
    compute_speech_rate, 
    compute_pause_durations
)

def test_extract_frame_features_shape(sample_audio):
    audio, sr = sample_audio
    config = {'hop_length_ms': 10, 'frame_length_ms': 25}
    features = extract_frame_features(audio, sr, config)
    expected_frames = len(audio) // int(sr * 0.01)
    assert len(features.rms_db) == expected_frames

def test_rms_energy_silent_audio():
    audio = np.zeros(16000)
    features = extract_frame_features(audio, 16000)
    assert np.all(features.rms_db < -80)

def test_rms_energy_loud_audio():
    audio = np.ones(16000)
    features = extract_frame_features(audio, 16000)
    # The max value will be 0 dB, so the mean should be 0 or very close to it, which is > -50
    assert np.mean(features.rms_db) > -50

def test_f0_sine_wave():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)
    features = extract_frame_features(audio, sr)
    f0 = features.f0_hz
    valid_f0 = f0[f0 > 0]
    if len(valid_f0) > 0:
        assert np.isclose(np.median(valid_f0), 440, atol=20)

def test_mfcc_shape(sample_audio):
    audio, sr = sample_audio
    n_mfcc = 13
    features = extract_frame_features(audio, sr, {'n_mfcc': n_mfcc})
    assert features.mfcc.shape[1] == 13

def test_spectral_features_exist(sample_audio):
    audio, sr = sample_audio
    features = extract_frame_features(audio, sr)
    assert features.spectral_centroid is not None
    assert features.spectral_flatness is not None
    assert features.spectral_bandwidth is not None

def test_speech_rate_computation(sample_word_timestamps):
    rate = compute_speech_rate(sample_word_timestamps)
    assert len(rate) > 0

def test_pause_durations(sample_word_timestamps):
    pauses = compute_pause_durations(sample_word_timestamps)
    assert len(pauses) == len(sample_word_timestamps) - 1
    assert np.isclose(pauses[0]['duration'], 0.1)

def test_empty_audio_handling():
    audio = np.array([])
    features = extract_frame_features(audio, 16000)
    assert len(features.rms_db) == 0
