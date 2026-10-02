"""
Acoustic feature extraction module for SpeechMirror.

Extracts per-frame and per-word acoustic features from audio recordings.
Supported features: RMS energy, F0/pitch, MFCC, spectral centroid/flatness/bandwidth, ZCR,
speech rate, pause durations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import pyworld as pw

logger = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    """Per-frame acoustic features."""
    times: np.ndarray
    rms_db: np.ndarray
    f0_hz: np.ndarray
    f0_log_st: np.ndarray
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    spectral_centroid: np.ndarray
    spectral_flatness: np.ndarray
    spectral_bandwidth: np.ndarray
    zcr: np.ndarray


@dataclass
class WordFeatures:
    """Per-word acoustic features."""
    word: str
    start: float
    end: float
    duration: float
    confidence: Optional[float]
    mean_rms_db: float
    mean_f0_hz: float
    mean_f0_log_st: float
    mean_spectral_centroid: float
    mean_spectral_flatness: float
    mean_spectral_bandwidth: float
    zcr: float


@dataclass
class Pause:
    """Pause duration between words."""
    start: float
    end: float
    duration: float
    previous_word: str
    next_word: str


@dataclass
class FeatureSet:
    """Complete feature set for an audio recording."""
    frame_features: FrameFeatures
    word_features: Optional[List[WordFeatures]] = None
    speech_rate_window: Optional[List[float]] = None
    pause_durations: Optional[List[Pause]] = None


def hz_to_log_st(hz: np.ndarray, ref_hz: float = 50.0) -> np.ndarray:
    """Convert Hz to log-semitones."""
    st = np.zeros_like(hz)
    mask = hz > 0
    if np.any(mask):
        st[mask] = 12.0 * np.log2(hz[mask] / ref_hz)
    return st


def extract_frame_features(audio: np.ndarray, sr: int = 16000, config: Optional[Dict[str, Any]] = None) -> FrameFeatures:
    """
    Extract all per-frame features from an audio signal.
    
    Args:
        audio: Audio signal array.
        sr: Sample rate.
        config: Configuration dictionary (overrides defaults).
        
    Returns:
        FrameFeatures containing extracted features.
    """
    if config is None:
        config = {}
        
    frame_length_ms = config.get('frame_length_ms', 25)
    hop_length_ms = config.get('hop_length_ms', 10)
    n_mfcc = config.get('n_mfcc', 13)
    
    n_fft = int(sr * frame_length_ms / 1000.0)
    hop_length = int(sr * hop_length_ms / 1000.0)
    
    if len(audio) == 0:
        logger.warning("Empty audio array provided.")
        empty_arr = np.array([])
        return FrameFeatures(
            times=empty_arr, rms_db=empty_arr, f0_hz=empty_arr, f0_log_st=empty_arr,
            mfcc=empty_arr, mfcc_delta=empty_arr, mfcc_delta2=empty_arr,
            spectral_centroid=empty_arr, spectral_flatness=empty_arr,
            spectral_bandwidth=empty_arr, zcr=empty_arr
        )
    
    # Pad audio if too short for frame length
    if len(audio) < n_fft:
        logger.warning("Audio shorter than frame length. Padding.")
        audio = np.pad(audio, (0, n_fft - len(audio)))
        
    # Pyworld F0 (dio + stonemask)
    audio_f64 = audio.astype(np.float64)
    _f0_hz, _times = pw.dio(audio_f64, sr, frame_period=hop_length_ms)
    f0_hz = pw.stonemask(audio_f64, _f0_hz, _times, sr)
    f0_log_st = hz_to_log_st(f0_hz)
    
    # Librosa features
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, center=True)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length, center=True)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True)[0]
    
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    
    # Align lengths
    n_frames_pw = len(_times)
    
    def pad_or_trunc(arr: np.ndarray) -> np.ndarray:
        n = arr.shape[-1]
        if n < n_frames_pw:
            if arr.ndim == 1:
                return np.pad(arr, (0, n_frames_pw - n), mode='edge')
            else:
                return np.pad(arr, ((0, 0), (0, n_frames_pw - n)), mode='edge')
        elif n > n_frames_pw:
            if arr.ndim == 1:
                return arr[:n_frames_pw]
            else:
                return arr[:, :n_frames_pw]
        return arr
    
    return FrameFeatures(
        times=_times,
        rms_db=pad_or_trunc(rms_db),
        f0_hz=f0_hz,
        f0_log_st=f0_log_st,
        mfcc=pad_or_trunc(mfcc).T,
        mfcc_delta=pad_or_trunc(mfcc_delta).T,
        mfcc_delta2=pad_or_trunc(mfcc_delta2).T,
        spectral_centroid=pad_or_trunc(spectral_centroid),
        spectral_flatness=pad_or_trunc(spectral_flatness),
        spectral_bandwidth=pad_or_trunc(spectral_bandwidth),
        zcr=pad_or_trunc(zcr)
    )

def extract_word_features(audio: np.ndarray, sr: int, word_timestamps: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> List[WordFeatures]:
    """
    Extract per-word features using word-level timestamps from alignment.
    
    Args:
        audio: Audio signal array.
        sr: Sample rate.
        word_timestamps: List of dicts with 'word', 'start', 'end', 'confidence'.
        config: Configuration dictionary.
        
    Returns:
        List of WordFeatures.
    """
    word_features = []
    frames = extract_frame_features(audio, sr, config)
    
    for wt in word_timestamps:
        start_t = wt.get('start', 0.0)
        end_t = wt.get('end', 0.0)
        word = wt.get('word', '')
        conf = wt.get('confidence')
        
        mask = (frames.times >= start_t) & (frames.times <= end_t)
        
        if not np.any(mask):
            logger.warning(f"No frames found for word '{word}' between {start_t} and {end_t}.")
            word_features.append(WordFeatures(
                word=word, start=start_t, end=end_t, duration=max(0, end_t - start_t), confidence=conf,
                mean_rms_db=-100.0, mean_f0_hz=0.0, mean_f0_log_st=0.0,
                mean_spectral_centroid=0.0, mean_spectral_flatness=0.0,
                mean_spectral_bandwidth=0.0, zcr=0.0
            ))
            continue
            
        mean_rms_db = float(np.mean(frames.rms_db[mask]))
        
        voiced_mask = mask & (frames.f0_hz > 0)
        if np.any(voiced_mask):
            mean_f0_hz = float(np.mean(frames.f0_hz[voiced_mask]))
            mean_f0_log_st = float(np.mean(frames.f0_log_st[voiced_mask]))
        else:
            mean_f0_hz = 0.0
            mean_f0_log_st = 0.0
            
        mean_spectral_centroid = float(np.mean(frames.spectral_centroid[mask]))
        mean_spectral_flatness = float(np.mean(frames.spectral_flatness[mask]))
        mean_spectral_bandwidth = float(np.mean(frames.spectral_bandwidth[mask]))
        zcr = float(np.mean(frames.zcr[mask]))
        
        word_features.append(WordFeatures(
            word=word, start=start_t, end=end_t, duration=max(0, end_t - start_t), confidence=conf,
            mean_rms_db=mean_rms_db, mean_f0_hz=mean_f0_hz, mean_f0_log_st=mean_f0_log_st,
            mean_spectral_centroid=mean_spectral_centroid, mean_spectral_flatness=mean_spectral_flatness,
            mean_spectral_bandwidth=mean_spectral_bandwidth, zcr=zcr
        ))
        
    return word_features


def compute_speech_rate(word_timestamps: List[Dict[str, Any]], window_size: int = 5) -> List[float]:
    """
    Compute local speech rate (words/sec) over a sliding window.
    
    Args:
        word_timestamps: List of dicts with 'word', 'start', 'end'.
        window_size: Number of words in the sliding window.
        
    Returns:
        List of local speech rates.
    """
    if len(word_timestamps) < 2:
        return []
        
    rates = []
    n_words = len(word_timestamps)
    
    for i in range(n_words):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_words - 1, i + window_size // 2)
        
        window_words = end_idx - start_idx + 1
        window_start = word_timestamps[start_idx].get('start', 0.0)
        window_end = word_timestamps[end_idx].get('end', 0.0)
        window_duration = max(0.001, window_end - window_start)
        
        rate = window_words / window_duration
        rates.append(rate)
        
    return rates


def compute_pause_durations(word_timestamps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute pause durations between consecutive words.
    
    Args:
        word_timestamps: List of dicts with 'word', 'start', 'end'.
        
    Returns:
        List of pause dictionaries.
    """
    pauses = []
    
    for i in range(1, len(word_timestamps)):
        prev_word = word_timestamps[i - 1]
        curr_word = word_timestamps[i]
        
        end_t = prev_word.get('end', 0.0)
        start_t = curr_word.get('start', 0.0)
        
        pause_duration = start_t - end_t
        if pause_duration > 0:
            pauses.append({
                "start": end_t,
                "end": start_t,
                "duration": pause_duration,
                "previous_word": prev_word.get('word', ''),
                "next_word": curr_word.get('word', '')
            })
            
    return pauses


def extract_all(audio: np.ndarray, sr: int, word_timestamps: Optional[List[Dict[str, Any]]] = None, config: Optional[Dict[str, Any]] = None) -> FeatureSet:
    """
    Extract complete feature set from audio.
    
    Args:
        audio: Audio signal array.
        sr: Sample rate.
        word_timestamps: Optional word-level timestamps.
        config: Configuration dictionary.
        
    Returns:
        FeatureSet containing all extracted features.
    """
    frame_features = extract_frame_features(audio, sr, config)
    
    word_features = None
    speech_rate_window = None
    pause_durations = None
    
    if word_timestamps:
        word_features = extract_word_features(audio, sr, word_timestamps, config)
        speech_rate_window = compute_speech_rate(word_timestamps)
        # Convert pauses dict back to Pause dataclass or change FeatureSet to List[Dict]
        pause_dicts = compute_pause_durations(word_timestamps)
        pause_durations = [Pause(**p) for p in pause_dicts]
        
    return FeatureSet(
        frame_features=frame_features,
        word_features=word_features,
        speech_rate_window=speech_rate_window,
        pause_durations=pause_durations
    )
