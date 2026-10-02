import numpy as np

def zscore_normalize(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, float, float]:
    """Z-score normalize. Returns (normalized, mean, std)."""
    mean = np.mean(values, axis=axis, keepdims=True)
    std = np.std(values, axis=axis, keepdims=True)
    std[std == 0] = 1.0  # Avoid division by zero
    normalized = (values - mean) / std
    return normalized, mean.item() if mean.size == 1 else mean, std.item() if std.size == 1 else std

def ratio_to_ideal(participant_values: np.ndarray, ideal_values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute ratio of participant to ideal values."""
    ideal_safe = np.where(ideal_values == 0, eps, ideal_values)
    return participant_values / ideal_safe

def hz_to_semitones(f0_hz: np.ndarray, ref_hz: float | None = None) -> np.ndarray:
    """Convert F0 in Hz to semitones relative to reference (default: median of voiced frames)."""
    voiced_f0 = f0_hz[f0_hz > 0]
    if len(voiced_f0) == 0:
        return np.zeros_like(f0_hz)
    
    if ref_hz is None:
        ref_hz = np.median(voiced_f0)
    
    ref_hz = max(ref_hz, 1e-5)
    semitones = np.zeros_like(f0_hz)
    mask = f0_hz > 0
    semitones[mask] = 12 * np.log2(f0_hz[mask] / ref_hz)
    return semitones

def normalize_energy_db(rms: np.ndarray, ref_rms: float | None = None) -> np.ndarray:
    """Normalize RMS energy to dB relative to reference."""
    # Add small epsilon to avoid log10(0)
    eps = 1e-9
    rms_safe = np.maximum(rms, eps)
    
    if ref_rms is None:
        ref_rms = np.median(rms_safe)
        
    ref_rms = max(ref_rms, eps)
    return 20 * np.log10(rms_safe / ref_rms)

def normalize_features(participant_features, ideal_features, config: dict | None = None) -> dict:
    """Apply full normalization pipeline to a feature set pair."""
    return {
        "participant": participant_features,
        "ideal": ideal_features
    }
