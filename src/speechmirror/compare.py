import numpy as np
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import librosa
from tslearn.metrics import dtw_path

# Assuming these are available from the project
# from .align import AlignmentResult, WordAlignment

@dataclass
class WordAlignment:
    word: str
    start: float
    end: float

@dataclass
class AlignmentResult:
    words: list[WordAlignment]

@dataclass
class WordComparison:
    word: str
    word_index: int
    ideal_start: float
    ideal_end: float
    participant_start: float
    participant_end: float
    # Feature deltas
    speech_rate_ratio: float  # participant / ideal duration ratio
    pause_excess_s: float     # excess pause before this word vs ideal
    f0_variance_ratio: float  # participant F0 var / ideal F0 var
    f0_range_ratio: float     # participant F0 range / ideal F0 range
    energy_delta_db: float    # participant energy - ideal energy (dB)
    spectral_centroid_ratio: float
    spectral_flatness_delta: float
    mfcc_distance: float      # DTW-aligned cosine distance
    zcr_delta: float

@dataclass
class SentenceComparison:
    sentence_index: int
    sentence_text: str
    words: list[WordComparison]
    aggregate_deltas: dict[str, float]

def compute_mfcc_distance(
    ideal_mfcc: np.ndarray,
    participant_mfcc: np.ndarray,
    method: str = 'cosine'
) -> float:
    """DTW-aligned MFCC distance between two word segments."""
    if ideal_mfcc.size == 0 or participant_mfcc.size == 0:
        return 0.0
    
    if ideal_mfcc.ndim == 1:
        ideal_mfcc = ideal_mfcc.reshape(-1, 1)
    if participant_mfcc.ndim == 1:
        participant_mfcc = participant_mfcc.reshape(-1, 1)
        
    path, dist = dtw_path(ideal_mfcc, participant_mfcc, global_constraint=None)
    
    # Calculate cosine distance along the DTW path
    dists = []
    for i, j in path:
        u = ideal_mfcc[i]
        v = participant_mfcc[j]
        # cosine distance
        uv = np.dot(u, v)
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        if uu == 0 or vv == 0:
            d = 0.0
        else:
            d = 1.0 - uv / np.sqrt(uu * vv)
        dists.append(d)
        
    return float(np.mean(dists)) if dists else 0.0

def compute_word_duration_ratio(
    ideal_word: WordAlignment,
    participant_word: WordAlignment
) -> float:
    """Compute duration ratio between participant and ideal word."""
    ideal_dur = ideal_word.end - ideal_word.start
    part_dur = participant_word.end - participant_word.start
    if ideal_dur <= 0:
        return 1.0
    return part_dur / ideal_dur

def extract_word_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    start_idx = int(start * sr)
    end_idx = int(end * sr)
    return audio[start_idx:end_idx]

def extract_word_features(audio_segment: np.ndarray, sr: int) -> dict:
    if len(audio_segment) == 0:
        return {
            'mfcc': np.zeros((13, 1)),
            'f0': np.zeros(1),
            'rms': np.zeros(1),
            'centroid': np.zeros(1),
            'flatness': np.zeros(1),
            'zcr': np.zeros(1)
        }
    
    # Extract basic features for the word
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13).T
    f0, _, _ = librosa.pyin(y=audio_segment, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    rms = librosa.feature.rms(y=audio_segment)[0]
    centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=audio_segment)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio_segment)[0]
    
    return {
        'mfcc': mfcc,
        'f0': f0,
        'rms': rms,
        'centroid': centroid,
        'flatness': flatness,
        'zcr': zcr
    }

def compare_aligned_pair(
    ideal_audio: np.ndarray,
    participant_audio: np.ndarray,
    sr: int,
    ideal_alignment: AlignmentResult,
    participant_alignment: AlignmentResult,
    sentences: list[str],
    config: dict | None = None
) -> list[SentenceComparison]:
    """Compare participant vs ideal using word-aligned features."""
    from .normalize import normalize_energy_db
    
    if config is None:
        config = {}
        
    comparisons = []
    
    ideal_words = ideal_alignment.words
    participant_words = participant_alignment.words
    
    # Align word sequences
    # Assuming words match perfectly in order
    
    word_comps = []
    ideal_prev_end = 0.0
    part_prev_end = 0.0
    
    for i, (iw, pw) in enumerate(zip(ideal_words, participant_words)):
        if iw.word != pw.word:
            # Handle mismatch if necessary
            pass
            
        # Extract audio segments
        i_audio = extract_word_audio(ideal_audio, sr, iw.start, iw.end)
        p_audio = extract_word_audio(participant_audio, sr, pw.start, pw.end)
        
        i_feats = extract_word_features(i_audio, sr)
        p_feats = extract_word_features(p_audio, sr)
        
        dur_ratio = compute_word_duration_ratio(iw, pw)
        
        ideal_pause = max(0.0, iw.start - ideal_prev_end)
        part_pause = max(0.0, pw.start - part_prev_end)
        pause_excess = part_pause - ideal_pause
        
        # F0 stats
        i_f0_var = np.var(i_feats['f0'][i_feats['f0'] > 0]) if np.any(i_feats['f0'] > 0) else 0.0
        p_f0_var = np.var(p_feats['f0'][p_feats['f0'] > 0]) if np.any(p_feats['f0'] > 0) else 0.0
        f0_var_ratio = p_f0_var / i_f0_var if i_f0_var > 0 else 1.0
        
        i_f0_range = np.ptp(i_feats['f0'][i_feats['f0'] > 0]) if np.any(i_feats['f0'] > 0) else 0.0
        p_f0_range = np.ptp(p_feats['f0'][p_feats['f0'] > 0]) if np.any(p_feats['f0'] > 0) else 0.0
        f0_range_ratio = p_f0_range / i_f0_range if i_f0_range > 0 else 1.0
        
        # Energy
        i_energy = normalize_energy_db(np.array([np.mean(i_feats['rms'])]))[0]
        p_energy = normalize_energy_db(np.array([np.mean(p_feats['rms'])]))[0]
        energy_delta = p_energy - i_energy
        
        # Spectral
        i_cent = np.mean(i_feats['centroid'])
        p_cent = np.mean(p_feats['centroid'])
        cent_ratio = p_cent / i_cent if i_cent > 0 else 1.0
        
        flat_delta = np.mean(p_feats['flatness']) - np.mean(i_feats['flatness'])
        zcr_delta = np.mean(p_feats['zcr']) - np.mean(i_feats['zcr'])
        
        mfcc_dist = compute_mfcc_distance(i_feats['mfcc'], p_feats['mfcc'])
        
        wc = WordComparison(
            word=iw.word,
            word_index=i,
            ideal_start=iw.start,
            ideal_end=iw.end,
            participant_start=pw.start,
            participant_end=pw.end,
            speech_rate_ratio=dur_ratio,
            pause_excess_s=pause_excess,
            f0_variance_ratio=f0_var_ratio,
            f0_range_ratio=f0_range_ratio,
            energy_delta_db=energy_delta,
            spectral_centroid_ratio=cent_ratio,
            spectral_flatness_delta=flat_delta,
            mfcc_distance=mfcc_dist,
            zcr_delta=zcr_delta
        )
        word_comps.append(wc)
        
        ideal_prev_end = iw.end
        part_prev_end = pw.end
        
    # Group into sentences (mocking sentence grouping for now)
    sc = SentenceComparison(
        sentence_index=0,
        sentence_text=" ".join(sentences) if sentences else "",
        words=word_comps,
        aggregate_deltas={}
    )
    comparisons.append(sc)
    
    return comparisons
