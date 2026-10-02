import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple, Optional, List

import librosa
import numpy as np
import scipy.signal
import pyworld as pw

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    audio: np.ndarray
    sr: int
    flaw_type: str
    severity: int
    flaw_start_sec: float
    flaw_end_sec: float
    params: dict = field(default_factory=dict)
    seed: int = 42


class PerturbationEngine:
    """
    Engine for introducing controlled synthetic flaws into ideal speech recordings.
    """

    def __init__(self, config_path: str = "configs/perturb.json", seed: int = 7):
        self.seed = seed
        self.config_path = Path(config_path)
        self.config = self._load_config()
        # Set seeds
        np.random.seed(self.seed)

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Default configuration based on specs
        return {
            "rushed_pace": {
                "1": {"speed_factor": 1.20},
                "2": {"speed_factor": 1.40},
                "3": {"speed_factor": 1.60},
                "4": {"speed_factor": 1.80}
            },
            "dead_pause": {
                "1": {"duration_s": 0.5},
                "2": {"duration_s": 1.0},
                "3": {"duration_s": 1.7},
                "4": {"duration_s": 2.5}
            },
            "flat_pitch": {
                "1": {"retained_variance_pct": 75.0},
                "2": {"retained_variance_pct": 50.0},
                "3": {"retained_variance_pct": 25.0},
                "4": {"retained_variance_pct": 5.0}
            },
            "mumbled_clarity": {
                "1": {"gain_db": -3.0, "lowpass_hz": 6000},
                "2": {"gain_db": -6.0, "lowpass_hz": 4500},
                "3": {"gain_db": -9.0, "lowpass_hz": 3500},
                "4": {"gain_db": -12.0, "lowpass_hz": 2500}
            }
        }

    def _crossfade(self, orig: np.ndarray, mod: np.ndarray, sr: int, start_idx: int, end_idx: int, fade_len_s: float = 0.03) -> np.ndarray:
        """Applies a crossfade between original and modified audio at the specified boundaries."""
        fade_samples = int(fade_len_s * sr)
        if fade_samples == 0:
            return mod

        out = mod.copy()
        
        # Crossfade at start (fade out original, fade in mod)
        if start_idx > 0:
            fade_start_idx = max(0, start_idx - fade_samples // 2)
            fade_end_idx = min(len(orig), start_idx + fade_samples // 2)
            actual_fade_len = fade_end_idx - fade_start_idx
            if actual_fade_len > 0:
                fade_in = np.linspace(0, 1, actual_fade_len)
                fade_out = 1.0 - fade_in
                out[fade_start_idx:fade_end_idx] = orig[fade_start_idx:fade_end_idx] * fade_out + mod[fade_start_idx:fade_end_idx] * fade_in
        
        # Crossfade at end (fade out mod, fade in original)
        if end_idx < len(orig):
            fade_start_idx = max(0, end_idx - fade_samples // 2)
            fade_end_idx = min(len(orig), end_idx + fade_samples // 2)
            actual_fade_len = fade_end_idx - fade_start_idx
            if actual_fade_len > 0:
                fade_out = np.linspace(1, 0, actual_fade_len)
                fade_in = 1.0 - fade_out
                out[fade_start_idx:fade_end_idx] = mod[fade_start_idx:fade_end_idx] * fade_out + orig[fade_start_idx:fade_end_idx] * fade_in
                
        return out

    def apply(self, audio: np.ndarray, sr: int, flaw_type: str, severity: int, 
              sentence_start: float, sentence_end: float, word_boundary: Optional[float] = None) -> PerturbationResult:
        """
        Applies a specified perturbation to the audio.
        """
        if flaw_type not in self.config:
            raise ValueError(f"Unknown flaw type: {flaw_type}")
        
        sev_str = str(severity)
        if sev_str not in self.config[flaw_type]:
            raise ValueError(f"Unknown severity {severity} for flaw type {flaw_type}")
            
        params = self.config[flaw_type][sev_str]
        
        np.random.seed(self.seed)
        
        if flaw_type == "rushed_pace":
            out_audio, flaw_start, flaw_end = self.apply_rushed_pace(
                audio, sr, sentence_start, sentence_end, params["speed_factor"]
            )
        elif flaw_type == "dead_pause":
            # If word_boundary not provided, default to middle of sentence
            insert_point = word_boundary if word_boundary is not None else (sentence_start + sentence_end) / 2.0
            out_audio, flaw_start, flaw_end = self.apply_dead_pause(
                audio, sr, insert_point, params["duration_s"]
            )
        elif flaw_type == "flat_pitch":
            out_audio, flaw_start, flaw_end = self.apply_flat_pitch(
                audio, sr, sentence_start, sentence_end, params["retained_variance_pct"]
            )
        elif flaw_type == "mumbled_clarity":
            out_audio, flaw_start, flaw_end = self.apply_mumbled_clarity(
                audio, sr, sentence_start, sentence_end, params["gain_db"], params["lowpass_hz"]
            )
        else:
            raise NotImplementedError(f"Flaw type {flaw_type} not implemented")

        return PerturbationResult(
            audio=out_audio,
            sr=sr,
            flaw_type=flaw_type,
            severity=severity,
            flaw_start_sec=float(np.round(flaw_start, 3)),
            flaw_end_sec=float(np.round(flaw_end, 3)),
            params=params,
            seed=self.seed
        )

    def apply_rushed_pace(self, audio: np.ndarray, sr: int, start: float, end: float, speed_factor: float) -> Tuple[np.ndarray, float, float]:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        prefix = audio[:start_idx]
        target = audio[start_idx:end_idx]
        suffix = audio[end_idx:]
        
        if len(target) == 0:
            return audio, start, end
            
        # Time stretch using librosa (uses phase vocoder, preserves pitch)
        stretched = librosa.effects.time_stretch(y=target, rate=speed_factor)
        
        # Combine
        out_audio = np.concatenate([prefix, stretched, suffix])
        
        new_end = start + (len(stretched) / sr)
        return out_audio, start, new_end

    def apply_dead_pause(self, audio: np.ndarray, sr: int, insertion_point: float, duration_s: float) -> Tuple[np.ndarray, float, float]:
        insert_idx = int(insertion_point * sr)
        pause_samples = int(duration_s * sr)
        
        prefix = audio[:insert_idx]
        suffix = audio[insert_idx:]
        
        # Create silence with tiny room tone (5% of RMS)
        rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0
        noise_amp = rms * 0.05
        room_tone = np.random.randn(pause_samples) * noise_amp
        
        # Combine
        out_audio = np.concatenate([prefix, room_tone, suffix])
        
        return out_audio, insertion_point, insertion_point + duration_s

    def apply_flat_pitch(self, audio: np.ndarray, sr: int, start: float, end: float, retained_variance_pct: float) -> Tuple[np.ndarray, float, float]:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        # Ensure float64 for pyworld
        target = audio[start_idx:end_idx].astype(np.float64)
        
        if len(target) < sr * 0.1: # too short
            return audio, start, end
            
        # Extract features
        _f0, t = pw.dio(target, sr)
        f0 = pw.stonemask(target, _f0, t, sr)
        sp = pw.cheaptrick(target, f0, t, sr)
        ap = pw.d4c(target, f0, t, sr)
        
        # Compress F0 towards mean
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) > 0:
            mean_f0 = np.mean(valid_f0)
            factor = retained_variance_pct / 100.0
            
            # Apply compression only to voiced regions
            mod_f0 = f0.copy()
            voiced = f0 > 0
            mod_f0[voiced] = mean_f0 + (f0[voiced] - mean_f0) * factor
            
            # Synthesize
            synthesized = pw.synthesize(mod_f0, sp, ap, sr)
            
            # Match lengths (pyworld can sometimes return slightly different lengths)
            min_len = min(len(target), len(synthesized))
            mod_target = target.copy()
            mod_target[:min_len] = synthesized[:min_len]
            
            # Put back into audio and crossfade
            out_audio = audio.copy().astype(np.float64)
            out_audio[start_idx:end_idx] = mod_target
            
            out_audio = self._crossfade(audio.astype(np.float64), out_audio, sr, start_idx, end_idx, fade_len_s=0.03)
            return out_audio.astype(np.float32), start, end
            
        return audio, start, end

    def apply_mumbled_clarity(self, audio: np.ndarray, sr: int, start: float, end: float, gain_db: float, lowpass_hz: float) -> Tuple[np.ndarray, float, float]:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        target = audio[start_idx:end_idx]
        if len(target) == 0:
            return audio, start, end
            
        # Apply lowpass filter
        nyq = 0.5 * sr
        norm_cutoff = lowpass_hz / nyq
        b, a = scipy.signal.butter(4, norm_cutoff, btype='low', analog=False)
        filtered = scipy.signal.filtfilt(b, a, target)
        
        # Apply gain reduction
        linear_gain = 10 ** (gain_db / 20.0)
        filtered = filtered * linear_gain
        
        # Put back and crossfade
        out_audio = audio.copy()
        out_audio[start_idx:end_idx] = filtered
        out_audio = self._crossfade(audio, out_audio, sr, start_idx, end_idx, fade_len_s=0.03)
        
        return out_audio, start, end

    def verify_monotonicity(self, results: List[PerturbationResult]) -> bool:
        """
        Verifies that higher severities produce stronger modifications.
        This is a simple heuristic check.
        """
        if len(results) < 2:
            return True
            
        flaw_type = results[0].flaw_type
        
        # Sort by severity
        sorted_res = sorted(results, key=lambda x: x.severity)
        
        if flaw_type == "rushed_pace":
            # Duration should decrease as severity increases
            durations = [r.flaw_end_sec - r.flaw_start_sec for r in sorted_res]
            return all(durations[i] > durations[i+1] for i in range(len(durations)-1))
            
        elif flaw_type == "dead_pause":
            # Duration should increase as severity increases
            durations = [r.flaw_end_sec - r.flaw_start_sec for r in sorted_res]
            return all(durations[i] < durations[i+1] for i in range(len(durations)-1))
            
        elif flaw_type == "flat_pitch":
            # Retained variance % should decrease
            pcts = [r.params.get("retained_variance_pct", 100) for r in sorted_res]
            return all(pcts[i] > pcts[i+1] for i in range(len(pcts)-1))
            
        elif flaw_type == "mumbled_clarity":
            # Gain reduction should be more negative, cutoff lower
            gains = [r.params.get("gain_db", 0) for r in sorted_res]
            cutoffs = [r.params.get("lowpass_hz", 8000) for r in sorted_res]
            return all(gains[i] > gains[i+1] and cutoffs[i] > cutoffs[i+1] for i in range(len(gains)-1))
            
        return True

    def verify_no_leakage(self, original: np.ndarray, perturbed: np.ndarray, 
                          flaw_start: float, flaw_end: float, sr: int, tolerance_db: float = 1.0) -> bool:
        """
        Verifies that the non-target regions are unmodified, allowing for a small tolerance due to crossfading/float issues.
        """
        start_idx = max(0, int((flaw_start - 0.05) * sr)) # 50ms margin for crossfade
        
        # For rushed_pace and dead_pause, the suffix is shifted, so we can't just subtract the whole array.
        # Check prefix only if lengths differ
        if len(original) != len(perturbed):
            prefix_orig = original[:start_idx]
            prefix_pert = perturbed[:start_idx]
            if len(prefix_orig) > 0:
                diff = np.abs(prefix_orig - prefix_pert)
                diff_db = 20 * np.log10(np.mean(diff) + 1e-10)
                # Note: diff_db will be very negative if identical. E.g. -80dB.
                # If they are different, it might be e.g. -20dB. We check if diff is close to zero.
                if np.max(diff) > 1e-3:
                    return False
            return True
        else:
            end_idx = min(len(original), int((flaw_end + 0.05) * sr))
            
            prefix_orig = original[:start_idx]
            prefix_pert = perturbed[:start_idx]
            
            suffix_orig = original[end_idx:]
            suffix_pert = perturbed[end_idx:]
            
            if len(prefix_orig) > 0 and np.max(np.abs(prefix_orig - prefix_pert)) > 1e-3:
                return False
                
            if len(suffix_orig) > 0 and np.max(np.abs(suffix_orig - suffix_pert)) > 1e-3:
                return False
                
            return True
