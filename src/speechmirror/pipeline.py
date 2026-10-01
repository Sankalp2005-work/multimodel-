import librosa
import numpy as np
import pyworld as pw
import whisper
import warnings

# Suppress FP16 warnings on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

class SpeechMirrorPipeline:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def extract_features(self, wav_path: str):
        """
        Extracts acoustic and text-aligned features for the given audio.
        Returns a dictionary containing frame-level and word-level features.
        """
        # 1. Acoustic Features
        y, sr = librosa.load(wav_path, sr=16000)
        
        # RMS Energy (Frame level)
        rms = librosa.feature.rms(y=y)[0]
        
        # F0 / Pitch (Frame level)
        _f0, t = pw.dio(y.astype(np.float64), sr)
        f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)
        
        # Mumbled / Clarity (Spectral centroid, flatness, bandwidth, ZCR)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        flat = librosa.feature.spectral_flatness(y=y)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # 2. Word-Level Alignment
        result = self.model.transcribe(wav_path, word_timestamps=True)
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip().lower(),
                    "start": w["start"],
                    "end": w["end"],
                    "duration": w["end"] - w["start"]
                })
                
        # 3. Calculate Pauses
        for i in range(1, len(words)):
            pause = words[i]["start"] - words[i-1]["end"]
            words[i]["prev_pause"] = pause if pause > 0 else 0
        if words:
            words[0]["prev_pause"] = 0

        # 4. Normalization (Speaker Agnostic)
        # Pitch: convert to semitones (relative to median of non-zero F0)
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) > 0:
            ref_f0 = np.median(valid_f0)
            f0_semitones = 12 * np.log2((f0 + 1e-9) / ref_f0)
            f0_semitones[f0 == 0] = -100  # unvoiced
        else:
            f0_semitones = f0
            
        # Energy: dB scale relative to max
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        return {
            "duration": len(y) / sr,
            "words": words,
            "frame_times": librosa.frames_to_time(np.arange(len(rms)), sr=sr),
            "features": {
                "rms_db": rms_db,
                "f0_semitones": f0_semitones,
                "centroid": cent,
                "flatness": flat,
                "bandwidth": bw,
                "zcr": zcr
            }
        }

    def compute_deltas(self, part_features: dict, ideal_features: dict):
        """
        Computes sentence/word level deltas between participant and ideal clip.
        This provides the contrastive comparison.
        """
        # For simplicity in this pipeline, we'll compare aggregated stats 
        # across 5-second overlapping windows, or word-matched windows.
        
        # Let's align words using dynamic time warping (DTW) or just sequential matching.
        # Since the transcript is identical, the word sequence should be nearly identical.
        part_words = part_features["words"]
        ideal_words = ideal_features["words"]
        
        # Match words sequentially (naive greedy match)
        # A more robust solution would use DTW, but whisper produces similar lengths.
        deltas = []
        min_len = min(len(part_words), len(ideal_words))
        
        for i in range(min_len):
            pw = part_words[i]
            iw = ideal_words[i]
            
            pace_delta = pw["duration"] - iw["duration"]
            pause_delta = pw["prev_pause"] - iw["prev_pause"]
            
            # Map word timing to frames to get pitch/energy deltas
            # Participant frames
            p_start_f = librosa.time_to_frames(pw["start"], sr=16000)
            p_end_f = librosa.time_to_frames(pw["end"], sr=16000)
            
            # Ideal frames
            i_start_f = librosa.time_to_frames(iw["start"], sr=16000)
            i_end_f = librosa.time_to_frames(iw["end"], sr=16000)
            
            # Extract pitch slice
            p_pitch = part_features["features"]["f0_semitones"][p_start_f:p_end_f]
            i_pitch = ideal_features["features"]["f0_semitones"][i_start_f:i_end_f]
            
            p_pitch_valid = p_pitch[p_pitch > -50] if len(p_pitch) > 0 else []
            i_pitch_valid = i_pitch[i_pitch > -50] if len(i_pitch) > 0 else []
            
            p_pitch_std = np.std(p_pitch_valid) if len(p_pitch_valid) > 0 else 0
            i_pitch_std = np.std(i_pitch_valid) if len(i_pitch_valid) > 0 else 0
            
            pitch_std_delta = p_pitch_std - i_pitch_std
            
            # Clarity (Spectral centroid)
            p_cent = part_features["features"]["centroid"][p_start_f:p_end_f]
            i_cent = ideal_features["features"]["centroid"][i_start_f:i_end_f]
            p_cent_mean = np.mean(p_cent) if len(p_cent) > 0 else 0
            i_cent_mean = np.mean(i_cent) if len(i_cent) > 0 else 0
            
            cent_delta_pct = (p_cent_mean - i_cent_mean) / (i_cent_mean + 1e-9)

            deltas.append({
                "word": pw["word"],
                "start": pw["start"],
                "end": pw["end"],
                "pace_delta": pace_delta,
                "pause_delta": pause_delta,
                "pitch_std_delta": pitch_std_delta,
                "clarity_delta_pct": cent_delta_pct
            })
            
        return deltas

    def detect_flaws(self, deltas: list, config: dict):
        """
        Rules-based detection of flaws based on deltas.
        Outputs regions, severities, and explanations.
        """
        flaws = []
        
        # Config Thresholds
        rushed_thresh = config.get("rushed_thresh", -0.1) # word duration shorter by 0.1s
        pause_thresh = config.get("pause_thresh", 0.5) # pause longer by 0.5s
        pitch_thresh = config.get("pitch_thresh", -1.5) # std deviation dropped by 1.5 semitones
        clarity_thresh = config.get("clarity_thresh", -0.2) # centroid dropped by 20%
        
        # Smooth deltas over a small window (e.g. 5 words) to find regions
        window_size = 5
        
        for i in range(len(deltas) - window_size + 1):
            window = deltas[i:i+window_size]
            
            avg_pace = np.mean([d["pace_delta"] for d in window])
            max_pause = np.max([d["pause_delta"] for d in window])
            avg_pitch = np.mean([d["pitch_std_delta"] for d in window])
            avg_clarity = np.mean([d["clarity_delta_pct"] for d in window])
            
            start_time = window[0]["start"]
            end_time = window[-1]["end"]
            
            # Detect Rushed Pace
            if avg_pace < rushed_thresh:
                sev = min(4, int(abs(avg_pace / rushed_thresh)))
                if sev > 0:
                    flaws.append({
                        "type": "rushed_pace",
                        "severity": sev,
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "explanation": f"Pace accelerated: word durations decreased by {abs(avg_pace)*1000:.0f}ms on average vs ideal."
                    })
                    
            # Detect Dead Pause
            if max_pause > pause_thresh:
                sev = min(4, int(max_pause / pause_thresh))
                if sev > 0:
                    flaws.append({
                        "type": "dead_pause",
                        "severity": sev,
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "explanation": f"Unnatural pause: detected silence of {max_pause:.2f}s longer than ideal."
                    })
                    
            # Detect Flat Pitch
            if avg_pitch < pitch_thresh:
                sev = min(4, int(abs(avg_pitch / pitch_thresh)))
                if sev > 0:
                    flaws.append({
                        "type": "flat_pitch",
                        "severity": sev,
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "explanation": f"Monotone speech: pitch variation dropped by {abs(avg_pitch):.1f} semitones vs ideal."
                    })
                    
            # Detect Mumbled Clarity
            if avg_clarity < clarity_thresh:
                sev = min(4, int(abs(avg_clarity / clarity_thresh)))
                if sev > 0:
                    flaws.append({
                        "type": "mumbled_clarity",
                        "severity": sev,
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "explanation": f"Muffled articulation: spectral centroid dropped by {abs(avg_clarity)*100:.0f}% vs ideal."
                    })

        # Non-Maximum Suppression (Merge overlapping flaws of same type)
        merged = []
        for ftype in ["rushed_pace", "dead_pause", "flat_pitch", "mumbled_clarity"]:
            type_flaws = [f for f in flaws if f["type"] == ftype]
            if not type_flaws:
                continue
                
            # Merge overlapping or adjacent regions
            type_flaws.sort(key=lambda x: x["start"])
            current = type_flaws[0]
            
            for f in type_flaws[1:]:
                # If they overlap or are very close (within 1s)
                if f["start"] <= current["end"] + 1.0:
                    current["end"] = max(current["end"], f["end"])
                    current["severity"] = max(current["severity"], f["severity"])
                else:
                    merged.append(current)
                    current = f
            merged.append(current)
            
        return merged
