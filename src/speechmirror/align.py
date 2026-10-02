import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class WordAlignment:
    word: str
    start: float  # seconds
    end: float    # seconds
    confidence: float  # 0-1

@dataclass
class AlignmentResult:
    words: list[WordAlignment]
    transcript_ref: str
    transcript_hyp: str
    wer: float
    mean_confidence: float
    method: str  # 'whisperx' or 'uniform_fallback'
    audio_hash: str
    duration_s: float

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentResult":
        data["words"] = [WordAlignment(**w) for w in data["words"]]
        return cls(**data)

class Aligner:
    def __init__(self, method: str = 'whisperx', model_size: str = 'base',
                 language: str = 'en', device: str = 'cpu',
                 cache_dir: str = 'data/processed/alignments'):
        self.method = method
        self.model_size = model_size
        self.language = language
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None

    def _init_whisperx(self):
        if self.whisper_model is not None:
            return
        try:
            import whisperx
            logger.info(f"Loading WhisperX models ({self.model_size} on {self.device})...")
            self.whisper_model = whisperx.load_model(self.model_size, self.device, compute_type="float32")
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code=self.language, device=self.device)
        except ImportError:
            logger.warning("WhisperX not installed. Cannot use whisperx alignment method.")
            raise

    def align(self, audio_path: str, transcript: str, use_cache: bool = True) -> AlignmentResult:
        """Align audio to transcript using WhisperX."""
        cache_key = self._cache_key(audio_path, transcript)
        
        if use_cache:
            cached_result = self._load_cache(cache_key)
            if cached_result:
                logger.info(f"Loaded alignment from cache for {audio_path}")
                return cached_result
                
        try:
            if self.method == 'whisperx':
                result = self.align_whisperx(audio_path, transcript)
            else:
                raise ValueError(f"Unknown alignment method: {self.method}")
        except Exception as e:
            logger.warning(f"Alignment method '{self.method}' failed: {e}. Falling back to uniform.")
            import librosa
            audio, sr = librosa.load(audio_path, sr=None)
            result = self.align_uniform_fallback(audio, sr, transcript)
            # Update hash to actual cache key hash for consistency
            audio_hash = cache_key.split('_')[0] if '_' in cache_key else cache_key
            result.audio_hash = audio_hash
            
        if use_cache:
            self._save_cache(cache_key, result)
            
        return result

    def align_whisperx(self, audio_path: str, transcript: str) -> AlignmentResult:
        """WhisperX forced alignment implementation."""
        self._init_whisperx()
        import whisperx
        
        audio = whisperx.load_audio(audio_path)
        result = self.whisper_model.transcribe(audio, batch_size=1)
        hyp_text = " ".join([seg["text"].strip() for seg in result["segments"]])
        
        aligned_result = whisperx.align(
            result["segments"], 
            self.align_model, 
            self.align_metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        words = []
        confidences = []
        for segment in aligned_result["segments"]:
            for word_info in segment.get("words", []):
                # WhisperX sometimes omits start/end if alignment is ambiguous
                if "start" in word_info and "end" in word_info:
                    conf = word_info.get("score", 0.0)
                    words.append(WordAlignment(
                        word=word_info["word"],
                        start=word_info["start"],
                        end=word_info["end"],
                        confidence=conf
                    ))
                    confidences.append(conf)
                    
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        wer = self.compute_wer(transcript, hyp_text)
        
        # Calculate duration and audio hash
        import librosa
        duration_s = librosa.get_duration(path=audio_path)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        
        return AlignmentResult(
            words=words,
            transcript_ref=transcript,
            transcript_hyp=hyp_text,
            wer=wer,
            mean_confidence=mean_conf,
            method="whisperx",
            audio_hash=audio_hash,
            duration_s=duration_s
        )

    def align_uniform_fallback(self, audio: np.ndarray, sr: int, transcript: str) -> AlignmentResult:
        """Uniform time distribution fallback (for UI demos ONLY)."""
        logger.warning("Using UNIFORM FALLBACK for alignment. This is a rough estimate.")
        words_list = transcript.split()
        duration_s = len(audio) / sr
        
        word_duration = duration_s / max(1, len(words_list))
        
        words = []
        current_time = 0.0
        for w in words_list:
            words.append(WordAlignment(
                word=w,
                start=current_time,
                end=current_time + word_duration,
                confidence=0.0
            ))
            current_time += word_duration
            
        return AlignmentResult(
            words=words,
            transcript_ref=transcript,
            transcript_hyp=transcript,
            wer=1.0,
            mean_confidence=0.0,
            method="uniform_fallback",
            audio_hash="unknown",
            duration_s=duration_s
        )

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate (Levenshtein distance)."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    cost = 0
                else:
                    cost = 1
                d[i][j] = min(
                    d[i-1][j] + 1,       # deletion
                    d[i][j-1] + 1,       # insertion
                    d[i-1][j-1] + cost   # substitution
                )
                
        wer = d[len(ref_words)][len(hyp_words)] / max(1, len(ref_words))
        return float(wer)

    def _cache_key(self, audio_path: str, transcript: str) -> str:
        """Generate cache key from audio hash + transcript hash."""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        trans_hash = hashlib.sha256(transcript.encode('utf-8')).hexdigest()
        return f"{audio_hash}_{trans_hash}"

    def _save_cache(self, key: str, result: AlignmentResult) -> None:
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _load_cache(self, key: str) -> Optional[AlignmentResult]:
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AlignmentResult.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load cache {cache_path}: {e}")
        return None
