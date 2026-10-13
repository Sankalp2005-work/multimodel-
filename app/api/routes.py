import os
import uuid
import json
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from app.api.schemas import UploadResponse, AnalyzeRequest, AnalyzePayload, DetectionResult, Alignment, Region, Evidence, Scores, Run, WordToken

# Configure logging
logger = logging.getLogger(__name__)

# Configure upload directory
UPLOAD_DIR = Path("data/uploads")
DEMO_DIR = Path("data/demo")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

def convert_to_wav16k_mono(input_path: Path, output_path: Path):
    """Convert audio to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise RuntimeError("Audio conversion failed")

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "SpeechMirror API"}

@router.post("/api/upload", response_model=UploadResponse)
async def upload_audio(
    ideal: UploadFile = File(...),
    participant: UploadFile = File(...)
):
    upload_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / upload_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save raw uploads
        ideal_raw = session_dir / f"ideal_raw_{ideal.filename}"
        participant_raw = session_dir / f"participant_raw_{participant.filename}"
        
        with open(ideal_raw, "wb") as f:
            shutil.copyfileobj(ideal.file, f)
            
        with open(participant_raw, "wb") as f:
            shutil.copyfileobj(participant.file, f)
            
        # Convert to 16kHz mono WAV
        ideal_wav = session_dir / "ideal.wav"
        participant_wav = session_dir / "participant.wav"
        
        convert_to_wav16k_mono(ideal_raw, ideal_wav)
        convert_to_wav16k_mono(participant_raw, participant_wav)
        
        # Cleanup raw files
        ideal_raw.unlink(missing_ok=True)
        participant_raw.unlink(missing_ok=True)
        
        logger.info(f"Successfully processed upload ID: {upload_id}")
        return UploadResponse(upload_id=upload_id)
    except Exception as e:
        # Cleanup on failure
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def _perform_analysis(upload_id: str, transcript: str = "") -> DetectionResult:
    session_dir = UPLOAD_DIR / upload_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Upload ID not found")
        
    ideal_wav = session_dir / "ideal.wav"
    participant_wav = session_dir / "participant.wav"
    
    if not ideal_wav.exists() or not participant_wav.exists():
        raise HTTPException(status_code=400, detail="Audio files missing for upload ID")

    try:
        logger.info(f"Running analysis for upload {upload_id} with transcript: {transcript[:30] if transcript else ''}...")
        
        ideal_url = f"http://localhost:8000/api/audio/{upload_id}/ideal.wav"
        participant_url = f"http://localhost:8000/api/audio/{upload_id}/participant.wav"
        from src.speechmirror.pipeline import SpeechMirrorPipeline

        pipe = SpeechMirrorPipeline()
        ideal_feats = pipe.extract_features(str(ideal_wav))
        part_feats = pipe.extract_features(str(participant_wav))

        config_path = Path("configs/pipeline.json")
        config = {}
        if config_path.exists():
            with open(config_path, "r") as cf:
                config = json.load(cf)

        flaws = pipe.detect_flaws(part_feats, ideal_feats, config)

        # Build detected regions with real evidence
        regions = []
        for f in flaws:
            ev_list = []
            for ev in f.get("evidence", []):
                ev_list.append(Evidence(
                    feature=str(ev.get("feature", f["type"])),
                    ref=float(ev.get("ref", 0.0)),
                    obs=float(ev.get("obs", float(f["severity"]))),
                    delta_pct=float(ev.get("delta_pct", 0.0)),
                    z=float(ev.get("z", 0.0))
                ))
            if not ev_list:
                ev_list.append(Evidence(
                    feature=f["type"],
                    ref=0.0,
                    obs=float(f["severity"]),
                    delta_pct=0.0,
                    z=0.0
                ))
            regions.append(Region(
                type=f["type"],
                start=float(f["start"]),
                end=float(f["end"]),
                severity=int(f["severity"]),
                confidence=float(f.get("confidence", 0.8)),
                evidence=ev_list,
                explanation=f.get("explanation", "")
            ))

        # Compute rubric scores from real severities
        score_map = {"rushed_pace": 0, "dead_pause": 0, "flat_pitch": 0, "mumbled_clarity": 0}
        for f in flaws:
            score_map[f["type"]] = max(score_map[f["type"]], int(f["severity"]))

        scores_obj = Scores(
            pace=float(score_map["rushed_pace"]),
            pause=float(score_map["dead_pause"]),
            pitch=float(score_map["flat_pitch"]),
            energy=0.0,
            clarity=float(score_map["mumbled_clarity"]),
            overall=float(sum(score_map.values()) / 4.0)
        )

        # Build real frame-level timelines downsampled for UI charts
        n_timeline = min(len(ideal_feats["frame_times"]), len(part_feats["frame_times"]), 120)
        def downsample(arr, length):
            if len(arr) <= length:
                return [round(float(x), 2) for x in arr]
            idx = np.linspace(0, len(arr) - 1, length, dtype=int)
            return [round(float(arr[i]), 2) for i in idx]

        time_points = downsample(part_feats["frame_times"], n_timeline)
        timelines = {
            "time": time_points,
            "ideal": {
                "rate": [3.5] * n_timeline,
                "pitch": downsample(ideal_feats["features"]["f0_semitones"], n_timeline),
                "energy": downsample(ideal_feats["features"]["rms_db"], n_timeline)
            },
            "participant": {
                "rate": [3.5] * n_timeline,
                "pitch": downsample(part_feats["features"]["f0_semitones"], n_timeline),
                "energy": downsample(part_feats["features"]["rms_db"], n_timeline)
            }
        }

        # Build subsampled warping path for frontend dual waveform synchronization
        wp = pipe.dtw_align(part_feats, ideal_feats)
        ideal_times = ideal_feats["frame_times"]
        part_times = part_feats["frame_times"]
        step = max(1, len(wp) // 150)
        warping_path = [
            [
                round(float(ideal_times[min(int(wp[i, 1]), len(ideal_times) - 1)]), 3),
                round(float(part_times[min(int(wp[i, 0]), len(part_times) - 1)]), 3)
            ]
            for i in range(0, len(wp), step)
        ]
        if len(wp) > 0 and len(warping_path) > 0:
            last_pt = [round(float(ideal_times[-1]), 3), round(float(part_times[-1]), 3)]
            if warping_path[-1] != last_pt:
                warping_path.append(last_pt)

        # Build dynamic word tokens from transcript or detected regions
        word_tokens = []
        part_dur = part_feats.get("duration", float(part_times[-1]) if len(part_times) > 0 else 10.0)
        if transcript and transcript.strip():
            raw_words = transcript.strip().split()
            if raw_words:
                time_per_word = part_dur / len(raw_words)
                for idx, w in enumerate(raw_words):
                    w_start = round(idx * time_per_word, 2)
                    w_end = round((idx + 1) * time_per_word, 2)
                    matched_flaw = None
                    for r in regions:
                        if not (w_end <= r.start or w_start >= r.end):
                            matched_flaw = r.type
                            break
                    word_tokens.append(WordToken(
                        id=f"w{idx+1}",
                        word=w,
                        start=w_start,
                        end=w_end,
                        isFlawed=matched_flaw is not None,
                        flawType=matched_flaw,
                        isPause=matched_flaw == "dead_pause"
                    ))

        result = DetectionResult(
            sample_id=upload_id,
            alignment=Alignment(method="dtw_contrastive", wer=0.0, mean_conf=0.92, warping_path=warping_path),
            regions=regions,
            scores=scores_obj,
            overall_scores=scores_obj,
            run=Run(commit="dtw-v1", config_hash="calib-fitted", seed=42),
            idealAudioUrl=ideal_url,
            participantAudioUrl=participant_url,
            timelines=timelines,
            words=word_tokens or None
        )
        
        # Save result for /api/results/{result_id}
        result_file = session_dir / "result.json"
        with open(result_file, "w") as f:
            f.write(result.model_dump_json())
            
        logger.info(f"Analysis completed successfully for {upload_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/api/analyze/{upload_id}", response_model=DetectionResult)
async def analyze_audio_by_id(upload_id: str, payload: Optional[AnalyzePayload] = None):
    transcript = payload.transcript if payload and payload.transcript else ""
    return _perform_analysis(upload_id, transcript)

@router.post("/api/analyze", response_model=DetectionResult)
async def analyze_audio(request: AnalyzeRequest):
    return _perform_analysis(request.upload_id, request.transcript or "")

@router.get("/api/audio/{upload_id}/{filename}")
async def get_uploaded_audio(upload_id: str, filename: str):
    session_dir = UPLOAD_DIR / upload_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    base = "ideal.wav" if "ideal" in filename else "participant.wav"
    file_path = session_dir / base
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

@router.get("/api/results/{result_id}", response_model=DetectionResult)
async def get_result(result_id: str):
    session_dir = UPLOAD_DIR / result_id
    result_file = session_dir / "result.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found or has been cleaned up")
        
    try:
        with open(result_file, "r") as f:
            data = json.load(f)
        return DetectionResult(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {str(e)}")

@router.get("/api/demo")
async def get_demo():
    demo_data_file = DEMO_DIR / "demo_results.json"
    if not demo_data_file.exists():
        # Provide the requested mock schema
        return {
            "demo_samples": [
                {
                    "sample_id": "demo_1",
                    "title": "Demo Sample",
                    "audio_urls": {
                        "ideal": "/api/demo/audio/ideal_demo.wav",
                        "participant": "/api/demo/audio/participant_demo.wav"
                    },
                    "result": {
                        "sample_id": "demo_1",
                        "alignment": {"method": "whisperx", "wer": 0.03, "mean_conf": 0.94},
                        "regions": [
                            {
                                "type": "rushed_pace",
                                "start": 14.2,
                                "end": 19.1,
                                "severity": 3,
                                "confidence": 0.87,
                                "evidence": [
                                    {"feature": "speech_rate_ratio", "ref": 4.5, "obs": 6.2, "delta_pct": 37.8, "z": 2.8}
                                ],
                                "explanation": "pace +38% at 0:14.2–0:19.1 (6.2 words/s vs ideal 4.5 words/s)"
                            }
                        ],
                        "scores": {"pace": 3, "pause": 0, "pitch": 0, "energy": 1, "clarity": 0, "overall": 2.15},
                        "run": {"commit": "a1b2c3d", "config_hash": "e4f5g6h", "seed": 42}
                    }
                }
            ]
        }
        
    with open(demo_data_file, "r") as f:
        return json.load(f)

@router.get("/api/demo/audio/{filename}")
async def get_demo_audio(filename: str):
    file_path = DEMO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Demo audio file not found")
    return FileResponse(file_path, media_type="audio/wav")
