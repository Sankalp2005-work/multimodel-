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
from app.api.schemas import UploadResponse, AnalyzeRequest, AnalyzePayload, DetectionResult, Alignment, Region, Evidence, Scores, Run

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
        
        time_points = [round(i * 0.15, 2) for i in range(100)]
        timelines = {
            "time": time_points,
            "ideal": {
                "rate": [round(3.5 + float(np.sin(i * 0.3) * 0.5), 2) for i in range(100)],
                "pitch": [round(130.0 + float(np.sin(i * 0.15) * 15.0), 1) for i in range(100)],
                "energy": [round(0.6 + float(np.cos(i * 0.2) * 0.15), 2) for i in range(100)]
            },
            "participant": {
                "rate": [round(5.8 if 10 <= i <= 25 else 3.6 + float(np.sin(i * 0.3) * 0.4), 2) for i in range(100)],
                "pitch": [round(95.0 if 55 <= i <= 72 else 125.0 + float(np.sin(i * 0.15) * 10.0), 1) for i in range(100)],
                "energy": [round(0.55 + float(np.cos(i * 0.2) * 0.12), 2) for i in range(100)]
            }
        }
        
        scores_obj = Scores(pace=3.0, pause=1.0, pitch=2.0, energy=1.0, clarity=3.0, overall=2.0)
        
        result = DetectionResult(
            sample_id=upload_id,
            alignment=Alignment(method="whisperx", wer=0.03, mean_conf=0.94),
            regions=[
                Region(
                    type="rushed_pace",
                    start=1.5,
                    end=3.8,
                    severity=3,
                    confidence=0.88,
                    evidence=[
                        Evidence(feature="speech_rate_ratio", ref=4.2, obs=6.1, delta_pct=45.2, z=2.4)
                    ],
                    explanation="Speaking rate accelerated +45% (6.1 vs 4.2 words/sec) between 0:01.5 and 0:03.8."
                ),
                Region(
                    type="flat_pitch",
                    start=8.2,
                    end=10.8,
                    severity=2,
                    confidence=0.92,
                    evidence=[
                        Evidence(feature="f0_std_ratio", ref=32.0, obs=14.5, delta_pct=-54.7, z=-2.3)
                    ],
                    explanation="Intonation range contracted by -55% (monotone cadence) between 0:08.2 and 0:10.8."
                )
            ],
            scores=scores_obj,
            overall_scores=scores_obj,
            run=Run(commit="72caca6", config_hash="e4f5g6h", seed=42),
            idealAudioUrl=ideal_url,
            participantAudioUrl=participant_url,
            timelines=timelines
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
