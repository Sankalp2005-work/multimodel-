import os
import uuid
import json
import shutil
import logging
import subprocess
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from app.api.schemas import UploadResponse, AnalyzeRequest, DetectionResult, Alignment, Region, Evidence, Scores, Run

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

@router.post("/api/analyze", response_model=DetectionResult)
async def analyze_audio(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    upload_id = request.upload_id
    session_dir = UPLOAD_DIR / upload_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Upload ID not found")
        
    ideal_wav = session_dir / "ideal.wav"
    participant_wav = session_dir / "participant.wav"
    
    if not ideal_wav.exists() or not participant_wav.exists():
        raise HTTPException(status_code=400, detail="Audio files missing for upload ID")

    try:
        # NOTE: Import your pipeline here when fully integrated
        # from src.speechmirror.pipeline import run_pipeline
        # result = run_pipeline(ideal_wav, participant_wav, request.transcript)
        
        logger.info(f"Running analysis for upload {upload_id} with transcript: {request.transcript[:30]}...")
        
        # Mock result for now
        result = DetectionResult(
            sample_id=upload_id,
            alignment=Alignment(method="fallback", wer=0.03, mean_conf=0.94),
            regions=[
                Region(
                    type="rushed_pace",
                    start=14.2,
                    end=19.1,
                    severity=3,
                    confidence=0.87,
                    evidence=[
                        Evidence(feature="speech_rate_ratio", ref=4.5, obs=6.2, delta_pct=37.8, z=2.8)
                    ],
                    explanation="pace +38% at 0:14.2–0:19.1 (6.2 words/s vs ideal 4.5 words/s)"
                )
            ],
            scores=Scores(pace=3, pause=0, pitch=0, energy=1, clarity=0, overall=2.15),
            run=Run(commit="unknown", config_hash="unknown", seed=42)
        )
        
        # Save result (optional, since it will be returned and session deleted)
        result_file = session_dir / "result.json"
        with open(result_file, "w") as f:
            f.write(result.model_dump_json())
            
        # NFR-3 privacy: schedule session deletion after response is sent
        background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
        
        logger.info(f"Analysis completed successfully for {upload_id}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
