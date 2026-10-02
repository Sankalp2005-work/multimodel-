from typing import List, Optional
from pydantic import BaseModel

class Evidence(BaseModel):
    feature: str
    ref: float
    obs: float
    delta_pct: float
    z: float

class Region(BaseModel):
    type: str
    start: float
    end: float
    severity: int
    confidence: float
    evidence: List[Evidence]
    explanation: str

class Alignment(BaseModel):
    method: str
    wer: float
    mean_conf: float

class Scores(BaseModel):
    pace: float
    pause: float
    pitch: float
    energy: float
    clarity: float
    overall: float

class Run(BaseModel):
    commit: str
    config_hash: str
    seed: int

class DetectionResult(BaseModel):
    sample_id: str
    alignment: Alignment
    regions: List[Region]
    scores: Scores
    run: Run

class AnalyzeRequest(BaseModel):
    upload_id: str
    transcript: str

class UploadResponse(BaseModel):
    upload_id: str
    message: str = "Upload successful"
