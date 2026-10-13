"""
tests/test_server_integration.py -- Full-stack API & server integration tests.
Verifies health, OpenAPI docs, demo payload, SPA mounting, and upload -> analyze workflow.
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns HTTP 200 and ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_docs_endpoint():
    """Verify OpenAPI Swagger documentation is reachable at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def test_demo_endpoint():
    """Verify /api/demo returns rich pre-calibrated coaching data."""
    response = client.get("/api/demo")
    assert response.status_code == 200
    data = response.json()
    assert "alignment" in data
    alignment = data["alignment"]
    assert "warping_path" in alignment
    assert len(alignment["warping_path"]) > 50, "Warping path should have high-resolution alignment points"
    assert "regions" in data, "Detected coaching regions should be present"
    assert "words" in data, "Word tokens should be present for interactive chip alignment"


def test_static_spa_mounted():
    """Verify root / serves the React production build HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()
    assert "root" in response.text.lower()


def test_upload_and_analyze_workflow():
    """Verify POST /api/upload followed by POST /api/analyze executes full DTW pipeline."""
    p_wav = Path("data/processed/wav/T1_rushed.wav")
    i_wav = Path("data/processed/wav/T1_ideal_1.wav")

    if not p_wav.exists() or not i_wav.exists():
        pytest.skip("Test WAV files not found in data/processed/wav/")

    # 1. Upload audio pair
    with open(i_wav, "rb") as fi, open(p_wav, "rb") as fp:
        files = {
            "ideal": ("ideal.wav", fi, "audio/wav"),
            "participant": ("participant.wav", fp, "audio/wav")
        }
        upload_resp = client.post("/api/upload", files=files)

    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert "upload_id" in upload_data
    upload_id = upload_data["upload_id"]

    # 2. Trigger contrastive analysis
    analyze_resp = client.post("/api/analyze", json={
        "upload_id": upload_id,
        "transcript": "Good morning everyone, and thank you for joining the quarterly review."
    })

    assert analyze_resp.status_code == 200
    result = analyze_resp.json()
    assert "alignment" in result
    assert "warping_path" in result["alignment"]
    assert len(result["alignment"]["warping_path"]) > 0
    assert "regions" in result
    assert "words" in result
    assert len(result["words"]) > 0
    assert result["sample_id"] == upload_id
