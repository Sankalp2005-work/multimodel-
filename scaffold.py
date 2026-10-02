import os

base_dir = r"d:\multimodel"

files = {
    r"src\speechmirror\__init__.py": "__version__ = '0.1.0'\n",
    r"configs\pipeline.json": """{
  "version": "0.1.0",
  "alignment": {
    "method": "whisperx",
    "model_size": "base",
    "language": "en",
    "max_wer": 0.08
  },
  "features": {
    "sample_rate": 16000,
    "frame_length_ms": 25,
    "hop_length_ms": 10,
    "n_mfcc": 13,
    "f0_method": "pyworld",
    "speech_rate_window_words": 5
  },
  "normalization": {
    "method": "zscore_per_recording",
    "f0_unit": "semitones_re_median",
    "energy_ref": "median_ideal_speech"
  },
  "detection": {
    "rushed_pace": {
      "speech_rate_ratio_threshold": 1.25,
      "min_region_words": 3,
      "severity_boundaries": [1.15, 1.3, 1.5, 1.7]
    },
    "dead_pause": {
      "excess_pause_threshold_s": 0.4,
      "min_pause_duration_s": 0.3,
      "severity_boundaries": [0.4, 0.8, 1.4, 2.0]
    },
    "flat_pitch": {
      "f0_variance_ratio_threshold": 0.6,
      "f0_range_ratio_threshold": 0.5,
      "min_voiced_frames": 10,
      "severity_boundaries": [0.65, 0.45, 0.25, 0.10]
    },
    "mumbled_clarity": {
      "energy_drop_db": -2.5,
      "spectral_centroid_ratio_threshold": 0.8,
      "mfcc_distance_threshold": 1.5,
      "severity_boundaries": [1.2, 1.8, 2.5, 3.5]
    }
  },
  "scoring": {
    "weights": {
      "pace": 0.25,
      "pause": 0.20,
      "pitch": 0.20,
      "energy": 0.15,
      "clarity": 0.20
    },
    "scale": [0, 1, 2, 3, 4]
  },
  "seed": 42,
  "deterministic": true
}""",
    r"configs\perturb.json": """{
  "version": "0.1.0",
  "perturbations": {
    "rushed_pace": {
      "description": "Pitch-preserving time-compress the target sentence",
      "method": "pyrubberband_time_stretch",
      "speed_factors": {
        "1": 1.20,
        "2": 1.40,
        "3": 1.60,
        "4": 1.80
      }
    },
    "dead_pause": {
      "description": "Insert digital-silence-plus-room-tone at a word boundary mid-sentence",
      "method": "silence_insertion",
      "pause_durations_s": {
        "1": 0.5,
        "2": 1.0,
        "3": 1.7,
        "4": 2.5
      },
      "room_tone_mix": 0.05
    },
    "flat_pitch": {
      "description": "Compress F0 contour toward sentence mean",
      "method": "pyworld_f0_compression",
      "retained_variance_pct": {
        "1": 75,
        "2": 50,
        "3": 25,
        "4": 5
      }
    },
    "mumbled_clarity": {
      "description": "Gain reduction + high-frequency smear",
      "method": "gain_plus_lowpass",
      "params": {
        "1": {"gain_db": -3, "lowpass_hz": 6000},
        "2": {"gain_db": -6, "lowpass_hz": 4500},
        "3": {"gain_db": -9, "lowpass_hz": 3500},
        "4": {"gain_db": -12, "lowpass_hz": 2500}
      },
      "crossfade_ms": 30
    }
  },
  "seed": 7,
  "output_format": {
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16,
    "format": "wav"
  }
}""",
    r"requirements.txt": """numpy>=1.24,<2.0
scipy>=1.10
librosa>=0.10
soundfile>=0.12
pyworld>=0.3
praat-parselmouth>=0.4
whisperx>=3.1
torch>=2.0
torchaudio>=2.0
pandas>=2.0
scikit-learn>=1.3
statsmodels>=0.14
pyrubberband>=0.3
fastapi>=0.100
uvicorn>=0.23
python-multipart>=0.0.6
plotly>=5.15
jinja2>=3.1
pydantic>=2.0
ruff>=0.1
black>=23.0
pytest>=7.0
pytest-cov>=4.0""",
    r"Makefile": """.PHONY: setup test lint demo benchmark clean

PYTHON := python
PIP := pip

setup:
\t$(PIP) install -r requirements.txt
\t@echo "Setup complete."

test:
\t$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
\t$(PYTHON) -m pytest tests/ -v --cov=src/speechmirror --cov-report=term-missing

lint:
\truff check src/ scripts/ tests/
\tblack --check src/ scripts/ tests/

format:
\truff check --fix src/ scripts/ tests/
\tblack src/ scripts/ tests/

demo:
\tcd app && $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
\t@echo "API running at http://localhost:8000"
\t@echo "Dashboard: open app/dashboard in browser"

benchmark-val:
\t$(PYTHON) scripts/benchmark.py --split val --seed 42

benchmark-test:
\t$(PYTHON) scripts/benchmark.py --split test --seed 42

perturb:
\t$(PYTHON) scripts/generate_perturbations.py --config configs/perturb.json --seed 7

validate:
\t$(PYTHON) scripts/validate_dataset.py --manifest data/manifests/manifest.csv

stats:
\t$(PYTHON) scripts/stats_report.py

clean:
\tfind . -type d -name __pycache__ -exec rm -rf {} +
\tfind . -type f -name '*.pyc' -delete""",
    r"docker-compose.yml": """version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      - ./results:/app/results
    environment:
      - PYTHONPATH=/app
      - SPEECHMIRROR_ENV=production
    command: uvicorn app.api.main:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: ./app/dashboard
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - api
    environment:
      - REACT_APP_API_URL=http://api:8000""",
    r"Dockerfile": """FROM python:3.11-slim

RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libsndfile1 \\
    rubberband-cli \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]""",
    r"setup.py": """from setuptools import setup, find_packages
setup(
    name="speechmirror",
    version="0.1.0",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)""",
    r"app\__init__.py": "",
    r"app\api\__init__.py": "",
    r"scripts\__init__.py": "",
    r"tests\__init__.py": "",
    r"data\raw\.gitkeep": "",
    r"data\manifests\.gitkeep": "",
    r"results\runs\.gitkeep": "",
    r"results\figures\.gitkeep": "",
    r"docs\pack\.gitkeep": "",
}

for rel_path, content in files.items():
    path = os.path.join(base_dir, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created {path}")
