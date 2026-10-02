# SpeechMirror Technical Document

## 1. Dataset Construction Methodology
SpeechMirror utilizes a contrastive dataset approach. We first record an "ideal" reference take for a given transcript. We then generate "flawed" takes by either:
- **Synthetic Perturbation**: Using deterministic DSP techniques (time-compression, digital silence insertion, F0 compression, and low-pass filtering) to inject specific flaws (`rushed_pace`, `dead_pause`, `flat_pitch`, `mumbled_clarity`) with exact ground-truth boundaries.
- **Human Performance**: Actors explicitly perform these flaws in specific sentences during a secondary take.

## 2. Rubric Definitions
The system evaluates five key dimensions:
- **Pace**: Measured via speech rate (words/sec) and evaluated for rushed delivery.
- **Pause**: Measured via inter-word silence durations to detect dead pauses.
- **Pitch**: Evaluated using F0 variance and range to detect monotone/flat delivery.
- **Energy**: Measured via per-frame RMS to identify volume drops.
- **Clarity**: Measured via spectral centroid and MFCC distance to detect mumbled speech.

## 3. Architecture
- **Backend**: FastAPI
- **Audio processing**: `librosa`, `pyworld`, `scipy` for feature extraction and synthetic perturbations.
- **Forced Alignment**: WhisperX for word-level timestamps.
- **Frontend**: React, `wavesurfer.js` for audio visualization, and `plotly.js` for feature timelines.

## 4. Scoring Methodology
Scoring is performed by taking the word-aligned feature deltas between the participant and ideal takes, applying configurable thresholds (calibrated on the training set), and detecting contiguous flaw regions. The overall score is a weighted sum of the individual dimension scores.

## 5. Results
- **Label Accuracy (A1)**: target ≥ 95%
- **Detection Accuracy (A2)**: target ≥ 95% on sealed test set
- **Temporal IoU (A3)**: target ≥ 0.60
