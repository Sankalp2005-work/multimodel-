# requirements.md

IDs are referenced from prompts and gate files. MUST = required for submission; SHOULD = strongly recommended; MAY = optional.

## 1. Functional requirements (system)
| ID | Pri | Requirement |
|---|---|---|
| FR-1 | MUST | Ingest an ideal and a participant recording of the same transcript (WAV/MP3/M4A → mono 16 kHz) |
| FR-2 | MUST | Forced alignment of both recordings to the transcript at word level (WhisperX; MFA acceptable). Report alignment confidence and WER |
| FR-3 | MUST | Extract per-frame and per-word features: RMS energy, F0/pitch, MFCC (+Δ), spectral centroid/flatness/bandwidth, ZCR, speech rate, pause durations |
| FR-4 | MUST | Speaker-agnostic normalization (per-recording z-score or ratio to the aligned ideal; log-F0 in semitones) |
| FR-5 | MUST | Contrastive comparison: word-aligned feature deltas participant vs ideal (not raw-time subtraction) |
| FR-6 | MUST | Detect flaw regions for ≥ 4 flaw types: rushed pace, dead pause, flat pitch, mumbled clarity; output start/end/severity (0–4)/type/confidence |
| FR-7 | MUST | Causal explanation per region: which feature(s) moved, by how much, vs which reference value (e.g. "pace +38% at 0:14–0:19") |
| FR-8 | MUST | Interactive dashboard: waveform/spectrogram overlay of ideal vs participant, highlighted flaw regions, click-to-play region, feature timelines, explanation panel |
| FR-9 | MUST | Overall rubric scores (pace, pause, pitch, energy, clarity) with weights in `configs/pipeline.json` |
| FR-10 | MUST | Honest-limitations panel in UI (mic sensitivity, alignment error propagation, no semantic/emotion judgment) |
| FR-11 | SHOULD | Multiple flaws per recording; overlapping-region handling |
| FR-12 | SHOULD | Confidence/abstain: when alignment confidence is low, say so instead of guessing |
| FR-13 | SHOULD | Compare against multiple ideal takes (variation envelope) to cut false positives |
| FR-14 | MAY | Export report (JSON + PDF) |

## 2. Dataset requirements
| ID | Pri | Requirement |
|---|---|---|
| DR-1 | MUST | Contrastive pairs: every flawed clip has an ideal clip of the SAME transcript (same speaker preferred) |
| DR-2 | MUST | One controlled flaw per clip, in one designated sentence; rest of the clip is ideal |
| DR-3 | MUST | Severity gradient 0–4 (0 = ideal/near-perfect, 4 = severely botched), parameters defined in `reference.md` |
| DR-4 | MUST | ≥ 4 transcript groups (rec. 6); ≥ 2 speakers (rec. 3); ≥ 40 validated paired clips for G4, ≥ 80 total target |
| DR-5 | MUST | Splits by transcript (and test also by speaker where possible): train / val / test; test sealed |
| DR-6 | MUST | Manifest + flaw-region CSVs per `templates/`, with sha256, source_type, source_license, consent, mic, room |
| DR-7 | MUST | Real human recordings are the primary data; synthetic perturbation fills severity gradient and is labeled `synthetic_perturbation`; TTS labeled `tts` |
| DR-8 | MUST | Hand-labeled ground-truth intervals, dual-annotated (see quality_gates.md) |
| DR-9 | MUST | Stress-test subset: different mic, light noise, different room, different speaker pace, accent variation, clean ideal clips for false-positive testing |
| DR-10 | SHOULD | ≥ 2 ideal takes per speaker-transcript to measure natural variation |
| DR-11 | SHOULD | Dataset card (datasheet): purpose, collection, consent, limitations, licence |
| DR-12 | MUST | Licence: all audio is own recordings with consent, or CC0/CC-BY with attribution; no scraped TED/YouTube audio redistributed in the repo |

## 3. Metric requirements
| ID | Pri | Requirement |
|---|---|---|
| MR-1 | MUST | A1 label accuracy ≥ 95% per `quality_gates.md` |
| MR-2 | MUST | A2 flaw-type accuracy ≥ 95% on sealed test, Wilson CI reported |
| MR-3 | MUST | A3 mean temporal IoU ≥ 0.60; median timestamp error ≤ 300 ms |
| MR-4 | MUST | Per-type precision/recall/F1 + confusion matrix |
| MR-5 | MUST | Severity MAE ≤ 0.6 levels (stretch ≤ 0.4) |
| MR-6 | MUST | False-positive rate on ideal clips ≤ 5% |
| MR-7 | MUST | Repeat-run: identical flaw list on 5 reruns; numeric drift ≤ 1e-6 (CPU deterministic mode) |
| MR-8 | SHOULD | Ablations: without normalization, without alignment (uniform fallback), single-feature baselines |

## 4. Non-functional
| ID | Pri | Requirement |
|---|---|---|
| NFR-1 | MUST | One-command setup (`docker compose up` or `make demo`); README verified on clean environment |
| NFR-2 | MUST | Analysis of a 70 s pair finishes ≤ 60 s on CPU (WhisperX small/base); cache alignment results |
| NFR-3 | MUST | Uploads deleted after processing; no audio leaves the machine (privacy section in docs) |
| NFR-4 | MUST | Unit tests for features, perturbations, metrics; CI green |
| NFR-5 | SHOULD | Type hints, linting, ≥ 70% coverage on core modules |
| NFR-6 | SHOULD | Pinned `requirements.txt`/lockfile, fixed seeds, saved run manifests |

## 5. Submission requirements (Devpost)
| ID | Pri | Requirement |
|---|---|---|
| SR-1 | MUST | Project description |
| SR-2 | MUST | Public GitHub repo with README setup instructions |
| SR-3 | MUST | 3–10 min unlisted YouTube demo: dataset process, stress-testing, dashboard catching deviations, honest limits |
| SR-4 | MUST | ≤ 6-page technical document: dataset construction, rubrics, architecture, scoring methodology |
| SR-5 | MUST | All team members on Devpost with real names |
| SR-6 | MUST | "Prior work" declaration for any pre-Oct 1 code |

## 6. Environment
- Python 3.10/3.11, ffmpeg, libsndfile
- Core: numpy, scipy, librosa, soundfile, pyworld or parselmouth (F0), whisperx, torch (CPU ok), pandas, scikit-learn, statsmodels
- Perturbation: pyrubberband (needs rubberband CLI) or librosa time-stretch; pyworld for pitch flattening
- Backend: FastAPI; frontend: Streamlit baseline → React + wavesurfer.js/plotly for polished dashboard (SHOULD)
- Tooling: ruff, black, pytest, pre-commit, Docker

## 7. Out of scope
Emotion/meaning judgment, language other than English, real-time streaming, medical/clinical claims.
