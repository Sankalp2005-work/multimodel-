# skills.md — Skills needed (human team + AI agent)

Each skill lists: **what**, **tools**, **done when**, **pitfalls**. Assign a human owner to each; the agent assists.

## S1 Dataset engineering (owns 30% of score)
- **What:** Design contrastive dataset, manage manifests, splits, provenance, datasheet.
- **Tools:** pandas, soundfile, ffmpeg, sha256, `validate_dataset.py`.
- **Done when:** G2 and G4 pass; manifest validates; datasheet written.
- **Pitfalls:** transcript leakage across splits; undocumented synthetic clips; editing labels to fit the model.

## S2 Recording & vocal performance
- **What:** Produce consistent ideal takes and clean single-flaw takes.
- **Tools:** Same mic/room per speaker, pop filter or phone at fixed distance, stopwatch, quiet room, Audacity for trimming only.
- **Done when:** takes pass SNR/clipping checks; flaw is audible to blind listener.
- **Pitfalls:** moving relative to mic (breaks "mumble"); flaw bleeding into other sentences; reading like a news anchor.

## S3 Audio DSP & feature engineering
- **What:** F0, MFCC, spectral, energy, pace, pause features; speaker normalization; DTW word-wise comparison.
- **Tools:** librosa, pyworld/parselmouth, scipy.
- **Done when:** feature tests pass on synthetic signals with known ground truth; features are monotonic with severity (G3/G5).
- **Pitfalls:** absolute thresholds; octave errors in F0; unvoiced frames counted in pitch variance.

## S4 Forced alignment
- **What:** WhisperX (or MFA) word timestamps for ideal & participant; confidence & WER reporting.
- **Done when:** ideal-take WER ≤ 8%; alignment cache reproducible.
- **Pitfalls:** fallback uniform alignment leaking into benchmarks; error propagation into detection.

## S5 Controlled perturbation
- **What:** Deterministic, severity-parametrized flaw injection with exact ground truth.
- **Tools:** pyrubberband/librosa, pyworld, scipy.signal.
- **Done when:** G3 passes; seed → identical bytes.
- **Pitfalls:** audible artifacts that make synthetic trivially detectable (so also test on real).

## S6 Annotation & label QA
- **What:** Dual annotation, adjudication, acoustic verification, blind listening.
- **Done when:** A1 ≥ 95% with κ ≥ 0.80.
- **Pitfalls:** annotators seeing manifest labels; adjudicating toward model output.

## S7 Detection, causal explanation, calibration
- **What:** Rule/threshold + lightweight statistical detector; evidence-bearing explanations.
- **Tools:** numpy, scikit-learn (logistic/GBM only if interpretable), config-driven thresholds.
- **Done when:** G5, G6 pass.
- **Pitfalls:** tuning on test; opaque models that can't explain; overfitting to one speaker.

## S8 Evaluation & statistics
- **What:** IoU, P/R/F1, confusion matrices, Wilson CI, bootstrap, ablations, stress tests.
- **Done when:** `results/runs/` reproducible; every doc number traceable to a run.

## S9 Frontend / dashboard (your strength)
- **What:** Overlay waveform (ideal vs participant), region highlighting, click-to-play, explanation cards, severity chips, limitation panel.
- **Tools:** Streamlit baseline; React + wavesurfer.js + plotly/visx for polish; design tokens, dark mode.
- **Done when:** G7; a stranger understands the main finding in 10 seconds.
- **Pitfalls:** pretty but unexplained; loading heavy models in the UI thread.

## S10 Technical writing & demo storytelling
- **What:** ≤ 6-page doc, datasheet, README, 3–10 min video with a clear arc.
- **Done when:** G8, G9; someone else reproduces from the README.

## S11 Research ethics & licensing
- **What:** Consent forms, licences, privacy, "prior work" declaration.
- **Done when:** G0 and consent flags filled for every speaker.

## Agent capability profile (what to ask the agent to be)
| Role in session | Use for |
|---|---|
| Senior audio-ML engineer | S3, S4, S5, S7 |
| Data QA engineer | S1, S6, S8 |
| Frontend engineer | S9 |
| Technical editor | S10 |
| Skeptical reviewer | Run at the end of every phase: "find leakage, fabricated numbers, untested claims" |

## Suggested team split (4 people)
P1 data+recording lead · P2 DSP/detector · P3 dashboard/frontend · P4 eval, docs, video. Everyone records as a speaker.
