# workflow.md

## End-to-end flow
```
P0 Rules/ownership ─► P1 Transcripts+protocol ─► P2 Record ideal+flawed (real) ─┐
                                              └► P3 Perturbation engine (synthetic)├─► P4 Annotate + label QA ─► [Oct 5 GO/NO-GO]
                                                                                   ┘
   GO ─► P5 Align+features+calibrate (train/val) ─► P6 Sealed-test benchmark + stress ─► P7 Dashboard ─► P8 Docs+video ─► P9 Audit+submit
   NO-GO ─► prompts/fallback_trackA.md
```

## Schedule
| Date | Phase | Output | Owner |
|---|---|---|---|
| Oct 1 | P0, P1 | ownership resolved, transcripts T1–T6, consent signed, recording checklist | all |
| Oct 2 | P2 start, P3 start | ideal takes all speakers; perturbation engine v0 | P1 / P2 |
| Oct 3 | P2 | flawed takes (4 flaws × sev 2 and 4) | P1 + speakers |
| Oct 4 | P2 finish, P3 finish, P4 start | manifests built, perturbations generated, annotation round 1 | P1 / P4 |
| Oct 5 | P4 | **G4 go/no-go** | all |
| Oct 6–7 | P5 | alignment cached, features, frozen thresholds on train/val | P2 |
| Oct 8 | P5 → P6 | calibration frozen, tag `calib-v1` | P2 |
| Oct 9 | P6 | sealed test run, stress tests, stats | P4 |
| Oct 10–12 | P7 | dashboard polish | P3 |
| Oct 12 | P8 | doc draft, video script | P4 |
| Oct 13 | P8, P9 | video recorded, README verified by someone else, Devpost draft | all |
| Oct 14 | P9 | **submit by noon**, EOD is buffer | all |

## Phase detail
Each phase: **Inputs → Steps → Outputs → Gate → Failure path**. Prompts for each step live in `prompts/`.

### P0 Rules, ownership, repo audit (prompts/01)
1. Confirm repo ownership/licence; read hackathon rules on prior work (quote the rule in `docs/prior_work.md`).
2. Add LICENSE (if team owns it), CLAUDE.md, pack docs.
3. Run existing tests; baseline repo audit; list gaps.
4. Branch `hackathon-build`; tag `pre-hackathon-baseline` for transparent diff.
Gate G0.

### P1 Transcripts & protocol (prompts/02)
1. Validate T1–T4, generate T5–T6 with same constraints.
2. Check phonetic coverage and numeric/proper-noun diversity.
3. Produce sentence-indexed JSON and a printable reading sheet for speakers.
4. Prepare consent form and recording checklist; assign speaker IDs.
Gate G1.

### P2 Recording & ingestion (prompts/03)
1. Each speaker: 2 ideal takes per transcript, then flawed takes (4 types × sev 2 and 4) on designated sentences.
2. Run `ingest_audio.py`: convert, trim, QC (clipping, SNR, duration), hash.
3. WhisperX WER check on ideal takes; re-record failures.
4. Build `manifest.csv` with `split`.
Gate G2.

### P3 Perturbation engine (prompts/04)
1. Implement 4 perturbations per `reference.md §2.1` with deterministic seeds.
2. Generate sev 1 and 3 (and optional sev 2/4 duplicates) from real ideal takes; label `synthetic_perturbation`.
3. Verify monotonicity, leakage, artifacts.
Gate G3.

### P4 Annotation & label QA (prompts/05)
1. Blind annotation round: annotators A and B label type + interval via simple tool.
2. Compute κ, boundary agreement; adjudicate.
3. Run `verify_flaw_labels.py`; blind listening on stratified sample.
4. Compute A1; drop/re-record failures; freeze v1 dataset (tag `dataset-v1`, hashes).
Gate G4 — **Oct 5**.

### P5 Pipeline and calibration (prompts/06)
1. Align all clips; cache.
2. Implement normalization, contrastive deltas, detectors, explainers.
3. Fit thresholds on train; select on val; freeze config; tag `calib-v1`.
Gate G5.

### P6 Benchmark & stress tests (prompts/07)
1. One run on sealed test; save run manifest.
2. Stress sets: mic change, noise (SNR 20/10 dB), room, new speaker, accent, tempo; ideal-only FP test.
3. Stats: Wilson CI, bootstrap IoU CI, confusion matrices, ablations.
Gate G6.

### P7 Dashboard (prompts/08)
Implements FR-8…FR-10. Gate G7.

### P8 Docs, video, Devpost (prompts/09)
6-page doc, datasheet, README, demo script, Devpost text. Gate G8.

### P9 Final audit & submission (prompts/10)
Reviewer pass, clean-machine run, tag, submit. Gate G9.

## Daily rhythm (recommended)
Morning: 15-min standup (gate status, blockers). Work in 90-min blocks. Evening: commit, update `results/gates/`, agent "skeptical reviewer" pass.

## Risk register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Repo ownership/prior-work rule | Med | Disqualification | G0 first; email organizers; declare openly |
| Recordings inconsistent | Med | A1 low | Checklist, same mic, 2 takes, dual QA |
| WhisperX fails on mumble | Med | A3 low | Anchor to ideal sentence boundaries; confidence + abstain |
| Test set too small for CI | High | Weak claim | ≥ 60 test clips, ≥ 2 speakers, report real-only |
| Time overrun | Med | Missed submit | Oct 5 gate; submit by noon Oct 14 |
| Overfitting to synthetic | Med | Fails on real | Calibrate on real+synthetic, report real-only separately |
