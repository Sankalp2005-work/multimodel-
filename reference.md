# reference.md

## 1. Rubric (Track C) — where points come from
| Criterion | Weight | What wins it |
|---|---|---|
| Data Engineering & Stress Testing | **30%** | Real contrastive dataset, severity gradient, dual-annotated intervals, stress sets, documented construction, datasheet |
| Causal Explainability & Temporal Grounding | 25% | Timestamped regions + measured causes, IoU/timestamp-error numbers |
| Feature Extraction | 20% | MFCC/FFT/pitch/F0/pace/pauses, correct, speaker-normalized |
| Dashboard | 15% | Overlay, flaw highlight, click-to-play, explanation, limits |
| Reproducibility & Code Quality | 10% | One-command run, tests, seeds, run manifests |
Source: dossier; **re-verify against the official Track C PDF before submission.**

## 2. Flaw taxonomy and severity parameters
One flaw type per clip, applied to ONE designated sentence (`flaw_sentence`).

| Severity | Meaning |
|---|---|
| 0 | Ideal / near-perfect |
| 1 | Barely noticeable |
| 2 | Clearly noticeable |
| 3 | Distracting |
| 4 | Severely botched |

### 2.1 Controlled-perturbation parameters (synthetic_perturbation, Phase 3)
| Flaw | Operation | Sev 1 | Sev 2 | Sev 3 | Sev 4 |
|---|---|---|---|---|---|
| `rushed_pace` | Pitch-preserving time-compress the target sentence (speed factor) | 1.20× | 1.40× | 1.60× | 1.80× |
| `dead_pause` | Insert digital-silence-plus-room-tone at a word boundary mid-sentence | 0.5 s | 1.0 s | 1.7 s | 2.5 s |
| `flat_pitch` | Compress F0 contour toward sentence mean (retained log-F0 variance) | 75% | 50% | 25% | ≤5% |
| `mumbled_clarity` | Gain reduction + high-frequency/consonant smear (low-pass cutoff), applied with 30 ms crossfade | −3 dB, LP 6 kHz | −6 dB, LP 4.5 kHz | −9 dB, LP 3.5 kHz | −12 dB, LP 2.5 kHz |
Notes: perturbed region on the participant timeline is the modified span (for pause: the inserted silence; for pace: compressed sentence span). Subsequent audio shifts in time; compare by aligned words, never raw time.

### 2.2 Human-performance guide (real recordings)
| Flaw | Perform it as | Do NOT |
|---|---|---|
| rushed_pace | Race through the target sentence, ~1.4× (sev 2) / ~1.8× (sev 4), words still intelligible | Skip words; change the transcript |
| dead_pause | Stop mid-sentence at a natural break for 1 s (sev 2) / 2.5 s+ (sev 4) | Pause at sentence ends (not a flaw) |
| flat_pitch | Monotone, robotic delivery on the target sentence | Whisper or change loudness |
| mumbled_clarity | Slur consonants, soften articulation, lower projection | Move away from mic; speak into collar |
Use a metronome-free, timer-assisted approach: phone stopwatch for pauses; read sentence once aloud at ideal pace first to know the baseline duration.

## 3. Features and expected direction under each flaw
| Feature | Definition | Rushed | Pause | Flat | Mumble |
|---|---|---|---|---|---|
| Local speech rate | words/s or syllables/s over sliding word window, ratio to ideal aligned span | ↑↑ | ~ | ~ | ~ |
| Pause duration | silent gaps between aligned words (ms), excess over ideal gap | ↓ | ↑↑ | ~ | ~ |
| F0 variance | variance of log-F0 in semitones over voiced frames in region (z vs ideal) | ~ | ~ | ↓↓ | ~ |
| F0 range | 10th–90th percentile of semitone F0 | ~ | ~ | ↓↓ | ~ |
| RMS energy | dB RMS per frame, normalized per recording by median of ideal speech | ~ | ~ | ~ | ↓ |
| Spectral centroid | Hz, mean over voiced frames, ratio to ideal | ~ | ~ | ~ | ↓ |
| Spectral flatness | geometric/arithmetic mean power ratio | ~ | ~ | ~ | ↑ |
| MFCC Δ distance | DTW-aligned (word-wise) Euclidean/cosine distance on MFCC 1–13 vs ideal | ↑ | ~ | ~ | ↑↑ |
| ZCR | zero-crossing rate | ~ | ~ | ~ | ↓/↑ |
Discriminators for commonly confused pairs:
- Pace vs Pause: pace changes *rate within words/gaps compressed*; pause adds *a long silent gap*. Use gap length plus word-duration ratio.
- Flat vs Mumble: flat changes *F0 variance only*; mumble changes *energy + spectral + MFCC* with F0 variance mostly preserved.

## 4. Metric definitions
- **Temporal IoU** for matched pred/true region of same type: |∩| / |∪|. Match greedily by max IoU; unmatched = FP/FN.
- **Region detection** (A3): precision = TP/(TP+FP), recall = TP/(TP+FN), TP if type matches and IoU ≥ 0.3 (also report at 0.5).
- **Timestamp error:** |start_pred − start_true| and |end_pred − end_true|, report median and p90.
- **Flaw-type accuracy (A2):** clip-level: predicted dominant flaw type == truth (ideal clips: predicted "none").
- **Severity MAE:** mean |pred_sev − true_sev| over TP regions.
- **FP rate on ideal:** fraction of ideal/clean clips with ≥ 1 flagged region.
- **Wilson 95% interval** for k/n: center = (p + z²/2n)/(1 + z²/n); half = z·sqrt(p(1−p)/n + z²/4n²)/(1 + z²/n), z=1.96.
- **Cohen's κ:** (po − pe)/(1 − pe) over annotator flaw-type labels.
- **Boundary agreement:** share of annotator pairs whose start and end both lie within ±150 ms.
- **Acoustic verification z:** (mean feature in labeled interval, participant − aligned ideal) / pooled SD of ideal-vs-ideal differences.

## 5. File schemas
### manifest.csv
`sample_id, text_id, speaker_id, take_id, transcript, ideal_audio, participant_audio, flaw_type, severity, flaw_sentence, source_type, source_license, consent, mic, room, noise_condition, sha256, duration_s, split`
- `flaw_type` ∈ {none, rushed_pace, dead_pause, flat_pitch, mumbled_clarity}
- `source_type` ∈ {real, synthetic_perturbation, tts, stress_real}
- `split` ∈ {train, val, test}

### flaw_regions.csv (one row per region; multiple per sample allowed)
`flaw_id, sample_id, flaw_type, start_sec, end_sec, severity, feature_expected, annotation_source, adjudicated`
- Times on participant timeline. `annotation_source` ∈ {construction, annotator_A, annotator_B, adjudicated}.

### Detector output (JSON)
`{sample_id, alignment:{method, wer, mean_conf}, regions:[{type, start, end, severity, confidence, evidence:[{feature, ref, obs, delta_pct, z}], explanation}], scores:{pace,pause,pitch,energy,clarity,overall}, run:{commit, config_hash, seed}}`

## 6. Repo layout (target)
```
SpeechMirror/
  CLAUDE.md  README.md  LICENSE  Makefile  docker-compose.yml
  configs/pipeline.json  configs/perturb.json
  data/raw/  data/processed/  data/manifests/{manifest.csv,flaw_regions.csv}
  src/speechmirror/{align,features,normalize,compare,detect,explain,perturb,metrics}.py
  scripts/{build_real_dataset,validate_dataset,align_dataset,extract_features,benchmark,
           generate_perturbations,verify_flaw_labels,split_dataset,stats_report}.py
  app/{api,dashboard}/
  tests/
  results/{gates,runs,figures}/
  docs/{technical_doc.md,datasheet.md,limitations.md,pack/}
```

## 7. Command chain (from repo; verify flags with --help)
```
python scripts/build_real_dataset.py --source-manifest data/manifests/source.csv
python scripts/validate_dataset.py
python scripts/generate_perturbations.py --config configs/perturb.json --seed 7
python scripts/verify_flaw_labels.py --manifest data/manifests/manifest.csv
python scripts/align_dataset.py --method whisperx
python scripts/extract_features.py
python scripts/benchmark.py --method whisperx --split val      # during calibration
python scripts/benchmark.py --method whisperx --split test     # ONCE per release
python scripts/stats_report.py --run results/runs/<id>
```

## 8. Useful notes
- WhisperX alignment degrades on very low-volume mumble; use `no_speech` and confidence reporting, and anchor to sentence boundaries from the ideal when word confidence is low.
- Use log-F0 in semitones relative to speaker median; pyworld or Praat-Parselmouth for stable F0.
- Keep ≥ 2 ideal takes per speaker-transcript to estimate natural variation (ideal-vs-ideal envelope).
- Do not use TED/YouTube audio inside the dataset; it is licensed. Use it only in the video as a short commentary clip if rules/licence allow, or skip.
