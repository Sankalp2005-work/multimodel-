# CLAUDE.md — Project rules for SpeechMirror (Track C)

## Mission
Contrastive speech analytics: compare a participant recording to an ideal reference of the SAME transcript, detect flaw regions, ground them in time, and explain them causally with measured acoustic evidence.

## Non-negotiable rules
1. **Test split is sealed.** Never tune thresholds, weights, prompts or code on `split=test`. Calibrate on `train`, select on `val`. Report test once per release, with the git commit hash.
2. **No fabricated numbers.** Every metric in docs/README/video must come from a saved run in `results/` with config + commit hash. If a result is bad, report it.
3. **Synthetic is labeled synthetic.** Test tones, TTS, and perturbation-generated clips carry `source_type` accordingly. Never present them as human recordings.
4. **Provenance.** Every audio file has sha256, source_type, license, speaker_id, consent flag in the manifest.
5. **Deterministic.** Fixed seeds, pinned dependencies, same output on rerun (tolerance in requirements.md MR-7).
6. **Speaker-agnostic.** Features are normalized per speaker/recording against the ideal reference; no absolute-Hz or absolute-dB thresholds.
7. **Fallback alignment is not alignment.** The "uniform estimate" fallback may be used for UI demos only, never for benchmarks. Benchmarks must use WhisperX (or MFA) alignment.
8. **Small, verifiable steps.** After each change: run tests + `validate_dataset.py`. Do not move on with red tests.
9. **Ask before destructive actions** (deleting audio, rewriting manifests). Write new versions, don't overwrite.
10. **Authorship transparency.** Pre-existing code from before Oct 1 is declared in README under "Prior work". New work is in clearly dated commits.

## Conventions
- Python 3.10+, type hints, ruff + black, unittest/pytest.
- Audio: mono, 16 kHz, 16-bit PCM WAV. Loudness is NOT normalized in stored files (keep raw level; normalize inside features).
- Times in seconds (float, 3 decimals) on the **participant audio timeline**.
- Config in `configs/`, no magic numbers in code.
- Every script has `--help`, a seed arg, and writes a JSON run manifest.

## Definition of done (any task)
Code + tests + docs line + gate evidence file in `results/gates/`.
