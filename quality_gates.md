# quality_gates.md — What ">95% accuracy" actually means here

"Dataset accuracy" is three different things. Mixing them up is how teams end up with inflated, indefensible numbers. We target all three, measure each separately, and report each with a confidence interval.

## The three accuracies
| ID | Name | Question it answers | Target | How to reach it |
|---|---|---|---|---|
| **A1** | Label accuracy | Is the flaw label/interval on each clip actually correct? | **≥ 95%** (aim ≥ 98%) | Controlled single flaw, scripted sentence, dual annotation, acoustic verification, blind listening, drop failures |
| **A2** | Detection accuracy | Does the system identify the right flaw type on clips it never saw? | **≥ 95%** flaw-type accuracy on sealed test, **stated with Wilson 95% CI** | Calibrated, speaker-normalized features; enough test clips |
| **A3** | Grounding quality | Does it point at the right time span? | Mean temporal IoU **≥ 0.60**, median timestamp error **≤ 300 ms** (stretch: IoU ≥ 0.70) | Real forced alignment, sentence-level anchoring |

Also required: **false-positive rate on ideal/clean clips ≤ 5%**, and **repeat-run consistency** (same input → same output).

## Honesty about A2 and small test sets
With N test clips, 95% accuracy means very few errors. N=20 → 1 error allowed, and the 95% CI is roughly 76–99%. So:
- Minimum test set: **≥ 60 clips** (target 80+) across **≥ 2 held-out transcripts and ≥ 2 speakers**.
- Report `k/N`, accuracy, Wilson CI, and confusion matrix. If lower bound < 90%, say "target met on point estimate; CI wide" — do not round up.
- Synthetic perturbation clips have exact ground truth and count toward test, but always report **real-only** and **synthetic-only** numbers separately. Judges trust the real-only number.
- Never claim 95% on a metric whose test clips were used for tuning.

## Gates (each produces `results/gates/Gx.json` + a one-paragraph note)
| Gate | Phase | Pass criteria (all must hold) |
|---|---|---|
| **G0** | 0 | Repo ownership/licence confirmed in writing; hackathon rule on prior work read and quoted; "Prior work" statement drafted |
| **G1** | 1 | ≥ 4 transcript groups (rec. 6), each 5–6 sentences, 55–80 s read time, sentence-indexed; ≥ 90% phoneme-class coverage across set; consent form signed by every speaker |
| **G2** | 2 | ≥ 2 speakers (rec. 3) × ≥ 4 transcripts ideal takes; all files 16 kHz mono; no clipping (peak < −1 dBFS); SNR ≥ 25 dB on ideal; WhisperX WER vs transcript ≤ 8% on ideal takes; manifest passes `validate_dataset.py` |
| **G3** | 3 | Perturbation engine reproducible (same seed → byte-identical WAV); each perturbation's measured effect in target interval is monotonic with severity; leakage check passes (perturbed clip's non-target region matches ideal within tolerance) |
| **G4** | 4 (**Oct 5 go/no-go**) | ≥ 40 validated paired clips, ≥ 3 transcript groups; **A1 ≥ 95%** on audit (see below); Cohen's κ ≥ 0.80 on flaw type, boundary agreement ≥ 85% within ±150 ms; blind-listen type agreement ≥ 95% on retained clips. **If red → pivot to Track A** |
| **G5** | 5 | Thresholds fitted on `train` only, frozen in `configs/pipeline.json` with commit hash; val metrics logged; no test access in git history of calibration scripts |
| **G6** | 6 | A2 ≥ 95% (point estimate) on sealed test with CI reported; A3 targets reported; FP ≤ 5%; stress tests (mic change, noise, new speaker, accent, speaking-rate shift) tabulated; repeat-run identical |
| **G7** | 7 | Dashboard runs from one command; overlay + flaw highlighting + causal explanation + honest-limitations panel; works with demo clips offline; 60 fps-feel interaction, no console errors |
| **G8** | 8 | ≤ 6-page doc, 3–10 min video, README setup verified on a clean machine/container by someone else |
| **G9** | 9 | Devpost complete, all members real names, public repo, unlisted YouTube link works logged-out, tag `v1.0-submission` |

## A1 audit procedure (label accuracy)
1. **By construction:** perturbation-generated clips have exact intervals → label correct by definition once G3 passes.
2. **Real clips — three independent checks, a clip is "accurate" only if all pass:**
   - *Dual annotation:* two annotators label flaw type + interval without seeing each other's labels or the manifest. Disagreements are adjudicated by a third pass; unresolved clips are dropped.
   - *Acoustic verification:* in the labeled interval the expected feature differs from the aligned ideal by ≥ 1.5 pooled-SD (z-score) in the expected direction, and non-target regions stay within ±1.0 SD. Run by `verify_flaw_labels.py`.
   - *Blind listening:* a person who has not seen labels hears a stratified 30% sample (all if < 80 clips) and names the flaw type. They must match.
3. **A1 = clips passing all checks / clips audited.** Failing clips are re-recorded or dropped, never "fixed" by editing the label to match the system output.
4. Record dropped-clip count and reasons. Reporting this is a strength, not a weakness.

## Anti-leakage checklist (check at G5 and G6)
- [ ] Same transcript never appears in two splits
- [ ] Same speaker's test transcript not used in calibration
- [ ] No threshold derived from test or from val-after-selection
- [ ] Perturbation parameters at test time are not the ones used to tune thresholds (use different seeds + severities mix)
- [ ] Ideal reference file for a test clip was not used as a training positive
- [ ] Manifest columns that leak the answer (`flaw_type`, `severity`, filename) are not read by the detector

## If a gate fails
| Failure | Do this |
|---|---|
| A1 < 95% | Find which flaw type fails; re-record those clips with clearer performance; tighten the script; never relabel to fit |
| A2 < 95% | Inspect confusion matrix; usually pace↔pause or flat↔mumble. Add a discriminating feature **using train/val only**; if still short, report honestly and show per-type results |
| A3 low | Check alignment WER first; then boundary policy (sentence vs word anchoring) |
| High FP | Widen the ideal-vs-ideal variation envelope using multiple ideal takes per speaker |
