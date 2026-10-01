# SpeechMirror — Track C Execution Pack
IIT Mandi Multimodal AI Hackathon 2026 · Track C: Contrastive Speech Analytics & Temporal Flaw Grounding
Build window: Oct 1 → Oct 14, 2026 (submit early on Oct 13/14). Judging Oct 15–20.

## What this pack is
A complete, ordered set of context files + paste-ready prompts to drive an AI coding agent (Claude Code / Cursor / any agent) through every phase, with measurable quality gates so the dataset and the detector are demonstrably accurate, not just claimed to be.

## Files
| File | Purpose |
|---|---|
| `CLAUDE.md` | Rules the agent must always follow. Drop in the repo root. |
| `skills.md` | Skills/roles needed (human + agent), with done-criteria and pitfalls |
| `requirements.md` | Numbered functional, dataset, metric and submission requirements |
| `reference.md` | Rubric, flaw taxonomy, severity parameters, features, schemas, metric formulas, CLI chain |
| `workflow.md` | Phase-by-phase workflow, dependencies, daily schedule, fallback |
| `quality_gates.md` | How ">95% accuracy" is defined and verified (read this first) |
| `prompts/00_master_prompt.md` | Paste at the start of every agent session |
| `prompts/01..09_*.md` | One prompt per phase |
| `prompts/fallback_trackA.md` | Pivot prompt if the Oct 5 gate fails |
| `templates/` | manifest.csv, flaw_regions.csv, annotations, transcripts, consent, checklist |

## How to use
1. Copy this folder's contents into the SpeechMirror repo (`docs/pack/`) and put `CLAUDE.md` in the repo root.
2. Read `quality_gates.md`, then `workflow.md`.
3. For each phase: open a fresh agent session → paste `prompts/00_master_prompt.md` → paste the phase prompt → review output against that phase's gate → commit.
4. Never advance a phase while its gate is red.

## Phase map
| # | Phase | Dates | Gate |
|---|---|---|---|
| 0 | Rules, ownership, repo audit | Oct 1 | G0 |
| 1 | Transcripts & protocol | Oct 1 | G1 |
| 2 | Recording & ingestion | Oct 2–4 | G2 |
| 3 | Controlled perturbation engine | Oct 2–4 (parallel) | G3 |
| 4 | Annotation & label QA | Oct 4–5 | **G4 (Oct 5 go/no-go)** |
| 5 | Pipeline + calibration | Oct 6–8 | G5 |
| 6 | Benchmark + stress tests + stats | Oct 8–9 | G6 |
| 7 | Dashboard | Oct 10–12 | G7 |
| 8 | Docs, video, Devpost | Oct 12–13 | G8 |
| 9 | Final audit & submit | Oct 13–14 | G9 |

## Honest caveat
Nothing here can guarantee a result. "95%" is defined precisely in `quality_gates.md` as verified targets with confidence intervals. If a gate is not met, the pack says what to fix, not how to hide it.
