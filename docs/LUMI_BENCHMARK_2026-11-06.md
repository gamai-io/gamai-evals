# LUMI-G grounding and tool-calling benchmark (2026-11-06)

Hardware: LUMI-G (AMD MI250X, GCDs scheduled as GPUs). All numbers below were produced on that system.

## Grounding (Qwen2.5-VL-3B + LoRA), tolerance 50 px, greedy decoding

Evaluation protocol: image preprocessing matched to training (`max_pixels=602112`, `min_pixels=200704`); both coordinate conventions scored on the same generations. The pixel budget matters: the same checkpoint scores 3.7% under the harness default budget and 23.2% under the training budget. Always state the budget.

| checkpoint | ground /600 | UI-Vision /300 | ScreenSpot /300 | convention |
|---|---:|---:|---:|---|
| L2n ck4750 | 147 (24.5%) | 60 (20.0%) | 87 (29.0%) | normalized, scored norm |
| L2n ck7750 | 139 (23.2%) | - | - | normalized, scored norm |
| L1 ck1500 | 68 (11.3%) | 37 (12.3%) | - | absolute px, scored abs |
| L1 ck9500 | 77 (12.8%) | - | - | absolute px, scored abs |
| base Qwen2.5-VL-3B | 0 | 0 | 0 | JSON protocol not followed |
| UI-TARS-1.5-7B | 0 | - | - | JSON protocol not followed |

Notes:
- The 0% rows are a protocol limitation: both models emit their native output format, not our JSON, so the harness cannot score them. They are not evidence about those models' grounding ability.
- Normalized-coordinate training beat absolute-pixel training by about 2x on every set when the pixel budget is matched.
- Checkpoints 4750 -> 7750 show a plateau (24.5% -> 23.2%); more steps at this configuration did not help.
- These rates are low in absolute terms and are reported as a negative result.

## Tool calling (MuJo 9B LoRA), development sets

Sets are development sets, not independent holdouts: training data share templates and contain duplicated prompts.

| adapter | v24 tool (n=63) | v24 args | v25 tool (n=86) | v26 tool (n=50) |
|---|---:|---:|---:|---:|
| v5e3 | 46 (73.0%) | 24/54 | 84 (97.7%) | 31 (62.0%) |
| v6 | 51 (81.0%) | 31/54 | 83 (96.5%) | 36 (72.0%) |
| v7 | 53 (84.1%) | 30/54 | 73 (84.9%) | 39 (78.0%) |
| v8 | 52 (82.5%) | 21/54 | 86 (100%) | 37 (74.0%) |

Notes:
- v25 is format-symmetric (SQL echoed back, URL tasks) and saturates; 100% there does not demonstrate general capability.
- `args` is exact string comparison on expected keys; extra keys are not penalised.
- The earlier attribution of 46/63 and 81/86 to v5e3/v6 was wrong (stale adapter path) and is superseded by `docs/CORRECTION-2026-11-06.md` in agentmujo-training.
- Aggregates with Wilson 95% intervals: `results/cua-trend-2026-11-06.json` and, in agentmujo-training, `benchmarks/results/lumi-mujo-2026-11-06-devset.json`.

## What this does not prove

- Not end-to-end agent success, not production readiness, not a safety guarantee.
- Small sets; no significance testing beyond the reported intervals.
- No prompts, images, or model weights are redistributed. Base model licence is non-commercial research.

## Reproduce

`python cua_eval_v12.py --base <model> --adapter <adapter> --ground-set evals/phase1_eval_600.json --tol 50 --match-train --coord-mode both --out out.json`
