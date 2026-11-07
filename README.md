# gamai-evals

Evaluation harness for GUI grounding and tool-calling, with an honest contract: every number states its dataset, tolerance, and what it does **not** prove.

## What is here
- `cua_eval_v11.py` — grounding + tool/refusal/multi-step evaluator (batch size 1 by default; resumable JSONL).
- `evals/phase1_eval_600.json` — 600 UI-Vision element-grounding tasks (MIT).
- `evals/screenspot_300.json` — 300 ScreenSpot-v2 tasks (Apache-2.0).
- `evals/uivision_300.json` — 300 UI-Vision tasks (MIT).

Each manifest entry has `image`, `instruction`, target `x/y/w/h`, `source`, `license`, `platform`.

## Images are not bundled
Manifests reference upstream images (e.g. `/data/data/raw/...`). Download the public datasets yourself and point the evaluator at them:

```bash
export EVAL_IMAGE_ROOT=/path/to/datasets   # replaces /data/data/raw
python cua_eval_v11.py --base <model> --adapter <adapter> \
  --ground-set evals/phase1_eval_600.json --tol 50 --batch-size 1 --out out/v1.json
```

## Results - verification status (2026-11-06)

**Earlier figures are withdrawn.** The previously listed numbers (adapter 49.0% / 294 of 600; ScreenSpot 76.3%; UI-Vision 26.3%; tool 60/63; refusal 8/8; multi-step 11/11) could not be reproduced and their artifacts are not in this repository. Do not cite them.

Pixel budget dominates the measurement. Three regimes, same checkpoints:

| regime | image budget | ck4750 ground | ck7750 ground |
|---|---|---:|---:|
| harness default | max 1,003,520 / min 3,136 | - | 22/599 = 3.7% (norm), 57/599 = 9.5% (abs) |
| training budget | max 602,112 / min 200,704 | **147/600 = 24.5%** | **139/600 = 23.2%** |
| union of both conventions (default budget) | - | 70/599 = 11.7% | 73/599 = 12.2% |

At the training budget the normalized adapter also scores **20.0% on UI-Vision (60/300)** and **29.0% on ScreenSpot (87/300)**.

**The 3.8% figure was a measurement artifact.** The checkpoint emits absolute pixels (47.7% of raw coordinates exceed 1000), while the first run scaled them as 0-1000. Scoring the same generations in pixel space gives 9.5%.

**Absolute-pixel lineage** (trained and scored in pixels, training budget): ck1500 68/600 = 11.3%, ck9500 77/600 = 12.8% (UI-Vision 12.3%). Normalized-coordinate training is about 2x better when the budget is matched.

**Protocol-limited baselines:** base Qwen2.5-VL-3B and UI-TARS-1.5-7B both score 0 because they emit their native output format instead of our JSON. This is not evidence about their grounding ability.

Full report: `docs/LUMI_BENCHMARK_2026-11-06.md`. No prompts, images, or weights are redistributed. The adapter's base model is under a non-commercial research licence.

## What this does not prove
- It is not a field lie-rate measurement.
- It does not make an agent honest; it measures specific grounding/tool behaviors under a fixed prompt and tolerance.

## License
Apache-2.0. Dataset entries retain their upstream licenses (MIT / Apache-2.0).
