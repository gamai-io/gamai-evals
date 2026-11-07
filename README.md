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

## Results - verification status (2026-11-06, corrected)

**Earlier figures are withdrawn.** The previously listed numbers (adapter 49.0% / 294 of 600; ScreenSpot 76.3%; UI-Vision 26.3%; tool 60/63; refusal 8/8; multi-step 11/11) could not be reproduced on 2026-11-06 and their artifacts are not in this repository. Do not cite them.

**Measurement bug found and corrected.** The first 2026-11-06 run scored 23/600 (3.8%) using `--coord-mode norm`, but the checkpoint emits absolute pixels: 47.7% of raw coordinates exceed 1000 and 286/599 fall outside a 0-1000 range. Re-scoring the same generations in pixel space gives 9.0% (checkpoint 4750) and 9.5% (checkpoint 7750); accepting either convention gives 11.7% and 12.2%. The 3.8% figure is an artifact of the coordinate-mode mismatch, not a fair model measurement.

| checkpoint | abs | norm (as first reported) | union |
|---|---:|---:|---:|
| norm-ck4750 | 54/599 = 9.0% | 23/599 = 3.8% | 70/599 = 11.7% |
| norm-ck7750 | 57/599 = 9.5% | 22/599 = 3.7% | 73/599 = 12.2% |

**Second mismatch (being measured).** Training used `max_pixels=602112` while `cua_eval_v12.py` defaults to `1003520`; 564/600 eval images therefore have a different visual token grid (mean 1.64x more tokens at eval). A diagnostic run with matched budgets is in progress; results will be appended.

An earlier absolute-pixel checkpoint scored 9-15/600 ground, 2-8/300 UI-Vision, 5-10/300 ScreenSpot. Normalized-coordinate training did **not** learn the normalized convention: the model keeps emitting pixel coordinates.

Aggregates: `results/cua-trend-2026-11-06.json`. No prompts or images are redistributed. Adapter weights are not distributed (base model is under a non-commercial research licence).

**Invalid baseline:** UI-TARS-1.5-7B scored 0/600 in an earlier run because of a prompt/format mismatch; it is not evidence about that model and is excluded.

## LUMI-G benchmark
Full protocol and numbers: `docs/LUMI_BENCHMARK_2026-11-06.md` (grounding and tool calling, with explicit limitations).

## What this does not prove
- It is not a field lie-rate measurement.
- It does not make an agent honest; it measures specific grounding/tool behaviors under a fixed prompt and tolerance.

## License
Apache-2.0. Dataset entries retain their upstream licenses (MIT / Apache-2.0).
