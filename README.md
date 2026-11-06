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

**Earlier figures are withdrawn pending re-verification.** The numbers previously listed here (adapter 49.0% / 294 of 600; ScreenSpot 76.3%; UI-Vision 26.3%; tool 60/63; refusal 8/8; multi-step 11/11) could not be reproduced on 2026-11-06, and the artifacts that produced them are not included in this repository. Do not cite them.

**Measured 2026-11-06** (Qwen2.5-VL-3B base plus a LoRA adapter, normalized 0-1000 coordinates, tolerance 50 px, batch size 1):

| manifest | hits | rate |
|---|---:|---:|
| phase1_eval_600 (grounding) | 23/600 | 3.8% |
| uivision_300 | 8/300 | 2.7% |
| screenspot_300 | 15/300 | 5.0% |

An earlier absolute-pixel adapter scored 9-15/600 grounding, 2-5/300 UI-Vision and 5-10/300 ScreenSpot. Normalized coordinates did better than absolute pixels on this setup; all rates are low. Aggregates: `results/cua-trend-2026-11-06.json`. No prompts or images are redistributed. Adapter weights are not distributed (base model is under a non-commercial research licence).

**Invalid baseline:** UI-TARS-1.5-7B scored 0/600 in an earlier run because of a prompt/format mismatch; it is not evidence about that model and is excluded.

## What this does not prove
- It is not a field lie-rate measurement.
- It does not make an agent honest; it measures specific grounding/tool behaviors under a fixed prompt and tolerance.

## License
Apache-2.0. Dataset entries retain their upstream licenses (MIT / Apache-2.0).
