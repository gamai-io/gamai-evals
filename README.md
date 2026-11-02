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

## Results reported on gamai.io
CUA adapter: 49.0% (294/600) vs base model 0.0%; ScreenSpot 76.3% (229/300); UI-Vision 26.3% (79/300); tool 60/63; refusal 8/8; multi-step 11/11.

These are **internal evaluations**: base model and adapter weights are not distributed here, so third parties cannot reproduce them bit-for-bit. We label them as such on the site.

## What this does not prove
- It is not a field lie-rate measurement.
- It does not make an agent honest; it measures specific grounding/tool behaviors under a fixed prompt and tolerance.

## License
Apache-2.0. Dataset entries retain their upstream licenses (MIT / Apache-2.0).
