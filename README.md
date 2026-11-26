# gamai-evals

Evaluation harness for GUI grounding and tool-calling, with an honest contract: every number states its dataset, tolerance, and what it does **not** prove.

## What is here
- `cua_eval_v11.py` — grounding + tool/refusal/multi-step evaluator (batch size 1 by default; resumable JSONL).
- `tools/trend_table.py` — computes every rate and Wilson 95% interval in `docs/TREND_TABLE.md` from `results/cua-trend-2026-11-06.json`; `--check` fails if the committed table drifts from the generator.
- `evals/phase1_eval_600.json` — 600 element-grounding tasks (MIT).
- `evals/screenspot_300.json` — 300 ScreenSpot-v2 tasks (Apache-2.0).
- `evals/uivision_300.json` — 300 UI-Vision tasks (MIT).

**These are not three independent sets.** `phase1_eval_600` is exactly the **union** of
`screenspot_300` and `uivision_300` (verified by `tests/test_manifest_overlap.py`). Scoring
all three and reporting three numbers counts the same items twice; cite either the 600 as a
whole or the two halves, not both as separate evidence.

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
| training budget | max 602,112 / min 200,704 | **147/600 = 24.5%** | not measured* |
| union of both conventions (default budget) | - | 70/599 = 11.7% | 73/599 = 12.2% |

\* `results/cua-trend-2026-11-06.json` records no matched-budget cell for ck7750. The
139/600 = 23.2% that used to sit in this cell is not backed by any committed artifact, so
it is withdrawn here instead of repeated.

At the training budget the normalized adapter also scores **20.0% on UI-Vision (60/300)** and **29.0% on ScreenSpot (87/300)**.

Every rate in this section comes from a `hits/n` cell in `results/cua-trend-2026-11-06.json`,
and the intervals come from `tools/trend_table.py` (Wilson 95%, z = 1.959963984540054).
The rendered table is `docs/TREND_TABLE.md`; CI fails if it goes stale.

**The 3.8% figure was a measurement artifact.** The checkpoint emits absolute pixels (47.7% of raw coordinates exceed 1000), while the first run scaled them as 0-1000. Scoring the same generations in pixel space gives 9.5%.

**Absolute-pixel lineage** (trained and scored in pixels; the JSON does not record a matched
budget for these checkpoints): ck1500 9/600 = 1.5%, ck9500 15/600 = 2.5%, ck9500 UI-Vision
8/300 = 2.7%. The 68/600, 77/600 and 12.3% figures previously printed here do not appear
anywhere in `results/cua-trend-2026-11-06.json` and are withdrawn. Against the same JSON the
normalized matched-budget cell is about 10x the absolute-pixel cells (24.5% vs 1.5-2.5%), but
the JSON records no matched budget for the absolute lineage, so this is not a clean
budget-matched comparison; the earlier "about 2x" rested on figures that are not in the JSON.

**Protocol-limited baselines:** base Qwen2.5-VL-3B and UI-TARS-1.5-7B both score 0 because they emit their native output format instead of our JSON. This is not evidence about their grounding ability.

Full report: `docs/LUMI_BENCHMARK_2026-11-06.md`. No prompts, images, or weights are redistributed. The adapter's base model is under a non-commercial research licence.

## What this does not prove
- It is not a field lie-rate measurement.
- It does not make an agent honest; it measures specific grounding/tool behaviors under a fixed prompt and tolerance.

## License
Apache-2.0. Dataset entries retain their upstream licenses (MIT / Apache-2.0).

## For a buyer who says "prove the bot won't click the wrong button" (AI Lounge, 24 Nov 2026)

That sentence is a test request, not a claim. What this repo gives an auditor:

- a frozen task manifest per run (nothing changes between runs),
- GUI-grounding tasks (click the right element) and refusal/multi-step tasks,
- per-run JSON with the denominator, so a result is `x / n`, never a bare percentage.

The figures in the tables above are therefore given as `hits / n` with the convention named.
If a buyer wants their own number, the honest route is the one-week audit (28+ actions, each
CONFIRMED / REFUTED / UNVERIFIABLE with a signed receipt) rather than a benchmark on our tasks.
