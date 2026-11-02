#!/usr/bin/env python3
"""EVAL_V11 - incremental + resumable + batched grounding eval (nasljednik v10).

Ista logika/prompt kao v10, ali:
  * svaki rezultat se ODMAH upisuje u <out>.jsonl (timeout cuva djelomicno)
  * --resume (default ON ako jsonl postoji): preskace vec uradjene uzorke
  * --batch-size N: batch inference (default 1 = kao v10); fallback na 1 ako batch padne
  * --limit N: samo prvih N (smoke test)
  * progress u log svakih --log-every uzoraka

Primjer:
  python3 cua_eval_v11.py --base /models/models/qwen25vl3b --adapter /data/out/v25_lumi \
    --ground-set /data/evals/phase1_eval_600.json --blank /models/blank.png --tol 50 \
    --out /data/out/eval_v25_phase1.json --batch-size 8
"""
import argparse, json, os, re, sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

IMG_ROOT = os.environ.get("EVAL_IMAGE_ROOT", "")
def _img(p):
    if IMG_ROOT and p.startswith("/data/data/raw"):
        return IMG_ROOT.rstrip("/") + p[len("/data/data/raw"):]
    return p

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated")
ap.add_argument("--adapter", default="")
ap.add_argument("--tool-set", default="")
ap.add_argument("--ground-set", default="")
ap.add_argument("--tol", type=int, default=50)
ap.add_argument("--blank", default="")
ap.add_argument("--out", default="/home/ubuntu/eval/res_v11.json")
ap.add_argument("--jsonl", default="")
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--log-every", type=int, default=25)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--no-resume", action="store_true")
ap.add_argument("--max-pixels", type=int, default=0)
a = ap.parse_args()
JSONL = a.jsonl or (a.out + ".jsonl")
if a.no_resume and os.path.exists(JSONL):
    open(JSONL, "w").close()

proc = AutoProcessor.from_pretrained(a.base)
if getattr(proc, "tokenizer", None) is not None:
    proc.tokenizer.padding_side = "left"
if a.max_pixels:
    try: proc.image_processor.max_pixels = a.max_pixels
    except Exception: pass

model = AutoModelForImageTextToText.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="auto")
if a.adapter:
    model = PeftModel.from_pretrained(model, a.adapter)
    # ADAPTER_KEY_REMAP_V1: Unsloth saves keys without .default -> remap if needed
    try:
        from safetensors.torch import load_file as _lf
        import os as _os
        _sd = _lf(_os.path.join(a.adapter, "adapter_model.safetensors"))
        _ns = dict(model.named_parameters())
        if sum(1 for k in _sd if k not in _ns) > 0:
            _rm = {}
            for k, v in _sd.items():
                k2 = k.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                _rm[k2] = v
            model.load_state_dict(_rm, strict=False)
            print("[ADAPTER_KEY_REMAP_V1] applied remap for", a.adapter)
    except Exception as _e:
        print("[ADAPTER_KEY_REMAP_V1] skip:", _e)
model.eval()
SYS = 'Ti si GAMA agent. Za klik vrati {"tool":"tap","args":{"x":X,"y":Y}}. Za alat {"tool":"...","args":{...}}. Za vise koraka {"steps":[...]}.'


def parse(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build(msgs):
    if a.blank:
        has = any(isinstance(p, dict) and p.get("type") == "image"
                  for m in msgs for p in (m.get("content") or []))
        if not has:
            msgs = [{"role": msgs[0]["role"],
                     "content": [{"type": "image", "image": a.blank}] + list(msgs[0].get("content") or [])}] + msgs[1:]
    pre = [] if os.environ.get("EVAL_NO_SYS", "1") == "1" else [{"role": "system", "content": [{"type": "text", "text": SYS}]}]
    text = proc.apply_chat_template(pre + msgs, tokenize=False, add_generation_prompt=True)
    imgs = []
    for m in msgs:
        for p in m.get("content") or []:
            if isinstance(p, dict) and p.get("type") == "image":
                imgs.append(Image.open(_img(p["image"])).convert("RGB"))
    return text, imgs


@torch.no_grad()
def gen(batch):
    texts = [t for t, _ in batch]
    imgs_ll = [im for _, im in batch]
    if any(len(im) > 0 for im in imgs_ll):
        inp = proc(text=texts, images=imgs_ll, return_tensors="pt", padding=True).to(model.device)
    else:
        inp = proc(text=texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=96, do_sample=False)
    n = inp["input_ids"].shape[1]
    dec = [d.strip() for d in proc.batch_decode(out[:, n:], skip_special_tokens=True)]
    if os.environ.get("EVAL_DEBUG_RAW") == "1":
        for _d in dec:
            print("[RAW_OUT]", repr(_d[:300]), flush=True)
    return dec


def run_one(msgs):
    return gen([build(msgs)])[0]


def run_batch(msgs_list):
    if a.batch_size <= 1 or len(msgs_list) == 1:
        return [run_one(m) for m in msgs_list]
    try:
        return gen([build(m) for m in msgs_list])
    except Exception as e:
        sys.stderr.write("batch fail -> per-sample: %r\n" % (e,))
        sys.stderr.flush()
        return [run_one(m) for m in msgs_list]


ROWS = {}
if not a.no_resume and os.path.exists(JSONL):
    for line in open(JSONL, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            ROWS[r["key"]] = r
        except Exception:
            pass


def emit(row):
    ROWS[row["key"]] = row
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def do_tool():
    if not (a.tool_set and os.path.exists(a.tool_set)):
        return
    items = json.load(open(a.tool_set, encoding="utf-8"))
    if a.limit:
        items = items[:a.limit]
    todo = [(i, it) for i, it in enumerate(items) if ("t%d" % i) not in ROWS]
    done = 0
    i = 0
    while i < len(todo):
        chunk = todo[i:i + max(1, a.batch_size)]
        outs = run_batch([[{"role": "user", "content": [{"type": "text", "text": it["q"]}]}] for _, it in chunk])
        for (idx, it), g in zip(chunk, outs):
            j = parse(g)
            exp = it["tool"]
            if exp == "multi":
                st = (j or {}).get("steps") if isinstance(j, dict) else None
                ok = isinstance(st, list) and len(st) >= 2
            elif exp == "refuse":
                ok = isinstance(j, dict) and j.get("tool") == "refuse"
            else:
                ok = isinstance(j, dict) and j.get("tool") == exp
            emit({"key": "t%d" % idx, "kind": "tool", "q": it["q"], "exp": exp,
                  "ok": ok, "json": j is not None, "raw": g[:120]})
            done += 1
            if a.log_every and done % a.log_every == 0:
                sys.stderr.write("[tool %d/%d]\n" % (done, len(todo)))
                sys.stderr.flush()
        i += len(chunk)


def do_ground():
    if not (a.ground_set and os.path.exists(a.ground_set)):
        return
    items = json.load(open(a.ground_set, encoding="utf-8"))
    if a.limit:
        items = items[:a.limit]
    todo = [(i, it) for i, it in enumerate(items) if ("g%d" % i) not in ROWS]
    done = 0
    i = 0
    while i < len(todo):
        chunk = todo[i:i + max(1, a.batch_size)]
        reqs = []
        for _, it in chunk:
            reqs.append([{"role": "user", "content": [
                {"type": "image", "image": it["image"]},
                {"type": "text", "text": "Klikni na: %s" % it["instruction"]}]}])
        outs = run_batch(reqs)
        for (idx, it), g in zip(chunk, outs):
            j = parse(g) or {}
            xy = (j.get("args") or {}) if j.get("tool") == "tap" else {}
            try:
                x, y = int(float(xy.get("x"))), int(float(xy.get("y")))
                ok = abs(x - it["x"]) <= a.tol and abs(y - it["y"]) <= a.tol
            except Exception:
                ok = False
            emit({"key": "g%d" % idx, "kind": "ground", "q": it["instruction"],
                  "exp": [it["x"], it["y"]], "got": xy, "ok": ok, "source": it.get("source", "unknown"), "platform": it.get("platform")})
            done += 1
            if a.log_every and done % a.log_every == 0:
                sys.stderr.write("[ground %d/%d]\n" % (done, len(todo)))
                sys.stderr.flush()
        i += len(chunk)


do_tool()
do_ground()
res = {"tool_total": 0, "tool_ok": 0, "json_ok": 0, "ground_total": 0, "ground_ok": 0,
       "refusal_ok": 0, "refusal_total": 0, "multi_ok": 0, "multi_total": 0}
for r in ROWS.values():
    if r["kind"] == "ground":
        res["ground_total"] += 1
        res["ground_ok"] += 1 if r["ok"] else 0
    elif r["kind"] == "tool":
        res["tool_total"] += 1
        res["tool_ok"] += 1 if r["ok"] else 0
        res["json_ok"] += 1 if r.get("json") else 0
        if r.get("exp") == "multi":
            res["multi_total"] += 1
            res["multi_ok"] += 1 if r["ok"] else 0
        elif r.get("exp") == "refuse":
            res["refusal_total"] += 1
            res["refusal_ok"] += 1 if r["ok"] else 0
res["tool_acc"] = round(res["tool_ok"] / max(res["tool_total"], 1), 3)
res["json_rate"] = round(res["json_ok"] / max(res["tool_total"], 1), 3)
res["ground_acc"] = round(res["ground_ok"] / max(res["ground_total"], 1), 3)
res["multi_acc"] = round(res["multi_ok"] / max(res["multi_total"], 1), 3)
res["refusal_acc"] = round(res["refusal_ok"] / max(res["refusal_total"], 1), 3)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
