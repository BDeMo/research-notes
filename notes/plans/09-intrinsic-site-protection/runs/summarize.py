#!/usr/bin/env python3
import json, glob, os, numpy as np
A = "/home/devuser/janus/artifacts"

def cap(m):
    print(f"===== {m} capeval =====")
    rows = []
    for f in sorted(glob.glob(f"{A}/{m}/cap_*.json")):
        d = json.load(open(f))
        rows.append((d["ckpt"], d.get("mmlu"), d.get("trivia"), d.get("math")))
    for c, mm, tr, ma in rows:
        print(f"  {c:36s} mmlu={mm} trivia={tr} math={ma}")
    return {c: (mm, tr, ma) for c, mm, tr, ma in rows}

def niah(m):
    out = {}
    for tag in ["_base", "_ablate_retr", "_ablate_sink"]:
        p = f"{A}/{m}/niah{tag}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            out[tag] = float(np.mean(list(d["grid"].values())))
    return out

print("################ FORGETTING / H3 (capeval) ################")
for m in ["qwen2.5-1.5b", "qwen2.5-1.5b-instruct", "qwen3-8b", "qwen2.5-7b-instruct",
          "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]:
    if os.path.isdir(f"{A}/{m}"):
        cap(m)

print("\n################ NIAH causal ablation (retrieval vs sink) ################")
for m in ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen2.5-1.5b"]:
    n = niah(m)
    if n:
        b = n.get("_base"); r = n.get("_ablate_retr"); s = n.get("_ablate_sink")
        print(f"  {m:14s} base={b}  ablate_retr={r}  ablate_sink={s}")

print("\n################ H3 Pareto (1.5b math): retain(MMLU+Trivia) vs plasticity(MATH) ################")
c = cap("qwen2.5-1.5b")
base = c.get("base")
if base:
    print(f"  {'variant':28s} {'MMLU':6s} {'Trivia':7s} {'retain_avg':10s} {'MATH(plast)':10s}")
    for k, v in c.items():
        mm, tr, ma = v
        ret = np.mean([x for x in [mm, tr] if x is not None]) if (mm is not None or tr is not None) else None
        print(f"  {k:28s} {str(mm):6s} {str(tr):7s} {str(round(ret,4) if ret else ret):10s} {str(ma):10s}")
