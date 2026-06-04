import json, os, numpy as np
from collections import defaultdict
A = os.environ.get("JANUS_OUT", "/root/janus/artifacts")
COHORT = ["qwen3.5-9b", "glm4-9b", "qwen3-8b", "qwen2.5-7b-instruct"]
TAUT = {tuple(sorted(t)) for t in [
    ("eff_rank", "repr_entropy"), ("w_eff_rank", "w_stable_rank"),
    ("ll_kl_to_final", "ll_top1_depth"), ("tuned_lens_kl", "tuned_lens_depth"),
    ("massive_count", "massive_max"), ("grad_mag", "fisher"),
    ("hd_grad_mag", "hd_fisher"), ("ll_kl_to_final", "tuned_lens_kl"),
    ("ll_top1_depth", "tuned_lens_depth"), ("ll_kl_to_final", "tuned_lens_depth"),
    ("ll_top1_depth", "tuned_lens_kl")]}

pair_vals = defaultdict(dict)
present_models = []
for m in COHORT:
    p = f"{A}/{m}/facts.json"
    if not os.path.exists(p):
        continue
    present_models.append(m)
    d = json.load(open(p))
    corr = d["corr"]
    for a in corr:
        for b, v in corr[a].items():
            if a < b:
                pair_vals[(a, b)][m] = v

print(f"cohort present: {present_models}\n")
rows = []
for (a, b), mv in pair_vals.items():
    if tuple(sorted([a, b])) in TAUT:
        continue
    if len(mv) < max(3, len(present_models) - 1):
        continue
    vals = np.array(list(mv.values()))
    if np.sign(vals).min() != np.sign(vals).max():
        continue  # inconsistent sign across families -> not a robust fact
    rows.append((a, b, float(vals.mean()), float(vals.min()), float(vals.max()), len(mv)))

rows.sort(key=lambda r: -abs(r[2]))
print("=== CROSS-FAMILY CONSISTENT FACTS (same sign in all, |mean rho| desc) ===")
print(f"{'metric A':20s} {'metric B':20s} {'mean':>6s} {'min':>6s} {'max':>6s}  n")
for a, b, me, mn, mx, n in rows[:35]:
    print(f"{a:20s} {b:20s} {me:+.2f} {mn:+.2f} {mx:+.2f}  {n}")

# Janus coupling: what predicts forgetting (hd_fisher / hd_dW_drift) consistently
print("\n=== JANUS COUPLING: predictors of forgetting-vulnerability (cross-family) ===")
for target in ["hd_fisher", "hd_dW_drift"]:
    agg = defaultdict(list)
    for m in present_models:
        d = json.load(open(f"{A}/{m}/facts.json"))
        c = d["corr"].get(target, {})
        for k, v in c.items():
            if k != target:
                agg[k].append(v)
    print(f"\n  -> {target}:")
    items = [(k, np.mean(v), len(v)) for k, v in agg.items() if len(v) >= 3
             and np.sign(v).min() == np.sign(v).max()]
    items.sort(key=lambda x: -abs(x[1]))
    for k, v, n in items[:12]:
        print(f"     {k:20s} {v:+.2f}  (n={n})")
