#!/usr/bin/env python3
"""Janus grid aggregator -> big tables (T1-T4) organized by LC vs CF frontier.
Reads grid_{dataset}.npz for each model in $JANUS_OUT. Robust to partial completion."""
import os, json, glob, numpy as np
from collections import defaultdict
A = os.environ.get("JANUS_OUT", "/root/janus/artifacts")
OUT = os.path.join(A, "_grid_tables"); os.makedirs(OUT, exist_ok=True)

FRONTIER = json.load(open(glob.glob(f"{A}/*/grid_*.json")[0]))["frontier"]
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return np.nan
    a, b = a[m], b[m]
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean(); d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / d) if d > 0 else np.nan

def load_cell(p):
    z = np.load(p, allow_pickle=True)
    head = {k[2:]: z[k] for k in z.files if k.startswith("H_")}       # [L,H]
    lay = {k[2:]: z[k] for k in z.files if k.startswith("L_")}        # [L]
    par = {k[2:]: z[k] for k in z.files if k.startswith("P_")}        # [L]
    parh = {k[2:]: z[k] for k in z.files if k.startswith("PH_")}      # [L,H]
    return head, lay, par, parh

models = sorted([d for d in os.listdir(A) if os.path.isdir(f"{A}/{d}") and glob.glob(f"{A}/{d}/grid_*.npz")])
cells = []
for m in models:
    for p in sorted(glob.glob(f"{A}/{m}/grid_*.npz")):
        ds = os.path.basename(p)[5:-4]
        head, lay, par, parh = load_cell(p)
        cells.append((m, ds, head, lay, par, parh))
print(f"models={models}  cells={len(cells)}")

# ---- pooled PER-HEAD correlation (LC + CF + per-head param metrics) ----
head_metrics = sorted(set().union(*[set(c[2]) | set(c[5]) for c in cells])) if cells else []
def head_vec(c, k):
    if k in c[2]: return c[2][k].ravel()
    if k in c[5]: return c[5][k].ravel()
    return None
HP = {k: np.concatenate([v for c in cells if (v := head_vec(c, k)) is not None]) for k in head_metrics}
nH = len(head_metrics)
corrH = np.full((nH, nH), np.nan)
for i, a in enumerate(head_metrics):
    for j, b in enumerate(head_metrics):
        n = min(len(HP[a]), len(HP[b])); corrH[i, j] = spearman(HP[a][:n], HP[b][:n])

# ---- T4: LC x CF coupling (per head), pooled + per-dataset consistency ----
LC = [k for k in head_metrics if FRONTIER.get(k) == "LC"]
CF = [k for k in head_metrics if FRONTIER.get(k) == "CF"]
coupling = {}
for lc in LC:
    for cf in CF:
        vals = []
        for c in cells:
            a, b = head_vec(c, lc), head_vec(c, cf)
            if a is None or b is None: continue
            n = min(len(a), len(b)); r = spearman(a[:n], b[:n])
            if np.isfinite(r): vals.append(r)
        if len(vals) >= 3:
            coupling[f"{lc}~{cf}"] = {"mean": round(float(np.mean(vals)), 3),
                                      "min": round(float(np.min(vals)), 3),
                                      "max": round(float(np.max(vals)), 3),
                                      "n": len(vals),
                                      "consistent": bool(np.sign(np.min(vals)) == np.sign(np.max(vals)))}
coup_sorted = sorted(coupling.items(), key=lambda x: -abs(x[1]["mean"]))

# ---- pooled PER-LAYER correlation (representation/info/activation/param + head-pooled) ----
lay_metrics = sorted(set().union(*[set(c[3]) | set(c[4]) for c in cells])) if cells else []
def lay_vec(c, k):
    if k in c[3]: return c[3][k]
    if k in c[4]: return c[4][k]
    return None
LP = {k: np.concatenate([v for c in cells if (v := lay_vec(c, k)) is not None]) for k in lay_metrics}
# add head-pooled LC/CF to the per-layer table
for k in head_metrics:
    pooled = []
    for c in cells:
        v = head_vec(c, k)
        if v is not None: pooled.append(v.reshape(c[3][lay_metrics[0]].shape[0], -1).mean(1) if lay_metrics else v)
    if pooled: LP["hd_" + k] = np.concatenate(pooled)
allL = sorted(LP); nL = len(allL)
corrL = np.full((nL, nL), np.nan)
mlen = min(len(v) for v in LP.values()) if LP else 0
for i, a in enumerate(allL):
    for j, b in enumerate(allL):
        corrL[i, j] = spearman(LP[a][:mlen], LP[b][:mlen])

# ---- T3: cross-dataset stability of each metric (does it rank layers consistently?) ----
# ---- save ----
np.savez(f"{OUT}/corr_head.npz", names=np.array(head_metrics), corr=corrH)
np.savez(f"{OUT}/corr_layer.npz", names=np.array(allL), corr=corrL)
summary = {
    "models": models, "n_cells": len(cells),
    "frontier": {k: FRONTIER.get(k) for k in head_metrics + lay_metrics},
    "T4_LC_CF_coupling_top": dict(coup_sorted[:25]),
    "T4_consistent_couplings": {k: v for k, v in coup_sorted if v["consistent"] and abs(v["mean"]) >= 0.15},
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print("\n=== T4: LC (inference) x CF (training) coupling, pooled across cells ===")
for k, v in coup_sorted[:20]:
    flag = "OK" if v["consistent"] else "  "
    print(f"  [{flag}] {k:34s} mean={v['mean']:+.3f}  [{v['min']:+.2f},{v['max']:+.2f}]  n={v['n']}")
print(f"\nwrote {OUT}/summary.json + corr_head.npz + corr_layer.npz")
