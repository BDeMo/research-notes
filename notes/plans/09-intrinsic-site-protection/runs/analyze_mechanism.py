"""Mechanism + confound controls on EXISTING grid data (CPU, instant).
For each model, pooled over its grid_*.npz cells:
  raw   rho(LC, CF)
  partial rho(LC, CF | out_norm)        -> is the coupling just activation magnitude?
  within-layer rho(LC, CF)              -> same heads, or just same layers (depth)?
LC in {retrieval, v_norm, attn_distance}; CF in {fisher, dW_drift}; control=out_norm.
"""
import os, glob, json, numpy as np
A = os.environ.get("JANUS_OUT", "/root/janus/artifacts")

def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8: return np.nan
    x, y = x[m], y[m]
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean(); d = np.linalg.norm(rx)*np.linalg.norm(ry)
    return float((rx@ry)/d) if d > 0 else np.nan

def partial(x, y, z):
    rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
    den = np.sqrt(max(1-rxz**2, 1e-9)*max(1-ryz**2, 1e-9))
    return (rxy - rxz*ryz)/den if den > 0 else np.nan

def within_layer(x2d, y2d):
    vals = [sp(x2d[l], y2d[l]) for l in range(x2d.shape[0]) if np.isfinite(x2d[l]).any()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

LC = ["retrieval", "v_norm", "attn_distance"]; CF = ["fisher", "dW_drift"]
models = sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{A}/*/grid_*.npz")})
print(f"models: {models}\n")
for m in models:
    cells = glob.glob(f"{A}/{m}/grid_*.npz")
    H = {}  # metric -> list of [L,H] arrays
    for p in cells:
        z = np.load(p, allow_pickle=True)
        for k in LC+CF+["out_norm"]:
            if "H_"+k in z.files: H.setdefault(k, []).append(z["H_"+k])
    if not all(k in H for k in LC+CF+["out_norm"]): continue
    flat = {k: np.concatenate([a.ravel() for a in v]) for k, v in H.items()}
    print(f"### {m} ({len(cells)} cells) ###")
    for cf in CF:
        for lc in LC:
            raw = sp(flat[lc], flat[cf]); par = partial(flat[lc], flat[cf], flat["out_norm"])
            wl = np.nanmean([within_layer(a, b) for a, b in zip(H[lc], H[cf])])
            print(f"  {lc:14s} ~ {cf:9s}: raw={raw:+.3f}  partial|out_norm={par:+.3f}  within-layer={wl:+.3f}")
    print()
