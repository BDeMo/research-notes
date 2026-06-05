"""Rigorous a-priori coupling mechanism, per model (no SFT -> no recipe confound).
On N generic probe texts: per-head LC metrics + out_norm (activation control).
A-priori grad/Fisher on gsm8k pairs (forward+backward, NO optimizer step = no
destruction) = where SFT gradients would land. Then for each (LC, CF) pair report:
  raw rho, partial rho|out_norm (activation confound), within-layer rho (depth
  confound), and a bootstrap-over-heads 95% CI on raw rho.
Run per model across the full-attention ladder (skip hybrid qwen3.5 for head-level).
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from janus import core, criteria

LC = ["retrieval", "v_norm", "attn_distance", "sink"]; CF = ["grad_mag", "fisher"]

def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8: return np.nan
    x, y = x[m], y[m]; rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean(); d = np.linalg.norm(rx)*np.linalg.norm(ry)
    return float((rx@ry)/d) if d > 0 else np.nan

def partial(x, y, z):
    rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
    den = np.sqrt(max(1-rxz**2, 1e-9)*max(1-ryz**2, 1e-9)); return (rxy-rxz*ryz)/den if den > 0 else np.nan

def within_layer(x2d, y2d):
    v = [sp(x2d[l], y2d[l]) for l in range(x2d.shape[0])]; v = [a for a in v if np.isfinite(a)]
    return float(np.mean(v)) if v else np.nan

def boot_ci(x, y, n=1000):
    x = x.ravel(); y = y.ravel(); idx = np.arange(len(x)); out = []
    rng = np.random.RandomState(0)
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True); out.append(sp(x[s], y[s]))
    out = [v for v in out if np.isfinite(v)]
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]

def main():
    name = sys.argv[1]
    model, tok = core.load_model(core.MODELS.get(name, name), eager=True); model._janus_name = name
    cfg = model.config; L, H, hd = cfg.num_hidden_layers, cfg.num_attention_heads, core.head_dim_of(cfg)
    M = criteria.compute_head_metrics(model, tok, n_texts=6)
    pairs = core.load_grid_dataset(tok, "gsm8k", n=24)[1]
    feats = core._pairs_to_feats(tok, pairs, 512)
    gm, fi, _ = core._grad_fisher_feats(model, feats, tok, L, H, hd, 12, False)
    M["grad_mag"], M["fisher"] = gm, fi
    res = {"model": name, "layers": L, "heads": H, "couplings": {}}
    for cf in CF:
        for lc in LC:
            res["couplings"][f"{lc}~{cf}"] = {
                "raw": round(sp(M[lc], M[cf]), 3),
                "partial_outnorm": round(partial(M[lc], M[cf], M["out_norm"]), 3),
                "within_layer": round(within_layer(M[lc], M[cf]), 3),
                "ci95": boot_ci(M[lc], M[cf])}
    out = os.path.join(os.environ.get("JANUS_OUT", "."), f"mechanism_{name}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[mech] {name} done -> {out}")
    for k, v in res["couplings"].items():
        print(f"  {k:22s} raw={v['raw']:+.3f} partial={v['partial_outnorm']:+.3f} within={v['within_layer']:+.3f} ci={v['ci95']}")

if __name__ == "__main__":
    main()
