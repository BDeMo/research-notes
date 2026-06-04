#!/usr/bin/env python3
"""Janus visualization gallery. Reads $JANUS_OUT artifacts -> PNG figures + index.md.
Renders whatever exists; missing artifacts are skipped gracefully."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.environ.get("JANUS_OUT", "/home/devuser/janus/artifacts")
FIG = os.path.join(ART, "figs"); os.makedirs(FIG, exist_ok=True)
MODELS_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
                "qwen2.5-1.5b", "qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct",
                "glm4-9b", "glm4-32b", "qwen3.5-4b", "qwen3.5-9b"]
SIZE_B = {"qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4.0, "qwen3-8b": 8.0,
          "qwen3-14b": 14.0, "qwen2.5-1.5b": 1.5, "qwen2.5-1.5b-instruct": 1.5,
          "qwen2.5-7b-instruct": 7.0, "glm4-9b": 9.0, "glm4-32b": 32.0,
          "qwen3.5-4b": 4.0, "qwen3.5-9b": 9.0}
made = []

def jload(p):
    try: return json.load(open(p))
    except Exception: return None

def models_present():
    return [m for m in MODELS_ORDER if os.path.isdir(os.path.join(ART, m))]

def save(fig, name, title):
    p = os.path.join(FIG, name)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(p, dpi=110); plt.close(fig)
    made.append((name, title))
    print("[viz] wrote", p)

# 1. site maps per model
def fig_sitemaps():
    for m in models_present():
        npz = os.path.join(ART, m, "detect.npz")
        if not os.path.exists(npz): continue
        z = np.load(npz)
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        for a, key, ttl in [(ax[0,0], "sink", "Attention-sink mass (streaming heads)"),
                            (ax[0,1], "retr", "Retrieval-head score (long-ctx read site)"),
                            (ax[1,0], "induction", "Induction-head score")]:
            im = a.imshow(z[key], aspect="auto", cmap="viridis")
            a.set_title(ttl); a.set_xlabel("head"); a.set_ylabel("layer")
            fig.colorbar(im, ax=a, fraction=0.046)
        cm = z["chan_max"]; order = np.argsort(cm)[::-1][:20]
        ax[1,1].bar(range(len(order)), cm[order], color="crimson")
        ax[1,1].set_yscale("log"); ax[1,1].set_title("Top massive-activation channels (log)")
        ax[1,1].set_xticks(range(len(order))); ax[1,1].set_xticklabels(order, rotation=90, fontsize=6)
        ax[1,1].set_xlabel("residual channel"); ax[1,1].set_ylabel("max |activation|")
        save(fig, f"01_sitemap_{m}.png", f"Intrinsic sites — {m}")

# 2. sink vs retrieval disjointness
def fig_disjoint():
    ms = models_present()
    fig, axes = plt.subplots(1, max(len(ms),1), figsize=(4.5*max(len(ms),1), 4.2), squeeze=False)
    for i, m in enumerate(ms):
        npz = os.path.join(ART, m, "detect.npz"); j = jload(os.path.join(ART, m, "detect.json"))
        if not os.path.exists(npz): continue
        z = np.load(npz); ax = axes[0][i]
        ax.scatter(z["sink"].ravel(), z["retr"].ravel(), s=6, alpha=0.4)
        rho = j["sink_vs_retr_spearman"] if j else 0
        jac = j["sink_vs_retr_jaccard_top15"] if j else 0
        ax.set_title(f"{m}\nρ={rho}  Jaccard(top15)={jac}")
        ax.set_xlabel("sink mass"); ax.set_ylabel("retrieval score")
    save(fig, "02_sink_vs_retrieval.png", "Sink heads vs Retrieval heads are DISJOINT (DuoAttention check)")

# 3. coupling scatter: retrieval score vs SFT drift (THE headline)
def fig_coupling():
    items = []
    for m in models_present():
        npz = os.path.join(ART, m, "detect.npz")
        if not os.path.exists(npz): continue
        retr = np.load(npz)["retr"]; sink = np.load(npz)["sink"]
        for dr in sorted(glob.glob(os.path.join(ART, m, "drift_*.npz"))):
            tag = os.path.basename(dr)[6:-4]
            hd = np.load(dr)["head_drift"]
            items.append((m, tag, retr, sink, hd))
    if not items: return
    n = len(items); cols = min(3, n); rows = (n + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2*cols, 4.4*rows), squeeze=False)
    for k,(m,tag,retr,sink,hd) in enumerate(items):
        ax = axes[k//cols][k%cols]
        x = retr.ravel(); y = hd.ravel()
        ax.scatter(x, y, s=8, alpha=0.4, color="teal")
        rho = spearman(x, y)
        # highlight top retrieval heads
        thr = np.sort(x)[::-1][max(int(0.05*len(x)),1)]
        ax.scatter(x[x>=thr], y[x>=thr], s=18, color="crimson", label="top-5% retrieval heads")
        ax.set_title(f"{m} | {tag}\nρ(retrieval, drift)={rho:.3f}")
        ax.set_xlabel("retrieval-head score (read-side)")
        ax.set_ylabel("‖ΔW‖ after SFT (write-side)")
        ax.legend(fontsize=7)
    save(fig, "03_coupling_scatter.png", "H2: does read-side importance PREDICT write-side disruption?")

# 4. drift heatmap with retrieval overlay
def fig_driftmap():
    for m in models_present():
        npz = os.path.join(ART, m, "detect.npz")
        if not os.path.exists(npz): continue
        retr = np.load(npz)["retr"]
        drs = sorted(glob.glob(os.path.join(ART, m, "drift_*.npz")))
        if not drs: continue
        n = len(drs); fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), squeeze=False)
        for i, dr in enumerate(drs):
            tag = os.path.basename(dr)[6:-4]; hd = np.load(dr)["head_drift"]
            ax = axes[0][i]
            im = ax.imshow(hd, aspect="auto", cmap="magma"); fig.colorbar(im, ax=ax, fraction=0.046)
            thr = np.sort(retr.ravel())[::-1][max(int(0.05*retr.size),1)]
            ys, xs = np.where(retr >= thr)
            ax.scatter(xs, ys, s=14, edgecolor="cyan", facecolor="none", label="retrieval heads")
            ax.set_title(f"{tag}"); ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.legend(fontsize=7)
        save(fig, f"04_driftmap_{m}.png", f"SFT weight-drift per head + retrieval heads — {m}")

# 5. NIAH grids
def fig_niah():
    for m in models_present():
        files = sorted(glob.glob(os.path.join(ART, m, "niah_*.json")))
        if not files: continue
        n = len(files); fig, axes = plt.subplots(1, n, figsize=(4.6*n, 4.0), squeeze=False)
        for i, f in enumerate(files):
            j = jload(f); ax = axes[0][i]
            if not j: continue
            lengths = j["lengths"]; depths = j["depths"]
            M = np.zeros((len(depths), len(lengths)))
            for a, Ln in enumerate(lengths):
                for b, d in enumerate(depths):
                    M[b, a] = j["grid"].get(f"{Ln}|{d}", np.nan)
            im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(lengths))); ax.set_xticklabels(lengths)
            ax.set_yticks(range(len(depths))); ax.set_yticklabels(depths)
            ax.set_xlabel("context length"); ax.set_ylabel("needle depth")
            ax.set_title(os.path.basename(f)[5:-5] + f"\nmean={np.nanmean(M):.2f}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        save(fig, f"05_niah_{m}.png", f"NIAH accuracy (depth × length) — {m}")

# 6. forgetting bars + 7 pareto
def collect_caps(m):
    caps = {}
    for f in glob.glob(os.path.join(ART, m, "cap_*.json")):
        j = jload(f)
        if j: caps[j["ckpt"]] = j
    return caps

def fig_forgetting():
    for m in models_present():
        caps = collect_caps(m)
        if "base" not in caps or len(caps) < 2: continue
        metrics = ["mmlu", "trivia", "math"]
        ckpts = ["base"] + [c for c in caps if c != "base"]
        fig, ax = plt.subplots(figsize=(1.6*len(ckpts)+3, 4.6))
        x = np.arange(len(metrics)); w = 0.8/len(ckpts)
        for i, c in enumerate(ckpts):
            vals = [caps[c].get(k) or 0 for k in metrics]
            ax.bar(x + i*w, vals, w, label=c)
        ax.set_xticks(x + 0.4); ax.set_xticklabels(["MMLU (retain)","TriviaQA (retain)","MATH (new domain)"])
        ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.set_ylim(0,1)
        save(fig, f"06_forgetting_{m}.png", f"Forgetting vs plasticity after SFT — {m}")

def fig_pareto():
    for m in models_present():
        caps = collect_caps(m)
        if "base" not in caps: continue
        base = caps["base"]
        pts = []
        for c, j in caps.items():
            if c == "base": continue
            retain = np.mean([j.get("mmlu") or 0, j.get("trivia") or 0])
            base_ret = np.mean([base.get("mmlu") or 0, base.get("trivia") or 0])
            plast = j.get("math") or 0
            pts.append((c, plast, retain, base_ret))
        if not pts: continue
        fig, ax = plt.subplots(figsize=(6.5,5))
        base_ret = pts[0][3]
        ax.axhline(base_ret, ls="--", color="gray", label="base retention")
        for c, plast, retain, _ in pts:
            color = "crimson" if "retrieval" in c else ("navy" if "none" in c else "green")
            ax.scatter(plast, retain, s=120, color=color)
            ax.annotate(c.replace("math_full_","").replace("math_lora_",""), (plast, retain),
                        fontsize=8, xytext=(5,5), textcoords="offset points")
        ax.set_xlabel("new-domain accuracy (MATH) — plasticity")
        ax.set_ylabel("retained accuracy (MMLU+TriviaQA) — anti-forgetting")
        ax.legend(fontsize=8)
        save(fig, f"07_pareto_{m}.png", f"Retention–plasticity Pareto (protection variants) — {m}")

# 8. coupling summary across models/tags
def fig_summary():
    rows = []
    for m in models_present():
        for f in glob.glob(os.path.join(ART, m, "coupling_*.json")):
            j = jload(f)
            if j: rows.append((m, j["tag"], j["spearman_retr_vs_drift"], j["spearman_sink_vs_drift"]))
    if not rows: return
    fig, ax = plt.subplots(figsize=(max(7,1.1*len(rows)),4.8))
    labels = [f"{m}\n{t}" for m,t,_,_ in rows]
    x = np.arange(len(rows))
    ax.bar(x-0.2, [r[2] for r in rows], 0.4, label="ρ(retrieval, drift)", color="teal")
    ax.bar(x+0.2, [r[3] for r in rows], 0.4, label="ρ(sink, drift)", color="orange")
    ax.axhline(0, color="k", lw=0.5); ax.axhline(0.4, ls=":", color="red", label="H2 gate 0.4")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ"); ax.legend(fontsize=8)
    save(fig, "08_coupling_summary.png", "Coupling strength: read-side criterion vs SFT disruption")

def spearman(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ra=a.argsort().argsort().astype(float); rb=b.argsort().argsort().astype(float)
    ra-=ra.mean(); rb-=rb.mean(); d=np.linalg.norm(ra)*np.linalg.norm(rb)
    return float((ra@rb)/d) if d>0 else 0.0

def niah_mean(m, suffix):
    j = jload(os.path.join(ART, m, f"niah{suffix}.json"))
    if not j: return None
    return float(np.mean(list(j["grid"].values())))

# 10. intrinsic metric x metric correlation matrix (THE coupling map)
def fig_intrinsic_corr():
    for m in models_present():
        npz = os.path.join(ART, m, "intrinsic.npz")
        if not os.path.exists(npz): continue
        z = np.load(npz, allow_pickle=True)
        present = list(z["present"]); corr = z["corr"]
        n = len(present)
        fig, ax = plt.subplots(figsize=(0.85*n+3, 0.85*n+2))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(present, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = corr[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
        save(fig, f"10_intrinsic_corr_{m}.png", f"Intrinsic metric×metric coupling map — {m}")

# 11. cross-model headline couplings vs drift / fisher
def fig_intrinsic_summary():
    ms = [m for m in models_present() if os.path.exists(os.path.join(ART, m, "intrinsic.json"))]
    if not ms: return
    targets = ["dW_drift", "fisher"]
    keys = ["retrieval", "sink", "induction", "attn_entropy", "kv_norm", "out_norm", "grad_mag"]
    fig, axes = plt.subplots(1, len(targets), figsize=(7*len(targets), 4.8), squeeze=False)
    for ti, tgt in enumerate(targets):
        ax = axes[0][ti]
        x = np.arange(len(keys)); w = 0.8/max(len(ms),1)
        any_data = False
        for mi, m in enumerate(ms):
            j = jload(os.path.join(ART, m, "intrinsic.json"))
            vs = j.get(f"vs_{tgt}", {})
            if not vs: continue
            any_data = True
            ax.bar(x + mi*w, [vs.get(k, 0) for k in keys], w, label=m)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x + 0.4); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Spearman ρ(metric, {tgt})"); ax.set_title(f"What predicts {tgt}?")
        if any_data: ax.legend(fontsize=6)
    save(fig, "11_intrinsic_vs_outcome.png", "Which intrinsic metric predicts SFT disruption (drift/Fisher)?")

# 12. FACTS — per-layer broad-metric correlation matrix + trajectories
def fig_facts_corr():
    for m in models_present():
        p = os.path.join(ART, m, "facts.npz")
        if not os.path.exists(p): continue
        z = np.load(p, allow_pickle=True)
        names = list(z["names"]); corr = z["corr"]; n = len(names)
        fig, ax = plt.subplots(figsize=(0.62*n+3, 0.62*n+2))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=6)
        for i in range(n):
            for j in range(n):
                v = corr[i, j]
                if not np.isnan(v) and abs(v) >= 0.6 and i != j:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5,
                            color="white" if abs(v) > 0.8 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        save(fig, f"12_facts_corr_{m}.png", f"Broad per-layer metric correlations (facts) — {m}")

def fig_facts_traj():
    for m in models_present():
        p = os.path.join(ART, m, "facts.npz")
        if not os.path.exists(p): continue
        z = np.load(p, allow_pickle=True)
        names = list(z["names"])
        groups = [("representation", ["F_eff_rank", "F_anisotropy", "F_intrinsic_dim", "F_cka_adjacent", "F_act_kurtosis"]),
                  ("information", ["F_ll_kl_to_final", "F_ll_top1_depth", "F_tuned_lens_kl", "F_tuned_lens_depth"]),
                  ("parameters", ["F_w_stable_rank", "F_w_ht_alpha", "F_down_ht_alpha", "F_w_eff_rank"])]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for gi, (title, keys) in enumerate(groups):
            ax = axes[gi]; plotted = False
            for k in keys:
                if k in z.files:
                    v = z[k]; v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
                    ax.plot(range(len(v)), v, marker=".", label=k[2:]); plotted = True
            ax.set_title(title); ax.set_xlabel("layer"); ax.set_ylabel("normalized")
            if plotted: ax.legend(fontsize=6)
        save(fig, f"13_facts_traj_{m}.png", f"Layer trajectories by angle (normalized) — {m}")

def fig_facts_topfacts():
    ms = [m for m in models_present() if os.path.exists(os.path.join(ART, m, "facts.json"))]
    if not ms: return
    # collect correlations that are strong AND consistent across models
    from collections import defaultdict
    acc = defaultdict(list)
    for m in ms:
        j = jload(os.path.join(ART, m, "facts.json"))
        if not j: continue
        for a, b, v in j.get("top_correlations", []):
            key = tuple(sorted([a, b]))
            acc[key].append(v)
    rows = [(k, np.mean(v), len(v)) for k, v in acc.items() if len(v) >= max(2, len(ms)//2)]
    rows.sort(key=lambda x: -abs(x[1])); rows = rows[:20]
    if not rows: return
    fig, ax = plt.subplots(figsize=(9, 0.45*len(rows)+1.5))
    y = range(len(rows))
    ax.barh(list(y), [r[1] for r in rows],
            color=["crimson" if r[1] > 0 else "navy" for r in rows])
    ax.set_yticks(list(y)); ax.set_yticklabels([f"{a} — {b} (n={n})" for (a, b), v, n in rows], fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("mean Spearman ρ across models")
    save(fig, "14_facts_consistent.png", "Cross-model CONSISTENT facts (strong, recurring per-layer correlations)")

# 9. cross-scale scaling ladder (the long-context-at-scale angle)
def fig_scaling():
    ms = [m for m in models_present() if m in SIZE_B]
    rows = []
    for m in ms:
        j = jload(os.path.join(ART, m, "detect.json"))
        if not j: continue
        mx = j["massive_channels"][0][1] if j.get("massive_channels") else np.nan
        jac = j.get("sink_vs_retr_jaccard_top15", np.nan)
        base = niah_mean(m, "_base")
        abl_r = niah_mean(m, "_ablate_retr")
        abl_s = niah_mean(m, "_ablate_sink")
        rows.append((m, SIZE_B[m], mx, jac, base, abl_r, abl_s))
    if not rows: return
    rows.sort(key=lambda r: r[1])
    sizes = [r[1] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(sizes, [r[2] for r in rows], "o-", color="crimson")
    ax[0].set_yscale("log"); ax[0].set_title("Max massive-activation magnitude vs size")
    ax[0].set_xlabel("params (B)"); ax[0].set_ylabel("max |activation| (log)")
    for r in rows: ax[0].annotate(r[0].replace("qwen",""), (r[1], r[2]), fontsize=6)
    ax[1].plot(sizes, [r[3] for r in rows], "s-", color="purple")
    ax[1].set_title("Sink–retrieval Jaccard (disjointness) vs size")
    ax[1].set_xlabel("params (B)"); ax[1].set_ylabel("Jaccard top-15"); ax[1].set_ylim(-0.02, 1)
    # NIAH causal ablation contrast
    for key, idx, c, lbl in [("base",4,"green","NIAH base"),("retr",5,"crimson","ablate retrieval"),("sink",6,"orange","ablate sink")]:
        ys = [r[idx] if r[idx] is not None else np.nan for r in rows]
        ax[2].plot(sizes, ys, "o-", color=c, label=lbl)
    ax[2].set_title("Long-ctx causal: retrieval-head ablation kills NIAH, sink does not")
    ax[2].set_xlabel("params (B)"); ax[2].set_ylabel("NIAH accuracy"); ax[2].legend(fontsize=8); ax[2].set_ylim(0,1)
    save(fig, "09_scaling_ladder.png", "Cross-scale: intrinsic sites & long-context causality vs model size")

def write_index():
    lines = ["# Janus overnight exploration — figure gallery\n",
             "Long-context (read) ↔ forgetting (write) coupling at intrinsic sites. Auto-generated.\n"]
    sections = [
        ("Intrinsic site maps (where the machinery lives)", "01_sitemap_"),
        ("Sink vs retrieval heads are disjoint", "02_sink_vs_retrieval"),
        ("H2 — coupling scatter (read predicts write?)", "03_coupling_scatter"),
        ("SFT drift maps + retrieval overlay", "04_driftmap_"),
        ("Long-context: NIAH grids", "05_niah_"),
        ("Forgetting vs plasticity bars", "06_forgetting_"),
        ("H3 — retention–plasticity Pareto", "07_pareto_"),
        ("Coupling summary across models", "08_coupling_summary"),
        ("Cross-scale scaling ladder", "09_scaling_ladder"),
        ("Intrinsic metric×metric coupling map", "10_intrinsic_corr_"),
        ("Which intrinsic metric predicts SFT disruption", "11_intrinsic_vs_outcome"),
        ("FACTS — broad per-layer metric correlations", "12_facts_corr_"),
        ("FACTS — layer trajectories by angle", "13_facts_traj_"),
        ("FACTS — cross-model consistent facts", "14_facts_consistent"),
    ]
    for title, pref in sections:
        figs = sorted([n for n,_ in made if n.startswith(pref)])
        if not figs: continue
        lines.append(f"\n## {title}\n")
        for fn in figs:
            lines.append(f"![{fn}](figs/{fn})\n")
    open(os.path.join(ART, "INDEX.md"), "w").write("\n".join(lines))
    print("[viz] wrote", os.path.join(ART, "INDEX.md"))

def main():
    for fn in [fig_sitemaps, fig_disjoint, fig_coupling, fig_driftmap, fig_niah,
               fig_forgetting, fig_pareto, fig_summary, fig_scaling,
               fig_intrinsic_corr, fig_intrinsic_summary,
               fig_facts_corr, fig_facts_traj, fig_facts_topfacts]:
        try: fn()
        except Exception as e: print("[viz] section failed", fn.__name__, e)
    write_index()
    print(f"[viz] {len(made)} figures in {FIG}")

if __name__ == "__main__":
    main()
