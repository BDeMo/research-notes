"""Generate the July-w01 comparison figures from the harvested RECIPE_EVAL numbers.
Run: python make_julw01_figs.py  (writes fig8..fig12 into this dir)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
def save(fig, name):
    p = os.path.join(HERE, name)
    fig.tight_layout(); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)

# ---------- Fig 8: KV-free family vs context length (RULER retrieval, keep 0.5) ----------
lengths = ["4k", "8k", "16k"]
kvfree_len = {
    "LLMLingua-2":      [0.94, 0.96, 0.96],
    "LLMLingua-orig":   [1.00, 0.98, np.nan],   # 16k OOM
    "Selective-Context":[1.00, 1.00, 1.00],
    "BM25-RAG":         [0.98, 0.94, 0.94],
    "ToMe":             [0.00, 0.00, 0.00],
}
fig, ax = plt.subplots(figsize=(6.2, 4.0))
x = np.arange(len(lengths))
for m, ys in kvfree_len.items():
    ax.plot(x, ys, marker="o", linewidth=2, label=m)
ax.set_xticks(x); ax.set_xticklabels(lengths)
ax.set_ylim(-0.03, 1.05); ax.set_xlabel("context length (RULER)"); ax.set_ylabel("retrieval accuracy")
ax.set_title("Fig 8 — KV-free family is length-robust on retrieval\n(prompt compression flat 0.94-1.0; ToMe destroys the needle)")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
ax.annotate("ToMe: merging removes the needle", (1, 0.0), (0.6, 0.18),
            fontsize=8, color="crimson", arrowprops=dict(arrowstyle="->", color="crimson"))
save(fig, "fig8_kvfree_length.png")

# ---------- Fig 9: KV-free family by task (nc8, keep 0.5) ----------
tasks = ["squad", "hotpot", "narrativeqa", "quality"]
kvfree_task = {   # None => missing cell
    "LLMLingua-2":      [0.58, 0.52, 0.16, 0.19],
    "LLMLingua-orig":   [0.62, 0.59, 0.16, 0.15],
    "LongLLMLingua":    [0.35, 0.59, 0.16, 0.15],
    "Selective-Context":[0.49, 0.46, np.nan, 0.16],
    "ToMe":             [0.23, 0.40, 0.12, 0.21],
    "BM25-RAG":         [0.75, 0.59, 0.11, 0.08],
}
full_ref =   [0.72, 0.61, 0.17, 0.12]
noctx_ref =  [0.19, 0.27, 0.10, 0.23]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
n = len(kvfree_task); w = 0.8 / n; x = np.arange(len(tasks))
for i, (m, ys) in enumerate(kvfree_task.items()):
    ax.bar(x + i * w - 0.4 + w / 2, [0 if (v!=v) else v for v in ys], w, label=m)
ax.plot(x, full_ref, "k_", markersize=22, markeredgewidth=2, label="full (ref)")
ax.plot(x, noctx_ref, "kx", markersize=8, label="no_ctx (ref)")
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.85)
ax.set_title("Fig 9 — KV-free family by task (keep 0.5)\nRAG leads extractive; nothing beats full on abstractive; full hurts on QuALITY")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, ncol=4, loc="upper right")
save(fig, "fig9_kvfree_task.png")

# ---------- Fig 10: head-to-head "which works" heatmap (16k, 50% budget) ----------
h2h_methods = ["kvzip(KV)", "knorm(KV)", "snapkv(KV)", "LLMLingua-2", "RAG", "ToMe", "window", "full"]
h2h_benches = ["RULER", "numerical", "hotpot", "narrativeqa", "QuALITY"]
H = np.array([
    [1.00, 0.81, np.nan, 0.15, 0.27],  # kvzip
    [0.90, 0.81, 0.21,  0.15, 0.29],   # knorm
    [0.17, 0.00, 0.47,  0.13, 0.31],   # snapkv
    [0.85, 0.77, 0.50,  0.18, 0.19],   # ll2
    [0.98, 0.92, 0.59,  0.17, 0.06],   # rag
    [0.00, 0.00, 0.41,  0.13, 0.21],   # tome
    [0.42, 0.42, 0.61,  0.13, 0.10],   # window
    [1.00, 0.77, np.nan,0.14, 0.08],   # full
])
fig, ax = plt.subplots(figsize=(6.8, 4.6))
im = ax.imshow(H, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(h2h_benches))); ax.set_xticklabels(h2h_benches, rotation=20, ha="right")
ax.set_yticks(range(len(h2h_methods))); ax.set_yticklabels(h2h_methods)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H[i, j]
        ax.text(j, i, "-" if v != v else f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="black")
ax.set_title("Fig 10 — Head-to-head 'which works' (16k, 50% budget)\nretrieval: kvzip/knorm/RAG/LLMLingua win, SnapKV & ToMe fail; abstractive: nothing; QuALITY: full hurts")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
save(fig, "fig10_headtohead.png")

# ---------- Fig 11: cheap importance-signal probe (needle-AUROC) ----------
signals = ["query-relevance", "surprisal", "hidden-norm", "embed-norm"]
ruler_auc =     [0.95, 0.64, 0.33, 0.28]
numerical_auc = [0.80, 0.84, 0.66, 0.26]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
x = np.arange(len(signals)); w = 0.38
ax.bar(x - w/2, ruler_auc, w, label="RULER (word needle)")
ax.bar(x + w/2, numerical_auc, w, label="numerical needle")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
ax.set_xticks(x); ax.set_xticklabels(signals, rotation=12)
ax.set_ylabel("needle-AUROC"); ax.set_ylim(0, 1.0)
ax.set_title("Fig 11 — A cheap O(L) signal finds the needle\nquery-relevance wins word needles (0.95), surprisal wins numeric (0.84); neither wins both")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
save(fig, "fig11_signal_probe.png")

# ---------- Fig 12: repeat-recon vs random-M — K128 dose-response (left) + high-K bracket (right) ----------
fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.3))
# left: K128 lambda dose-response (real vs random vs conf)
lam = [0.0, 0.25, 0.5, 1.0]
real_auc = [0.498, 0.523, 0.464, 0.493]; rand_auc = [0.481, 0.516, 0.512, 0.539]; conf_auc = [0.655, 0.644, 0.614, 0.654]
axl.plot(lam, real_auc, "o-", lw=2, color="tab:blue", label="repeat-recon (real M)")
axl.plot(lam, rand_auc, "s--", lw=2, color="tab:orange", label="random-M control")
axl.plot(lam, conf_auc, "^-", lw=2, color="tab:green", label="confidence")
axl.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
axl.set_xlabel("lambda (K128/2000-step)"); axl.set_ylabel("gate AUROC"); axl.set_ylim(0.4, 0.72)
axl.set_title("K128: real-M = random-M = chance (no signal)"); axl.grid(alpha=0.3); axl.legend(fontsize=8)
# right: high-K bracket — (real-M minus random-M) for base vs +repeat
cfgs = ["K256@4k", "K256@6k", "K512@4k"]
base_delta = [-0.006, -0.045, -0.001]
rep_delta  = [ 0.084, -0.093, -0.002]   # full bracket: only K256@4k positive -> noise
x = np.arange(len(cfgs)); w = 0.38
axr.bar(x - w/2, base_delta, w, color="tab:gray", label="base (no repeat)")
axr.bar(x + w/2, rep_delta, w, color="tab:blue", label="+repeat")
axr.axhline(0.0, color="black", lw=1)
axr.set_xticks(x); axr.set_xticklabels(cfgs)
axr.set_ylabel("real-M  −  random-M  (AUROC)"); axr.set_ylim(-0.12, 0.12)
axr.set_title("High-K: +repeat scatters around 0 (mean ~0) -> no signal")
axr.grid(alpha=0.3, axis="y"); axr.legend(fontsize=8)
fig.suptitle("Fig 12 — Repeat-recon vs the random-M leak control: no memory-specific gate signal (K128 flat; high-K scatters ~0)", fontsize=10)
save(fig, "fig12_repeat_ablation.png")

print("all figures written")
