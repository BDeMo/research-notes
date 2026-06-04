#!/usr/bin/env python3
"""Render the grid big-tables: LC x CF coupling heatmap + per-head/per-layer corr matrices."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = os.environ.get("JANUS_OUT", "/root/janus/artifacts")
T = os.path.join(A, "_grid_tables"); FIG = os.path.join(A, "figs"); os.makedirs(FIG, exist_ok=True)
S = json.load(open(f"{T}/summary.json")); FR = S["frontier"]

def corr_fig(npz, name, title):
    z = np.load(npz, allow_pickle=True); names = list(z["names"]); C = z["corr"]; n = len(names)
    # order by frontier so LC/CF/ST block together
    order = sorted(range(n), key=lambda i: (FR.get(names[i]) or "Z", names[i]))
    names = [names[i] for i in order]; C = C[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(0.5*n+3, 0.5*n+2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    lbl = [f"[{FR.get(x,'?')}] {x}" for x in names]
    ax.set_xticks(range(n)); ax.set_xticklabels(lbl, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(lbl, fontsize=6)
    for i in range(n):
        for j in range(n):
            if np.isfinite(C[i,j]) and abs(C[i,j])>=0.6 and i!=j:
                ax.text(j,i,f"{C[i,j]:.1f}",ha="center",va="center",fontsize=5,color="white" if abs(C[i,j])>0.8 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.98]); fig.savefig(f"{FIG}/{name}", dpi=110); plt.close(fig)
    print("wrote", name)

# LC x CF coupling matrix (rows=LC, cols=CF) from the head corr
def coupling_fig():
    z = np.load(f"{T}/corr_head.npz", allow_pickle=True); names = list(z["names"]); C = z["corr"]
    LC = [x for x in names if FR.get(x)=="LC"]; CF = [x for x in names if FR.get(x)=="CF"]
    if not LC or not CF: return
    M = np.array([[C[names.index(l), names.index(c)] for c in CF] for l in LC])
    fig, ax = plt.subplots(figsize=(1.1*len(CF)+3, 0.5*len(LC)+2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(len(CF))); ax.set_xticklabels(CF, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(LC))); ax.set_yticklabels(LC, fontsize=8)
    for i in range(len(LC)):
        for j in range(len(CF)):
            if np.isfinite(M[i,j]): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=7,
                                            color="white" if abs(M[i,j])>0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Janus coupling: long-context (LC, inference) heads × forgetting (CF, training)  [pooled, all cells]", fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{FIG}/G1_LC_CF_coupling.png", dpi=120); plt.close(fig)
    print("wrote G1_LC_CF_coupling.png")

coupling_fig()
try: corr_fig(f"{T}/corr_head.npz", "G2_corr_head.png", "Per-head metric correlations (frontier-ordered)")
except Exception as e: print("head fig fail", e)
try: corr_fig(f"{T}/corr_layer.npz", "G3_corr_layer.png", "Per-layer metric correlations (frontier-ordered)")
except Exception as e: print("layer fig fail", e)
