#!/usr/bin/env python3
"""Janus (Plan 09) Phase-0 site detectors — forward-only, validation run.

Goal: reproduce the three intrinsic-site phenomena on the *real target model*
(Qwen3-8B) and produce the first artifact for the long-context<->forgetting
coupling project:

  1. attention-sink mass per (layer, head)  -> "sink heads"        [streaming side]
  2. massive-activation channels per (layer) -> outlier residual dims
  3. retrieval-head score per (layer, head)  -> "retrieval heads"  [long-ctx side]

Bonus H2-precondition probe (cheap, no training): overlap between the top sink
heads and the top retrieval heads. DuoAttention claims these are largely
DISJOINT (retrieval vs streaming/sink). If true here, it sharpens the site
inventory for the coupling study.

Forward-only: no generation, no SFT. Safe to share a GPU.
"""
import json, os, sys, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("JANUS_MODEL", "Qwen/Qwen3-8B")
DEVICE = os.environ.get("JANUS_DEVICE", "cuda:0")
OUT = os.environ.get("JANUS_OUT", "/home/devuser/janus_phase0_out.json")
SEQ_LEN = int(os.environ.get("JANUS_SEQLEN", "480"))

GENERIC = [
    "The history of cartography is closely tied to the development of "
    "mathematics and astronomy. Early maps were often based on travelers' "
    "accounts and were highly inaccurate by modern standards. ",
    "Photosynthesis is the process by which green plants convert light energy "
    "into chemical energy stored in glucose. It occurs in the chloroplasts and "
    "releases oxygen as a byproduct. ",
    "Economic inflation refers to a sustained increase in the general price "
    "level of goods and services. Central banks attempt to manage inflation "
    "through monetary policy, including adjusting interest rates. ",
    "The Roman aqueducts were a remarkable feat of ancient engineering, "
    "carrying water across vast distances using a slight downhill gradient and "
    "a system of bridges, tunnels, and channels. ",
]

def filler(n_repeat):
    base = ("Grass is green and the sky is often blue during a clear day. "
            "People enjoy walking in parks when the weather is pleasant. ")
    return base * n_repeat

def build_needle(tok):
    """A NIAH-style probe. Returns (input_ids, needle_value_token_positions)."""
    pre = "Read the following passage carefully and answer the question.\n" + filler(8)
    needle = "\nThe secret access code for project Helios is 47192.\n"
    post = filler(8) + "\nQuestion: What is the secret access code for project Helios? Answer: The code is"
    full = pre + needle + post
    ids = tok(full, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids
    # locate the value tokens "47192" inside the assembled ids
    val_ids = tok(" 47192", add_special_tokens=False).input_ids
    seq = ids[0].tolist()
    pos = []
    for i in range(len(seq) - len(val_ids) + 1):
        if seq[i:i+len(val_ids)] == val_ids:
            pos = list(range(i, i+len(val_ids)))
            break
    if not pos:  # fallback: bare digits
        val_ids = tok("47192", add_special_tokens=False).input_ids
        for i in range(len(seq) - len(val_ids) + 1):
            if seq[i:i+len(val_ids)] == val_ids:
                pos = list(range(i, i+len(val_ids)))
                break
    return ids, pos

@torch.no_grad()
def main():
    print(f"[janus] loading {MODEL} on {DEVICE} (eager attn) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(DEVICE).eval()
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    print(f"[janus] layers={L} heads={H} hidden={cfg.hidden_size}", flush=True)

    sink = torch.zeros(L, H)          # mean attn prob to position 0
    n_seq = 0
    massive = torch.zeros(L, cfg.hidden_size)  # max |hidden| per channel, over layers

    # --- sink mass + massive activations on generic text ---
    for text in GENERIC:
        ids = tok(filler(2) + text, return_tensors="pt",
                  truncation=True, max_length=SEQ_LEN).input_ids.to(DEVICE)
        out = model(ids, output_attentions=True, output_hidden_states=True)
        T = ids.shape[1]
        for l in range(L):
            att = out.attentions[l][0]            # [H, T, T]
            # mean over query positions >= 4 (skip the first few), attn to key pos 0
            sink[l] += att[:, 4:, 0].mean(dim=1).float().cpu()
        for l in range(L):
            hs = out.hidden_states[l+1][0].abs().float().cpu()  # [T, hidden]
            massive[l] = torch.maximum(massive[l], hs.max(dim=0).values)
        n_seq += 1
    sink /= n_seq

    # --- retrieval score on needle ---
    ids, val_pos = build_needle(tok)
    ids = ids.to(DEVICE)
    out = model(ids, output_attentions=True)
    retr = torch.zeros(L, H)
    if val_pos:
        last = ids.shape[1] - 1
        for l in range(L):
            att = out.attentions[l][0]            # [H, T, T]
            # attn mass the LAST query token puts on the needle value positions
            retr[l] = att[:, last, val_pos].sum(dim=1).float().cpu()
    print(f"[janus] needle value-token positions: {val_pos}", flush=True)

    # ---- summaries ----
    def topk_heads(mat, k=15):
        flat = mat.flatten()
        vals, idx = torch.topk(flat, k)
        return [(int(i)//H, int(i)%H, round(float(v), 4)) for i, v in zip(idx, vals)]

    sink_top = topk_heads(sink)
    retr_top = topk_heads(retr) if val_pos else []

    # massive activations: channels whose max-abs >> median over channels (per layer max)
    chan_max = massive.max(dim=0).values            # [hidden] worst-case over layers
    med = chan_max.median()
    outlier_mask = chan_max >= (6.0 * med)
    outlier_idx = torch.nonzero(outlier_mask).flatten().tolist()
    outlier_vals = [(int(c), round(float(chan_max[c]), 2)) for c in outlier_idx]
    outlier_vals.sort(key=lambda x: -x[1])

    # H2-precondition: overlap of top sink heads vs top retrieval heads
    sink_set = {(l, h) for l, h, _ in sink_top}
    retr_set = {(l, h) for l, h, _ in retr_top}
    inter = sink_set & retr_set
    union = sink_set | retr_set
    jaccard = (len(inter) / len(union)) if union else 0.0

    # Spearman between sink and retrieval over ALL heads (precondition for H2 logic)
    def spearman(a, b):
        a = a.flatten(); b = b.flatten()
        ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
        ra -= ra.mean(); rb -= rb.mean()
        denom = (ra.norm() * rb.norm())
        return float((ra @ rb) / denom) if denom > 0 else 0.0
    rho_sink_retr = spearman(sink, retr) if val_pos else None

    gate = {
        "bos_sink_present": bool((sink > 0.3).any()),
        "n_strong_sink_heads(>0.5)": int((sink > 0.5).sum()),
        "n_massive_act_channels(>=6x median)": len(outlier_idx),
        "retrieval_heads_sparse(<5%)": (len([1 for _,_,v in retr_top if v>0.05]) / (L*H) < 0.05) if val_pos else None,
        "max_sink_mass": round(float(sink.max()), 4),
        "max_retrieval_score": round(float(retr.max()), 4) if val_pos else None,
    }

    result = {
        "model": MODEL, "layers": L, "heads": H, "hidden": cfg.hidden_size,
        "phase0_gate": gate,
        "top_sink_heads(layer,head,mass)": sink_top,
        "top_retrieval_heads(layer,head,score)": retr_top,
        "massive_activation_channels(channel,maxabs)": outlier_vals[:25],
        "median_channel_maxabs": round(float(med), 3),
        "H2_precondition": {
            "top15_sink_set": sorted(list(sink_set)),
            "top15_retrieval_set": sorted(list(retr_set)),
            "overlap": sorted(list(inter)),
            "jaccard_top15": round(jaccard, 4),
            "spearman_sink_vs_retrieval_allheads": round(rho_sink_retr, 4) if rho_sink_retr is not None else None,
            "note": "DuoAttention predicts sink(streaming) and retrieval heads are largely DISJOINT -> expect low jaccard, low/negative spearman.",
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[janus] wrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
