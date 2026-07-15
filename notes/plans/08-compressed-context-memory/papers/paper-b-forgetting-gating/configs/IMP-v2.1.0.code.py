# ============================================================================
# IMP-v2.1.0 — EXACT method code (verbatim snapshot for reproducibility)
# ----------------------------------------------------------------------------
# Source: gcm/experiments/run_baseline.py, branch `MODE == "imp"` (frozen base).
# Version: IMP-v2.1.0 = span-level Importance-routing Prefilter, Mode A (training-free).
# Config (env): GCM_BASELINE=imp  GCM_IMP_SPAN=32  GCM_LL_RATE=0.5  GCM_IMP_SIGNAL=both
#               GCM_MAXCTX=<window; 32768 for long-context, no artificial cut>
# Signals: query-relevance (embedding cosine, prefill-free) + surprisal (one forward), z-summed (F20).
# Selection: split into contiguous SPAN-token blocks; block score = max token score; keep top
#            (keep//SPAN) blocks WHOLE in reading order (preserves the answer span). keep = max(8, p*L).
# This file is a copy for traceability; the executed code lives in run_baseline.py. Do not import.
# ============================================================================

# --- scoring + selection (verbatim) ---------------------------------------
def _imp(it):
    ids = rt.tok("".join(map(str, getattr(it, "chunks", []) or [])), add_special_tokens=False,
                 truncation=True, max_length=MAXCTX, return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] == 0:
        return embed(rt.query_ids(it))
    L = ids.shape[1]
    with torch.no_grad():
        E = embed(ids)[0].float()
        qe = embed(rt.query_ids(it))[0].float().mean(0, keepdim=True)
        En = E / (E.norm(dim=-1, keepdim=True) + 1e-6); qn = qe / (qe.norm() + 1e-6)
        qdot = (En @ qn.T).squeeze(-1)                       # query relevance (F20: 0.95 word-needle)
        lg = base(inputs_embeds=embed(ids), use_cache=False).logits[0].float()
        lp = lg.log_softmax(-1); surp = torch.zeros(L, device=dev)
        surp[1:] = -lp[:-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)   # surprisal (F20: 0.84 numeric)
        z = lambda t: (t - t.mean()) / (t.std() + 1e-6)
        _sig = env("GCM_IMP_SIGNAL", "both")                 # signal ablation: query / surprisal / both
        score = z(qdot) if _sig == "query" else z(surp) if _sig == "surprisal" else z(qdot) + z(surp)
    keep = max(8, int(LL_RATE * L))
    SPAN = int(env("GCM_IMP_SPAN", "1"))                     # 1 = token-level (v2.0); >1 = span-level (v2.1.0)
    if SPAN <= 1:
        idx = score.topk(min(keep, L)).indices.sort().values  # keep top-p tokens, preserve order
    else:
        nb = (L + SPAN - 1) // SPAN                          # split into contiguous spans; span score = max token score
        pad = torch.full((nb * SPAN - L,), float("-inf"), device=dev)
        bs = torch.cat([score, pad]).view(nb, SPAN).max(dim=1).values
        nkeep = max(1, keep // SPAN)                         # keep whole top spans (preserve local coherence)
        blocks = bs.topk(min(nkeep, nb)).indices.sort().values
        idx = torch.cat([torch.arange(int(b) * SPAN, min(int(b) * SPAN + SPAN, L), device=dev) for b in blocks])
    return embed(ids[:, idx])

# --- scoring path used downstream -----------------------------------------
# MC benches:  model.mc_loglik(prefix=embed(ids[:, idx]), query, letters)  -> argmax letter accuracy
# gen benches: model._gen_batch([embed(ids[:, idx])], query, ...)          -> score_gen / gold-substring
# no_ctx (blind) and full (embed(cids[:, :MAXCTX])) columns are computed for every item as bounds.
