# Paper A — ARCHIVE (do-no-harm gate): facts frozen

> **Status: ARCHIVED (2026-07-06).** Facts documented and frozen; not actively pushed. The gate / C3 lives here (Paper B
> no longer discusses it). Reopen only if the conformal validation (below) is run and lands.

## Thesis (frozen)
A compressed-context memory is lossy and capacity-bound, so the durable contribution is **not a better compressor but a do-no-harm gate**: per input, trust the cheap compressed memory only when a robust intrinsic signal says it's safe, else fall back to full context — quality never below the feasible baseline, at compressed cost when safe. *The compressor is the carrier; the gate is the contribution.*

## Facts established (frozen — see `matrix-facts.md`)
- **F12 — the compressor is a commodity:** flipping any single loss moves compress by ±0.03; the pruned 2-loss core ≈ the kitchen-sink (Δ0.01). ⇒ effort belongs on the gate, not the compressor. ✅
- **F13 — "gated ≥ full" was TAUTOLOGICAL in-sample** (cannot drop below full by construction), so the v1.8 headline was not a real result. The honest version requires an **out-of-sample / conformal** threshold with stated coverage. ✅ (diagnosis) / ⏳ (the conformal validation itself — the ONE open item, `matrix-experiments.md` X-C3).
- Gate signals characterized: `conf`, `margin`, `neg_entropy`, `neg_recon`, `dlogit`, `targ`; best signal is **bench-dependent** (an argument for a learned/ensemble gate). ✅
- Baselines are faithful/annotated (`baseline-catalog-faithfulness.md`, D21): kvpress KV methods + LLMLingua-2 exact; gist/cart NOT faithful; txl ≠ StreamingLLM.

## Elegant method (frozen design — `method-elegance-plan-v1.8.x.md`)
One frozen base, four roles (encode/read/self-distill/self-verify), **two losses** (answer + reconstruct), **one gate signal** (the reconstruction score itself = train-loss ≡ inference-signal), **one guarantee** (conformal calibration). Validated: pruned ≈ kitchen-sink (T1 A/B). Not validated: the conformal guarantee (X-C3).

## Honest caveats carried into the archive
- The do-no-harm headline is **not proven** until X-C3 (conformal) runs; until then state it as *designed*, not *shown*.
- gist-lite / cartridge-lite are our-infra approximations — **not** Gisting/Cartridges (do not cite as such).
- CMG demo accuracy is near-chance (motivation only, never a headline).

## Where its material lives
- Facts: `matrix-facts.md` (F12, F13). Method: `method-elegance-plan-v1.8.x.md`. Results: `results-v1.8.0/`. Logs: `results-v2.0.0/logs/` (T1 A/B = `t1_ab_*`).

## To reopen Paper A
Run **X-C3** (held-out/conformal threshold → coverage + risk; gated ≥ feasible baseline out-of-sample) and **X-C6** (one gate across many compressors). If both land, the do-no-harm + generality claims become real and Paper A is submittable.
