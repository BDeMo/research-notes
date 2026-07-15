# Baseline faithfulness audit — our Cartridge/Gist vs the originals

**Verdict: both our baselines are under-powered in the SAME way** — we reduced them to a *frozen-base, single-layer
input-embedding soft prefix*, while the originals use much richer mechanisms. This likely explains why every
compressor (incl. our GCM) sits at ≈ no_ctx: we crippled all of them to the weakest injection.

## Cartridge — ours vs Eyuboglu et al. 2025 (arXiv 2506.06266)
| aspect | **original** | **ours** (`gcm/methods.py::Cartridge`) | impact |
|---|---|---|---|
| parameterization | **trainable KV-cache** `Z ∈ R^{L×p×d×2}` — key+value vectors at **every layer L** (prefix-tuning) | `K×d` **input-embedding** prefix, **layer-0 only** (`prefix_p`) | ours has ~L× fewer params and only injects at the embedding layer → must propagate through frozen layers; far weaker |
| training | self-study (synthetic QA) + **context-distillation** (KL to full-ctx teacher), base frozen | self-study + context-distillation ✓ | **matches** |
| stability | freeze attention-sink (first KV token) | — | minor |
| init | random or prefill | random·0.02 | ok |

➡️ **Fix = per-layer trainable KV-cache** — which is exactly **A4 (v1.7.1.4) KV-injection** that I wrongly deferred.
Faithful Cartridge ≈ A4. The paper reports it **matches ICL** (full context) — so it *should* beat no_ctx.

## Gist — ours vs Mu, Li, Goodman 2023 (arXiv 2304.08467; code jayelm/gisting)
| aspect | **original** | **ours** (`gcm/methods.py::Gist`) | impact |
|---|---|---|---|
| model | **instruction-finetune the LM** (the LM *is* the gist predictor) with the gist mask | **frozen base**; train only `K×d` gist embeddings (`gist_p`) | ours: a frozen base **never learns to compress** into gist tokens → crippled (matches our A3 finding) |
| data | **diverse instruction dataset** (Alpaca+); generalizes to novel prompts, no per-task retrain | **per-corpus**, 96 items, gold-CE | ours: per-corpus, no generalization |
| gist token | added to vocab; compression lives in the **finetuned model's activations** | a learned input embedding | different object |
| mask | mask lower-left (post-gist can't attend pre-gist) ✓ | gist mask ✓ | **matches** (only the mask is faithful) |

➡️ **Fix = LoRA-finetune the base + gist mask on diverse data** — i.e. **A3-style** (we already have the LoRA + the
gist mask). Faithful Gist ≈ A3 + gist-mask.

## The unifying point
Our **GCM, Cartridge, Gist are all the same crippled mechanism** (frozen base + layer-0 soft prefix). The faithful
baselines need precisely the mechanisms our A-group explored:
- faithful **Cartridge = A4 (per-layer KV-cache)**,
- faithful **Gist = A3 (LoRA-finetune) + gist-mask**.

So: (1) our baseline comparison so far is **unfair** (we beat/lose to crippled baselines); (2) "compression doesn't
work" is **premature** — it may just be that the *frozen-base layer-0 prefix* doesn't work, while KV-cache /
finetuning do (the papers say they match ICL). **A4 should be un-deferred and implemented as the faithful Cartridge.**

## Action items
1. **Faithful Cartridge (A4 KV-cache)** — ✅ **IMPLEMENTED + smoke-passed** (Qwen3-8B). `methods.py::Cartridge` now = per-layer
   trainable KV-cache `Z (L,n_kv,p,hd)`, prefill-init from the first chunk's KV, self-study + context-distillation,
   injected via `runtime.{cart_cache,kv_logits,generate_kv,mc_loglik_kv}` + harness `kv_prefix` dispatch. Smoke (20
   steps/8 items): distillKL 1.62→0.38, and it already emits **relevant text** (gold "Inertia"→"Inertial frame…";
   "The European Commission"→correct). ⚠️ standard-attention bases only (Qwen3-8B); Qwen3.5 is hybrid linear-attn.
2. **Faithful Gist** — ✅ **IMPLEMENTED + smoke-passed**. `methods.py::Gist` now LoRA-finetunes the base (rank≥16,
   `svc/lora.py`) UNDER the gist mask; ce drops 7.6→0.36 (undertrained at 20 steps → needs real run).
3. **TODO: full faithful-baseline runs** (proper steps, squad/hotpot) on **Qwen3-8B** (clean KV), then re-do the
   comparison before any "compression is the bottleneck" claim. (Code synced to test; sync to ray/sam-dev for scale.)
