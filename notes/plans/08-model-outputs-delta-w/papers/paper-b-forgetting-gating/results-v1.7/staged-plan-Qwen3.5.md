# v1.7 staged plan & metrics — Qwen3.5-9B (primary model)

**Model policy:** always run the newest model of an appropriate size. Current primary = **Qwen3.5-9B**
(Feb/Mar 2026, dense 9B, Apache-2.0; *hybrid linear-attention* arch → flash-linear-attn fast path unavailable on
the pods, falls back to torch ⇒ ~2–3× slower/item than Qwen3-8B, but functionally fine). Reference/older:
Qwen3-8B (the matrix), Qwen3-14B (same `qwen3` arch). Heavyweight agentic headline option: GLM-4-32B-0414 (sam-dev).

Official agentic standing (why Qwen3.5-9B): BFCL-V4 **66.1**, TAU2-Bench **79.1** — strongest *for its size*; beats
Qwen3-Next-80B agentic. (GLM-4-32B-0414 is strongest absolute: BFCL-v3 69.6, TAU-Bench retail 68.7/airline 51.2.)

---

## Generate / eval strategy (the harness `Runtime`, shared by every method & sweep)

- **Frozen base, cache-free.** Base loaded once, all params frozen. Every method differs *only* in the **prefix
  embeddings** placed before the query: `no_ctx`=nothing · `full_ctx`=the full context token-embeds ·
  `gist`=gist tokens (+4D mask blocking query→raw-context) · `cartridge`=trained soft prefix · `gcm`=K soft memory
  tokens M. Forwards re-run the growing sequence (`use_cache=False`) to dodge cross-version KV-cache API breakage.
- **Generative tasks** (`generate`): **greedy** argmax decode from `[prefix ; query]`, ≤ `max_new` (=16) tokens,
  stop on EOS. Scored (in `data.py`) on the **answer line** (first non-empty line) with squad-F1 / gold-in-pred —
  the answer-line extraction fixes the earlier verbosity confound (raw 16-tok F1 rewarded terseness).
- **Multiple-choice tasks** (`mc_loglik`): pick the option with the highest **length-normalised log-likelihood**
  of its tokens appended after `[prefix ; query]`.
- **Context**: chunks joined by newline, truncated to `max_ctx`; runs use `n_chunks=5` (~600–800 tok) ⇒ ~10–12×
  compression into K=64. Larger K (128/256) = easier ratio = more capacity.
- **GCM memory M** is produced non-autoregressively (K learnable tokens are appended to `[ctx;query]`, read out
  after the encoder = a truncated trainable copy of the base's first N blocks; query-mask toggles M0 vs Mq).

---

## Staged goals & the metric to watch at each step

Primary task signal = **`acc_compressor`** (the method's eval accuracy). Always reported next to `acc_no_ctx`,
`acc_full`, the baselines, and `best(compress,full)` (the per-item oracle-gate ceiling, GCM-only).

### IN-TASK (train-set = eval-set, e.g. trivia→trivia). Climb this ladder in order:

| Stage | Goal | **Metric / 提点** | **vs whom** |
|---|---|---|---|
| **I-1** | beat no-context | `acc_gcm − acc_no_ctx > 0` | **no_ctx** (the floor; "M is not harmful") |
| **I-2** | beat our baselines | `acc_gcm − max(acc_gist, acc_cart) > 0` | **Gist, Cartridge** (strict originals) |
| **I-3** | approach full-context | minimise gap `acc_full − acc_gcm` | **full_ctx** (the ceiling) |

Current blocker: at Qwen3-8B N16 bf16, GCM in-task ≈ 0.41 (seed-avg) **< no_ctx 0.48** ⇒ Stage I-1 not yet passed.
Levers being swept on Qwen3.5-9B: **K (memory width 64/128/256)**, **N (encoder depth)**, **adv (lam_adv/layer)**,
steps, compression ratio (n_chunks). Hypothesis: **K (width) is the dominant capacity lever**.

**Measured in-task ladder (trivia, bf16, n=96), Qwen3-8B reference** (`out/q35/q3_8b_baselines_trivia`):
`no_ctx 0.474 · Gist 0.322 · GCM-adv ~0.41(seed-avg) · Cartridge 0.541 · full 0.654` (best(compress,full): Cart 0.735).
⇒ on Qwen3-8B the baseline to beat is **Cartridge 0.541** (Gist is weak here); GCM clears neither no_ctx nor Cartridge yet.
Memory note: **K256@N16 OOMs on Qwen3.5-9B@80G** (bigger model + linear-attn) → high-K runs use N12, or N16 on the 100G card.

### CROSS-TASK / CROSS-DOMAIN (train trivia → eval hotpot / squad / quality / tool / …). Goal = **be highest**, with **fallback ON by default**:

| Metric | Definition | **Target / vs whom** |
|---|---|---|
| raw `acc_gcm` | compressed-only accuracy off-distribution | report vs no_ctx / baselines / full |
| **gated accuracy** | apply the gate: compress when signal says safe, else **fall back to full** | **≥ acc_full** (do-no-harm) and ideally **> always-full** (pick up the OOD-compressible items) |
| **gate F1 / P / R** | compress-vs-fallback as a classifier (track B: y=1 iff comp ≥ full−ε) | maximise F1 at **precision-first** (do-no-harm = high P) |
| **gate AUROC** | how well the signal ranks safe-to-compress | the headline gate-quality number |
| `best(compress,full)` | oracle-gate ceiling | the upper bound the gate chases |

**Gating signals to compare (report F1 + AUROC for each, pick the winner):**
- **Intrinsic / verifiability:** `neg_recon` (M0 reconstruction), `dcode`/`dlogit` (M0↔Mq gap), `mnorm` (geometry).
- **Generic uncertainty:** `conf`, `margin`, `neg_entropy` on the deployed `[Mq;q]` run.
- **Learned discriminator (adversarial):** `disc_negp` = compress iff D thinks "full" (recommended) · `disc_conf`
  = compress iff D confident either way · `disc_p` = compress iff D thinks "compressed". Run harness with `--signals`.
- Baselines for the gate: always-compress, always-full, and `best(compress,full)`.
  *Finding so far (Qwen3-8B):* disc-gate is marginal (best `S_conf` AUROC≈0.70, **ties** generic `neg_entropy`);
  no clear architectural advantage yet — revisit once the compressor itself clears Stage I-1.

---

## Batch-1 findings (2026-06-11) — and a task-selection correction

- **Width K helps on Qwen3.5, not on Qwen3-8B.** Qwen3.5 trivia: K64=0.256 → **K128=0.537** (adv); no-adv K128=0.387.
  Qwen3-8B trivia: K64≈0.41 ≈ K256=0.401 (no gain). So the capacity lever is model-dependent; K128 is the workhorse.
- **`best(compress,full)` ceilings are high** (Qwen3.5 K128 trivia 0.734) ⇒ a good gate has lots to gain even when
  raw compress loses — the gate story survives.
- **Trivia is the WRONG in-task benchmark.** On both models context barely helps and on **Qwen3.5 no_ctx 0.594 >
  full 0.538** (the model knows trivia parametrically; retrieved chunks distract it). "Beat no_ctx" is then ill-posed:
  compression of distracting context cannot beat ignoring it. The Qwen3-8B `n_chunks=2` easy-ratio diagnostic also
  has GCM 0.291 ≪ no_ctx 0.491 ⇒ not a ratio problem.
- **Correction:** the in-task ladder must run on **context-dependent** tasks where **full ≫ no_ctx** (answer lives in
  the context): squad_v2 / hotpot_qa / narrativeqa. There no_ctx is low ⇒ real headroom for compression to beat it.
  Pivoted the in-task push to these (batch 2).
- **Likely true bottleneck:** injection mechanism. GCM injects M as *input-embedding* soft prefix; **Cartridge (trained
  KV-cache) 0.541 ≫ GCM 0.40** on the same trivia in-task. Candidate next lever after batch 2: KV-cache injection.

## Experiment queue (keep all 9 GPUs busy)

**ray (4×80G) + test (1×100G) — Qwen3.5-9B in-task capability push (Stage I-1/I-2), trivia, n=96, 800 steps, bf16:**
- `q35_K64_N24_adv` · `q35_K128_N24_adv` · `q35_K256_N24_adv` (K width sweep @ adv L24)
- `q35_K128_N24` (no-adv, isolates adv at K128)
- `q35_baselines` (no_ctx, full_ctx, **Gist, Cartridge** @1500 steps) → the Stage I-2 targets

**sam-dev (4×93G):** finishing the Qwen3-8B **11-bench transfer matrix** (authoritative cross-task/domain on the
older model); when free → Qwen3.5-9B **tool/agentic** (bfcl/apibank/toolace/rca, datasets cached here) + cross-domain.

**Next batches (queued):** winning-K × {N, steps, n_chunks} refinement → Stage I-3; then `--signals` rerun of the
winning config for the gate F1 table; then cross-task/domain eval with fallback. (TIR-Bench/V\* deferred — need new
text-tool wiring / a multimodal compress-vision-tokens path.)
