# Paper C (v3.0) — research line: linear attention & its variants

**Scope decision (2026-07-14):** future work focuses on **linear attention and its variants**. Paper C = version tag **v3.0**, demo **v0.3.0**. This doc organizes the research thread, positioning, the bridge from our current work, candidate contributions, and open questions. References: [`references-linear-attention.md`](references-linear-attention.md).

## 1. The mainline (what the field is doing, 2024→2026)
The dominant thread is **recurrent/linear sequence models with an expressive, data-dependent fixed-size state** that trade softmax attention's O(L²)/growing-KV for O(L)/O(1)-state:
- **Kernel/linear** (Linear Transformer→RetNet→GLA) → **gated linear RNNs** (HGRN2, Mamba/Mamba-2/3).
- **Delta-rule / test-time-regression** line is the current frontier: **DeltaNet → Gated DeltaNet (Qwen3.5/3.6's module) → GDN-2, KDA/Kimi-Linear, DeltaProduct, Comba, MesaNet, RWKV-7** — the state is updated by an online least-squares/delta step, i.e. **memory as test-time learning**.
- **Hybrids win in practice**: nobody ships pure-linear at scale; **linear : full-attn interleave** (Samba, MiniMax-01 lightning, Qwen3.5 GDN 3:1, Jamba) — a few global-attention layers restore what linear state can't hold.
- **Sparse-attention peers** (NSA, MoBA, FoX) attack the same long-context cost from the softmax side.

**Two forces define the frontier:** (a) **expressivity/state-tracking** (delta/Householder products vs TC⁰ limits — "Illusion of State"), and (b) **recall capacity** (fixed state ⇒ the recall–throughput tradeoff, Based/Zoology). Hybrids + bigger/structured state are how the field buys both.

## 2. The gap / tension Paper C can own
Linear & hybrid models are efficient but have a **structural recall bottleneck**: a fixed-size state cannot losslessly hold a long context, so **long-range retrieval/associative recall degrades** (Based recall–throughput; Illusion-of-State for state-tracking). Our own data already shows this concretely:
- **F28/F43:** on Qwen3.5-9B (GDN 3:1 hybrid), our findings reproduce, **but IMP's RULER-needle gap is WIDER than on quadratic** (79 vs 96.8) → **state-collapse / recall limit of the linear layers**.
- **F10:** KV-eviction methods (kvzip/knorm) — the literary-MC winners in Paper B — **cannot even run on linear** (no KV cache). Input-side selection (our IMP-lite, RAG) is the *only* compression family that works on linear.

⇒ **The opening:** on linear/hybrid bases, the long-context problem is not "shrink the KV cache" (there isn't one) but "**feed the fixed-state model the right, small context**" and/or "**augment its state's recall**". This is exactly where our training-free input-side compressor (v3.0 IMP-lite) lands — and it's a regime the KV-compression literature structurally cannot address.

## 3. Candidate contributions (pick 1–2 after a scoping pass)
- **C-1 (diagnostic, our strength):** a systematic **recall/state-tracking failure map of linear & hybrid architectures** (GDN, GDN-2, Mamba-3, RWKV-7, KDA, MiniMax, Qwen3.5 vs quadratic) across RULER/∞Bench/BABILong/associative-recall — *where* and *why* the fixed state fails, and how the **hybrid full-attn ratio** modulates it. Model-invariance style, extended to architectures.
- **C-2 (method, training-free):** **input-side memory for linear models** — IMP-lite (full-doc lexical chunk retrieval, no forward, no KV) as a plug-in that restores long-range recall the fixed state loses; show it closes the linear-vs-quadratic gap (F28) where KV-methods can't run.
- **C-3 (method, light-training / architectural):** an **external/side recall module** for linear bases (learned retrieval into the state, or a delta-rule-friendly memory), bridging to the test-time-training view (TTT/Titans) — a small, principled add-on that raises recall capacity without going quadratic.
- **C-4 (analysis):** connect the **delta rule = online least-squares memory** view to recall capacity; where DeltaProduct/GDN-2/KDA improvements actually help on real long-context recall (not just synthetic state-tracking).

**Occam bias:** prefer C-1 (diagnostic) + C-2 (training-free plug-in) as the spine — both build directly on our validated fact-base and need no base retraining; C-3/C-4 are stretch.

## 4. Bridge from current work (what we already have)
- Working harness on **Qwen3.5-4B/9B GDN** (frozen), full eval pipeline, RULER/∞Bench/BABILong/LongBench loaders, IMP-lite (runs on linear), RAG, and the diagnosis that KV-methods don't run on linear.
- `linear-attention-and-kvcache-background.md` (GDN mechanics, KV differences), F10/F28/F43.
- Immediate reusable experiment: **multi-architecture recall sweep** (extend `fam_*`/`gen_*` to Mamba-3, RWKV-7, KDA/Kimi-Linear, MiniMax, pure-GDN vs hybrid-ratio) on RULER/∞Bench/BABILong.

## 4b. Model scale & family (Paper C directives, 2026-07-14)
- **Scale:** > 10B and < 30B params (≈13B–27B band) for headline. No sub-10B headline; nothing ≥30B. (Scoped to Paper C; Paper B's Qwen3-8B stands.)
- **Family (updated):** **Paper C uses the GDN linear/hybrid family ONLY — Qwen3.5 and Qwen3.6 — NO Qwen3-dense control.** The comparison is *within* the linear family (versions/scales), not linear-vs-quadratic.
- **Availability (2026-07-14):** cache has **Qwen3.5-4B, Qwen3.5-9B** only; **Qwen3.6 NOT cached** (need to locate/download the in-band Qwen3.6-27B checkpoint — HF-Hub had 403 issues; secure it for headline). Scoping module runs use Qwen3.5-4B/9B now; 3.6-27B is the in-band headline once available.
- **In-band candidates to prioritize:** Qwen3-14B (dense quadratic control), **Qwen3.6-27B / 35B-A3B** (GDN 3:1 hybrid — the in-band *linear/hybrid* headline), any ~13–27B Mamba-3 / RWKV-7 / KDA-Kimi-Linear / Jamba-active-in-band checkpoint.
- **Out-of-band (demote to small ablations only):** Qwen3.5-**9B** (just under 10B), Qwen3.5-4B, Qwen3-1.7/4/8B; MiniMax-01 (~456B, too big).
- Action: the checkpoint inventory (§6) must **filter to the 10–30B band** and confirm which in-band linear/hybrid weights are open & runnable in our harness (Qwen3.6-27B is the key target to secure).

## 5. Open questions to resolve in a scoping pass
1. Which linear/hybrid checkpoints are open & runnable in our harness (Qwen3.5 ✓; Mamba-3? RWKV-7 ✓ via fla; KDA/Kimi-Linear? MiniMax-01 weights?)?
2. Is the recall gap **fixable input-side** (IMP-lite) or does it need a **state-side** augment (C-3)? — decide empirically.
3. Does the **hybrid full-attn ratio** (Qwen3.5's 3:1) predict the recall gap? A clean knob to study.
4. Relationship to **test-time-training memory** (TTT/Titans) — is our input-side compressor complementary or redundant to a better state update?

## 6. First actions
1. ✅ references collected ([`references-linear-attention.md`](references-linear-attention.md)).
2. ▶ Verify arXiv IDs; read in full: GDN (2412.06464), GDN-2 (2605.22791), Mamba-3 (2603.15569), KDA/Kimi-Linear (2510.26692), Based+Zoology (2402.18668/2312.04927), Illusion-of-State (2404.08819).
3. ▶ Inventory runnable linear/hybrid checkpoints in the cluster HF cache.
4. ▶ Scoping experiment: recall failure map (RULER/∞Bench/BABILong) across the available linear/hybrid bases vs a quadratic control, with IMP-lite as the input-side fix — decide C-1 vs C-2 spine.
