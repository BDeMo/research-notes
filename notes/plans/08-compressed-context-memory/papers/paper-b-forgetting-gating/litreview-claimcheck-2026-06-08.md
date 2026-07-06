# Lit review + claim check (2026-06-08)

> Question we are answering: how do current latent-memory methods work, and are our
> paper's claims correct against the 2025--2026 literature? Sources are web-searched
> primaries (arXiv/ACL/AAAI), listed inline. Bottom line: the *forgetting* and
> *augmentation-can-hurt* claims are well-supported; the **novelty claim needs
> tightening**, because gating *when to augment* is now a mature line for
> **retrieval** (TARG, L-RAG, CTRLA, Adaptive-RAG). Our defensible novelty is
> narrower and must be stated precisely.

## 1. How current latent memory actually works (the compression line)
All of these optimise **how much** a fixed token budget can carry; none gate
**when** to use it, and none target a do-no-harm floor.
- **Gist** (Mu et al., NeurIPS'23): attention-mask trick, compress a prompt into a
  few gist tokens, up to 26x; tokens are tied to the fine-tuned model.
- **AutoCompressor** (Chevalier et al., 2023): recursive segment summary vectors,
  long contexts (~30k), soft-prompt accumulation.
- **ICAE** (Ge et al., 2024): in-context autoencoder; **frozen decoder**, compress
  long context into memory slots, ~15x.
- **500xCompressor** (2024): compress into **KV values** (not embeddings), 6--480x.
- **Cartridges** (Stanford/Hazy, arXiv 2506.06266, Jun 2025): **train a small KV
  cache offline per corpus on a frozen LLM** via *self-study* (synthetic
  conversations + context-distillation), amortised across queries; 38.6x less
  memory, 26.4x throughput, composable. **This is the closest prior to our
  learned-memory-module-on-frozen-base.** It is prefix-tuning on the KV cache; it
  does **not** add a per-input gate or a fallback.

> Verdict: our description of the compression line ("optimise how much, not when")
> is correct. Cartridges is the must-cite nearest neighbour on the *module* axis.

## 2. The "when to use it" line is MATURE -- for retrieval (the risk to our novelty)
Deciding *whether/when* to augment, per input, is an active, crowded area. Our
gate is not the first gate; it is a gate on a different object.
- **TARG** (Training-free Adaptive Retrieval Gating, arXiv 2511.09803): a short
  **no-context draft**, read prefix logits, score by **mean entropy / top-1-vs-top-2
  margin / small-N variance**, retrieve only if uncertainty exceeds a threshold.
  **This is essentially our Track-B base-uncertainty gate (`margin_0`, `conf_0`,
  `entropy_0`).** We already flagged TARG; this confirms our cheap-signal gate is
  *not* novel on its own.
- **L-RAG** (Lazy RAG, entropy-based lazy loading, arXiv 2601.06551): **two-tier** --
  a compact **summary first**, fall back to **full retrieval** only when predictive
  entropy is high. **This is structurally our Track B** (cheap compressed tier ->
  full context on uncertainty). Closest threat to Track B's novelty; must cite +
  distinguish (their cheap tier is a text summary + vector store; ours is a
  *learned latent* memory).
- **CTRLA** (Findings-ACL'25): adaptive RAG via **internal representation control**
  (honesty/confidence directions) for retrieval timing -- a learned-signal gate.
- **Adaptive-RAG / Search-R1 / ReALM / GRIP** (2025--2026): when/how-much to
  retrieve via RL or control tokens.

> Verdict: "we introduce the question of *when* to use augmentation" is **false as
> stated** -- it is mature for retrieval. Reframe (see §5).

## 3. Augmentation can hurt -> the do-no-harm framing is well-supported
- **Astute RAG** (ACL'25, arXiv 2410.07176): "imperfect retrieval augmentation is
  **inevitable, common, and harmful**"; knowledge conflicts; Astute RAG is the only
  method that **matches/exceeds no-RAG under the worst case**. Direct support for
  the do-no-harm goal (and a target framing: "match the base in the worst case").
- **When Retrieval Hurts** (medical QA, 2025): RAG **helps 5--6% but harms 6--7%,
  net negative**; crucially, **retrieval scores fail to predict utility (AUC 0.53)**;
  harm rises with base confidence. Great motivation, and a clean contrast: our
  `delta_last` predicts utility at **AUROC 0.71**, i.e. a *learned-memory write
  signal* predicts usefulness better than retrieval scores do.

> Verdict: the "augmentation can hurt, so gate it" motivation is strongly supported;
> cite Astute RAG + "When Retrieval Hurts" as motivation, not as our finding.

## 4. Fine-tuning forgets -> supported, but a mitigation line exists (baseline gap)
- **Catastrophic forgetting under (continual) fine-tuning** is confirmed empirically
  for 1B--14B (worsens with scale; Qwen2.5 3B->14B) -- supports our LoRA-forgets claim.
- **But forgetting-aware LoRA exists:** **O-LoRA** (EMNLP'23, orthogonal subspaces),
  **OPLoRA** (AAAI'26, orthogonal projection preserves top-k singular triples),
  **CLoRA** (ACL'25, null-space subspace regularisation). These *reduce* forgetting
  while still training in-weight.

> Verdict: "LoRA forgets" holds against **vanilla** LoRA (what we ran). A reviewer
> will ask why we did not compare to a **forgetting-aware** LoRA. Either add an
> O-LoRA/OPLoRA baseline, or scope the claim to vanilla LoRA and cite these as the
> in-weight alternative we deliberately avoid (we don't touch the base at all).

## 5. Claim-by-claim verdict for the paper
| paper claim | verdict | action |
|---|---|---|
| Latent-memory/compression optimises *how much*, not *when* | **holds** | keep; cite Gist/ICAE/AutoCompressor/Cartridges/500x |
| We add the *when-to-use* question + fallback | **overstated** | reframe: when-to-augment is mature for **retrieval** (TARG/L-RAG/CTRLA); our object is a **learned latent memory**, not text retrieval |
| Do-no-harm floor (never below base) | **holds (framing)** | cite Astute RAG ("match no-RAG worst case") as the analogous goal; our floor is *by construction* via the frozen base, not learned robustness |
| Fine-tuning catastrophically forgets off-task | **holds vs vanilla LoRA** | add O-LoRA/OPLoRA baseline OR scope to vanilla + cite the mitigation line |
| `delta_last` predicts usefulness, transfers (AUROC 0.71) | **holds + strengthened** | contrast with "When Retrieval Hurts" AUC 0.53 for retrieval scores |
| Two-track (base <-> mem <-> full) | **partially novel** | L-RAG is summary<->full; distinguish: ours is *learned latent* mem + base fallback + one shared signal |

## 6. The defensible novelty statement (use this wording)
Not "we gate augmentation" (done) and not "we compress context" (done). Precisely:

> We are the first to put a **per-input do-no-harm gate on a *learned latent
> memory*** (vs. text retrieval), giving a **frozen-base floor by construction**
> (vs. learned robustness like Astute RAG), driven by an **intrinsic
> memory-write signal** (`delta_last`) that transfers across model families (vs.
> base-uncertainty-only signals like TARG, or retrieval scores that fail to predict
> utility), and we show the gate is a *necessity* via a cross-model boundary where
> the unguarded memory hurts (Mistral).

## 7. Must-cite additions (currently missing from the draft)
TARG (2511.09803), L-RAG (2601.06551), CTRLA (Findings-ACL'25), Astute RAG
(ACL'25), "When Retrieval Hurts" (2025), Cartridges (2506.06266), O-LoRA
(EMNLP'23), OPLoRA (AAAI'26), CLoRA (ACL'25), 500xCompressor, ICAE, AutoCompressor.
Add bib entries + a Related Work paragraph on **adaptive/selective augmentation**
that explicitly says our object (learned latent memory) differs from theirs
(retrieval), and a sentence in §forgetting on **forgetting-aware LoRA**.

## 8. Action items
1. **Reframe the novelty** in abstract/intro/related to §6 wording (stop implying we
   invented when-to-augment).
2. **Add the adaptive-augmentation Related Work paragraph** (TARG/L-RAG/CTRLA/Astute)
   with an explicit "our delta = learned latent memory, not retrieval."
3. **Close the LoRA-baseline gap**: run an O-LoRA/OPLoRA forgetting-aware baseline, or
   scope to vanilla LoRA + cite the mitigation line.
4. **Distinguish Track B from L-RAG** in one sentence.
5. Use the **AUC 0.53 vs AUROC 0.71** contrast as evidence the memory-write signal is
   a better utility predictor than retrieval scores.
