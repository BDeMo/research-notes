# v1.7 Compressor training design (working note)

> Records the 2026-06-10 brainstorm (two losses, the conditional expectations, the three-level gap
> diagnostic, the gap-as-self-supervised-signal, the deploy cascade) and a PROPOSED training procedure.
> Basis for ongoing co-design. Pairs with [`v1.7-gated-compressor-module-2026-06-10.md`](v1.7-gated-compressor-module-2026-06-10.md).
> House rule: no em-dashes. Code: `mem-test/svc/`. Status: design under active discussion.

## 1. The two-mode compressor (recap)
One module, two modes (the ablation axis):
- **Unconditional (cacheable):** $M_0 = \mathrm{Compress}(C)$. Query-agnostic, reusable.
- **Conditional:** $M_q$ = the SAME encoder run with the query UNMASKED (see §9; this **supersedes** the
  earlier $M_0+\alpha\cdot\mathrm{Refine}$ residual sketch). $M_0$ and $M_q$ share one encoder.
At deploy the K soft tokens REPLACE the context (base sees $[\,M\,;q\,]$), not LoRA, base fully frozen.

## 2. The two losses
- **$\mathcal{L}_{\text{uncond}}$ (lossless aspiration):** $M_0$ should be a query-agnostic sufficient statistic for $C$.
  Operationalised as **behavioural losslessness** over a probe-query set $Q_C$:
  $\mathrm{KL}\big(p_{\text{full}}(\cdot|q)\,\|\,p_{M_0}(\cdot|q)\big)$ averaged over $q\in Q_C$, with **reconstruction**
  (decode pooled / per-chunk ctx from $M_0$) as a denser proxy.
- **$\mathcal{L}_{\text{cond}}$ (per-query performance, minimal deviation):**
  $$\mathcal{L}_{\text{cond}}=\underbrace{\mathrm{KL}\big(p_{\text{full}}(\cdot|q)\,\|\,p_{M_q}(\cdot|q)\big)}_{\text{align oracle on }q}+\lambda_{\min}\underbrace{\lVert M_q-M_0\rVert^2}_{\text{deviate from }M_0\text{ only when needed}}\;[\,+\,\lambda_{\text{ce}}\,\mathrm{CE}(\text{gold}\mid M_q,q)\,].$$
  Distil to the **full-context teacher** (not only gold) so $M_q$ matches full's BEHAVIOUR, which is what makes the
  gap signals (below) meaningful. The optional CE-to-gold lets $M_q$ exceed full when possible.

The min-deviation term is the keystone: where $M_0$ suffices, moving $M_q$ earns no distillation gain but pays the
penalty, so $M_q=M_0$; where $M_0$ is insufficient, the distillation gain dominates and $M_q$ moves. So the
$M_0\leftrightarrow M_q$ gap appears **exactly where $M_0$ is lossy on $q$**, i.e. it is calibrated by construction.

## 3. Conditional expectations
1. **No-op in the lossless regime.** If $M_0$ is already lossless on $q$, then $M_q=M_0$ (gap $\to 0$). If conditional
   beats unconditional, that is a DIAGNOSTIC that $M_0$ was lossy on $q$ (capacity-bound).
2. **Pushes the capacity wall out.** At fixed $K$, conditioning holds accuracy at higher compression ratio / longer
   context, because it only preserves query-relevant info (a task-conditional rate-distortion argument).
3. **Raises the compress-decision F1** (the headline metric): only query-relevant info must survive, so more inputs
   are safely compressible -> higher coverage at fixed precision.
4. **New failure mode:** "compressed the wrong thing" (misread query, dropped a multi-hop bridge). Expect conditional
   to help single-span queries more than multi-hop, and the gate must catch relevance errors, not just capacity.

## 4. The three-level gap diagnostic (analysis)
For $(C,q)$: $\Delta_{\text{code}}=\lVert M_q-M_0\rVert$; $\Delta_{\text{logit}}=D\big(p_{M_q}(\cdot|q)\,\|\,p_{M_0}(\cdot|q)\big)$; $\Delta_{\text{bb}}=\mathrm{score}(M_q)-\mathrm{score}(M_0)$ (signed by gold).

| $\Delta_{\text{code}}$ | $\Delta_{\text{logit}}$ | $\Delta_{\text{bb}}$ | reading |
|---|---|---|---|
| large | small | $\approx 0$ | lossless-redundant: $M_0$ already enough, conditioning wasted -> use cacheable $M_0$ |
| large | large | $+$ (cond right, uncond wrong) | capacity-rescue: $M_0$ lossy on $q$, conditioning concentrated budget. $\Delta_{\text{bb}}$ = value of conditioning |
| large | large | $-$ (cond wrong, uncond right) | relevance failure: conditioning dropped the needed info -> the gate must catch this |
| small | - | uncond still wrong | conditioning inert (training problem: $\alpha$ stuck, query ignored) |
| small | large | any | sensitive/unstable: tiny code change -> big logit swing -> low robustness, gate be cautious |

## 5. Gaps as deploy signals (the key move)
- **$\Delta_{\text{bb}}$ depends on the oracle (gold/full)** -> EVAL + ANALYSIS only, and for generating the gate's
  TRAINING labels. NOT a deploy signal.
- **$\Delta_{\text{code}}$ and $\Delta_{\text{logit}}$ are deploy signals** (no gold). Cost order:
  $\Delta_{\text{code}}$ (compressor only, free) < $\Delta_{\text{logit}}$ (2 base prefills) < full-context (expensive).
- **Why they are calibrated:** with $\mathcal{L}_{\text{cond}}$ driving $M_q\to$ full and $\mathcal{L}_{\text{uncond}}$
  driving $M_0\to$ full-where-possible: $M_0$ enough $\Rightarrow p_{M_q}\approx p_{M_0}\approx p_{\text{full}}\Rightarrow\Delta_{\text{logit}}\approx 0$;
  $M_0$ insufficient but $M_q$ ok $\Rightarrow\Delta_{\text{logit}}\approx \mathrm{KL}(p_{\text{full}}\|p_{M_0})$ = $M_0$'s loss to oracle,
  obtained WITHOUT running full. $\Delta_{\text{code}}$ is the cheaper pre-screen ($\approx 0\Rightarrow$ skip the $M_q$ prefill).
- **Caveat:** if BOTH $M_0$ and $M_q$ fail (agree on a wrong answer), $\Delta_{\text{logit}}$ is falsely small. This case is
  caught by low recon-coverage / low query-coverage + the full-context-calibrated gate. Conditioning thus does double
  duty: a better per-query compression AND a probe of whether $M_0$ was lossy.

## 6. Deploy cascade (3-way routing by cost)
$\Delta_{\text{code}}$ (free) $\to$ $\Delta_{\text{logit}}$ (1-2 prefills) $\to$ full-context (fallback). Cache $M_0$;
per query compute $M_q$ + gaps; route among {cacheable $M_0$, per-query $M_q$, full-context}.

## 7. PROPOSED training procedure (DRAFT, for discussion)
Frozen base throughout; only the compressor + gate train; BPTT through the frozen base for the distillation terms.

- **Stage 0 (data):** per context $C$, build a probe-query set $Q_C$ = benchmark queries for $C$ + teacher-generated
  self-study queries; cache the full-context teacher's answer/logits (the oracle) for each $q$.
- **Stage 1 (lossless $M_0$):** reconstruction + behavioural distillation $\mathrm{KL}(p_{\text{full}}\|p_{M_0})$ over $Q_C$.
- **Stage 2 (conditional $M_q$):** ADD $\mathcal{L}_{\text{cond}}$ (distil-to-full + min-deviation [+ CE-gold]).
  NOTE (corrects an earlier draft): the query-mask means $M_0$ and $M_q$ SHARE the encoder (§9), so we CANNOT
  freeze $M_0$ independently. Instead **keep $\mathcal{L}_{\text{uncond}}$ on (mixed) in Stage 2** so the shared
  encoder does not lose losslessness, and let the **min-deviation LOSS** (not freezing) keep $M_q\approx M_0$
  where $M_0$ suffices.
- **Stage 3 (self-verifying gate):** features = $[\Delta_{\text{code}},\Delta_{\text{logit}}, \text{recon-cov}, \text{query-cov}, \text{geometry}]$;
  label $y$ = do-no-harm (from full at train); loss = $\mathrm{BCE}(g,y) + \lambda_{\text{sep}}\cdot$ separation/ranking on the
  signals. Optionally let the separation gradient flow INTO the compressor (verifiability-aware co-design) and alternate
  Stage 2/3 (co-evolution) so the gate stays in-distribution.

## 8. Open design questions (for discussion)
1. **$M_0$ frozen vs jointly trained in Stage 2:** stable lossless substrate vs more capacity but risk corrupting $M_0$.
2. **Verifiability separation loss into the compressor vs head-only:** co-design (compressor learns to fail detectably)
   vs safer (no risk of the compressor gaming its own signal).
3. **The oracle teacher for long contexts:** full context may exceed the base window / be costly. Chunked teacher?
   retrieval teacher? truncate? This bounds what "lossless" even means for long ctx.
4. **Probe-query distribution $Q_C$ for $\mathcal{L}_{\text{uncond}}$:** how many / how generated (self-study coverage
   determines whether $M_0$ is lossless for the queries we will actually see).
5. **$\lambda_{\min}$ tuning/annealing:** too high -> conditioning never fires; too low -> noisy gap. It calibrates the signal.
6. **Reconstruction granularity:** pooled vs per-chunk (sharper worst-covered-chunk feature vs cost).
7. **Curriculum vs joint:** staged (1->2->3) vs joint-from-start vs alternating; which keeps the gate calibrated.

## 9. Architecture refinement (2026-06-10, resolves several of the above)
**Compressor = an autoencoder whose encoder is a trainable copy of the base's first N blocks.**
- **Encoder (the compressor):** initialise from the base's first $N$ transformer blocks (strong init, not a
  random pooler) + the base embedding; append $K$ learnable memory tokens. Input layout $[\,C\;;\;q\;;\;\text{mem}_{1..K}\,]$;
  run the $N$ blocks; read the $K$ memory positions as $M$. Output is exactly $K$ tokens.
  - (Init ablation: first-$N$-blocks vs the from-scratch Perceiver pooler currently in `svc/compressor.py`.)
- **Query-mask toggles the two modes, one module, one forward:** with $C$ first, masking the query positions
  (so memory and $C$ cannot attend to $q$) = **unconditional** ($M_0$); unmasking = **conditional** ($M_q$).
  Needs a custom attention mask over $[C;q;\text{mem}]$ (build + verify like the Gist mask).
- **Decoder = a TRAINABLE transformer (the autoencoder's decoder), optionally init from base blocks:** it
  reconstructs $C$ from $M$; the reconstruction CE is the **lossless loss** $\mathcal{L}_{\text{uncond}}$. If $M$
  reconstructs $C$ exactly there is no information loss. **The encoder AND the decoder are the autoencoder and
  are both TRAINED**; the decoder is a separate trainable probe, NOT the frozen base.
- **Trainable vs frozen:** trained = encoder blocks + $K$ memory tokens + decoder blocks + verifier head.
  Frozen = the base model (the task forward where $M$ is injected, the base-readable behaviour signals, and the
  full-context teacher).
- **Conditioning IS the query-mask, not a residual module:** $M_0$ and $M_q$ are the SAME encoder run with the
  query masked vs unmasked (this supersedes the $M_q = M_0 + \alpha\cdot\mathrm{Refine}$ residual sketched in
  §1). The min-deviation $\lVert M_q - M_0\rVert$ remains as a LOSS term, not as an architectural residual.
- **Window = the model's context window.** The encoder is base-architecture first-$N$ layers, so it inherits
  the base window and the full-context teacher always runs (resolves the long-context-oracle question).
- **Encoder is a separate trainable COPY of the base's first $N$ layers** (init from base weights; $N$ is an
  ablation knob), NOT an adapter toggled on the shared base. The frozen base is a distinct model used for the
  task / signals / teacher.
- **The query-mask is encoder-internal only.** It only decides whether the ENCODER attends to the query while
  compressing (masked = $M_0$, unmasked = $M_q$). The **frozen base always receives the query** at answer time
  ($[\,M\,;q\,]$, $M$ replacing the context) in BOTH modes; conditioning changes WHAT was compressed, not
  whether the base sees the query.

**Gate signal (inference) = an empirical search.** Run $M$ through the frozen base once and read the
base-readable signals (conf / margin / entropy / kl_w0 / first_tok_agree) + query-coverage + the
$\Delta_{\text{code}}/\Delta_{\text{logit}}$ gaps. WHICH signals + HOW is determined empirically:
**(1) correlation analysis** (which signals correlate with do-no-harm / usefulness) ->
**(2) actual inference + training** -> **(3) finalise the signal set.** This signal selection IS the tuning
that makes the main table work.
- **mem-embedding's signal code is NOT directly reusable** (different architecture + training method now). We
  must **re-implement / reproduce** the signal menu and the correlation analysis for THIS architecture, and
  check whether the old findings reproduce.
- Although the signal is computed at inference, an **auxiliary verifiability regularizer (a small-coefficient
  term, $\lambda_{\text{sep}}$, not a main loss)** shapes the compressor at training so the signals become
  predictive (see §11 for the adversarial variant).

## 10. Locked decisions (2026-06-10)
- **Schedule:** implement all three (staged / joint / alternating) and **ablate on ONE dataset** first.
- **Verifiability regularizer:** an auxiliary, small-coefficient term ($\lambda_{\text{sep}}$) whose gradient
  flows INTO the compressor + monitoring (co-design, bounded gaming). "Auxiliary / small-coefficient", not a
  main loss (renamed from the confusing "weak weight").
- **Gate signal correlation code:** re-implement / reproduce for this architecture (do NOT reuse mem-embedding's).
- **Long-context oracle:** keep $|C| \le$ model window; full-context teacher always fits (use the base directly).
- **Compressor = a trainable autoencoder:** trainable encoder (init from the base's first $N$ blocks) + $K$
  memory tokens + a **trainable decoder** (reconstruction = lossless loss). Only the base (task / signals /
  teacher) is frozen. Conditioning = query-mask. The Perceiver pooler stays as an encoder-init ablation.
- **Gate signal = empirical:** correlation analysis first, then inference + training, then finalise the signal
  set; this doubles as the main-table tuning.

### Remaining construction knobs (defaults, ablate later)
- $N$ (how many base blocks): default small (e.g. 4-8 / about a quarter), ablate.
- Encoder attention over $[C;q]$: bidirectional within $C$ (it is an encoder) with memory attending back;
  query-maskable. (vs keep the base's causal mask.)
- Reconstruction: frozen-base CE on "repeat the context" (vs a small separate decoder).
  (Note: per §9 the decoder is TRAINABLE, so reconstruction CE is through the trainable decoder, not the base.)

## 11. Gate training as OOD detection: adversarial / conditional-GAN options (BRAINSTORM, under discussion)
The gate is fundamentally an **OOD / competence detector**: fire (fall back) when the compression is outside the
region where it behaves like full context. Can we train it adversarially (the conditional-GAN idea)? Yes, with
one crucial constraint that keeps it honest.

**Headline mapping: the gate is a conditional discriminator in an adversarial-distillation game.**
- For $(C,q)$ there are two behaviours: the **full-context behaviour $b_{\text{full}}$** (oracle, FIXED) and the
  **compressed behaviour $b_M$** (from $M$). "Behaviour" = answer-span logits / hidden / the base-readable signals.
- **Discriminator $D(b,q)$** (conditioned on $q$, and optionally domain = the conditional-GAN condition): real =
  $b_{\text{full}}$, fake = $b_M$.
- **Compressor (generator)** is trained to make $b_M$ indistinguishable from $b_{\text{full}}$ (fool $D$) ->
  this drives $M$ toward **behavioural losslessness** (a GAN form of the KL-distillation).
- **$D$'s score on $b_M$ = how distinguishable from full = the unsafe / gate signal.** Distinguishable ->
  unsafe -> fall back.
- **Why it is NOT gameable** (this is the key, and it resolves the earlier gaming worry): the "real" class is
  the TRUE full-context behaviour, an external anchor. The only way the compressor can fool $D$ is to GENUINELY
  match full. $D$ cannot be bribed into a free pass, so it stays an honest do-no-harm detector. Contrast a
  naive cGAN where the generator fools the discriminator into uselessness; here the anchor prevents that.

**Conditional vs unconditional** maps to the cGAN condition: $D(\cdot,q)$ judges "is $M_q$ sufficient for THIS
$q$"; an unconditional $D$ judges "is $M_0$ globally lossless".

**Three concrete adversarial roles (can combine):**
1. **Adversarial distillation (D on behaviour):** the headline above. $D$ replaces/augments the fixed KL with a
   learned discrepancy that can capture richer behavioural mismatch than answer-token KL. $D$ = the gate.
2. **Adversarial autoencoder on the code $M$ (anomaly detection):** a discriminator regularizes $M$'s
   distribution toward a prior / the in-distribution code manifold, so OOD contexts produce **off-manifold
   codes** that are detectable (sharpens $\Delta_{\text{code}}$ / geometry). This is the GANomaly / adversarial-
   autoencoder lineage; pair with the recon-AE-OOD lesson (combine reconstruction error with a latent
   Mahalanobis distance, since reconstruction error alone is a weak OOD signal).
3. **Conditional hard-case adversary (OOD coverage):** a generator produces hard / OOD $(C,q)$ pairs
   (conditioned on domain) that push the compressor to its failure boundary; $D$ trains on these to extend
   **cross-domain** do-no-harm coverage (the gate must hold out of domain). This is adversarial hard-negative
   mining, NOT generator-fooling.

**Practical / caveats:**
- Keep the **supervised anchors stable**: KL-distill to full + BCE do-no-harm on the oracle label form the
  backbone; the adversarial term is the **auxiliary small-coefficient regularizer** (the renamed verifiability
  term). Adversarial mainly buys a sharper decision boundary + OOD coverage.
- GAN instability, mode collapse, and off-manifold adversarial examples (constrain the hard-case adversary to
  realistic contexts so the gate does not over-fire on unrealistic inputs).
- Plan: prove the supervised gate first (correlation analysis -> BCE gate), then test whether the adversarial
  variant improves the compress-decision F1 / OOD generalization (an ablation, not the default).

### 11.1 Discriminator input = the base's internal hidden at layer(s) $\ell$ after M-insertion (2026-06-10)
Refinement of the discriminator. Instead of reading only the output logits, $D$ reads the FROZEN base's hidden
state at layer $\ell$ (or several layers) at the query / answer positions, after inserting $M$:
$D\big(h^{(\ell)}_{\text{query-pos}}(M),\,q\big)$, with **real = $h^{(\ell)}$ from the full-context run** (same
layers, same query positions) and **fake = the $M$-induced $h^{(\ell)}$**.

Why this is stronger than an output-only discriminator:
1. **No full-context run at deploy.** $D$ internalises the full-context hidden manifold during training; at
   inference we only run $[M;q]$ to layer $\ell$ and score $D(h^{(\ell)}_M)$. This is the point of a learned
   discriminator over an explicit $\mathrm{KL}(\cdot\|\text{full})$, which would need the full run we are avoiding.
2. **Early-exit = cheap gate.** Shallow $\ell$ lets us decide before finishing the forward; depth $\ell$ trades
   discriminability vs cost. Multi-layer = a depth profile; pick the cheapest sufficiently-discriminative layer
   (a clean ablation). Connects to the E1 early-detection idea.
3. **Catches internal divergence** the output misses (M corrupted the representation but greedy answer happens
   to match).
4. **Anchoring to FULL (not $M_0$) sidesteps the "both fail" caveat for Track B:** if $M$ and full both err
   (hard task), their behaviour is close so $D$ says safe (M is no worse than full = correct); only "M wrong,
   full right" makes $h_M$ deviate from full and trips $D$.

Design choices: which layer(s) (start mid-layer, then sweep to a multi-layer profile); which positions
(query / answer positions, especially the last query token = the decision state; pool over query positions;
NOT ctx positions, which differ in length between full and $M$); single multi-layer $D$ (concat) vs per-layer
$D$ (feature-matching style, more stable). This is the LEARNED version of mem-embedding's per-layer signals
(logit-lens, drift_l); the reproduced correlation analysis should include per-layer hidden features and $D$ is
their learned aggregator.

### 11.2 Gate-signal taxonomy: three kinds by training method, four combinations (2026-06-10)
The gate signal is not one thing; it has **three kinds, distinguished by how they are produced/trained**:
- **K1 Oracle-supervised output ("directly output"):** a head trained on the oracle do-no-harm label
  (BCE / regression) that directly outputs the gate score. Training = supervised on the oracle.
- **K2 Untrained behavioural (correlation-selected, "input-agnostic"):** hand-read behavioural statistics of
  the M-run (conf / margin / entropy / kl_w0 / first_tok_agree + recon-coverage + query-coverage + geometry +
  $\Delta_{\text{code}}/\Delta_{\text{logit}}$). NO training; selected / combined purely by CORRELATION with the
  do-no-harm label. Because it needs no training it is ORTHOGONAL and can be stacked on K1 or K3.
- **K3 Adversarial discriminator ("the discriminator we train"):** the cGAN-style $D$ on the base's layer-$\ell$
  hidden (real = full, fake = M). Training = adversarial.

The three have **different training methods** (supervised / none / adversarial).

**Four combinations** = the two trained kinds $\times$ the orthogonal untrained K2 (on/off):
1. **K1** (supervised head alone)
2. **K1 + K2** (supervised + behavioural stats)
3. **K3** (adversarial discriminator alone)
4. **K3 + K2** (adversarial + behavioural stats)
(K2 alone = the untrained, correlation-only gate = the cheapest reference point.)

**Plan:** run the **correlation analysis on ALL candidate signals** (the K2 menu + K1's output + K3's score)
vs the do-no-harm label FIRST, see what predicts, then the fine-grained ablation over the four combinations.
The chosen kind + combination is the main-table tuning. (K2 is computable first since it needs no training and
also supplies the feature pool for K1 / K3.)

## 12. Anticipated pitfalls + pre-design (2026-06-10, before coding)
Grouped; **bold = could break the design if missed.** Mitigations are what we pre-design around.

### Architecture / feasibility
- **P1 Encoder = copy of first-N HF layers is version-fragile** (RoPE, position_ids, attention-mask format,
  cache API changed across transformers). Hand-rolling `layer.forward` will bite. **Pre-design:** build the
  encoder as a REAL small HF model (a config with `num_hidden_layers=N`, copy `model.layers[:N]` + `embed_tokens`
  + rotary), call it via the standard forward with a custom 4D mask. Smoke this layer-copy + a tiny forward on
  ray-test FIRST, before any training code.
- **P2 $M$ is layer-$N$ hidden but is injected as a layer-0 input embedding** into the frozen base (space /
  semantics mismatch even though dims match). **Pre-design:** add a learned projection $M\to$ input-embedding
  space; rely on the task distillation to align $M$ to what the base reads; use ONE shared $M$ for recon + task
  + signals.
- **P3 (CRITICAL) RoPE position confound for the discriminator AND the task forward.** The query sits at
  different absolute positions in full ($[C;q]$, pos $|C|..$) vs $M$ ($[M;q]$, pos $K..$); RoPE rotates the
  query hidden by position, so $D$ could learn "short vs long prefix" instead of "M vs full", and the frozen
  base reads the $M$-prefixed query at the wrong positions. **Pre-design:** remap the $M$-run query `position_ids`
  to START at $|C|$ (the full-run positions), so the query sits at the SAME absolute positions in both runs.
- **P4 Reconstruction (K tokens -> full ctx) is the hard part**; the trainable decoder may floor high and
  pollute the "lossless" signal. **Pre-design:** autoregressive "repeat the context" CE; size the decoder;
  treat recon as a coverage signal, not a hard zero-target; consider per-chunk.
- **P13 Bidirectional attention on causal-trained copied layers is off-distribution.** **Pre-design:** keep the
  encoder CAUSAL (layout $[C;q;\text{mem}]$, causal; unconditional = additionally mask mem->query). Simpler and
  in-distribution. (Overrides the "bidirectional" knob in §10.)

### Training
- **P5 (CRITICAL, = R2) Shared encoder vs "freeze $M_0$" staging conflict** (see §7 fix): $M_0,M_q$ share the
  encoder, so staging = "add $\mathcal{L}_{\text{cond}}$ while keeping $\mathcal{L}_{\text{uncond}}$ on"; the
  min-deviation LOSS (not freezing) keeps $M_q\approx M_0$.
- **P6 Teacher cost:** distill + the discriminator need the full-context run per $(C,q)$. **Pre-design:**
  precompute + cache the teacher's top-k logits AND the layer-$\ell$ hidden at query positions once.
- **P7 Memory:** trainable N-layer encoder + trainable decoder + frozen full base + Adam states + 2 encoder
  passes ($M_0,M_q$). **Pre-design:** small $N$, gradient checkpointing, cache teacher, a smaller base for dev,
  modest $K$.
- **P14 Trivial-solution risk** (autoencoder encodes positions/ids not content). **Pre-design:** monitor recon
  vs task, diverse data, check $M$ generalizes across queries.
- **P8 Loss balancing** (recon vs distill vs min-dev vs gate can fight). **Pre-design:** per-loss monitoring;
  tune $\lambda_{\min}$ (too high -> conditioning never fires); class-balance the gate BCE.

### Gate / discriminator
- **P9 $\Delta_{\text{logit}}$ may be redundant given the layer-$\ell$ discriminator.** $D(h^{(\ell)}_M)$ already
  scores "unlike full" without a full run; do not over-build parallel signals. **Pre-design:** make the
  layer-$\ell$ feature extractor the PRIMARY gate input; $\Delta_{\text{code}}/\Delta_{\text{logit}}$ are
  auxiliary/analysis.
- **P15 Deploy must not need full-context features.** **Pre-design:** the gate's deploy feature vector =
  functions of the $M$-run only; full-context features are train-time labels/anchors only. Audit the pipeline.
- **P8b Gate degeneracy** (always-fallback / always-compress). **Pre-design:** report use-rate / help-rate /
  precision-recall; the F1 metric catches it.

### Infra / eval
- **P10 signal correlation must be re-implemented** for this arch (not reuse mem-embedding); include per-layer
  hidden features. **Pre-design:** fresh corr script in `svc`.
- **P11 Eval needs the decision-metric reporter + the Cartridge/Gist baselines**, currently in
  `mem-embedding/gcm`. **Pre-design:** port / re-implement the baselines + the confusion-matrix/F1/cost-coverage
  reporter into `svc`.
- **P12 (BLOCKER) Pod env + base availability on ray-test:** transformers version (4D mask, layer copy,
  output_hidden_states), and the base weights path (the session hit "model not found" + version breaks).
  **Pre-design:** smoke the layer-copy + mask + tiny forward + base load on ray-test BEFORE building the
  pipeline; pin the transformers version.
- **P16 Reuse the feature extractor across the supervised gate and the later adversarial $D$** (same
  layer-$\ell$ features) so the GAN phase plugs in without rework.
