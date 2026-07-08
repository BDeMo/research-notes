# Paper B Summary Matrix v1.7 (writing-agent handoff)   ⛔️ SUPERSEDED

> ⛔️ **SUPERSEDED (2026-06-13) — DO NOT CITE THESE NUMBERS.** v1.7 had a **train/eval leakage bug** (≈71% of
> BFCL eval items seen in training) that inflated every tool result. The clean redo + the reframed thesis
> (**a compressor-agnostic robustness layer**, not a better compressor) live in
> **[`summary-matrix-v1.7.3.md`](summary-matrix-v1.7.3.md)** (one-page brief) and
> **[`results-v1.7.3/results-v1.7.3.md`](results-v1.7.3/results-v1.7.3.md)** (full results). Kept only as a record
> of the (now-retracted) v1.7 brief.

> Purpose: single-source brief for the writing agent, **reset for the v1.7 "gated compressor module"
> reframe**. Read the design first: `v1.7-gated-compressor-module-2026-06-10.md`.
> The v1.5/v1.6 ledger is archived in [`summary-matrix.md`](summary-matrix.md); still-valid prior evidence is
> carried in §7 below, flagged to re-validate under the v1.7 protocol.
>
> Base model for all numbers = Qwen3-8B unless noted. **HONESTY FIRST: every result cell below is PENDING
> until actually run. Do not invent numbers.** House rule: no em-dashes.
> Status: v1.7 just reframed (2026-06-10); baseline code rebuilt; the gated module + the metric reporter are
> not implemented yet.

---

## 0. One-paragraph thesis (v1.7)
We propose a **gated compressor module (GCM)**: a context compressor on a **frozen** base that emits both a
compressed latent memory $M$ ($K$ tokens) and a **calibrated reliability score $g$ computed from its own
internal structure** (reconstruction coverage + query coverage + memory geometry), and at inference
**detects-and-falls-back** to full context when compression is unsafe. Because the base is untouched and the
injection vanishes when $g\to0$, the worst case is the exact base (do-no-harm by construction). We evaluate it
as a **compress / fall-back decision** and compare it head to head with the **original** Cartridge and Gist
compressors at the compressor level.

---

## 1. Positioning (v1.7)
- **Headline object:** one **gated compressor module** (not a setting axis, not a generic gate bolted onto
  others). The gate is intrinsic to our compressor; baselines stay vanilla.
- **Mechanism we sell:** robust context compression for agentic tool-use via **detect-and-fall-back**.
- **Scope:** adaptive latent memory for LLM agents (fast = few latent tokens, safe = do-no-harm gate,
  accurate when needed = falls back to full context). Venue fit = agentic memory.
- **Honest novelty:** the compressor mechanisms are not ours (cite Gist, Cartridges, ICAE). Ours = a compressor
  that **predicts its own reliability from its structure and falls back when unsafe**, with a do-no-harm floor
  by construction on a frozen base. Gate+fallback exists at text-level (TAAC) and reasoning-level (SLT/SeLaR);
  our white space = a **learned latent** memory + the structural intrinsic signal + the systematic treatment.

---

## 2. Evaluation objective (the headline metric): the compress-decision confusion matrix
Fall-back floors accuracy, so we optimize the **quality of the compress / fall-back decision**. Gate = binary
classifier, positive action = "compress" (use $M$). Label $y=\mathbb{1}[n_w\ge n_{\text{full}}-\varepsilon]$
(Track B; Track A uses $n_w\ge n_0$).

| | compression acceptable $y{=}1$ | compression fails $y{=}0$ |
|---|---|---|
| decide compress | TP (cheap+correct) | **FP = the harm** |
| decide fall back | FN (compute waste) | TN (harm avoided) |

- **precision** = TP/(TP+FP) = compress accurately = do-no-harm. **recall** = TP/(TP+FN) = compress as much as
  possible. **F1** = the judgment number (also the GCM gate's training target).
- **Asymmetry:** FP costs accuracy, FN costs only compute -> operate at **max recall s.t. precision $\ge p^\*$**
  (or $F_\beta,\ \beta<1$). Full picture = **cost-coverage / risk-coverage frontier** (sweep $\tau$); AURC = curve summary.
- **Anchors per bench:** base rate $P(y{=}1)$ (compressibility); always-compress; always-fall-back; **best(compress,full)** (the perfect-gate ceiling).
- **best(compress, full) (precise; renamed from the confusing "oracle").** The accuracy of a PERFECT per-input gate (precision = recall = 1).
  With per-input scores $n_0,n_w,n_{\text{full}}$ (no-context / compressed / full-context) and the do-no-harm
  label $y_i=\mathbb{1}[\,n_w^i \ge n_{\text{full}}^i-\varepsilon\,]$ (Track B):
  $$\mathrm{oracle}=\frac{1}{N}\sum_i \big(y_i\, n_w^i + (1-y_i)\, n_{\text{full}}^i\big),$$
  i.e. use the compressed answer exactly where compression does no harm, else re-read full context. At
  $\varepsilon{=}0$ this equals $\frac1N\sum_i \max(n_w^i, n_{\text{full}}^i)$ (per-input best of {compress, full});
  Track A swaps $n_{\text{full}}\!\to\!n_0$ (fallback = no-context base). It USES gold (so it is a ceiling we
  cannot deploy), it is always $\ge$ always-compress and $\ge$ always-fall-back, and our learned gate's accuracy
  lies between always-compress and best(compress,full); the gap to it is exactly the gate's error. Code:
  `svc/decision.py::report_block` (`acc_best`).

---

## 3. MAIN COMPARISON (compressor-level; all cells PENDING)
Same frozen base, same $K$, same training data + split, same benches. Rows are methods, not a setting axis.
**Baselines are vanilla (no gate).** OURS applies detect-and-fall-back.

**3a. Accuracy panel (per bench): n_0 / n_w / n_full**
| row | what | gate? | result |
|---|---|---|---|
| no-ctx | bare base | floor | PENDING |
| Cartridge (Eyuboglu+ 2025, ORIGINAL) | trained KV-cache via self-study + context-distillation; base frozen (THEIR original design); always used | none | PENDING |
| Gist (Mu+ 2023, ORIGINAL) | gist tokens + gist mask + **fine-tuned base** (instruction-tuning, base NOT frozen); always used | none | PENDING |
| **OURS (GCM)** | gated compressor + fall-back | intrinsic | PENDING |
| full-ctx | full context in prompt | ceiling | PENDING |

> **Preliminary INTERIM relation-split (2026-06-10, `out/grid1`, N=384 = 4 benches, Qwen3-8B, track B, NON-strict
> baselines, trivia-trained):** acc[no_ctx/comp/full/best], cost[comp/full]:
> - **in_task** (trivia): gist 0.171/**0.309**/0.215/0.425 (102/1017); cartridge 0.171/0.199/0.215/0.254
> - **cross_task_in_domain** (hotpot,squad): gist 0.125/**0.118**/0.282/0.338 (98/468); cartridge 0.125/**0.122**/0.282/0.304
> - **cross_task_cross_domain** (quality): gist 0.271/0.219/0.146/0.250 (152/1112); cartridge 0.271/0.188/0.146
>
> **Three findings = the v1.7 story, with data:** (1) **negative transfer is real**: out-of-task both compressors
> fall BELOW no-ctx (gist 0.118<0.125; cartridge 0.122<0.125) -- the do-harm the gate must catch. (2) **in-task
> compression can beat full** (gist 0.309 > full 0.215) at ~10x lower cost. (3) **large best(compress,full)
> headroom** everywhere (in_task 0.425 vs comp 0.309; overall best 0.338 vs full 0.231) -> a good gate has much
> to gain. These are INTERIM baselines (NOT strict Cartridge-KV / Gist-LoRA); to be replaced (Q2). Reporter:
> `svc/decision.py`. **Ceiling-sanity flag:** `quality` full-ctx 0.146 < no-ctx 0.271 (base_rate 0.97) -> that MC
> bench's full-context is broken/noisy; fix before trusting cross-domain (see `critical-review-v1.7` flaw 5).
>
> **GCM development log (method renamed OURS -> GCM; class `GCM`, method name `gcm`):**
> - **v1** (L_cond distill, bf16, N=4): 0.174 < no-ctx 0.196 -- collapsed (no reconstruction anchor + bf16 KL
>   spiked to 3.2).
> - **v2** (added trainable decoder + reconstruction $\mathcal{L}_{\text{uncond}}$ + min-dev, fp32): fixed
>   stability (cond~0.1, rec 44->0.9 = M0 reconstructs context) but still 0.162 < no-ctx in_task -- the L_cond
>   distilled a WEAK full-ctx teacher over a teacher-forced response (too easy).
> - **v3** (gold-answer CE as the task loss, like Gist; teacher-distill off): task CE 8->0.087, but eval still
>   loses. Full relation split (`out/oursv3`, N=384): overall no_ctx/comp/full/best = 0.173/0.159/0.232/0.304;
>   **in_task** 0.179/**0.164**/0.216/0.263; **cross_task_in_domain** 0.125/**0.095**/0.283/0.310 (strong negative
>   transfer); **cross_task_cross_domain (quality MC)** 0.260/**0.281**/0.146/0.333 (only here comp>no_ctx, but
>   that bench's full-ctx ceiling is broken).
>
> **Diagnosis: N=4 capacity ceiling, NOT a training bug.** Gist's K tokens are computed by the FULL frozen base
> (~36 layers) so its memory is deep + base-native (in_task 0.309); GCM's M comes from a 4-layer encoder copy
> injected as input embeddings -- far shallower. The training machinery works (gold CE -> 0, reconstruction ->
> ~1.0, fp32 stable); the gap is compressor capacity.
>
> **CAPACITY SWEEP RESULT (2026-06-10, in_task trivia) -- breaks the N=4 ceiling, GCM WINS:**
> no_ctx/comp/full/best, cost[comp/full]:
> - N=4 (v3): 0.179/0.164/0.216/0.263 (102/1017) -- loses
> - N=8 (fp32, partial n=70): 0.179/0.176/0.217/0.277 -- ~tied no_ctx (transitional)
> - **N=16 (bf16, n=96): 0.177/0.231/0.218/0.304 (102/1017)** -- **GCM 0.231 > full 0.218 > no_ctx 0.177, at ~10x
>   lower cost.** The headline: compression beats FULL context, much cheaper.
> - K=128 (N=4, fp32, partial n=93): 0.180/0.203/0.214/0.277 (166/1016) -- width also clears no_ctx, cheaper than depth
>
> So the v1/v2/v3 failures were purely CAPACITY (N=4 too shallow), not the training (gold-CE/L_uncond/fp32 all
> fine). Notes: the winning N=16 was bf16 whose teacher-forced `task` loss looked underfit (4.4) -> that metric
> does NOT predict eval; generation does.
>
> **CAPACITY x RELATION (2026-06-10, trivia-trained, Qwen3-8B) -- the gate motivation in one table.** GCM/full/no_ctx:
> - **in_task** (trivia): N8 0.197 / N12 0.208 / N16 0.231 / **N24 0.247** vs full 0.218, no_ctx 0.177
>   -> GCM compression BEATS full, scales with N, at ~10x lower cost (102 vs 1017 tok).
> - **cross_task_in_domain** (hotpot,squad): GCM ~0.10-0.14 (all N) vs **full 0.284**, no_ctx 0.126
>   -> compression COLLAPSES to ~no_ctx out-of-task; full is far better -> gate must FALL BACK TO FULL.
> - **cross_task_cross_domain** (quality MC): GCM ~0.21-0.23 vs full 0.146 (broken), **no_ctx 0.260**
>   -> no_ctx wins -> gate must NOT COMPRESS. (quality ceiling broken, see T20.)
>
> So: compression is great in-task, harmful/useless out-of-task -> you NEED the gate. `best(compress,full)`
> headroom out-of-task (~0.32 vs GCM 0.13) is exactly what a good gate captures.
>
> **ROUND-2 (2026-06-11):**
> - **Depth has an optimum ~N=24, not monotone forever:** in_task trivia N16 0.231 -> N24 **0.247** -> N32 0.208
>   (N32 degrades; over-parameterized/unstable on 96 items). N=24 is the trivia sweet spot.
> - **Autoencoder ablation (justifies the design):** removing $\mathcal{L}_{\text{uncond}}$+min-dev (lam_rec=lam_dev=0,
>   `abl_nae`) drops in_task 0.231 -> 0.195. The reconstruction + min-dev contributes ~+0.036.
> - **The "GCM beats full" win is TASK-DEPENDENT (be honest):** in_task GCM/full/no_ctx by anchor --
>   trivia 0.247/0.218 (WIN) · hotpot 0.148/0.201 (>no_ctx, <full) · squad 0.114/0.367 (LOSES; extractive span,
>   K=64 too lossy) · quality-MC 0.219/0.146-but-no_ctx-0.260 (no_ctx wins). So compression helps most on
>   parametric-ish QA (trivia), least on extractive/long (squad/hotpot). This makes the PER-ITEM GATE more
>   important, not less: compressibility varies by item/task, so detect-and-fallback is the point.
> - **Gate CODED + RUNNING:** `GCM.signals()` logs verifiability (neg_recon, dcode, dlogit) + base-uncertainty
>   (conf/margin/neg_entropy) + geometry (mnorm); harness `--signals` stores them per item; `decision.py
>   --signal-key K` already scores AUROC/F1/cost-coverage. Runs `gate_N16fp32/N16bf16/N24` (with signals) +
>   `abl_K256`/`abl_dec4` in flight on all 5 GPUs. Gate AUROC/F1 analysis pending their completion.
>
> **SWEEP RESULTS (2026-06-11, 100-job grid, best hyperparams per method x dataset; comp=compressed,
> full=fallback, best=gate-ceiling). CSV `out/sweep_summary.csv`.**
> - **GCM (ours)** best = N24/K64 (quality: N12/K128):
>   - trivia in_task comp/full/no_ctx/best = **0.293/0.218/0.177/0.372** (GCM>full); cross_task_in_domain
>     0.142/0.284 (neg transfer); cross_domain 0.281/0.146. **gate neg_entropy AUROC 0.718 F1 0.778.**
>   - hotpot in_task 0.165/0.200 (<full); squad 0.136/0.367 (extractive, loses badly); quality 0.260/0.146(=no_ctx).
>     gate: hotpot neg_dcode AUROC 0.683 F1 0.801; squad neg_recon |AUROC 0.327| (verifiability fires).
> - **Gist (interim) = strongest compressor in-task** (full-base memory): trivia comp **0.370** > full 0.215 >
>   GCM 0.293; hotpot 0.260>0.198; squad 0.157<0.368; quality 0.312. best=s1500/K64-128.
> - **Cartridge (interim) weakest**: trivia in_task 0.205 < full 0.218.
> - **All compressors collapse cross_task_in_domain** (comp ~0.12-0.19 << full ~0.28) -> the gate is needed by all.
>
> **HONEST TAKEAWAY (reshapes the claim):** Gist's memory is computed by the FULL frozen base (~36 layers) so it
> out-compresses GCM's N<=24 encoder in-task. GCM's EDGE is NOT raw compression -- it is the **intrinsic GATE**
> (verifiability $\Delta$recon/$\Delta$code + uncertainty -> AUROC 0.68-0.72 / F1 0.78-0.80) that vanilla
> Gist/Cartridge lack. So the v1.7 headline must be **decision-level (gated do-no-harm / F1 / cost-coverage)**,
> NOT "we compress better". Open: can we bolt base-readable signals onto Gist as a gated-baseline (fair gate
> comparison)? GCM-specific signals (recon/dcode) must beat that to justify the architecture.
>
> **PHASE-2 deployed (2026-06-11, 27 jobs / 5 GPUs, ~4-5h): completes the main table + ablations.**
> - **MAIN GRID** (`out/phase2/main_*`): best config/method (GCM N24/K64; Gist/Cart s1500/lr1e-4/K64) x 6 domain
>   anchors (trivia,hotpot=wiki; quality,narrativeqa=literary; ms_marco=web; musr_mm=synthetic) x eval ALL 7
>   loadable benches -> full in/cross-task/cross-domain x compressed/fallback panel over 4 domains.
> - **ABLATIONS (GCM)** (`out/phase2/abl_*`): agnostic(M0) vs conditioned(Mq) [`--gcm-agnostic`]; encoder init
>   copy vs random [`--enc-init`]; compression-ratio / capacity wall (n_chunks 2/4/8/12 at fixed K); width K 32/256.
>   (N-depth, fp32-vs-bf16, no-AE/loss-component, decoder-depth already covered by the sweep.)
> - **Offline-bench constraint:** tool (bfcl/apibank/toolace) + ops (rca) are NOT in either pod's HF cache ->
>   excluded; the table spans wiki/literary/web/synthetic. ruler_niah loads on ray only -> excluded for
>   cross-pod consistency. (Fetching tool/ops needs network.)
> - Reporter: `python svc/sweep_report.py --root out/phase2` (same selector/CSV).

**3b. Decision panel (OURS gate, the headline): per bench**
`base_rate P(y=1)` · `precision@p*` · `recall@p*` · `F1` · `coverage (compress rate)` · `mean token cost` ·
`AURC` · vs `oracle` / `always-compress` / `base-uncertainty (TARG) gate`. All PENDING.

**3c. Transfer split (relation taxonomy, already wired into the loaders):**
`in_task` (train==eval bench) · `cross_task_in_domain` (same domain, diff bench) · `cross_task_cross_domain`
(diff domain). Claim to test: gate closes (coverage drops, precision holds) as relation moves outward, so
cross-domain accuracy is preserved. All PENDING.

**Claims the table must make (once filled):** (1) at compressed budget OURS matches the best vanilla compressor
on accuracy AND never below fall-back when the gate fires; vanilla Cartridge/Gist show negative transfer with
no recourse. (2) The **gate ablation** (OURS minus gate) reproduces the baselines' negative transfer = the gate
is what makes the compressor safe. (3) The transfer split shows do-no-harm holds out of domain.

---

## 3d. Experiment queue (v1.7 GCM, status-tagged) -- TALLY: DONE 9 - RUNNING 2 - TODO 20 (of ~31)
> Pods: `sam-dev-test` (1x H200-144G, tf 4.56) + `sam-dev-ray` (4x H100-80G, tf 5.10) -- both work; ns aird-ray-dev,
> kubeconfig `~/.kube/aird-ray-dev.yaml`; code `/mnt/persist/v17`; base `/mnt/persist/checkpoints/Qwen3-8B`; env
> `llm`, `PYTHONPATH=llm-infra/src:mem-embedding/src:svc/src`, `HF_HUB_OFFLINE=1`. Harness `mem_embedding.gcm.harness`;
> reporter `svc/decision.py --compressor gcm`. fp32 AdamW memory wall: H100 fp32 up to ~N=8-12; N>=16 fp32 needs H200.

### DONE (9)
- D1. Code: GCM training machinery -- encoder=copy first-N base blocks + K mem + query-mask M0/Mq; trainable
  decoder reconstruction ($\mathcal{L}_{\text{uncond}}$); min-dev; gold-answer CE task loss; fp32 master copy.
- D2. Code: harness + decision reporter (confusion/P/R/F1/AUROC/AURC/best) + 3-way relation taxonomy.
- D3. Interim-baseline grid (`out/grid1`): no_ctx/full/Cartridge-interim/Gist-interim x 4 benches x relation.
- D4. Capacity curve in_task: N in {4,8,16,24}, fp32 & bf16, + K=128, + N8+K128. -> N scales monotone.
- D5. Capacity x relation split (`out/rel_N{8,12,16bf16,16fp32,24}`): in/cross-task/cross-domain. **KEY RESULT**
  (see GCM dev-log): in_task GCM>full (N24 0.247), cross-task GCM~no_ctx<<full -> motivates the gate.
- D6. fp32 vs bf16 @ N=16 (0.233 vs 0.231 -- ~equal).
- D7. Rename OURS -> GCM (class `GCM`, method `gcm`).
- D8. LaTeX `main_v1.7.tex` pushed (commit 689f568) + critical review (`critical-review-v1.7`).
- D9. GCM-v1/v2/v3 development (collapse -> stability -> gold-CE) documented.

### RUNNING (2 batches, all 5 GPUs)
- R1. Transfer matrix: train {hotpot, squad, quality, bfcl} x eval 4-bench relation (N=16 bf16) -- `out/grid_*`.
- R2. Capacity frontier N=32 (trivia, bf16) -- `out/rel_N32`.

### TODO (20) -- critical path = baselines + gate
**Baselines (strict-original):**
- T1. Cartridge strict: trainable KV-cache + self-study + context-distillation (base frozen). [code+run]
- T2. Gist strict: gist tokens + gist mask + per-training-set LoRA-FT. [code+run]
**Main grid (headline table):**
- T3. {no_ctx, full, Cartridge-strict, Gist-strict, GCM-best(N=16/24)} x anchors x relation, scaled (n~300, steps~1500).
- T4. Multi train-corpus full transfer matrix (extends R1 to all anchors + GCM-best N).
- T5. Cross-model: GLM4-9B / Qwen2.5-7B / Mistral-7B-Instruct (cached). do-no-harm + F1 generality.
- T6. CIs: multi-seed on the main grid.
**The GATE (headline contribution):**
- T7. Signal extraction: $\Delta_{\text{code}}/\Delta_{\text{logit}}$ (M0<->Mq) + recon-cov + query-cov + geometry +
  base-readable conf/margin/entropy + layer-$\ell$ hidden. [code]
- T8. Correlation analysis: which signals predict the compress/fallback do-no-harm label.
- T9. Supervised gate (BCE) -> F1/AUROC/cost-coverage; the 4 signal-family combos (K1/K1+K2/K3/K3+K2).
- T10. Gate vs references: base-uncertainty (TARG), always-compress, always-fallback, best(compress,full) ceiling.
- T11. Gate ablation: GCM-minus-gate reproduces negative transfer (keystone -- the gate is what makes it safe).
- T12. Adversarial / cGAN gate: internal-layer discriminator (M-induced vs full-ctx hidden). [later]
**Ablations:**
- T13. init: first-N copy vs random/Perceiver pooler.
- T14. agnostic (M0) vs conditioned (Mq) at eval. [needs harness flag]
- T15. loss components: $\mathcal{L}_{\text{uncond}}$ on/off, min-dev on/off, gold-CE vs distill (lam_* sweeps).
- T16. decoder depth n_dec_layers in {1,2,4}.
- T17. K (memory width) sweep {16,32,64,128,256}; deeper N {32,36}.
- T18. compression-ratio sweep (ctx length at fixed K) -> capacity wall + conditioning coincide->diverge.
- T19. schedule: staged / joint / alternating.
**Sanity:**
- T20. Ceiling sanity: `quality` MC full-ctx 0.146 < no_ctx -> fix the MC full-context eval before trusting cross-domain.

---

## 4. The 4-step program (status board)
| step | goal | metric | status |
|---|---|---|---|
| **1. Define + measure** | confusion-matrix reporter over logged $n_0/n_w/n_{\text{full}}$ + features; oracle + trivial baselines | F1, AURC, oracle ceiling | ⬜ not started (no model change; **recommended start**) |
| **2. Precision first** | threshold for precision $\ge p^\*$ (do-no-harm) | recall@$p^\*$ | ⬜ |
| **3. Recall at fixed precision** | fused structural gate (recon + query + geometry) + calibrated head; ablation geometry < +query < +recon | F1 up, AURC down at $p^\*$ | ⬜ (needs GCM module) |
| **4. Raise base rate** | better compression (distillation/capacity) so $P(y{=}1)$ rises | frontier shift | ⬜ |

---

## 5. Contributions (v1.7)
- **C1 Gated compressor module:** a frozen-base compressor that emits memory + a structural reliability score
  and falls back; do-no-harm by construction. (status: design under review)
- **C2 Decision-quality objective:** frame robust compression as a compress/fall-back confusion matrix with
  precision/recall/F1 + cost/risk-coverage; report oracle ceilings. (status: reporter pending = Step 1)
- **C3 Structural fused gate:** reconstruction-coverage + query-coverage + geometry fused into one calibrated
  head, no extra base forward; ablation shows each family adds. (status: pending)
- **C4 Faithful head-to-head:** OURS vs original Cartridge and Gist at the compressor level (same base/K/data),
  with the in/cross-task and in/cross-domain split. (status: baselines rebuilt; runs pending)
- **C5 Honest negatives / limits:** capacity wall, gate-degeneracy checks, asymmetric-cost caveat. (carried)

---

## 6. Baselines (run in their ORIGINAL form; **no gating applied to baselines**)
> **Frozen base is OUR method's principle, NOT the baselines'.** Each baseline is run exactly as its paper
> defines it; we do not impose our frozen-base constraint on them.
- **Compressors (rows):** no-ctx (floor) · full-ctx/ICL (ceiling) · **Cartridge** (original) = a **trained
  KV-cache** via self-study + context-distillation; base frozen because that is Cartridges' OWN design (not us
  imposing it); always used · **Gist** (original) = gist tokens + gist attention mask + **per-training-set LoRA fine-tuning of the base**
  (gist-masked instruction-tuning CE; base adapted via LoRA, NOT frozen; both the gist token embeddings and the
  LoRA adapter are trained); always used. (LoRA chosen over full FT for feasibility across the grid; disclosed.)
- **Gate references (for OURS' decision panel only):** oracle · always-compress · always-fall-back ·
  base-uncertainty (TARG-style) single-signal gate. These contextualize OURS' F1; they are not applied to the
  baseline compressors.
- Code: `mem-test/mem-embedding/scripts/mem_baselines.py` (`--mode prefix|gist`, `--train-objective
  ce|distill`); loaders + relation taxonomy in `llm-infra/.../datasets.py` and `mem_baselines.DOMAIN`.

---

## 7. Carried-over evidence from v1.5/v1.6 (RE-VALIDATE under the v1.7 protocol)
Still relevant as motivation/limits, but **not** to be quoted as v1.7 results until re-run on the new
baselines + metric:
- **Forgetting is real (motivation):** SFT-LoRA drops no-ctx accuracy (trivia 0.475 -> 0.250); ungated OOD
  negative transfer. Source archive: old matrix §2/§4.
- **Do-no-harm by construction:** $g\to0$ = exact base. (architectural, still holds)
- **Capacity wall:** full-ctx >> compressed on exact recall (RULER ~0 vs 0.995). (still holds; bounds claims)
- **Cross-model signal:** `delta_last` direction-consistent across 7 families, AUROC 0.59-0.80; logit-lens is
  model-specific. (now folded as the geometry feature family in the fused gate)
- **TARG competitive:** a trivial base-uncertainty gate was >= our old signal on 3 families. (keep as the gate
  reference in 3b; the v1.7 question is whether the fused structural gate beats it at fixed precision)

---

## 8. Do-NOT-overclaim (hard constraints)
1. All v1.7 numbers are PENDING; do not state any until run.
2. We do not claim the compressor mechanisms (cite Gist/Cartridges/ICAE). We claim the gated module + the
   decision treatment.
3. do-no-harm by construction is exact only for $g\to0$; the learned gate is "recovers most", report F1/AURC.
4. Baselines are run in their ORIGINAL form: **Gist fine-tunes the base** (NOT frozen), **Cartridge trains a
   KV-cache** (base frozen, which is Cartridges' own design). Frozen base is OUR method's principle only.
5. The 4D gist mask must be verified to actually fire (smoke check) before trusting Gist numbers.
6. Asymmetric costs: do not headline plain F1 without the precision-constrained / cost-coverage view.

---

## 9. Open decisions (need your call)
- **Reconstruction granularity:** coarse pooled-context vs per-chunk (sharper worst-chunk feature, small decoder).
- **Single $g$ vs 2-logit $(g_A,g_B)$** for Track A / Track B (default 2-logit).
- **$\varepsilon$ and $p^\*$:** the success margin and the do-no-harm precision target (e.g. $p^\*{=}0.95$).
- **Primary track:** lead with Track B (compression vs full) or Track A (do-no-harm vs no-ctx)?

---

## 10. Code status (this session)
- ✅ Cartridge rebuilt faithful: per-corpus + self-study/context-distillation (`--train-objective distill`).
- ✅ Gist rebuilt faithful: gist tokens + gist attention mask + frozen-base gist-embedding training; eager attn.
- ✅ Transfer taxonomy wired: `DOMAIN` + `relation()` (in_task / cross_task_in_domain / cross_task_cross_domain),
  records tagged; `grid_transfer`/`gate_baselines` extended.
- ⬜ Step-1 reporter (confusion matrix / P-R-F1 / cost-coverage / oracle) = NOT started.
- ⬜ GCM module in `wrapper.py` (fused gate + fall-back hook) = design under review.
- ⬜ Trainer gate-calibration loss; comparison harness; strip gate plumbing off baselines.

---

## 11. Asset index (fresh; paths TBD until runs land)
| asset | path | state |
|---|---|---|
| v1.7 design | `papers/paper-b-forgetting-gating/v1.7-gated-compressor-module-2026-06-10.md` | done |
| baseline compressors | `mem-embedding/scripts/mem_baselines.py` | rebuilt |
| transfer reporter | `mem-embedding/scripts/grid_transfer.py` (3-way relation added) | ready |
| Step-1 decision reporter | TBD (`scripts/decision_matrix.py`?) | not built |
| GCM module | `mem-embedding/src/mem_embedding/wrapper.py` | pending redesign |
| main comparison CSVs / figures | TBD | pending runs |
