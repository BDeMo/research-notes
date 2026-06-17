# Janus — metric curation: soundness × interestingness, filter, and metric-driven ideas

> We rate all 39 metrics on **soundness** (合理性: is it a well-justified, non-artifactual measurement?) and **interestingness** (有意思: does it reveal something non-obvious after simple analysis?). We **filter the naive/straightforward/redundant** ones and keep the metrics that are *simple-but-surprising* or *deep-and-sound*. Then we mine ideas from the survivors. Companion to [`metrics-reference.md`](metrics-reference.md) (definitions) and [`grid-metrics-2026-06-04.md`](grid-metrics-2026-06-04.md) (couplings).

## Rating rubric
- **Soundness** ★–★★★: ★ raw/ad-hoc or estimator-noisy · ★★ standard, well-defined · ★★★ principled, robust, hard to game.
- **Interesting** ★–★★★: ★ trivial/expected · ★★ informative covariate · ★★★ non-obvious; its behaviour or coupling teaches us something.
- **Verdict**: 🟢 KEEP (interesting + sound) · 🟡 SUPPORT (sound covariate, not a headline) · 🔴 FILTER (naive, redundant, or artifactual).
- Guiding principle (user): *cut the most naive/straightforward; keep what is interesting and, after simple analysis, makes sense.* A **raw quantity can still be KEEP if its consequence is non-obvious** (e.g. `v_norm`).

---

## The big table (39 metrics)

| metric | frontier | one-line interpretation | sound | interest | comment (合理性 & 为什么 interesting / naive) | verdict |
|---|---|---|---|---|---|---|
| **retrieval** | LC | attn to a planted needle = copy-from-context | ★★★ | ★★★ | the canonical read-side head; causally tied to NIAH. Interesting because it both *defines* long-ctx and *predicts* drift. | 🟢 |
| **attn_distance** | LC | how far back a head reaches | ★★★ | ★★★ | dead-simple to compute, yet it's a **central coupling node** (predicts dW_drift 0.53). Simple-but-surprising. | 🟢 |
| **v_norm** | LC | value-projection magnitude | ★★ | ★★★ | a *raw norm* — naive on its own — but it's the **single best forgetting predictor** (dW_drift 0.56). The poster child for "naive metric, non-naive consequence." | 🟢 |
| **sink** | LC | attn mass on BOS | ★★★ | ★★ | distinct family from retrieval (Jaccard 0). Well-studied; interesting that it's *separate* and still couples with act_drift. | 🟢 |
| **induction** | LC | copy of repeated tokens | ★★★ | ★★ | merges with retrieval into one copy-family (0.61). Sound; moderately novel here. | 🟢 |
| **prev_token** | LC | attn to t−1 | ★★ | ★★ | a primitive; *would* be naive, but it's the top predictor of **grad_mag/fisher** (0.46/0.44) — interesting which CF axis it hits. | 🟡 |
| **attn_entropy** | LC | sharpness of attention | ★★★ | ★★ | sound; partly overlaps token_mixing & receptive_field. Keep as the canonical sharpness measure. | 🟡 |
| **receptive_field** | LC | exp(attn_entropy) | ★ | ★ | **= exp(attn_entropy)** definitionally (ρ=0.998). Pure duplicate. | 🔴 |
| **kv_norm** | LC | key magnitude | ★★ | ★ | ≈ v_norm (0.67) and K/V co-scale; v_norm already carries it. | 🔴 |
| **out_norm** | ST | per-head output norm | ★★ | ★ | mechanically = W_O·(Σattn·V); rides v_norm (0.80). Naive. | 🔴 |
| **eff_rank** | ST | participation ratio of hidden SVs | ★★★ | ★★★ | the representation-collapse axis; non-obvious link to massive-acts & prediction depth. | 🟢 |
| **anisotropy** | ST | mean cosine of token reps | ★★★ | ★★ | representation degeneration; interesting that it tracks prediction crystallization. | 🟢 |
| **intrinsic_dim** | ST | TwoNN manifold dim | ★★ | ★★★ | non-linear dimensionality, robust to linear rank — a *different* view than eff_rank; estimator a bit noisy. | 🟢 |
| **spectral_decay** | ST | slope of log-SV | ★★ | ★ | ≈ −eff_rank (ρ=−0.81). Redundant with eff_rank. | 🔴 |
| **repr_entropy** | ST | entropy of hidden SVs | ★ | ★ | ≡ log(eff_rank). Pure duplicate (already pruned). | 🔴 |
| **update_norm** | ST | ‖h_l−h_{l−1}‖ | ★★ | ★ | layers change the residual — expected; rides resid_norm. | 🔴 |
| **resid_norm** | ST | residual-stream norm | ★★ | ★★ | grows monotonically (mostly trivial) **but** is the backbone of the collapse axis → keep as an index, not a headline. | 🟡 |
| **curvature** | ST | ‖h_{l+1}−2h_l+h_{l−1}‖ | ★★ | ★★ | trajectory bend = where non-linear processing happens; mildly interesting. | 🟡 |
| **cka_adjacent** | ST | similarity to next layer | ★★★ | ★★ | identifies "redundant"/copy layers; sound, moderately interesting. | 🟡 |
| **token_mixing** | LC | off-diagonal attn mass | ★★ | ★★ | mixing vs self/sink; overlaps attn_entropy but more interpretable. | 🟡 |
| **ll_top1_depth** | ST | logit-lens argmax = final | ★★★ | ★★★ | **"prediction depth"** — where the answer locks in. Deep + sound + non-obvious (couples with Fisher). | 🟢 |
| **ll_kl_to_final** | ST | KL(lens‖final) | ★★★ | ★★ | same axis as depth (ρ −0.84); keep one. | 🟡 |
| **tuned_lens_depth** | ST | de-biased prediction depth | ★★★ | ★★ | removes the untrained-lens bias; the *correct* depth measure (prefer over logit-lens). | 🟢 |
| **tuned_lens_kl** | ST | de-biased KL | ★★★ | ★ | ≈ tuned_lens_depth; keep one. | 🔴 |
| **lens_gap** | ST | tuned−logit KL | ★ | ★★ | derived (contains ll_kl); interesting as "how misleading the naive lens is" but not a primary metric. | 🔴 |
| **ll_entropy** | ST | lens prediction entropy | ★★ | ★ | ≈ inverse confidence; covered by depth. | 🔴 |
| **surprisal** | ST | model NLL/token on the data | ★★★ | ★★ | clean difficulty/OOD signal; good control for cross-dataset comparison. | 🟡 |
| **act_kurtosis** | ST | heavy-tailedness of activations | ★★★ | ★★ | = the massive-activation tail; ≈ massive_max (0.93). Keep one. | 🟡 |
| **massive_max** | ST | top channel / median | ★★★ | ★★★ | the massive-activation signature (Sun 2024); ties to eff-rank collapse — channel-level load-bearing site. | 🟢 |
| **massive_count** | ST | #outlier channels | ★★ | ★ | ≈ massive_max (0.65) + count is coarse. | 🔴 |
| **gini** | ST | activation inequality | ★★ | ★★ | concentration in few units; cleaner scalar than sparsity. | 🟡 |
| **act_sparsity** | ST | frac near-zero | ★★ | ★ | ≈ gini/massive cluster (0.79). Redundant. | 🔴 |
| **dead_frac** | ST | frac inactive channels | ★★ | ★★ | unused capacity — interesting for "where new domain can go," but niche/noisy. | 🟡 |
| **w_ht_alpha** | CF/ST | HT-SR power-law exponent | ★★ | ★★★ | "trainedness" of a layer (WeightWatcher); genuinely non-obvious that it predicts anything; estimator noisy → ★★ sound. | 🟢 |
| **head_alpha** | CF | per-head HT-SR α | ★★ | ★★ | per-head trainedness; interesting but noisy at head granularity. | 🟡 |
| **w_stable_rank** | ST | ‖W‖_F²/σ_max² | ★★★ | ★★ | robust rank; ≈ w_eff_rank (0.80). Keep one. | 🟡 |
| **w_eff_rank** | ST | participation ratio of W | ★★★ | ★★ | sound; ≈ weight_entropy (0.99). | 🟡 |
| **weight_entropy** | ST | entropy of W SVs | ★ | ★ | ≡ log(w_eff_rank). Duplicate. | 🔴 |
| **w_spectral_norm** | ST | σ_max(W) | ★★ | ★ | raw top singular value; naive. | 🔴 |
| **condition_num** | ST | σ_max/σ_min | ★ | ★ | dominated by a tiny σ_min → noisy/unreliable. | 🔴 |
| **down_stable_rank** | ST | MLP down-proj stable rank | ★★★ | ★★ | the *one* spectral metric that co-moves with prediction formation (−0.71) — keep as the interesting param-spectra hook. | 🟡 |
| **mlp_gain** | ST | ‖up‖/‖down‖ | ★ | ★ | raw norm ratio; naive. | 🔴 |
| **head_wnorm** | ST | ‖q‖+‖o‖ per head | ★ | ★ | raw weight norm; naive. | 🔴 |
| **ov_norm** | ST | ‖W_O W_V‖ | ★★ | ★ | OV circuit, but ≈ head_wnorm (0.90); naive in practice. | 🔴 |
| **fisher** | CF | empirical Fisher diag per head | ★★★ | ★★ | a-priori importance; the "predict-before-training" lever. | 🟢 |
| **grad_mag** | CF | mean ‖grad‖ per head | ★★ | ★ | ≈ fisher (0.98). Redundant. | 🔴 |
| **grad_noise** | CF | per-head gradient CV² | ★★ | ★★★ | **which heads are pushed inconsistently** across batches — non-obvious, different from magnitude. | 🟢 |
| **dW_drift** | CF | ‖ΔW‖ per head after SFT | ★★★ | ★★★ | what SFT *actually* did; the cleanest CF target; couples hardest with LC. | 🟢 |
| **act_drift** | CF | functional change per head after SFT | ★★★ | ★★★ | behavioural change; couples hardest with LC sites (sink/retrieval). | 🟢 |
| **niah_acc / retention** | LC/CF | behavioural outcomes | ★★★ | ★★★ | the ground-truth the whole story must move. | 🟢 |

### Summary of the cut
- 🟢 **KEEP (14):** retrieval, attn_distance, v_norm, sink, induction, eff_rank, anisotropy, intrinsic_dim, ll_top1_depth, tuned_lens_depth, massive_max, w_ht_alpha, fisher, grad_noise, dW_drift, act_drift, niah/retention. *(the interesting + sound core)*
- 🟡 **SUPPORT (covariates, keep but not headline):** prev_token, attn_entropy, resid_norm, curvature, cka_adjacent, token_mixing, surprisal, act_kurtosis, gini, dead_frac, head_alpha, w_stable_rank/eff_rank, down_stable_rank, ll_kl_to_final.
- 🔴 **FILTER (naive/redundant/artifactual — 15):** receptive_field, kv_norm, out_norm, spectral_decay, repr_entropy, update_norm, tuned_lens_kl, lens_gap, ll_entropy, massive_count, act_sparsity, weight_entropy, w_spectral_norm, condition_num, mlp_gain, head_wnorm, ov_norm, grad_mag.

**Filter logic in one line:** drop (a) *definitional duplicates* (receptive_field, repr_entropy, weight_entropy, lens_gap), (b) *mechanical twins* (kv_norm/out_norm vs v_norm; grad_mag vs fisher; massive_count vs massive_max; act_sparsity vs gini), and (c) *raw-norm naïveté with no surprising consequence* (w_spectral_norm, mlp_gain, head_wnorm, ov_norm, condition_num).

---

## Ideas mined from the survivors (metric-driven)

Distinct from the method-mechanism list in [`ideas-brainstorm-2026-06-04.md`](ideas-brainstorm-2026-06-04.md) — these are ideas that exist *only because a specific metric turned out interesting*.

**I1 — Zero-backward forgetting predictor.** *From:* v_norm + attn_distance (KEEP, forward-only) predict dW_drift as well as Fisher does. *Idea:* predict which heads will forget using a **single forward pass** (no gradients, no SFT) → a free "forgetting risk map." *Insight:* if a raw forward metric matches Fisher, protection becomes essentially free to target.

**I2 — Prediction-depth–gated protection.** *From:* ll/tuned_lens depth ("where the answer forms"). *Idea:* protect heads/layers **below the prediction-depth knee** (the still-computing region) and let late crystallized layers adapt freely. *Insight:* tie the protected set to *where computation finishes*, not to raw importance.

**I3 — Trainedness-aware fine-tuning (HT-SR α).** *From:* w_ht_alpha / head_alpha. *Idea:* **fine-tune the under-trained (high-α) heads, freeze the well-trained (low-α, heavy-tailed) ones** — adapt where there's "room," protect where the layer is already mature. *Insight:* an a-priori, data-free criterion orthogonal to the LC criterion; great ablation against it.

**I4 — Gradient-noise scheduling.** *From:* grad_noise (interesting, distinct from magnitude). *Idea:* per-head lr / clip ∝ 1/grad_noise — damp the heads that are pushed *inconsistently* (the ones that drift destructively). *Insight:* tests whether forgetting is driven by *instability* vs *magnitude*.

**I5 — Massive-activation channel protection (channel granularity).** *From:* massive_max (KEEP). *Idea:* protect the ~5 massive-activation **channels** (not heads) during SFT — they're the densest load-bearing sites. *Insight:* does the coupling live at channel or head granularity? cheap, orthogonal axis.

**I6 — Capacity-exhaustion router for new domains.** *From:* eff_rank / intrinsic_dim / dead_frac. *Idea:* send new-domain learning into the layers with **spare capacity** (high dead_frac, low eff_rank-utilization), away from the collapsed late layers. *Insight:* place adaptation where it does least damage.

**I7 — Act-drift "un-forgetting" steering vector.** *From:* act_drift (KEEP). *Idea:* the act_drift direction of the coupled heads = the "forgetting direction"; **subtract it at inference** to recover base long-context — training-free. *Insight:* if forgetting is low-rank in activation space, it's reversible without retraining.

**I8 — The collapse-axis as a single scalar.** *From:* eff_rank + anisotropy + massive_max + depth all co-move. *Idea:* define one **"computation-finished" index** (their first PC) per layer; use it as the universal protect/eval boundary across families. *Insight:* compresses 4 interesting metrics into one transferable knob.

**I9 — Copy-family freeze (retrieval∪induction∪prev_token).** *From:* F4 copy family. *Idea:* protect the *union* of the copy circuit as one functional unit. *Insight:* is the protectable object a *circuit* (multi-metric) rather than single heads?

**I10 — v_norm cap as forgetting *prevention by design*.** *From:* v_norm is the bridge. *Idea:* lightly **regularize V-norm during pretraining/CPT** so no head becomes a coupling bottleneck → a model that forgets *gracefully*. *Insight:* the dual of protection — *dilute* the coupling instead of guarding it.

**I11 — Surprisal-conditioned dose.** *From:* surprisal (data difficulty). *Idea:* scale protection strength by per-batch surprisal — protect harder when the new data is far OOD (where forgetting is worst). *Insight:* adaptive protection keyed to a free signal.

**I12 — Depth × drift map = a "forgetting fingerprint."** *From:* prediction depth × dW_drift per layer. *Idea:* the 2-D (depth, drift) signature predicts *which capabilities* a fine-tune will erode (early-forming = syntax, late = knowledge). *Insight:* forecast *what* gets forgotten, not just how much.

### Shortlist to try first
**I1** (zero-backward predictor — cheapest, biggest if it works) · **I2** (depth-gated protection — principled boundary) · **I3** (trainedness criterion — the cleanest a-priori rival to the LC criterion) · **I7** (act-drift steering — training-free un-forgetting, high wow-factor). These four are the most "interesting + sound + testable."
