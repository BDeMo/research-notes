# Janus — Metric reference (codebook) + full correlation ranking

> Complete reference for the 39 intrinsic metrics: **how each is computed, why it's
> designed that way, how to read it, and its range** — plus the **full correlation
> ranking** from the 41-cell grid. Companion to [`grid-metrics-2026-06-04.md`](grid-metrics-2026-06-04.md)
> and [`paper-draft-2026-06-04.md`](paper-draft-2026-06-04.md). Source code: [`runs/janus_run.py`](runs/janus_run.py).

## Legends
- **Frontier** — **LC** long-context/inference · **CF** forgetting/training · **ST** structural covariate.
- **Pass** (decoupled compute) — **A** forward (hidden+attn, K/V hooks) · **B** weights-only SVD · **C** backward (grad/Fisher) · **D** short SFT (drift) · **E** behavioural.
- **Granularity** — H per-(layer,head) · L per-layer · M per-matrix(→layer) · S scalar.
- ρ = Spearman rank correlation over sites, pooled across the 41 model×dataset cells.

---

## PART I — Full correlation ranking (what co-varies with what)

Each pair is tagged by **dependence type**: ⚙️ *definitional* (one is a function of the other — an artifact, exclude), 🔧 *mechanistic* (true by construction of the forward pass), 🧭 *empirical-LC-internal* (long-context heads cluster), ⭐ *empirical LC×CF coupling* (the scientific claim).

### A. Per-head ranking (top, of 153 pairs)
| ρ | pair | type |
|---|---|---|
| +0.998 | attn_entropy ~ receptive_field | ⚙️ receptive_field ≡ exp(attn_entropy) |
| +0.980 | fisher ~ grad_mag | ⚙️ both gradient-magnitude |
| +0.904 | head_wnorm ~ ov_norm | ⚙️ per-head weight-norm family |
| +0.799 | out_norm ~ v_norm | 🔧 output is built from V |
| +0.671 | kv_norm ~ v_norm | 🔧 K and V co-scale |
| +0.610 | **induction ~ retrieval** | 🧭 the copy/retrieval-head family |
| +0.560 | **dW_drift ~ v_norm** | ⭐ big-V heads drift most |
| +0.528 | **attn_distance ~ dW_drift** | ⭐ long-reach heads drift most |
| +0.509 | act_drift ~ dW_drift | 🔧 ΔW → Δactivation |
| +0.496 | dW_drift ~ out_norm | ⭐ big-output heads drift |
| +0.488 | retrieval ~ v_norm | 🧭 retrieval heads have big V |
| +0.480 | sink ~ v_norm | 🧭 sink heads have big V |
| +0.478 | **act_drift ~ attn_distance** | ⭐ long-reach heads' acts shift |
| +0.473 | **act_drift ~ sink** | ⭐ sink heads' acts shift |
| +0.473 | act_drift ~ v_norm | ⭐ |
| +0.467 | kv_norm ~ retrieval | 🧭 |
| +0.463 | **grad_mag ~ prev_token** | ⭐ prev-token heads take big grads |
| +0.457 | attn_distance ~ grad_noise | ⭐ |
| +0.448 | act_drift ~ kv_norm | ⭐ |
| +0.445 | dW_drift ~ kv_norm / retrieval | ⭐ |
| +0.441 | fisher ~ prev_token | ⭐ |

→ Once the ⚙️/🔧 artifacts are set aside, the dominant signal is **⭐ every long-context head metric (v_norm, attn_distance, retrieval, sink, prev_token, kv_norm) positively predicting every forgetting metric (dW_drift, act_drift, fisher, grad_mag, grad_noise)** — the Janus coupling. Centre of both clusters: **v_norm** and **attn_distance**.

### B. Per-layer ranking (top, of 1081 pairs)
| ρ | pair | type |
|---|---|---|
| +0.993 | w_eff_rank ~ weight_entropy | ⚙️ both spectral-entropy of W |
| -0.982 | lens_gap ~ ll_kl_to_final | ⚙️ lens_gap contains the term |
| -0.973 | tuned_lens_depth ~ tuned_lens_kl | ⚙️ KL vs top-1 of same lens |
| +0.940 | attn_entropy ~ token_mixing | 🔧 diffuse attention = more mixing |
| +0.926 | act_kurtosis ~ massive_max | 🔧 outliers drive kurtosis |
| +0.919 | resid_norm ~ update_norm | 🔧 updates accumulate in residual |
| +0.848 | curvature ~ resid_norm | 🔧 |
| -0.837 | ll_kl_to_final ~ ll_top1_depth | 🔧 closer ⇒ deeper |
| +0.814 | act_sparsity ~ massive_max | 🧭 massive-activation signature |
| -0.810 | eff_rank ~ spectral_decay | ⚙️ both from singular spectrum |
| +0.777 | cka_adjacent ~ massive_max | 🧭 collapse cluster |
| +0.774 | retrieval ~ v_norm (head-pooled) | 🧭 |
| +0.704 | **dW_drift ~ v_norm (pooled)** | ⭐ |
| +0.684/0.679 | **dW_drift / act_drift ~ retrieval** | ⭐ |
| +0.668 | **act_drift ~ attn_distance** | ⭐ |

Full machine-readable list: [`runs/grid_summary.json`](runs/grid_summary.json) + `_grid_tables/full_ranking.json` (on the pod).

---

## PART II — The codebook (every metric)

### Angle 1 — Attention sink / streaming  ·  pass A, gran H
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **sink** [LC] | mean over query pos ≥4 of attention prob to key pos 0: `att[:,4:,0].mean` | attention sinks (Xiao 2023) park probability on BOS; a per-head sink score | high = streaming/sink head (dumps attention on token 0) | [0,1] |
| **attn_entropy** [LC] | −Σ_k p·log p of each attention row, mean over queries | sharpness of attention; low = peaked (copy/retrieval), high = diffuse | high = spread-out attention | [0, log T] |

### Angle 2 — Retrieval / copy  ·  pass A, gran H
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **retrieval** [LC] | attn from last query to a planted **needle**'s value tokens, summed; avg over 3 needles | operationalizes "retrieval head" (Wu 2024) = the long-context read site | high = head that fetches the needle | [0,1] |
| **induction** [LC] | on a repeated random sequence, attn from t to t−period (the induction offset), mean | induction heads (Olsson 2022) implement copy of repeats; ICL substrate | high = induction head | [0,1] |
| **prev_token** [LC] | attn on the −1 diagonal (to t−1), mean | previous-token heads feed induction; a local-copy primitive | high = previous-token head | [0,1] |

### Angle 3 — Attention reach  ·  pass A, gran H
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **attn_distance** [LC] | Σ_k att·\|i−j\|, mean over queries (token units) | how far back a head reaches = its long-context footprint | high = long-range head | [0, T] |
| **receptive_field** [LC] | exp(attn_entropy) = effective #keys attended | intuitive "span"; **⚙️ redundant with attn_entropy — pruned** | high = attends to many keys | [1, T] |

### Angle 4 — KV / value geometry  ·  pass A (K/V hooks), gran H
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **kv_norm** [LC] | mean ‖K‖ of the head's key projection (per kv-group, broadcast to heads) | KV-cache magnitude = how much a head writes into the long-context memory | high = heavy KV-memory head | [0,∞) |
| **v_norm** [LC] | mean ‖V‖ | value magnitude; the content a head moves | high = high-content head | [0,∞) |
| **out_norm** [ST] | mean ‖per-head attention output‖ (o_proj input, reshaped per head) | how much the head contributes to the residual stream | high = high-impact head | [0,∞) |

### Angle 5 — Representation rank / geometry  ·  pass A, gran L
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **eff_rank** [ST] | exp(entropy of normalized singular values of centered hidden [T,d]) | participation ratio = how many dimensions the representation uses | high = high-dim/spread reps | [1, min(T,d)] |
| **anisotropy** [ST] | mean off-diagonal cosine of normalized token vectors | representation degeneration (tokens collapsing to one direction) | high = collapsed/aligned reps | [−1,1] (≈0–1) |
| **intrinsic_dim** [ST] | TwoNN estimator: N / Σ log(r₂/r₁) on token reps | manifold dimensionality, robust to linear rank | high = complex local manifold | [0, d] |
| **spectral_decay** [ST] | −slope of log singular values vs index | how fast the spectrum falls (steep = low-rank) | high = fast decay = concentrated | [0,∞) |

### Angle 6 — Representation dynamics  ·  pass A, gran L (between layers)
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **update_norm** [ST] | mean ‖h_l − h_{l−1}‖ | how much a layer changes the residual stream | high = "busy" layer | [0,∞) |
| **curvature** [ST] | mean ‖h_{l+1} − 2h_l + h_{l−1}‖ | bend in the representation trajectory (non-linear processing) | high = sharp redirection | [0,∞) |
| **cka_adjacent** [ST] | linear CKA(h_l, h_{l+1}) | representational similarity to the next layer | high = layer barely transforms reps | [0,1] |
| **token_mixing** [LC] | 1 − (self-attn + sink mass), mean over heads | how much positions actually exchange info (vs attend self/sink) | high = strong token mixing | [0,1] |

### Angle 7 — Information theory (logit / tuned lens)  ·  pass A, gran L (surprisal S)
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **ll_kl_to_final** [ST] | KL(softmax(unembed(norm(h_l))) ‖ final), mean over tokens | how far the layer's "current guess" is from the final prediction | high = prediction not yet formed | [0,∞) |
| **ll_top1_depth** [ST] | frac tokens where logit-lens argmax == final argmax | "prediction depth": where the answer locks in | high = answer already decided | [0,1] |
| **ll_entropy** [ST] | entropy of the logit-lens distribution | confidence of the intermediate readout | high = uncertain layer | [0, log V] |
| **tuned_lens_kl** [ST] | ridge-fit A: h_l→h_final, then unembed; KL to final | tuned lens removes the untrained-lens bias (Belrose 2023) | high = layer far from final (de-biased) | [0,∞) |
| **tuned_lens_depth** [ST] | top-1 agreement of the tuned lens to final | de-biased prediction depth | high = answer decided | [0,1] |
| **lens_gap** [ST] | tuned_lens_kl − ll_kl_to_final | how much the learned translator helps (lens calibration) | ⚙️ derived; large negative = naive lens misleading | (−∞,∞) |
| **surprisal** [ST] | model NLL per token on the data (cross-entropy) | the model's actual difficulty on this dataset | high = harder/OOD data | [0,∞) nats |

### Angle 8 — Activation distribution  ·  pass A, gran L
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **act_kurtosis** [ST] | 4th standardized moment of all activations | heavy tails ⇒ massive activations (Sun 2024) | high = outlier-dominated | [1,∞) (Gauss=3) |
| **act_sparsity** [ST] | frac \|x\| < 0.1·σ | how concentrated activity is in few units | high = sparse layer | [0,1] |
| **massive_max** [ST] | max channel (max-over-tokens \|act\|) / median channel | size of the largest massive activation | high = strong outlier channel | [1,∞) |
| **massive_count** [ST] | #channels with max\|act\| ≥ 6× median | how many outlier channels | high = many massive channels | [0, d] |
| **gini** [ST] | Gini coefficient of \|activations\| | inequality/concentration of activation mass | high = few units carry it | [0,1] |
| **dead_frac** [ST] | frac channels with max\|act\| < 0.05× median | inactive-capacity fraction | high = many dead channels | [0,1] |

### Angle 9 — Parameter spectra (per matrix → layer mean)  ·  pass B, gran M
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **w_stable_rank** [ST] | ‖W‖_F² / σ_max² | smooth rank robust to tail; capacity used by W | high = many comparable singular values | [1, min(m,n)] |
| **w_eff_rank** [ST] | exp(entropy of normalized σ) | participation ratio of the weight spectrum | high = spread spectrum | [1, min(m,n)] |
| **w_spectral_norm** [ST] | σ_max(W) | dominant gain of the matrix | high = strong leading direction | [0,∞) |
| **w_ht_alpha** [CF] | Hill power-law exponent of the eigenvalue (σ²) tail (HT-SR) | heavy-tailed self-regularization: lower α ⇒ better-trained layer | low = more heavy-tailed/trained | ≈[2,6] |
| **condition_num** [ST] | σ_max/σ_min | numerical conditioning of W | high = ill-conditioned | [1,∞) |
| **weight_entropy** [ST] | entropy of normalized σ | ⚙️ log of w_eff_rank — spectral flatness | high = flat spectrum | [0, log min(m,n)] |
| **down_stable_rank** [ST] | stable rank of the MLP **down_proj** | the output projection co-moves with prediction formation (a key fact) | high = high-rank MLP write | [1, min(m,n)] |
| **mlp_gain** [ST] | ‖up_proj‖_F / ‖down_proj‖_F | expansion-vs-contraction balance of the MLP | high = up-weighted MLP | [0,∞) |

### Angle 10 — Per-head parameter geometry  ·  pass B, gran H
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **head_alpha** [CF] | HT-SR α on the per-head [q;o] weight slice | per-head "trainedness"/heavy-tail | low = more heavy-tailed head | ≈[2,6] |
| **head_wnorm** [ST] | ‖q_head‖ + ‖o_head‖ | raw weight magnitude of the head | high = large-weight head | [0,∞) |
| **ov_norm** [ST] | ‖W_O,head · W_V,group‖ (the OV circuit) | the head's write-path strength (Elhage 2021) | high = strong output-value circuit | [0,∞) |

### Angle 11 — Forgetting / training dynamics  ·  pass C (a-priori) + D (outcome), gran H
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **fisher** [CF] | per head, mean-over-batches Σ g² on its q/o params (empirical Fisher diag) | curvature/importance: high-Fisher params are where loss is sensitive — SFT moves them | high = SFT-sensitive head (a-priori) | [0,∞) |
| **grad_mag** [CF] | per head, mean-over-batches mean\|g\| | gradient magnitude landing on the head | high = big-gradient head | [0,∞) |
| **grad_noise** [CF] | per-head gradient CV² across batches (Var/Mean²) | gradient stability; noisy heads are pushed inconsistently | high = unstable gradient | [0,∞) |
| **dW_drift** [CF] | ‖W_after − W_before‖ per head (q+o) after the short SFT | what fine-tuning **actually changed** (the outcome) | high = head SFT rewrote | [0,∞) |
| **act_drift** [CF] | per-head relative activation change \|a_after−a_before\|/\|a_before\| on a fixed probe | functional change of the head after SFT | high = head's behaviour changed | [0,∞) |

### Angle 12 — Behaviour / capability  ·  pass E, gran S (per model×dataset)
| metric | how computed | why | interpret | range |
|---|---|---|---|---|
| **niah_acc** [LC] | needle-in-haystack accuracy over length×depth grid | the long-context *outcome* (does it actually retrieve?) | high = strong long-context | [0,1] |
| **mmlu_acc / gsm8k_acc** | per-option log-likelihood / generation match | general-knowledge & new-domain ability | higher = better | [0,1] |
| **retention Δ** [CF] | after − before accuracy (any capability) | the forgetting *outcome* | <0 = forgot; ≥0 = preserved | [−1,1] |

---

## PART III — Design notes (why this set)
- **Two frontiers, matched conditions.** LC metrics are read off **inference** on (long) inputs; CF metrics are read off **training** (a-priori gradient/Fisher + post-SFT drift). They are deliberately *separable so the coupling is non-trivial* — nothing forces a head's inference reach to predict its training drift.
- **A-priori vs outcome on the CF side.** `fisher`/`grad_mag`/`grad_noise` are computable **before** any SFT (the "predict-before-training" claim); `dW_drift`/`act_drift` are the **outcome**. The coupling is strongest with the outcome, but the a-priori legs are what make the protection rule usable.
- **Structural covariates** explain *why*: the representation-collapse axis (eff_rank↓, anisotropy↑, massive-acts↑, down-proj spectrum) co-locates with where prediction forms (lens depth) and is the substrate both frontiers ride on.
- **Pruned/derived (don't double-count):** `receptive_field` (=exp attn_entropy), `weight_entropy` (≈log w_eff_rank), `lens_gap` (derived), `grad_mag`≈`fisher`. Keep one of each.
- **Ranges in practice:** rank correlations make absolute scales irrelevant for the coupling; we report raw ranges here only for reading single-metric layer/​head profiles.
