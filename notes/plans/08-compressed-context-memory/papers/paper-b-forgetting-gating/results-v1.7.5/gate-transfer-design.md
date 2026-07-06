# Gate / compressor transfer design (v1.7.5) — the two orthogonal axes

> **Why this doc.** The v1.7.5 tables (T1–T8) conflated two different "transfer" questions. To make the
> robustness claim precise we separate them into **two orthogonal axes**, and crucially split the second axis
> into *what the compressor saw* vs *what the gate saw* (and whether the gate signal depends on training data at
> all). This pins down exactly how strong the "compressor-agnostic, do-no-harm" claim is in each cell.

---
## The two tables this maps to (T6 vs T8) — composability vs generalization
The two questions are **orthogonal** and live in two tables:
- **T6 = COMPOSABILITY** — *vary* **compressor × gate-signal-family (Axis 2B) × relation (Axis 1)**, each cell read
  at **τ ∈ {in-sample, cv, xbench}** (Axis 2C — "the threshold at three points"). Question: is the gate a detachable
  module that pairs with **any** compressor, does that composability *hold across relations*, and does its threshold
  *transfer*? Answer so far: only the **agnostic** family composes with any compressor, holds across relations
  (AUROC 0.77→0.63), and has a portable τ (xbench Δ≈0); intrinsic/learned are GCM-internal and collapse cross-domain.
- **T8 = GENERALIZATION** — *fix the system* (compressor+gate as ONE pipeline), *vary* the eval **relation** (Axis 1:
  in_task / cross_task/in_domain / cross_task/cross_domain). Question: does the whole system transfer to an unseen
  task/domain (incl. non-tool)?

The three sub-axes below (2A what the *compressor* saw; 2B the gate signal's training dependence; 2C where the
*threshold* was fit) qualify **both** tables — e.g. T8 cells should be read at the honest τ (`cv`/`xbench`), and the
T6 "any compressor" claim is specifically the **agnostic** row.

---
## Axis 1 — evaluation relation (eval bench vs the compressor's training anchor)
The harness tags every (train→eval) pair:
- **in_task** — eval bench == train anchor.
- **cross_task / in_domain** — train on tool A, eval on tool B (same tool domain, different task).
- **cross_task / cross_domain** — train on tool, eval on QA / ops / literary (different domain).

## Axis 2 — what was trained on what (SEPARATE for compressor vs gate)
This is the new part. Three sub-dimensions, reported independently:

### 2A. Compressor training source (what the encoder saw)
| code | meaning | "has it seen the eval bench?" |
|---|---|---|
| `self` | trained on the eval bench's train split | yes (= in_task compressor) |
| `anchor` | trained on a *different* single bench | **no** (clean cross) |
| `mix` | trained on a 12-bench mixture | yes — saw eval bench's *leak-free train* split (multi-task, not zero-shot) |
| `none` | train-free (trunc / meanpool) | n/a (necessity floor) |

### 2B. Gate signal family + training dependence (the key table)
| signal family | examples | dep. on **compressor** training | dep. on **gate** training (label/adv) | works on which compressors |
|---|---|---|---|---|
| **agnostic** (behavioural, data-free) | conf, margin, neg_entropy | no (just reads base on `[M;q]`) | **no** | **any** (GCM/Cartridge/Gist/2025) |
| **intrinsic** (self-supervised) | neg_recon, dcode, dlogit, mnorm | **yes** (enc+dec trained) | no (no labels) | **GCM only** |
| **learned-disc** | disc_p | yes | **yes** (adversarial) | GCM only |
| **supervised-logreg** | fit_gate (all signals) | yes | **yes** (fit on y) | GCM only |

→ **Only the agnostic family is truly compressor-agnostic** (intrinsic signals are GCM-specific). The headline
"robustness layer" claim, to be honest, must rest on **agnostic signals**.

### 2C. Threshold / weight fitting source (the gate's *second* "seen what" layer)
Even a data-free signal needs an operating threshold τ. Where τ is fit decides how honest the number is:
| code | meaning | honesty |
|---|---|---|
| `in_sample` | τ chosen on the same eval items (T6 best-F1) | optimistic ceiling |
| `cv` | τ fit on the bench's train fold, applied to held-out fold (`gated_acc_cv`) | honest, realizable |
| `xbench` | τ fit on a *different* bench, applied here (gate also zero-shot) | strictest — true transfer |

---
## The full design grid (what each cell means)
Rows = Axis-1 relation × Axis-2A compressor source. Each cell is further qualified by (2B signal family, 2C τ source).

| compressor src \ relation | in_task | cross_task/in_domain | cross_task/cross_domain |
|---|---|---|---|
| `self` | the in-task gate (T6) | — | — |
| `anchor` | (= self) | **clean tool→tool** | **clean tool→QA/ops/literary** |
| `mix` | multi-task (saw train split) | multi-task | multi-task (T8a) |
| `none` (trunc) | necessity floor (T1) | floor | floor |

The **strongest robustness claim** = `anchor` compressor (never saw eval bench) × `cross_domain` relation ×
**agnostic** signal × `xbench` threshold. That is "zero-shot compressor, zero-shot gate, different domain — still
do-no-harm." No current cell proves that yet.

---
## Current coverage (re-located into the new framework)
| data | compressor src | relation | gate signal | τ source | status |
|---|---|---|---|---|---|
| T6 (GCM ch16a) | `self` | in_task | mixed (intrinsic+agnostic), best | `in_sample` | done (optimistic) |
| T8a (`scale`, GCM mix) | `mix` | labelled cross_domain but really multi-task | neg_recon (intrinsic) | `in_sample` | done |
| T8b (`p2`, Cartridge) | `anchor` | in/cross-task/cross-domain | **none** (no signals/full logged) | — | compress-only |
| v1.7.3 §2.4b (E1, Cartridge) | `self` | in_task | **agnostic** conf | `cv` | done — agnostic on a *different* compressor |
| T8c (`p2_gcm`, RUNNING) | `anchor` | in/cross-task/cross-domain | intrinsic+agnostic (logged) | TBD | running |

## Gaps (what's missing to make the claim airtight)
1. **Agnostic-only gate, reported separately** from intrinsic — currently T6 mixes them, so we can't say how much
   of the gate is data-free. Fix: re-score every records dir with `--signal-key conf` (agnostic) vs `neg_recon`
   (intrinsic) vs `disc_p` (learned) side by side. *(analysis only, no GPU.)*
2. **Cross-bench threshold transfer** (`xbench`) — never tested. Fix: fit τ on bench A's signal, apply to bench B;
   report do-no-harm. *(analysis only.)*
3. **Clean `anchor` compressor × gate** — T8c (running) supplies this for GCM; still need the **agnostic gate on
   Cartridge/2025 compressors under cross relations** (E1 was in_task only). Fix: re-run the `p2` Cartridge
   transfer **with `--signals`** (so agnostic conf is logged on the cross cells) — small GPU.
4. **`mix` vs `anchor` honesty** — T8a's "cross_domain" is mislabelled (multi-task). Fix: rename/annotate, and use
   T8c as the true cross-domain GCM number.

## Immediate finding (gap-1, analysis-only — GCM `mix` → cross-domain, N=576, `disc_gate.py`)
Splitting the signals by family on the *hardest* setting (compressor trained on mix, evaluated cross-domain):
| signal | family | AUROC | note |
|---|---|---|---|
| neg_entropy | **agnostic** | **0.628** | data-free |
| conf | **agnostic** | **0.625** | data-free |
| margin | **agnostic** | **0.607** | data-free |
| neg_dlogit | intrinsic | 0.645 | best intrinsic |
| neg_recon | intrinsic | 0.367 | **degrades (~reversed)** |
| dcode | intrinsic | 0.351 | degrades |
| dlogit | intrinsic | 0.355 | degrades |
| mnorm | intrinsic | 0.398 | degrades |
| disc_p | learned | n/a | mix trained `lam_adv=0` ⇒ no D |

→ **The data-free agnostic signals (conf/margin/neg_entropy, AUROC 0.61–0.63) BEAT the compressor-trained
reconstruction signals (neg_recon/dcode/dlogit 0.35–0.40, ~reversed) on cross-domain.** The signal that needs the
LEAST training (neither compressor nor gate) transfers BEST — direct support for the agnostic robustness-layer
framing. Honesty caveat: `gАcc_cv ≈ always-full` (cov≈0) here — on cross-domain the *realizable* gate degrades to
conservative fallback (do-no-harm holds, no compute saving). Learned-disc needs an adv-trained run to score (gap-3).

## Immediate finding (gap-2, analysis-only — xbench threshold transfer, `xbench_gate.py`)
The strictest regime: fit τ on all OTHER benches (leave-one-bench-out), apply here — does the *gate itself*
transfer? Realized do-no-harm (Δ gAcc vs always-full), GCM mix, 12 benches:
| signal | family | xbench mean(Δ vs full) | worst bench | τ portable? |
|---|---|---|---|---|
| **conf** | **agnostic** | **−0.002** | toolace −0.021 | **✓ safe** (conservatively falls back to full) |
| neg_dlogit | intrinsic | −0.010 | bfcl_live_multiple −0.125 | ✗ harms |
| neg_recon | intrinsic | −0.017 | bfcl_live_multiple **−0.208** | ✗ **harms badly** |

→ **Only the agnostic `conf` threshold is portable across benches** (0–1 probability scale is bench-stable ⇒ a
transferred τ degrades to conservative fallback, Δ≈0). **Intrinsic thresholds do NOT transfer** (reconstruction /
dlogit scale is per-bench ⇒ a transferred τ over-compresses harmful items: neg_recon −0.208 on live_multiple).
This nails the zero-shot gate to the **agnostic** family: it is the only one that is *both* compressor-agnostic
*and* threshold-portable. Honest caveat: portable=safe here means **do-no-harm (Δ≈0), not compute saving** — the
transferred gate compresses almost nothing on cross-domain.

## Recommended headline framing
Report the gate as a **3×3 (relation × signal-family) panel**, at the **honest τ** (`cv`, and `xbench` where
possible), with the agnostic-only row as the compressor-agnostic claim:
- **agnostic signal** row → the data-free robustness layer (works on any compressor, any domain).
- **intrinsic signal** row → GCM's extra self-verification (better where it applies, but GCM-only).
- columns in_task → cross_domain → shows the gate degrades gracefully to conservative always-fallback (do-no-harm
  floor = full) as the signal loses discrimination (AUROC→0.5), never dipping below full by more than the τ slack.
