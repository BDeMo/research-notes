# Results v1.8.0 — Adversarial vs Contrastive semantic alignment of the compressed memory

> **Builds on [v1.7.5](../results-v1.7.5/results-v1.7.5.md).** Goal: make the compressed memory **M** induce, in the
> base model's **later layers**, a hidden state **semantically close to the full-context one** — so M is not only
> answer-predictive (task CE) but **representationally faithful**. Two ways to push this, compared **head-to-head** at
> matched budget + tuned: **ADVERSARIAL** (a discriminator can't tell M-induced from full-induced) vs **CONTRASTIVE**
> (InfoNCE pulls M_i→full_i, pushes M_i from other full_j). Setup inherits v1.7.5
> ([experimental-setup](../results-v1.7.3/experimental-setup.md)) + the deltas below.

## Why
- The OOD probe (v1.7.5 T7) + T8a show M's later-layer hidden **diverges from full**, worst on cross-domain. If we can
  **align M's later-layer semantics to full**, compression may lose less detail *and* the do-no-harm gate signals
  (which read base behavior on `[M;q]`) get cleaner.
- An adversarial term already existed (`lam_adv`, "did not hurt" earlier); **contrastive is the new rival** — this
  version makes the comparison fair (same layer, same pooled hidden, matched budget) and tunes both.

## The two losses (same target: `align_layer=18`, query-position mean-pooled hidden)
| loss | mechanism | what it enforces |
|---|---|---|
| **adversarial** | a small MLP **D** reads the layer-18 hidden at query positions, labels M-induced=1 / full-induced=0; the compressor is trained to **fool D** (make D call M "full") | **distribution-level**: M-induced ≈ full-induced as a set |
| **contrastive** | **InfoNCE**: normalize(hM_i), normalize(hF_i); pos = hM_i·hF_i, negs = **other samples'** hF_j (SimCLR-style global teacher bank); cross-entropy with τ | **instance-level**: M_i like *its own* full, *unlike* other fulls |

Shared: same `align_layer`, same pooled query-position hidden, bf16, teacher (full) detached. The adversarial D is
also persisted as the **learned gate** (`disc_p`) — so this version *also* fills T6's learned-gate column.

## Implementation (`svc/method.py`, `gcm/{methods,harness}.py` — synced to both pods)
- New `TrainCfg`/harness knobs: `--lam-contrast`, `--align-layer`, `--contrast-temp`, `--contrast-bank` (+ existing `--lam-adv`).
- **adv**: discriminator step (BCE) + compressor fool-term; D persisted as the gate.
- **contrast**: `cbank_all` = all train samples' full pooled hidden (normalized); per step pos = hM·hF_i, negs =
  subsample(`cbank_all`, bank). **Global teacher bank** (not FIFO) so the loss starts at ~log(1+N) and **drops as M
  aligns** (FIFO had a fill-in artifact where the loss spuriously *rose*).
- **Backward-compatible**: `lam_adv=lam_contrast=0` ⇒ the original path (no hidden-state outputs) — does not affect
  the v1.7.5 / transfer runs.
- **Smoke**: both train end-to-end, no errors; task CE descends; `ctr` logged (starts ≈log(1+N)). Whether `ctr`
  *drops* (alignment actually happens) is the sweep's question.

## Sweep (24 jobs, `free` queue after `ab_min`) — `queue_v18.py`
- **variants**: `base` (no alignment) / `adv0.5` / `adv1.0` / `ctr0.5` / `ctr1.0` / `ctr2.0`
- **benches**: bfcl_live_multiple, hermes, bfcl_multiple, toolace (value-add + long — where M has something to compress)
- **matched combo**: enc16 / K64 / LoRA32 / lr3e-4 / 1600 steps / grad-accum 8 / lam-distill 0.5, align_layer 18.
- **metrics**: (1) **compress acc** — does alignment lift compression vs base? (2) **gate AUROC** — does it clean the
  gate signal? (3) **train `ctr`/`adv` loss** — does alignment actually happen, or is M too OOD to align?

## Results — PENDING (`free`, `out/clean/v18/`)
| bench | base | adv0.5 | adv1.0 | ctr0.5 | ctr1.0 | ctr2.0 |
|---|---|---|---|---|---|---|
| bfcl_live_multiple | — | — | — | — | — | — |
| hermes | — | — | — | — | — | — |
| bfcl_multiple | — | — | — | — | — | — |
| toolace | — | — | — | — | — | — |

*(compress acc; gate AUROC + train-loss-drop in a companion table. ETA tonight; analyzer: `decision.py` / `disc_gate.py`.)*
