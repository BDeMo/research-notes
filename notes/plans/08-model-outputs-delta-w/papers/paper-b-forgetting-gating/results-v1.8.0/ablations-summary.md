# GCM v1.8.0 — All ablations (consolidated)

Model: Qwen3-8B (dense) unless noted. Compressor = self-compressor, K memory tokens, read-LoRA, recurrent prefix encode.
Two eval regimes (do NOT mix): **MC** = module-choice log-likelihood on `cmg_rca`/`rca_openrca` (argmax==gold); **GEN** = free-form RCA report on `cmg_distill` (module_strict + collapse).

---

## 0. Headline: best model on CMG (MC module-ID, cmg_rca, n=12)

| model | train data | K / LoRA | no_ctx | full(8k) | **compress** | read |
|---|---|---|---|---|---|---|
| **mt_all_distill1** | distill+aug+rca, distill=1.0 | 256 / 64 | 0.083 | 0.333 | **0.417** | compress **>** full (M beats raw evidence) |
| **mc_augrca_lg** | aug+rca (MC) | 256 / 64 | 0.167 | 0.417 | **0.417** | compress **=** full |
| mc_aug | aug | 256 / 64 | 0.0 | 0.25 | 0.333 | |
| mc_aug_k128 | aug | 128 / 64 | 0.083 | 0.333 | 0.333 | |
| mc_allthree | aug+rca+distill (MC) | 256 / 64 | 0.083 | 0.25 | 0.333 | |
| mt_all_lora128 | all | 256 / 128 | 0.083 | 0.25 | 0.333 | |
| mc_aug_long | aug (long) | 256 / 64 | 0.083 | 0.417 | 0.333 | |
| mt_all_norec | all, **no recur** | 256 / 64 | 0.25 | 0.25 | 0.25 | recur off hurts |
| mc_all_final | all | 256 / 64 | 0.0 | 0.333 | 0.25 | |
| mt_all_long2 | all (long) | 256 / 64 | 0.167 | 0.333 | 0.25 | |
| mc_augrca | aug+rca | 256 / 64 | 0.083 | 0.25 | 0.167 | |
| mc_distillaug | distill+aug | 256 / 64 | 0.167 | 0.333 | 0.167 | |
| mt_all_k128 | all | 128 / 64 | 0.167 | 0.333 | 0.167 | |
| mt_distillonly | distill only | 256 / 64 | 0.083 | 0.333 | **0.0** | distill-only collapses on MC |

**Best = `mt_all_distill1`** (compress 0.417 AND compress>full — the only config where M beats reading raw evidence). `mc_augrca_lg` ties on compress (0.417=full). **Now served in the demo.**

> ⚠ n=12 → these swing ±1 case (compress for the same adapter observed 0.17–0.42 across independent re-evals). Not statistically separable. The reliable benchmark is OpenRCA below.

## Reliable benchmark: Public OpenRCA (MC, n=150)
| | no_ctx | truncated-full (8k) | compress |
|---|---|---|---|
| mt_all_distill1 | 0.16 | **0.80** | 0.18 |

→ Reading raw evidence (0.80) ≫ compressed M (0.18 ≈ no-context 0.16). The compressor does NOT yet recover the discriminative module signal; it buys fluency + 25–48× token savings.

---

## 1. Loss-term ablations (Qwen3-8B, bfcl anchor)
Each term toggled on/off vs the v1.7.5 answer-CE+distill anchor:
- **distill** (KL to teacher logits): 8b_distill0 vs 8b_distill1 — distill helps.
- **reconstruction** (decoder M→context): 8b_norecon / 8b_recon05 / 8b_recon / 8b_recon2 — small/neutral; 0.5 default.
- **VAE-KL** (M as latent + KL prior): 8b_vaelo / 8b_vaehi — KL must be tiny or it dominates.
- **deviation** (8b_dev / 8b_nodev), **adversarial** (8b_adv / 8b_noadv), **contrastive** (8b_contrast / 8b_nocon) — contrast is the lever for M diversity.
- **none** (8b_none) and **anchor** (8b_anchor175) = reference points.
- **phase split** (8b_split_full / 8b_split_recon): encoder-only phase-1 vs joint.
- Mirror set on the gate path: q3_8b_g_{full,none,distill0,distill1,contrast,no_contrast,adv,dev}.

## 2. Memory size K
K ∈ {16, 64, 128, 256, 384, 512}: e_k64/k128/k256/k384/k512, q38r_K16/K64, mc_aug_k128, mt_all_k128.
→ 256 is the sweet spot; 128 underfits MC (mt_all_k128 compress 0.167 vs 256's 0.417). K512 in the running gap-sweep.

## 3. Read-LoRA rank
{64, 128, 256}: default 64; mt_all_lora128 (128), f_lora128/f_lora256.
→ 128/256 did not beat 64 on cmg MC; more read capacity alone doesn't break the gxl-collapse.

## 4. Compression depth (layers used to build M)
half vs full: e_*_h vs e_*_f, q38_K64_depthF, dfull cells.
→ depth is NOT the lever (half ≈ full); half is cheaper.

## 5. Recurrent vs parallel encode (long context)
recur (e_*_hr) vs parallel (e_*_par / dd_par) vs **no-recur** (mt_all_norec).
→ recurrent prefix-carry needed for long evidence; mt_all_norec drops compress to 0.25.

## 6. Normalization of M (embedding-manifold)
{off, hard, learn, manifold}: v175fix_q38_normhard + GCM_NORM sweep.
→ hard-normalize-to-embedding-manifold is the stable default; off destabilizes.

## 7. Data-mix (what M is trained on)
single vs mix: m_aug_only / m_rca_only / m_distill_* ; mc_aug / mc_augrca / mc_distillaug / mc_allthree ; mt_all_*.
→ mixing distill+aug+rca (mt_all_distill1) is best on cmg MC; data-mixing did NOT break the GEN gxl-collapse.

## 8. Anti-collapse decoding (GEN)
greedy → letter/word spam. Fix: repetition_penalty + no_repeat_ngram_size. Tuned **1.2 / off** (1.3+ngram3 = rambling). collapse_rate 0.0 after fix.

## 9. Gate signal (keep-M-vs-fallback)
{margin, conf, neg_entropy, neg_recon, dlogit} swept per run → best usually **dlogit / margin**; gated_acc ≈ 0.42–0.50 (picks full when M is weak).

## 10. GEN regime ceiling (cmg_distill, free-form)
Across K{128–512} × recon{0–0.8} × steps{2000–5000} × distill: **module_strict ≤ 0.21**, degenerate constant-module ("gxl") mode-collapse, mem_effect ~0.86 (M changes surface text, not the verdict). → free-form module-ID hit an architectural ceiling; MC framing is the way to show value.

## 11. RUNNING — OpenRCA compress-gap sweep (g_*, 8 GPUs)
Target: close full(0.80) − compress(0.18) on OpenRCA. K{256,512} × LoRA{64,128,256} × contrast{0,0.5} × {mix, 5k-steps}: g_anchor / g_k512 / g_lora256 / g_con / g_capcon / g_lora128 / g_mix / g_long. Eval on rca_openrca; best → hot-swap into demo.
