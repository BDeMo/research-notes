# v1.7.3 — Reframed thesis + experiment plan (robustness layer, not compression)

## Thesis (the actual selling point)
**The contribution is NOT a better compressor. It is a compressor-agnostic ROBUSTNESS layer**: an external **confidence / self-verification signal** that, per item, **detects when compression has gone wrong** and **falls back to full context** — a do-no-harm guarantee. Compression is inherently lossy and sometimes fails catastrophically (our own §2.6: it fails at high compression ratio / long context). *Even given a better compressor, you still need this:* a signal to answer **"what if the compression is bad? how do we detect it, and how do we recover?"**

This flips our weak compression numbers into the **motivation**: because compression is unreliable, a detect-and-fallback layer is necessary. The positive results are then: (a) the gate **detects** failures, (b) the gated system stays **≈ full** (never catastrophically below), (c) this works **across compressors** and **across compressor quality**.

Reviewer concerns this addresses: R1/R2 (we no longer claim compression wins or saves compute as the headline), R6 (the gate is the contribution, scoped + improved), R4 (the gate is a *compressor-agnostic* method, evaluated on Cartridge/Gist/GCM, not just ours).

## Experiments (solve one by one)

| id | experiment | claim it supports | needs | status |
|---|---|---|---|---|
| **E1** | **Compressor-agnostic gate**: add a base-confidence signal computable for ANY compressor; apply the gate to **Cartridge / Gist / GCM** (later LLMLingua). Report failure-detection AUROC + gАcc + coverage per compressor. | "our robustness layer works on top of any compressor" | code (generic confidence) + re-run cart/gist w/ signals | **build now** |
| **E2** | **Robustness vs compressor quality**: use the K-sweep (K=16→512 = bad→good compressor) as a quality spectrum; show **gАcc ≈ full flat across quality**, while always-compress swings. | "even a bad/strong compressor + gate = robust" | K-sweep (running) + overlay | running |
| **E3** | **Value of gating**: harm the gate prevents = `full − always_compress` (mean + worst-case tail); gate recovers it. | "without the gate, compression can be catastrophic" | analysis | analysis |
| **E4** | **Confidence calibration**: reliability curve — when the signal says "trust", how often is compress ≥ full? | "the confidence score is calibrated/meaningful" | analysis | analysis |
| **E5** | **Best detector**: combine signals (neg_recon + conf + margin + dcode), supervised vs intrinsic vs trivial perplexity. Which detects failures best? | "the self-verification signal beats trivial confidence" | analysis (logreg over logged signals) | analysis |
| **E6** | **Where the gate matters most**: long-ctx / high-ratio (toolace, rca, K-sweep high-K) — always-compress tanks, gated ≈ full. | "the gate earns its keep exactly where compression is unreliable" | running + analysis | running |
| **E7** | **Gate on a STRONG external compressor (LLMLingua)** — does the robustness layer add value even to SOTA prompt compression? | "compressor-agnostic, including SOTA" | LLMLingua integration | later |

## Key metric reframe
- **Failure-detection AUROC**: signal's AUROC for `y = 1[compress < full]` (did compression hurt this item?). This is THE detector metric.
- **gАcc (held-out CV)** vs **always-compress** vs **full**: gated should sit at ≈full, always-compress can be far below. The **gap (gАcc − always_compress)** = robustness delivered.
- **Coverage at do-no-harm**: how much we can still compress while staying ≥ full−ε.

## Honest guardrail
The gate only *adds efficiency* where the signal has AUROC > 0.5 (today: live benches). Where the signal is chance, robustness is delivered by **conservative always-fallback** (= full) — still do-no-harm, but no compute win. E5 (better detector) is how we extend the value-add beyond the live benches.
