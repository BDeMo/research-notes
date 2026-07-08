# Long-Context LLM — References Knowledge Base (for Paper B v2.0.0)

> Curated reference library for the v2.0.0 reframe (**long-context necessity + domain generalization**, not efficiency).
> Each entry: **Name (authors, year) [arXiv/venue]** — one-line takeaway · *→ relevance to GCM (our compressed-memory + do-no-harm-gate method)*.
> Tags: `[bench]` `[soft-compress]` `[hard-compress]` `[memory/recur]` `[kv-evict]` `[pos]` `[lin-attn/ssm]` `[rag]` `[domain-gen]` `[gate/route]` `[probe/interp]` `[arch]`.
> Status: v1 (2026-06-24, ~115 entries). Grounded by web search for 2024-2026; classics from prior knowledge. ⚠ verify exact arXiv IDs before citing.

---

## A. Long-context benchmarks & evaluation `[bench]` — *what "long-context" means + how we'll measure v2.0.0*
1. **RULER (Hsieh et al., 2024) [COLM, arXiv:2404.06654]** — configurable synthetic NIAH/multi-hop/aggregation to measure a model's *effective* context size; perf degrades well before the advertised window. *→ our headline length-sweep figure: compression vs truncation/window as L grows.*
2. **∞Bench / InfiniteBench (Zhang et al., 2024) [arXiv:2402.13718]** — 12 tasks, avg >100k tokens, requires reasoning beyond retrieval. *→ over-window regime where input "doesn't fit".*
3. **LongBench (Bai et al., 2023) [arXiv:2308.14508]** — 21 tasks (QA/summ/retrieval/code), ~10k, bilingual; the standard first-gen suite. *→ task breadth.*
4. **LongBench v2 (Bai et al., 2025) [ACL'25]** — 503 human MC, 8k–2M words, expert acc only 53.7%; *deep reasoning*, not retrieval. *→ the "shallow retrieval ≠ long-context understanding" point.*
5. **HELMET (Yen et al., 2025)** — 7 application-centric categories at controlled lengths; argues most suites mis-rank models. *→ evaluation rigor for v2.*
6. **NoCha — "1001 pairs" (Karpinska et al., 2024) [EMNLP'24]** — book-length true/false minimal pairs on *recent* fiction (anti-leakage); NIAH success ≠ NoCha success. *→ global reasoning over long docs; leakage control.*
7. **BABILong (Kuratov et al., 2024) [NeurIPS'24]** — reasoning over facts hidden in up to 10M+ tokens of distractors. *→ extreme-length recurrence eval.*
8. **L-Eval (An et al., 2023) [arXiv:2307.11088]** — 4k–60k, 18 tasks, better metrics for open-ended long QA.
9. **ZeroSCROLLS / SCROLLS (Shaham et al., 2022/2023)** — long-doc summarization/QA (GovReport, QMSum, Qasper, NarrativeQA, QuALITY).
10. **LooGLE (Li et al., 2024) [ACL'24]** — short- vs long-dependency split; tests genuine long-range dependency.
11. **NovelQA / NovelQA (Wang et al., 2024)** — QA over full novels (>100k).
12. **NarrativeQA (Kočiský et al., 2018) [TACL]** — QA over full books/scripts (~14–18k tok in our loader). *→ our genuine long-doc QA bench.*
13. **QuALITY (Pang et al., 2022) [NAACL]** — long literary MC (~5k tok). *→ `quality`, our current long-ctx win (compress 3× truncated-full).*
14. **NIAH / Needle-in-a-Haystack (Kamradt, 2023)** — single-fact retrieval probe; necessary-not-sufficient. *→ baseline diagnostic; we go beyond it.*
15. **HashHop (Magic, 2024)** — multi-hop hash chains, leakage-proof long-context.
16. **Michelangelo / Latent-list (Vodrahalli et al., 2024)** — synthetic latent-structure long reasoning.
17. **LV-Eval (Yuan et al., 2024)** — controllable 16k–256k with distractor injection + keyword-recall metrics.
18. **Loong (Wang et al., 2024)** — multi-doc QA forcing use of *all* docs (no single-doc shortcut).
19. **DENIAHL / counting-stars / variable tracking** — RULER-style aggregation/multi-needle variants.
20. **Long-Range Arena (Tay et al., 2020) [ICLR'21]** — original efficiency-era long-seq classification suite (text/image/math). *→ historical context.*

## B. Soft-prompt / learned context compression `[soft-compress]` — *our direct family*
21. **Gisting (Mu et al., 2023) [NeurIPS]** — trainable "gist" tokens + gist-masking compress *instructions*; short prompts only. *→ our `gist-lite` baseline; we extend to long ctx + gate.*
22. **AutoCompressor (Chevalier et al., 2023) [EMNLP]** — recurrent segment summary vectors as soft prompts; needs many summary tokens. *→ our recurrent over-window encode is AutoCompressor-style (detached carry).*
23. **ICAE — In-Context AutoEncoder (Ge et al., 2024) [ICLR]** — LoRA encoder compresses ctx → memory slots, frozen decoder reads; ~4×. *→ closest architecture to GCM (frozen base + adapter + memory slots).*
24. **500xCompressor (Li et al., 2025) [ACL]** — LoRA compresses to as few as 1 token by feeding KV to decoder; extreme ratios. *→ extreme-compression reference; our K=128–256 is milder/lossier-honest.*
25. **xRAG (Cheng et al., 2024) [NeurIPS]** — project a retrieved doc to ONE token via a modality bridge; RAG-specific. *→ extreme single-token compression baseline.*
26. **CompAct (Yoon et al., 2024)** — active compression of retrieved docs into concise context.
27. **Compressed Context Memory / CCM (Kim et al., 2024) [ICLR]** — online key-value memory compression for streaming dialogue.
28. **UniICL / SoftPromptComp (various, 2024)** — unified demonstration+context compression for ICL.
29. **Selective-Context (Li et al., 2023) [EMNLP]** — drop low-self-information lexical units (bridge to hard-compress).
30. **Nugget / Nugget2D (Qin & Van Durme, 2023/2024)** — learned token "nuggets" select & compress salient positions.
31. **Cartridges / self-study (Eyuboglu et al., 2025)** — train a *fixed* KV "cartridge" per corpus offline (amortized, input-independent). *→ our `cart` baseline; great contrast (amortizable vs per-input).*
32. **LLoCO (Tan et al., 2024)** — learn compressed context offline + LoRA for long-doc QA.
33. **PCC / Prompt-Cache (Gim et al., 2024) [MLSys]** — reuse precomputed attention states of repeated prompt modules.
34. **SnapKV-as-compression / Activation Beacon (Zhang et al., 2024)** — "beacon" tokens compress activations into a sliding global memory (in-stream). *→ in-stream global memory variant.*
35. **Mooncake/Context-caching (2024)** — system-level KV reuse (serving-side compression).

## C. Hard prompt compression / token pruning `[hard-compress]`
36. **LLMLingua (Jiang et al., 2023) [EMNLP]** — small-LM perplexity prunes tokens; coarse-to-fine. *→ training-free hard baseline.*
37. **LongLLMLingua (Jiang et al., 2024) [ACL]** — question-aware + reorder to fight position bias for long ctx.
38. **LLMLingua-2 (Pan et al., 2024) [ACL Findings]** — token-classification distillation (bidirectional), faster + better generalization in/out-of-domain. *→ strong hard-compress + the in/out-of-domain framing.*
39. **RECOMP (Xu et al., 2024) [ICLR]** — extractive+abstractive compressors for RAG.
40. **FilCo (Wang et al., 2024)** — filter retrieved context by lexical/learned utility.
41. **TCRA / Provence (2024-2025)** — pruning retrieved passages with reranker signals.

## D. Memory & segment-level recurrence `[memory/recur]` — *our over-window mechanism's lineage*
42. **Transformer-XL (Dai et al., 2019) [ACL]** — segment-level recurrence: cache previous-segment hidden states; relative pos. *→ our TXL/StreamingLLM baseline ancestor.*
43. **Compressive Transformer (Rae et al., 2020) [ICLR]** — compress old memories into coarser slots (PG-19). *→ compress-old-context lineage.*
44. **Memorizing Transformers (Wu et al., 2022) [ICLR]** — kNN-retrieve old KV from a large external memory at one layer.
45. **Recurrent Memory Transformer / RMT (Bulatov et al., 2022) [NeurIPS]** — pass memory tokens between segments; scales to 1M+ via BPTT. *→ recurrent memory-token carry (our chunk carry).*
46. **Associative RMT / ARMT (Rodkin et al., 2024) [arXiv:2407.04841]** — per-layer associative matrices for segment memory; BABILong to 50M. *→ memory capacity vs token-count tradeoff.*
47. **Infini-attention (Munkhdalai et al., 2024) [arXiv:2404.07143]** — compressive linear-attn memory + local attn; unbounded ctx, *constant* memory. *→ the constant-memory ideal; we instead keep a discrete K-token soft memory.*
48. **Melodi (Chen et al., 2024) [arXiv:2410.03156]** — compress middle-layer KV into short+long-term memory (8× smaller). *→ layer-selective memory compression.*
49. **TransformerFAM (Hwang et al., 2024)** — feedback attention as working memory (no extra weights).
50. **REFORM (2025) [arXiv:2506.01215]** — compress+gather+recompute hybrid of recurrence and random-access at 256k.
51. **InfLLM (Xiao et al., 2024)** — training-free block-level memory with dynamic retrieval (random access).
52. **LM-Infinite (Han et al., 2023)** — Λ-shaped mask (sinks + recent) lets short-trained models run to 200M w/o tuning.
53. **Landmark Attention (Mohtashami & Jaggi, 2023) [NeurIPS]** — landmark tokens index blocks for retrieval-style attention.
54. **Unlimiformer (Bertsch et al., 2023) [NeurIPS]** — kNN over encoder hidden states for unbounded input (enc-dec).
55. **Memory³ / explicit memory (2024)** — externalize knowledge as retrievable memory to shrink params.

## E. KV-cache eviction / sliding-window / quantization `[kv-evict]` — *bounded-memory no-compression baselines*
56. **StreamingLLM (Xiao et al., 2023) [ICLR]** — attention **sinks** (first tokens) + recent window → infinite streaming, no retrain. *→ our TXL-lite baseline (4 sinks + recent W).*
57. **H2O — Heavy-Hitter Oracle (Zhang et al., 2023) [NeurIPS]** — evict by cumulative attention score. *→ attention-based eviction baseline.*
58. **Scissorhands (Liu et al., 2023) [NeurIPS]** — persistence-of-importance eviction.
59. **SnapKV (Li et al., 2024) [arXiv:2404.14469]** — observation-window votes select clustered prefill KV. *→ prefill compression baseline.*
60. **PyramidKV / PyramidInfer (Cai/Yang et al., 2024)** — layer-wise budget (more KV in low layers): "pyramidal information funneling". *→ supports our depth-readout choice + the per-layer attention-sparsity probe.*
61. **FastGen (Ge et al., 2024) [ICLR]** — adaptive per-head KV policy (local/special/broad heads). *→ head-type taxonomy for our attention probes.*
62. **Quest (Tang et al., 2024) [ICML]** — query-aware page selection for sparse decode.
63. **Keyformer (Adnan et al., 2024) [MLSys]** — keep "key" tokens via Gumbel-softmax scoring.
64. **KIVI (Liu et al., 2024) [ICML]** — 2-bit asymmetric KV quantization. *→ orthogonal compression axis.*
65. **KVQuant / GEAR (2024)** — low-bit KV with outlier/residual handling.
66. **Ada-KV (Feng et al., 2025)** — head-adaptive budget allocation.
67. **ChunkKV (Liu et al., 2025)** — retain contiguous semantic chunks vs isolated tokens. *→ "exact-string survives in raw chunks" — our bfcl_multiple failure mode.*
68. **SCOPE (Wu et al., 2025)** — separate prefill vs decode compression.
69. **DuoAttention (Xiao et al., 2024)** — split heads into retrieval (full KV) vs streaming (window). *→ head-role split relevant to the gate.*
70. **R-KV / reasoning-trace compression (Cai et al., 2025)** — attention + key-similarity dedup for CoT.
71. **KV-compression benchmarks (Yuan et al., 2024 arXiv:2407.01527; Semantic-Integrity/ShotKV 2025)** — show big drops on reasoning/high-density tasks under compression. *→ "compression hurts exact/dense info" — supports our identifier-copy failure analysis.*

## F. Position extrapolation `[pos]` — *why raw windows are bounded; how natively-long models get there*
72. **RoPE (Su et al., 2021) [arXiv:2104.09864]** — rotary relative position; foundation. *→ RoPE relativity is why our TXL rolling-window with original positions is correct.*
73. **ALiBi (Press et al., 2022) [ICLR]** — linear attention bias for length extrapolation.
74. **Position Interpolation / PI (Chen et al., 2023) [arXiv:2306.15595]** — compress position indices into trained range + brief FT.
75. **NTK-aware / NTK-by-parts / Dynamic-NTK (bloc97, 2023)** — frequency-dependent base scaling (community-originated).
76. **YaRN (Peng et al., 2023) [arXiv:2309.00071]** — NTK-by-parts + attention temperature; the standard 8–32× extension (Llama-3.1, used by Qwen/DeepSeek/gpt-oss). *→ how our base hits 128k–256k; our deployment used max-model-len 262144.*
77. **LongRoPE (Ding et al., 2024) [ICML]** — evolutionary per-dim scaling, progressive 4k→2M. *→ extreme extension reference.*
78. **CLEX (Chen et al., 2024) [ICLR]** — continuous length extrapolation via ODE on scaling.
79. **Self-Extend (Jin et al., 2024) [ICML]** — grouped attention for training-free extension.
80. **LongLoRA (Chen et al., 2024) [ICLR]** — shifted-sparse-attn + LoRA for cheap long-ctx FT. *→ cheap long-ctx adaptation, contrast to our frozen-base + memory.*

## G. Efficient / linear attention & state-space models `[lin-attn/ssm]` — *our base (Qwen3.5) lives here*
81. **S4 (Gu et al., 2022) [ICLR]** — structured state space; long-range with sub-quadratic cost.
82. **Mamba (Gu & Dao, 2023) [COLM]** — selective SSM, data-dependent gating, KV-cache-free, linear time. *→ the SSM line our base hybridizes.*
83. **Mamba-2 (Dao & Gu, 2024) [ICML]** — SSD: SSMs ≈ structured masked attention; faster.
84. **DeltaNet (Schlag et al., 2021; Yang et al., 2024)** — delta-rule linear transformer; precise KV association but no fast erase.
85. **Gated DeltaNet / GDN (Yang et al., 2024) [ICLR'25, arXiv:2412.06464]** — **gating (erase) + delta rule (targeted update)**; beats Mamba2/DeltaNet on long-ctx + length extrapolation. *→ THIS IS Qwen3.5's linear-attn (`gdn_prefill` kernel from our vLLM crash); explains why our base is KV-cache-free → KV-injection N/A → soft-prompt prefix only, TXL N/A on it.*
86. **RWKV-4/5/6 (Peng et al., 2023/2024)** — RNN-attention hybrid, channel-wise decay.
87. **RWKV-7 "Goose" (Peng et al., 2025)** — data-dependent low-rank (diagonal+low-rank) transition, delta-like.
88. **RetNet (Sun et al., 2023)** — retention with fixed decay; parallel/recurrent/chunkwise.
89. **GLA — Gated Linear Attention (Yang et al., 2024) [ICML]** — hardware-efficient gated linear attn.
90. **HGRN-1/2 (Qin et al., 2023/2024)** — hierarchically gated linear RNN.
91. **Jamba (Lieber et al., 2024)** — Mamba+Transformer+MoE hybrid block.
92. **Kimi Linear / KDA (2025)** — 3:1 linear:full-attention hybrid surpassing full attention at scale. *→ hybrid trend; our base is similar hybrid.*
93. **FG²-GDN (2026) [arXiv:2604.19021]** — doubly fine-grained control over GDN for long ctx. *→ frontier on our base's family.*
94. **Longformer / BigBird (Beltagy 2020; Zaheer 2020)** — sparse (local+global/random) attention; classic efficiency line.

## H. Retrieval-augmented vs long-context `[rag]`
95. **RAG (Lewis et al., 2020) [NeurIPS]** — retrieve-then-read; the alternative to stuffing context. *→ "compress vs retrieve" framing.*
96. **REALM / Atlas / RETRO (2020–2022)** — retrieval-pretrained / chunked cross-attention.
97. **Self-RAG (Asai et al., 2024) [ICLR]** — model decides when/what to retrieve + critique. *→ a gating analogue (route to retrieval); contrast with our do-no-harm gate.*
98. **"Lost in the Middle" (Liu et al., 2024) [TACL]** — U-shaped position bias: mid-context evidence is under-used. *→ motivates compression (re-surface mid evidence) + a probe (does M fix mid-context loss?).*
99. **RAG vs Long-Context (Xu et al., 2024; Li et al., 2024)** — when retrieval beats long-ctx and vice-versa. *→ scoping our claim.*
100. **OP-RAG / order-preserve (2024)** — order matters for long-ctx RAG.

## I. Domain generalization / transfer / forgetting `[domain-gen]` — *the v2.0.0 second pillar*
101. **Latent Traits & Cross-Task Transfer (2025) [arXiv:2509.13624]** — OOD transfer driven by *hidden* stats (output-length dist, label imbalance, linguistic sensitivities), NOT surface domain similarity; PCA "traits" across LoRA adapters. *→ directly motivates our cross-domain *intrinsic* analysis (features, not labels).*
102. **SFT Doesn't Always Hurt (2025) [arXiv:2509.20758]** — domain SFT degrades general caps; small-LR / TALR mitigate but don't eliminate. *→ our SFT-LoRA baseline's overfit/forgetting; "don't touch weights" framing.*
103. **Synergy-over-Discrepancy partitioning (2025) [arXiv:2511.07198]** — cluster synergistic domains, stage FT to avoid negative transfer + generalization bounds. *→ multi-domain training (our mix/curriculum).*
104. **SAE as a Crystal Ball / STS (2026) [arXiv:2603.02908]** — SAE-feature "transferability score" predicts cross-domain SFT gains *without training* (Pearson>0.7). *→ predict where compression transfers; an intrinsic predictor probe.*
105. **Catastrophic forgetting in LLM FT (Luo et al., 2023; Kotha et al., 2024)** — task/format forgetting after FT. *→ paper-B "forgetting" theme; gate avoids weight edits.*
106. **Cross-domain ICL / abstraction (2024-2025)** — larger models abstract reasoning structure; small models suffer structural mismatch. *→ base-size × transfer interaction (our cross-model study).*
107. **Task vectors / model arithmetic (Ilharco et al., 2023) [ICLR]** — edits as vectors in weight space. *→ contrast: ours edits *input* (memory), not weights.*
108. **DoLa / activation steering (2024)** — intrinsic representation interventions. *→ probe toolkit.*
109. **Domain-shift detection / OOD scores (energy, Mahalanobis; Lee 2018, Liu 2020)** — feature-distribution OOD detectors. *→ methods for our cross-vs-in-domain feature-distribution probes.*
110. **Wikitext/Pile domain-shift perplexity studies** — classic in/out-domain LM generalization. *→ baseline framing.*

## J. Gating / routing / cascades / calibration `[gate/route]` — *our contribution's neighbors*
111. **Model cascades / FrugalGPT (Chen et al., 2023)** — route easy queries to cheap models, hard to expensive. *→ our 3-way cost-ascending gate is a within-model cascade.*
112. **Speculative decoding (Leviathan/Chen, 2023)** — cheap draft + verify; quality-preserving. *→ "do-no-harm via verification" analogue.*
113. **Confidence/calibration for selective prediction (Hendrycks 2017; Geifman 2017)** — abstention/threshold theory. *→ our gate τ; *held-out* threshold selection (our CV fix) is the selective-prediction-done-right point.*
114. **Conformal prediction (Angelopoulos & Bates, 2023)** — distribution-free coverage guarantees via held-out calibration. *→ principled fix to make gated≥full hold out-of-sample (recommended for v2 gate).*
115. **Mixture-of-Depths / early-exit (Raposo 2024; Schuster 2022 CALM)** — adaptive compute per token. *→ adaptive-cost lineage.*
116. **Router/MoE load-balancing (Shazeer 2017; Fedus 2022)** — learned routing. *→ gate-as-router framing.*
117. **TARG / training-free uncertainty gating (base-margin)** — no-context base confidence as a gate. *→ our honest `targ` baseline signal.*

## K. Interpretability & probing `[probe/interp]` — *toolkit for the cross-domain vs in-domain intrinsic study*
118. **Attention is not Explanation / is Explanation (Jain 2019; Wiegreffe 2019)** — caveats on attention-as-evidence. *→ caution for our attention-distribution probes.*
119. **Induction heads & in-context learning (Olsson et al., 2022)** — circuits that copy/complete; key for retrieval. *→ probe: do in-domain vs cross-domain differ in induction-head usage?*
120. **Retrieval heads (Wu et al., 2025)** — a small set of heads do long-ctx retrieval; ablating them breaks NIAH. *→ probe: are retrieval heads engaged differently in/out-of-domain & under compression?*
121. **Attention sinks / massive activations (Xiao 2023; Sun 2024)** — sink tokens & huge-norm activations. *→ probe: sink mass shift across domains; M-token norms vs embedding manifold.*
122. **Logit Lens / Tuned Lens (nostalgebraist 2020; Belrose 2023)** — decode intermediate layers. *→ probe: where does M's answer info localize by depth?*
123. **Sparse Autoencoders / dictionary learning (Bricken 2023; Cunningham 2023)** — monosemantic features. *→ probe: do in/out-domain activate disjoint SAE features? (ties to STS #104).*
124. **CKA / SVCCA representation similarity (Kornblith 2019; Raghu 2017)** — compare hidden geometry across conditions. *→ probe: CKA(in-domain hidden, cross-domain hidden); CKA(full-ctx hidden, M-induced hidden).*
125. **Probing classifiers / intrinsic dimension (Hewitt 2019; Aghajanyan 2021)** — linear decodability / ID of representations. *→ probe: effective dim of M vs full-ctx by domain.*
126. **Activation patching / causal tracing (Meng et al., 2022 ROME)** — locate causal info flow. *→ probe: where compressed-context info enters the residual stream.*
127. **Entropy / attention dispersion across layers (PyramidKV §3; many)** — attention sparsifies with depth. *→ probe: in vs cross-domain attention-entropy curves.*

---

## L. Foundational efficiency & architecture spine (citation anchors) `[arch][lin-attn/ssm][pos]`
*Added from the two 2025 surveys' citation lineages — these are the most-cited "spine" works the field builds on.*
128. **Transformer (Vaswani et al., 2017) [NeurIPS]** — self-attention; absolute position embedding. The root.
129. **Relative position (Shaw et al., 2018; T5 Raffel 2020)** — distance-based position; precursor to RoPE.
130. **Sparse Transformer (Child et al., 2019)** — factorized sparse attention; first big O(L²) cut.
131. **Reformer (Kitaev et al., 2020) [ICLR]** — LSH attention + reversible layers.
132. **Linformer (Wang et al., 2020)** — low-rank attention → linear.
133. **Performer (Choromanski et al., 2021) [ICLR]** — FAVOR+ kernel linear attention.
134. **xPos (Sun et al., 2022)** — exponential-decay RoPE for extrapolation + windowed BCA.
135. **LongNet (Ding et al., 2023)** — dilated sliding-window attention → claims 1B tokens.
136. **ScalingRoPE (Liu et al., 2024)** — identifies a *critical dimension* setting RoPE's extrapolation limit. *→ theory of why windows are bounded.*
137. **FlashAttention 1/2/3 (Dao et al., 2022/2023; Shah et al., 2024)** — IO-aware exact attention; FA-3 tuned for H100. *→ why "full" within-window is cheap on our H100; the kernel layer under everything.*
138. **Ring Attention / Blockwise (Liu et al., 2023)** — sequence-parallel exact attention across devices → very long ctx in training.
139. **Native Sparse Attention / NSA (MiniMax/DeepSeek-style, 2025)** — hardware-aligned trainable dynamic token compression+selection; **hot 2025 frontier**. *→ "trainable sparsity" rival to compression.*
140. **MoBA (2025)** — mixture-of-block-attention, learned block routing for long ctx.
141. **MInference (Jiang et al., 2024) [NeurIPS]** — dynamic sparse *prefill* (A-shape/vertical-slash/block) for 1M ctx. *→ training-free long-ctx speedup.*
142. **GQA / MQA (Ainslie 2023 / Shazeer 2019)** — grouped/multi-query KV sharing; the standard KV-shrink in modern LLMs.
143. **MLA — Multi-head Latent Attention (DeepSeek-V2/V3, 2024)** — low-rank latent KV compression; big serving win. *→ architectural KV-compression (vs our token compression).*
144. **YOCO / CLA / MiniCache (2024)** — cross-layer KV sharing/merging.
145. **HiPPO (Gu et al., 2020) → LSSL (2021) → S4D (2022) → H3 (Fu et al., 2022)** — the SSM theory lineage into language modeling. *→ roots of our base's linear-attn line.*
146. **Hyena (Poli et al., 2023)** — long convolution attention-free operator.
147. **Griffin / Hawk / RecurrentGemma (De et al., 2024)** — gated linear recurrence + local attention hybrid (shipped model).
148. **Based / Zamba / Samba / Hymba / Jamba (2024)** — the **hybrid linear+attention** wave (the dominant practical compromise). *→ our Qwen3.5 is one of these.*
149. **MiniMax-01 (2025)** — lightning (linear) attention at 456B/4M ctx; largest hybrid to date.
150. **Titans (Behrouz et al., 2024, Google)** — neural **long-term memory module learned at test time** (surprise-gated). *→ closest "learned memory" cousin to GCM; test-time memory write (our H4/H16).*
151. **DeciMamba / ReMamba (2024)** — extend Mamba's effective length via Δ-based token selection / KV-style compression.
152. **StuffedMamba (Chen et al., 2024)** — diagnoses **state collapse** in linear models at long ctx (state saturates → garbage). *→ explains our F2 gen-collapse on RULER; linear-state capacity wall.*
153. **Focused Transformer / FoT (Tworkowski et al., 2023) [NeurIPS]** — contrastive memory + kNN to fight distraction at long ctx.
154. **CREAM (Wu et al., 2024)** — middle-sampling to fix "lost-in-the-middle" during long-ctx tuning.
155. **Short-context collaboration: PCW (Ratner 2022), NBCE (Su 2023), XL3M (2024), LLM×MapReduce (2024), LongAgent (2024)** — split-and-merge / multi-agent long-ctx (training-free). *→ "divide & conquer" alternative to a single compressed memory.*
156. **PagedAttention / vLLM (Kwon et al., 2023) [SOSP]** — KV paging; the serving substrate (what we deploy on).
157. **Star Attention (2024) / APE (2025)** — block-local + global anchor attention for efficient long-ctx inference.

## M. Surveys & field-trend anchors `[survey]`
158. **Thus Spake Long-Context LLM (Liu et al., 2025) [arXiv:2502.17129]** — full-lifecycle survey: **Architecture (length-extrapolation · KV-opt · memory · arch-innovation) + Infrastructure + Training + Evaluation** + 10 unanswered questions. *→ the field's own consensus taxonomy (basis of the main-thread judgment below).*
159. **A Survey of Context Engineering (2025) [arXiv:2507.13334]** — agent-era reframe: context retrieval/generation · processing · management · RAG · memory systems. *→ the application/agent main thread.*
160. **MECW — "Context Is What You Need" (2025) [arXiv:2509.21361]** — *Maximum Effective Context Window* ≪ advertised; effective length is task-dependent and can collapse near 0 with enough distractors. *→ quantifies "long ≠ usable"; core motivation.*
161. **Context-rot / context-discipline (2026) [arXiv:2601.11564]** — non-linear degradation + "context tax" (719% latency at 15k) under distractors; MoE anomalies. *→ cost+quality both degrade → necessity for compression-or-discipline.*
162. **Efficient Transformers survey (Tay et al., 2022) / earlier LC surveys (Huang 2023, Pawar 2024, Dong 2024)** — the prior survey lineage.
163. **Lost-in-the-Middle (Liu et al., 2024) [TACL]** — (also #98) U-shaped position usage; the most-cited "long≠usable" datapoint.

---

## ★ MAIN-THREAD JUDGMENT (citation-weighted + trend, 2026)
*Evidence: the consensus taxonomy of survey #158 (its §2–§5 = the architecture spine), citation-density of the lineages above, and the 2025–2026 trend papers (#139, #149, #150, #160, #161).*

**The spine, by citation density (most-built-on → niche):**
1. **Length extrapolation (position)** — RoPE→PI→NTK→**YaRN**(now *standard*, Llama-3.1/Qwen/DeepSeek)→LongRoPE. **Maturing/"solved-enough"**: heat cooling; it answers *"can the tokens go in"* but not *"are they used."*
2. **Efficient/linear attention + SSM, trending to HYBRID** — Sparse/Linformer/Performer → S4/H3/**Mamba** → **Gated-DeltaNet/RWKV-7** → **hybrid linear+full** (Jamba/Samba/Griffin/MiniMax-01; our Qwen3.5). **Hottest architectural thread 2025–26** (NSA, MiniMax-01). Known wall: **state collapse / finite-state capacity** (#152) → exact retrieval/copy is weak (matches our F1/F2).
3. **KV-cache optimization** — StreamingLLM/H2O/SnapKV/PyramidKV + sharing (GQA/MLA/YOCO) + quant (KIVI). **Hot, practical/serving-driven.** Known wall: hurts reasoning/high-density (matches our exact-string finding).
4. **Memory / context compression** (our family) — soft (Gist/ICAE/AutoCompressor/500x) + recurrent (RMT/Infini-attention/**Titans**). **Smaller, active niche**; the survey files it under "Memory Management." Known wall: **lossy → exact info + generalization** (our exact-niche).
5. **Evaluation** — **rising fast**: NIAH→RULER→∞Bench→NoCha→LongBench-v2→HELMET + MECW/context-rot. The "long≠usable" realization is *driving* the whole field's redirection.

**Trend verdict (where the heat is moving):** from **"extend the window" (done, via YaRN)** → to **"make long context efficient AND actually usable"** = (a) **hybrid linear/sparse architectures** (the model-building frontier), (b) **KV/serving optimization** (the deployment frontier), and (c) **harder evals + context-rot diagnosis** (the science frontier). **Pure context-compression-as-a-method is a *niche on the spine*, not the main current** — its open problems are exactly **lossiness (exact info), faithfulness, and cross-domain generalization**.

**Where this puts Paper B (the under-served corner):** the main thread optimizes *efficiency on tasks that already fit*; almost nobody owns **(i) necessity** (input physically can't fit → must compress), **(ii) cross-domain generalization of the compressed representation**, **(iii) provable do-no-harm (out-of-sample/conformal)**. All three are explicitly listed as open/under-explored. Our base being a **Gated-DeltaNet hybrid (thread #2)** with **state-collapse limits (#152)** is the architectural hook: on a KV-free linear model, a learned soft/compressed memory is the *natural* long-context interface, and the gate is the safety contract. → **v2.0.0 should sit at the intersection of thread #2 (hybrid linear) and thread #4 (memory/compression), claiming the necessity + generalization corner the spine leaves open.**

---

## Synthesis for v2.0.0 (how the field positions us)
- **The honest gap we fill:** prior compression (Gist/ICAE/AutoCompressor/500x/xRAG) sells **efficiency/ratio**; KV-evict & TXL/StreamingLLM sell **bounded memory**. Almost none ask *when compression is **necessary** (input physically won't fit / window truncates the evidence)* and *whether the compressed representation **generalizes across domains***. v2.0.0 = **necessity + domain generalization**, with a **do-no-harm gate** validated *out-of-sample* (conformal / held-out, per #113–114), not the discredited in-sample "≥ full".
- **Our base is a Gated-DeltaNet hybrid (#85):** KV-cache-free linear attention ⇒ KV-injection/TXL-window baselines are *architecturally N/A* on it ⇒ **soft-prompt compressed memory is the natural (only) interface** → a clean architectural motivation for our approach (unlike dense models where StreamingLLM/H2O are available).
- **Evaluation pivot:** lead with **RULER length-sweep + QuALITY/NarrativeQA/NoCha/∞Bench/LongBench-v2** (necessity), and **cross-domain transfer matrices + intrinsic probes** (§K) for generalization. Drop efficiency tables to the appendix.
- **Closest baselines to name explicitly:** ICAE (#23), AutoCompressor (#22), Gisting (#21), 500xCompressor (#24), Cartridges (#31) [soft]; LLMLingua-2 (#38) [hard]; StreamingLLM (#56)/H2O (#57)/SnapKV (#59) [KV/TXL]; RMT (#45)/Infini-attention (#47) [recurrent memory].
