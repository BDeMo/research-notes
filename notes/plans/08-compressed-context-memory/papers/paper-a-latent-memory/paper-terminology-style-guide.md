# Paper A terminology and first-use rules

> Use plain, exact language. Define a technical term the first time it appears in both the Abstract and the
> main paper. Keep an acronym only if it is used at least three times.

| term | first-use wording | later wording |
|---|---|---|
| prefill | “the initial pass that reads the prompt, called prefill” | prefill |
| KV cache | “the attention keys and values stored for earlier tokens, called a key–value (KV) cache” | KV cache |
| soft memory | “a short sequence of continuous vectors that replaces raw tokens” | soft memory |
| frozen model | “a model whose main pretrained weights stay fixed” | fixed/frozen base |
| adapter | “a small set of trainable weight updates” | adapter |
| LoRA | “a low-rank adapter (LoRA), a small set of trainable weight updates” | LoRA |
| bounded raw | “the raw tokens that fit the selected hardware or protocol budget” | bounded raw |
| held-out | “examples not used to train the model or fit the predictor” | held-out |
| GCM | “the full framework: recurrent compressor, read adapter, confidence gate, and raw fallback” | GCM |
| compressor without gate | “answer from compressed memory on every input” | Compressor (w/o gate) |
| compressor with gate | “use compressed memory when accepted; otherwise use bounded raw fallback” | Compressor (w/ gate) |
| routing | “the gate chooses between the compressed and raw paths” | routing |
| coverage | “the fraction of examples sent to memory” | memory coverage |
| fallback AUROC | “how well low memory confidence ranks examples where raw scores higher than memory” | fallback AUROC |
| fallback rate | “the fraction of test examples sent to bounded raw context” | fallback rate |
| harm | “raw succeeds but memory fails, or the raw score exceeds the memory score” | compression harm |
| rescue | “memory succeeds but raw fails” | memory rescue |
| SFT | “supervised fine-tuning (SFT) of the raw-context path” | SFT |
| BFCL | “Berkeley Function Calling Leaderboard (BFCL), a tool-use benchmark” | BFCL |
| QuALITY | “QuALITY, a long-document multiple-choice benchmark” | QuALITY |
| RULER NIAH | “the RULER needle-in-a-haystack (NIAH) exact-retrieval task” | RULER / NIAH |
| MLP | “multilayer perceptron (MLP)” | MLP |
| RoPE | “rotary position embeddings (RoPE)” | RoPE |
| GDN | “gated delta network (GDN) recurrent-attention layer” | GDN |
| KL | “Kullback–Leibler (KL) divergence” | KL |
| ROC-AUC | “area under the receiver operating characteristic curve (ROC-AUC)” | ROC-AUC |
| RAG | “retrieval-augmented generation (RAG)” | RAG |

## Required wording

- Use **bounded raw** when the path can truncate input.
- Use **true raw** only when every audited source token is read.
- Use **held-out empirical routing** for the current gate.
- State that the formal test certifies **0/24 groups** and returns all-raw.
- Describe cross-model results as “independent adapters on the same algorithm,” not shared embedding space.
- Describe GCM memory as `S×K`, not a fixed 128-token document representation.

## Forbidden or conditional wording

- Do not write “full context” for an 8k/4k bounded path.
- Do not use `GCM` and `Route` as contrasting method rows; write `Compressor (w/o gate)` and `Compressor (w/ gate)`.
- Do not write “risk-controlled GCM” or “finite-sample guarantee” for the current result.
- Do not write “model-agnostic” before the fixed-config seven-base grid finishes.
- Do not write “lossless,” “safe,” or “never worse” without the exact measured condition.
- Do not call internal replicas by the published method name.
- Do not call a lower reader-token count an end-to-end speedup without encoder and fallback timing.

## Style check

Before merging a section:

1. Search for every capitalized acronym.
2. Check that its first occurrence is expanded or replaced.
3. Replace vague adjectives with numbers or direct comparisons.
4. Check that “raw,” “bounded raw,” and “true raw” are used consistently.
5. Mark pilot, current, pending, invalid, and excluded evidence explicitly.
6. In result tables, bold only the best value among methods being directly compared. Never bold raw/SFT
   references, compression ratios, coverage, fallback rate, AUROC, or deltas.
7. Report task scores, coverage, fallback rates, and AUROC on a 0–100 scale. Report score differences as
   percentage points (`pp`), not decimal fractions such as `.035`.
