# Long context

> Architectures, training recipes, benchmarks, and limits of using **long input sequences** directly (typically 16K – 1M tokens). The original method for "giving the model more information at inference" before the $W$-axis ideas in [`inference-time-training`](../inference-time-training/) became viable.

## Structure

**Contained by**: (top-level).

**Contains**:
- **Architectural extensions** — RoPE scaling, YaRN, position interpolation, ALiBi, attention sinks.
- **Compression / KV-cache reduction** — quantized KV, sparse / strided attention, sliding window.
- **Prompt / token compression** — LLMLingua, gisting, summarization.
- **Activation / hidden-state compression** — Activation Beacon, AutoCompressors.
- **Linear-attention / state-space** — Mamba, RWKV, retention. Provide $\mathcal{O}(N)$ context but with different inductive bias.
- **Benchmarks** — LongBench, RULER, NIAH variants, ZeroSCROLLS.
- **Context distillation as alternative** — see [`context-distillation/`](../context-distillation/).

## Nearest neighbors

- [`context-distillation`](../context-distillation/) — the leading $W$-axis alternative to long context.
- [`inference-time-compute`](../inference-time-compute/) — long context is one form of $X$-scaling.

## Key concepts

- **Quadratic attention cost** — vanilla attention is $\mathcal{O}(N^2)$ in sequence length, the root cause of long-context expense.
- **KV cache size** — at inference, the KV cache scales linearly with context length and model dimension. 12 GB for a 128K-token cache on a 7B model is a useful order-of-magnitude.
- **Lost-in-the-middle** — even when a model technically supports long context, accuracy on information in the middle of the input drops substantially.
- **Context rot** — performance degrades roughly *monotonically* with input length in many regimes (Chroma 2025), not just middle-bias.
- **Effective vs nominal context length** — RULER and similar benchmarks find that "100K context" models often only use ~10–30K effectively.

## Foundational references

- Su, J. et al. (2024). **RoFormer / RoPE.** [Neurocomputing 568.](https://arxiv.org/abs/2104.09864)
- Press, O., Smith, N. A., Lewis, M. (ICLR 2022). **ALiBi.**
- Peng, B. et al. (2024). **YaRN: Efficient Context Window Extension.**
- [lost-mid] Liu, N. F. et al. (TACL 2024). **Lost in the Middle.**
- Xiao, G. et al. (2024). **Efficient Streaming Language Models with Attention Sinks.**
- Chen, S. et al. (2023). **Extending Context Window of Large Language Models via Position Interpolation.**
- [ruler] Hsieh, C.-P. et al. (COLM 2024). **RULER.**
- [longbench] Bai, Y. et al. (2023). **LongBench.**
- [context-rot] Hong, K., Troynikov, A., Huber, J. (Chroma 2025). **Context rot.**
- [li-tmlr] Li, T. et al. (TMLR 2025). **Long-context LLMs struggle with long in-context learning.**
- Pan, Z. et al. (ACL Findings 2024). **LLMLingua-2.**
- Mu, J., Li, X., Goodman, N. (NeurIPS 2024). **Gisting.**
- Zhang, P. et al. (ICLR 2025). **Activation Beacon.**
- Cao, B., Cai, D., Lam, W. (2025). **InfiniteICL.** [arXiv:2504.01707](https://arxiv.org/abs/2504.01707).
- Gu, A., Dao, T. (2024). **Mamba.**
- [ttt-layers] Sun, Y., Li, X. et al. (2024). **TTT layers** — competing $\mathcal{O}(N)$ approach.
- [diff-trans] Ye, T. et al. (ICLR 2025). **Differential Transformer.**

## Open questions / live debates

- **Is the quadratic cost fundamentally unavoidable for reasoning that requires global mixing**, or are architectural fixes (linear attention, state-space) genuinely competitive?
- **Internalization vs in-context** — long-document QA via Doc-to-LoRA vs. native long-context: which wins on what axis (quality, latency, memory, privacy)?
- **Benchmark gap** — most long-context evals are retrieval-flavored (NIAH). The performance on actual *reasoning* over long inputs is much worse and harder to measure.
- **Diminishing returns** — even at 1M tokens, models rarely use > 30% effectively. Why?
