# Paper A model embedding and position audit (2026-07-15)

> Purpose: define what is and is not shared across the cross-model GCM experiments. Source:
> local checkpoint `AutoConfig`/`AutoTokenizer` metadata on `sam-dev-d1525-gpu4`; tokenizer-map SHA-256
> and sampled embedding-weight comparisons. No model output metrics are reported here.

## Tokenizer and input-embedding inventory

| model | model family | hidden d | config embedding rows | tokenizer / usable entries | tokenizer-map group | tied input/output? |
|---|---|---:|---:|---|---|:--:|
| ToolACE-2-8B | Llama | 4096 | 128,256 | BPE · 128,256 | `e5f11f7a935e` | no |
| Llama-xLAM-2-8B | Llama | 4096 | 128,256 | BPE · 128,256 | `e5f11f7a935e` | no |
| Ministral-8B-Instruct | Ministral | 4096 | 131,072 | BPE · 131,072 | `38a74daf301c` | no |
| Qwen3-8B | Qwen3 | 4096 | 151,936 | Qwen2 BPE · 151,669 | `cab123b865b7` | no |
| Qwen3.5-9B | Qwen3.5-text | 4096 | 248,320 | BPE · 248,077 | `5a428c179906` | no |
| GLM-4-9B-0414 | GLM4 | 4096 | 151,552 | BPE · 151,343 | `800efcd0476f` | no |
| Qwen3.5-4B | Qwen3.5-text | 2560 | 248,320 | BPE · 248,077 | `5a428c179906` | yes |

Config vocab rows can exceed tokenizer length because checkpoints reserve/pad embedding rows.

### Exact tokenizer alignment

- ToolACE and xLAM: exact same 128,256 token→ID map.
- Qwen3.5-9B and Qwen3.5-4B: exact same 248,077 token→ID map.
- All other pairs use distinct vocabularies/pre-tokenizers.
- Ministral must load with `fix_mistral_regex=True`; the shared loader now does so.

### Embedding geometry is not implied by tokenizer alignment

- ToolACE vs xLAM, sampled 8,173 nonzero aligned rows: direct cosine mean **0.999979**,
  p01 **0.999846**, minimum **0.999709**. These two input-embedding coordinate systems are
  empirically almost identical, consistent with a shared Llama base lineage.
- Qwen3.5-9B vs Qwen3.5-4B: dimensions differ (4096 vs 2560). The same relational-geometry
  correlation is only **0.143**, despite an identical tokenizer.

**Paper rule:** cross-model means the same algorithm/configuration is retrained per base. Except for the
ToolACE/xLAM diagnostic, do not transfer or compare raw memory vectors, projection weights, or embedding
coordinates directly across bases.

## Position handling in current GCM

GCM adds no learned position embedding. Every base keeps its native position mechanism (principally RoPE in
the full-attention layers; sequence order/state dynamics in GDN layers).

### Encode

For one chunk:

```text
[prior projected memories ; current context ; query ; K memory queries]
```

`position_ids` are explicitly `0..L-1`. Memory-query rows are last and can causally attend all preceding
positions. For `M0`, only memory-query→query attention is blocked.

### Recurrent chunks

Each chunk call restarts at position zero. Earlier memories become an ordered prefix, so GCM preserves:

- local order inside each chunk;
- coarse order among chunk memories.

It does **not** retain each source token's original absolute index. The recurrent prefix is detached across
chunks to bound backprop memory.

### Read

The final sequence is:

```text
[M1 ; ... ; MS ; query]
```

It receives a new contiguous position range. Original 6k-token QuALITY articles are typically represented by
two K=128 chunk memories, then re-indexed as about 256 reader positions.

### Batch and reconstruction

- Short-context batch encoding uses left padding + an attention mask; the model derives valid-token positions.
- Reconstruction uses `[M ; repeated position-query slots]`. Slot vectors share content but receive distinct
  native positions.

**Paper rule:** describe GCM as order-preserving at chunk granularity, not as an absolute-position-preserving
compressor. Current benchmarks do not directly test exact source-token-index recovery.
