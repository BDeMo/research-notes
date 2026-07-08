# results-v2.0.0 — raw experiment logs (every cell, in-repo)

All **812** evaluation cells behind the v2.0.0 fact-base, figures and diagnosis report, saved here so **every log is inspectable in the repo** (not just on the ephemeral dev pods). Each line is one cell:

```
RECIPE_EVAL <tag> {'method': .., 'full': .., 'no_ctx': .., 'compression_ratio': .., 'mode': ..}
```
`method` = the baseline's score · `full` = uncompressed full-context reference · `no_ctx` = no-context floor.

## Log files (`logs/`)
| file | cells | wave (what it probes) |
|---|---|---|
| [`logs/catalog-177.txt`](logs/catalog-177.txt) | 177 | master budget table: 20 KV methods × ratio + method×bench + refs |
| [`logs/necessity-72.txt`](logs/necessity-72.txt) | 72 | feasible-baseline length-sweep (trunc/window/full/LLMLingua × 2k–32k) |
| [`logs/rag-18.txt`](logs/rag-18.txt) | 18 | BM25 RAG × length / budget / bench |
| [`logs/diagnosis-329.txt`](logs/diagnosis-329.txt) | 329 | the 10-story diagnosis campaign (`s1…s10`) |
| [`logs/dive-100.txt`](logs/dive-100.txt) | 100 | dive-ins (`da/db/dc/dd`): extreme-ratio, cliff-length, distractor-causal, RAG-hurts |
| [`logs/precision-30.txt`](logs/precision-30.txt) | 30 | precision wave (`pw`): pin the 16k cliff, kvzip cliff, distractor×KV |
| [`logs/expansion-86.txt`](logs/expansion-86.txt) | 86 | scaling/cross-family (`x1–x4`): Qwen3 1.7B→14B, Qwen2.5, Llama, GDN-4B |
| [`logs/ALL-812.txt`](logs/ALL-812.txt) | 812 | union of all of the above (de-duplicated) |

## Where these are used
- Tables: [`../baseline-factbase-v2.0.0.md`](../baseline-factbase-v2.0.0.md) (§1–9)
- Figures: [`../figures/`](../figures/) (auto-generated from these logs)
- Narrative: [`../baseline-diagnosis-report.md`](../baseline-diagnosis-report.md), [`../insights-longcontext-validity.md`](../insights-longcontext-validity.md)
- Decisions/provenance: [`../decisions-2026-06-24.md`](../decisions-2026-06-24.md) (D12–D20)

Base models: `Qwen/Qwen3-8B` (dense) and `Qwen/Qwen3.5-9B` (GDN) unless a cell tag says otherwise; public benches only; training-free unless the tag/notes say otherwise.
