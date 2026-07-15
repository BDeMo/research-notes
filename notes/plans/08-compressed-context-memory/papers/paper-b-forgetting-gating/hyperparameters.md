# Hyperparameters — explained against the IMP framework

**Framework recap (3 stages).** IMP-v2.1.0 wraps a **frozen** base `F` with:
**(a) SCORE** every context token by cheap O(L) signals → **(b) SELECT** the top-budget spans (drop the rest) → **(c) READ** the shortened sequence with `F`. Each hyperparameter lives in exactly one stage; below, "stage" says where.

## A. IMP method hyperparameters (ours)

| knob (env) | stage | what it controls | default | effect / trade-off |
|---|---|---|---|---|
| **keep fraction `p`** (`GCM_LL_RATE`) | (b) select | fraction of context tokens RETAINED (`keep=max(8,⌊pL⌋)`) | **0.5** | the budget dial: ↑p → closer to `full` accuracy but more prefill; ↓p → cheaper but risks dropping the needle. **This is the axis of the keep-ablation.** |
| **span width `w`** (`GCM_IMP_SPAN`) | (b) select | selection granularity: keep whole `w`-token spans, not isolated tokens | **32** | w=1 (token) → perfect on isolated-needle retrieval but **shatters coherent QA** (F24); w≈32 → keeps the answer *sentence* intact → rescues QA while retaining retrieval (F25). Insensitive 16–64; multi-hop likes larger. |
| **signal** (`GCM_IMP_SIGNAL`) | (a) score | which importance signal(s): `query` / `surprisal` / `both` | **both** | `query`-relevance wins word/phrase needles (AUROC 0.95), `surprisal` wins numeric/high-info needles (0.84); **neither wins both (F20)** → `both` = z(query)+z(surprisal) is load-bearing. |
| **MAXCTX** (`GCM_MAXCTX`) | (a)/(c) | how much raw context is scored & the base's read window | **32768** long-ctx | set to the true window so there is **no artificial truncation**; smaller only for speed on short benches. |
| (Mode B, planned) router-head dim · side-cache size · LoRA rank | learned combine | tiny trained module (base frozen) | — | Mode B only; not in current results. |

**How p and w interact (the two dials that matter):** `p` sets *how much* you keep; `w` sets *in what shape*. The paper's核心消融 is (p × w): span-level keeps the QA curve monotone in `p`, token-level does not (it plateaus/oscillates because coherence, not budget, is the bottleneck).

## B. Shared protocol hyperparameters (all methods, for fair comparison)

| knob | value | note |
|---|---|---|
| base model | `Qwen/Qwen3-8B` (frozen, bf16) | + linear arm `Qwen3.5-9B` (GDN) for generality |
| scoring | MC = letter log-likelihood; gen = EM/F1/ROUGE-L (+gold-substring) | native per bench |
| `N` (eval items) | FULL split for headline; **N=100 for ablations / N=500 for short sanity (disclosed)** | see `experiment-config-and-sampling.md` |
| `n_chunks`, temperature=0 (greedy) | per bench | deterministic decode |

## C. Baseline knobs — and how the keep-ablation maps them to ONE budget axis

To compare every method at the **same retained budget `k`**, the shared keep-fraction `k∈{0.1,0.25,0.5,0.75}` maps to each method's native knob:

| method | native knob | mapping from keep `k` | family |
|---|---|---|---|
| window (txl) | `GCM_TXL_WINDOW` (abs tokens) | `round(k·maxctx)` | truncate-to-window |
| RAG (BM25) | `GCM_RAG_BUDGET` (abs tokens) | `round(k·maxctx)` | retrieval |
| LLMLingua-2 | `GCM_LL_RATE` (frac) | `= k` | prompt compression |
| ToMe | `GCM_TOME_RATIO` (frac) | `= k` | token merging |
| **IMP (ours)** | `GCM_LL_RATE` (frac), span32 | `= k` | span importance-routing |
| kvzip | `GCM_KV_RATIO` (frac dropped) | `= 1−k` | KV eviction (reconstruction) |
| knorm | `GCM_KV_RATIO` (frac dropped) | `= 1−k` | KV eviction (key-norm) |

*(window/RAG budgets are absolute, so at a given `k` the retained token count matches the fractional methods against `maxctx`; kvzip/knorm express budget as the fraction *dropped*, hence `1−k`.)*

## Keep-rate ablation (running)
Launcher [`configs/keep_ablation.sh`](configs/keep_ablation.sh); 112 cells = 7 methods × 4 benches × 4 keeps; base Qwen3-8B, N=100; output `/mnt/persist/grid_keepabl/ka_*`. Benches: `ruler_niah`@16k (retrieval), `lb_hotpotqa`@16k (multi-hop), `squad_v2`@4k (extractive), `quality_hard`@8k (literary-MC). Results → [`keep-ablation-results.md`](keep-ablation-results.md) (built on completion). Prior single-method IMP keep-curve: fact-base §12.2 / F25 (monotone: squad 0.33/0.46/0.59 @ 0.25/0.5/0.75).
