# Paper A — Latent Memory (frozen-base soft-prompt memory wrapper)

**Status:** ⏸ **parked** (we write Paper B first). This is a stub to hold scope + evidence pointers.

**Working title:** *Bit-Capacity Limits of a Frozen-Base Soft-Prompt Memory Wrapper.*

## Thesis (one line)
A recurrent soft-prompt wrapper that compresses long context into K tokens on a **frozen base**
hits an empirical **bit-capacity wall**: it can specialize to one distribution but cannot encode
exact/extractive detail, and degrades when forced to hold everything.

## Core claims (evidence already in hand)
- **Capacity wall on exact recall** — RULER-NIAH → 0 even as full-context → 0.995; compressed memory ≪ raw context exactly where detail matters (matrix §7 full-context ceiling).
- **Distribution-bound competence** — in-dist 提点 (QA +2..+9 pt) but gain vanishes at the same-task line (matrix §8 capability boundary).
- **Capacity interference** — one wrapper trained on all datasets mixed is *worse* than per-dataset (matrix §8 mix-train); can't hold all at once.

## Honest novelty / required baselines
The wall + exact-recall failure are **documented** (gist "Silver Bullet" study, Info-Preservation).
Defensible: (i) the wall is a **read-interface** property (matched-K Gist hits it too), (ii) the
recurrent frozen-base wrapper + OPD/RL recipe, (iii) the precise bit-entropy curve.
**Baselines a reviewer demands:** Gist, ICAE, CCM, AutoCompressor; evaluate on the synthetic-recall axis.

## Links
- scope + lit: [`../../summary/2026-06-05/two-paper-litreview-2026-06-05.md`](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md) (§Paper A)
- related work: [`../../v2-related-work.md`](../../history/v2-related-work.md) · [`../../v0-opd-on-latent-memory-survey-2026-06-01.md`](../../history/v0-opd-on-latent-memory-survey-2026-06-01.md)
- evidence: `mem-test/mem-embedding/summary/matrix.md` §7 (ceiling), §8 (boundary + mix-train)
