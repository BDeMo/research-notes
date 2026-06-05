# Papers — Plan 08 line

This folder holds **one subfolder per paper** spun out of the mem-X / Plan-08 line.
When a new paper starts, **add a new `paper-<x>-<slug>/` folder here** (don't scatter paper
files in the plan root).

| folder | paper | status | one-line scope |
|---|---|---|---|
| [`paper-a-latent-memory/`](paper-a-latent-memory/) | **A — Latent Memory** | ⏸ parked | bit-capacity limits of a frozen-base soft-prompt memory wrapper (vs Gist/ICAE/CCM; the Silver-Bullet recall-failure axis) |
| [`paper-b-forgetting-gating/`](paper-b-forgetting-gating/) | **B — Forgetting & Gating** | 🟢 **active (writing first)** | do-no-harm adaptation: one intrinsic, cross-model signal that **gates** an augmentation (read) and **protects** sites during SFT (write), preventing catastrophic forgetting |

**Source of the split:** [`../summary/2026-06-05/two-paper-litreview-2026-06-05.md`](../summary/2026-06-05/two-paper-litreview-2026-06-05.md).
**Shared evidence ledger:** `mem-test/mem-embedding/summary/matrix.md` (§2 signals, §7/§7b gate, §7c forgetting+multi-depth, §8 boundary).

Each paper folder = `README.md` (thesis, contributions, status, links) + `outline.md`
(section skeleton + **claim→evidence map** + experiment gaps) + (later) a `draft/` for the `.tex`.
