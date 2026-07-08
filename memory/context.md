# Stable context

> Long-lived facts about the user, the repo, and active threads.
> Update when context changes; don't re-derive from scratch each session.

---

## User

- **Name**: Mingjia (Samuel Jayden) Shi · 石明佳
- **GitHub**: [@BDeMo](https://github.com/BDeMo) · HuggingFace [@BDeM](https://huggingface.co/BDeM)
- **Homepage**: https://bdemo.github.io/homepage/
- **Affiliation**: Ph.D. student, VAST Lab, University of Virginia. Advisor: Jundong Li.
- **Background**: Master at Sichuan University (federated learning, decentralized data analyses); previously interned at NUS-HPC-AI-Lab (Yang You) and HoumoAI.
- **Co-author background**: SpeeD (diffusion training acceleration, ICML 2024), R-MeeTo (vision Mamba), pFedBreD (NeurIPS 2023), MINE-SSM-Attention.
- **Personality**: ENFP. Likes structured brainstorms with concrete next steps. Open to wild ideas.

## Communication preferences

- Default language: Chinese for conversation, English for code/headings/file names/technical text.
- Wants concise but thorough; "写得详细一点" on plans.
- Prefers brainstorming → reading → planning workflow.
- Likes when ideas are ranked by novelty × feasibility × impact.
- Wants validation/cost/channels in every plan.
- Asks clarifying questions structurally — prefer concise multiple-choice when asking back.

## Repo

- **Name**: `BDeMo/research-notes` (private).
- **Remote**: `git@github.com:BDeMo/research-notes.git` (SSH).
- **Purpose**: personal research notebook — brainstorms, paper reading, idea lists, project plans, and a curated knowledge base.
- **History root**: this repo was started 2026-05-26.
- **Four pillars**:
  1. `memory/` — standing rules and what-to-remember.
  2. `docs/` — methodology (`workflow.md`) + history (`matrix/`).
  3. `known/` — public knowledge base organized by category with nearness graph.
  4. `notes/` — actual brainstorms and plans (the working output).

## Environment

- **Local path**: `/Users/s1shi/workspace/research-notes`
- **Workspace**: `/Users/s1shi/workspace/idea-brain-storm` (the working area where Cursor opens; not the repo itself).
- **OS**: macOS (Darwin).
- **Git config**: `Mingjia Shi <sam.shi@nokia.com>` (note the Nokia email — possibly leftover from internship; flag if it should be changed).
- **GitHub auth**: SSH key configured; **no `gh` CLI installed** → cannot create repos / PRs from the terminal. Repo creation must be done by the user via web UI.

## Active threads

### Inference-time training (started 2026-05-26)
- Central framing: $y = f(X; W)$, exhaust $X$, then look for best $W$.
- Brainstorm: 50 ideas, cataloged at [`notes/ideas/README.md`](../notes/ideas/README.md), full content at [`notes/ideas/inference-time-training.md`](../notes/ideas/inference-time-training.md).
- Three plans drafted (catalog at [`notes/plans/README.md`](../notes/plans/README.md)):
  - **Plan 01** (`notes/plans/01-x-saturation-curve/`) — X-saturation curve curriculum. ~$6.5K. Drafting.
  - **Plan 03** (`notes/plans/03-w-space-best-of-n/`) — test-time search on weights axis. ~$4.8K. Drafting.
  - **Plan 08** (`notes/plans/08-compressed-context-memory/`) — compressed-context memory (Papers A & B; ΔW self-modifying north-star parked). ~$24K, PhD-scale. Drafting.
- Next concrete step suggested in the matrix: 1-day Plan 01 pilot on GSM8K + Qwen2.5-7B (≤ 20 GPU-hours).

### Long-context inference + catastrophic forgetting (started 2026-06-03)
- **General method**, not RCA-specific: RCA (Nokia release) is the *application* where both pains coincide; the method must stand on the two universal problems alone.
- **Thesis (shared substrate)**: the intrinsic load-bearing sites — attention sinks / massive activations (dense), super experts (MoE) — carry long context *and* are the forgetting-vulnerable sites; a data-agnostic protection rule fixes both. Source + audit: [`notes/ideas/rca-transformer-intrinsic-2026-06-03.md`](../notes/ideas/rca-transformer-intrinsic-2026-06-03.md).
- **Design rules DR1–DR15** in §0 of that file are the durable constraints (data-agnostic · transformers-intrinsic · lightweight/model-agnostic · non-task-specific training · audit-first · general-first cross-X eval · 4-seed · AR→dLLM). Any new idea/plan in this line must comply.
- **Audit verdict**: every single mechanism (R1–R12 dense, M1–M9 MoE) is preempted; the contribution is the *unifying observation + intrinsic site-selection criterion*, not a new mechanism.
- **Plan 09 = "Janus"** ([`notes/plans/09-intrinsic-site-protection/`](../notes/plans/09-intrinsic-site-protection/)) — **executed 2026-06-03→04** (codename Janus; impl repos `~/workspace/janus`, `~/workspace/janus-methods`). Built a 39-metric / 12-angle intrinsic instrument; ran a 41-cell grid (4 families × 12 benchmarks).
  - **v1 measurement**: per-head long-ctx × forgetting coupling looked uniformly positive (ρ 0.17–0.56). ([`paper-draft-2026-06-04.md`](../notes/plans/09-intrinsic-site-protection/paper-draft-2026-06-04.md))
  - ⛔ **Falsified under controls** (partial-out activation magnitude + within-layer + per-model + CIs): **no robust head-level coupling**. The "same heads, two frontiers" thesis does **not** hold at head level. ([`mechanism-results-2026-06-04.md`](../notes/plans/09-intrinsic-site-protection/mechanism-results-2026-06-04.md))
  - ✅ **Surviving finding (F-shield)**: long-context structural heads are gradient-**shielded** during SFT (within-layer ρ(grad, attn_distance) ≈ −0.5…−0.6) → SFT gradients *avoid* them. **F-scale**: strong at 0.6–14B, neutralizes to ≈0 by 30B/32B (no reversal). **F-domain**: it's an *instruction/Q→A objective* effect (holds for hotpot/narrativeqa, vanishes for wikitext LM). **F-consequence**: explains why NIAH stays robust under instruction-SFT. ([`facts-2026-06-04.md`](../notes/plans/09-intrinsic-site-protection/facts-2026-06-04.md))
  - **H3 causal test inconclusive** (GSM8K-SFT didn't induce long-ctx forgetting). Method line (`janus-methods`: criterion×operator lib) blocked on a valid forgetting-inducing SFT recipe.
  - **Paper pivot**: from "same heads, two frontiers" → "long-context heads are intrinsically shielded from instruction-SFT gradients (why NIAH survives), a small/mid-model effect that neutralizes by 30B." Measurement infra/metrics still sound.
- New `known/` categories from this line: [`catastrophic-forgetting/`](../known/catastrophic-forgetting/) + [`transformer-internals/`](../known/transformer-internals/).
- Next concrete step: fix the Janus method-line SFT recipe (LoRA-all-modules/full-FT that *learns then forgets*); re-run the protection sweep; MoE super-expert probe needs expert-level metrics (current mechanism.py is attention-head only).

### Other active repos (detail in matrix active-threads + 2026-06-04 harvest)
- **`latent-mem-paper`** — Plan 08 paper "Bit-Capacity Limits of Soft-Prompt Memory", target **ICLR 2027**.
- **`test_env/EMNLP-dllm-BoN`** — dLLM Best-of-N verifier-readout paper ("confluence"), **EMNLP** draft.
- **`rca-demo-qwen`** — Nokia RCA application; curriculum LoRA adapters (Nezha→OpenRCA-500→RCAEval→LincYaw) on Qwen2.5/Qwen3.
- **`intern-project`** — Nokia intern cohort (MoECompiler, agent skills, telecom time-series FM, RCA deliverables).
- Cross-repo roll-up: [`docs/matrix/2026-06-04-cross-repo-harvest.md`](../docs/matrix/2026-06-04-cross-repo-harvest.md).

## Archived threads

(none yet)
