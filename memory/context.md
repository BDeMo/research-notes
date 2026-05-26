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
- **Purpose**: personal research notebook — brainstorms, paper reading, idea lists, project plans.
- **History root**: this repo was started 2026-05-26.

## Environment

- **Local path**: `/Users/s1shi/workspace/research-notes`
- **Workspace**: `/Users/s1shi/workspace/idea-brain-storm` (the working area where Cursor opens; not the repo itself).
- **OS**: macOS (Darwin).
- **Git config**: `Mingjia Shi <sam.shi@nokia.com>` (note the Nokia email — possibly leftover from internship; flag if it should be changed).
- **GitHub auth**: SSH key configured; **no `gh` CLI installed** → cannot create repos / PRs from the terminal. Repo creation must be done by the user via web UI.

## Active threads

### Inference-time training (started 2026-05-26)
- Central framing: $y = f(X; W)$, exhaust $X$, then look for best $W$.
- Three plans drafted:
  - **Plan 01** (`notes/plans/01-x-saturation-curve/`) — X-saturation curve curriculum. ~$6.5K. Drafting.
  - **Plan 03** (`notes/plans/03-w-space-best-of-n/`) — test-time search on weights axis. ~$4.8K. Drafting.
  - **Plan 08** (`notes/plans/08-model-outputs-delta-w/`) — self-modifying LLMs. ~$24K, PhD-scale. Drafting.
- Next concrete step suggested in the matrix: 1-day Plan 01 pilot on GSM8K + Qwen2.5-7B (≤ 20 GPU-hours).

## Archived threads

(none yet)
