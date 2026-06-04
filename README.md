# research-notes

Personal research notebook — brainstorms, paper reading, idea lists, and project plans.

Maintained by [Mingjia Shi](https://bdemo.github.io/homepage/) (@BDeMo).

---

## Layout

```
research-notes/
├── memory/                                          # standing instructions — READ FIRST
│   ├── README.md
│   ├── instructions.md                              # explicit user instructions
│   ├── conventions.md                               # repo conventions
│   ├── symbols.md                                   # notation for status/priority/phase/mode
│   └── context.md                                   # stable user / project context
├── docs/
│   ├── workflow.md                                  # methodology & operating principles
│   ├── maintenance.md                               # context-budget & pruning policy
│   └── matrix/                                      # session log + knowledge mother nest
│       ├── README.md
│       ├── knowledge-sources.md                     # running index of papers / blogs / etc
│       └── YYYY-MM-DD-<topic>.md                    # one entry per working session
├── known/                                           # public knowledge base by category
│   ├── README.md                                    # nearness graph + categories
│   ├── inference-time-training/
│   ├── hypernetworks/
│   ├── context-distillation/
│   ├── test-time-training/
│   ├── model-editing/
│   ├── lora-peft/
│   ├── inference-time-compute/
│   ├── long-context/
│   ├── self-improvement/
│   ├── catastrophic-forgetting/                     # preserving ability while fine-tuning
│   └── transformer-internals/                       # sinks, massive activations, super experts
└── notes/
    ├── ideas/                                       # idea catalog
    │   ├── README.md                                # index (TOC with meta per idea)
    │   ├── inference-time-training.md               # 50-idea brainstorm (X-W framing)
    │   ├── evaluation-2026-05-28.md                 # scored evaluation of all ideas
    │   └── rca-transformer-intrinsic-2026-06-03.md  # long-ctx + forgetting (design rules + audit)
    └── plans/                                       # detailed project plans
        ├── README.md                                # index (TOC with meta per plan)
        ├── 01-x-saturation-curve/                   # data curation by inference-time difficulty
        ├── 03-w-space-best-of-n/                    # test-time search along the weights axis
        ├── 08-model-outputs-delta-w/                # self-modifying LLMs
        └── 09-intrinsic-site-protection/            # long-ctx ↔ forgetting coupling → anti-forgetting
```

---

## Where to start

- **New here?** Read [`memory/README.md`](memory/README.md) **first** — it lists the standing rules and the session-start read order. Then [`docs/workflow.md`](docs/workflow.md) for the methodology.
- **Want to know what we've done?** Skim [`docs/matrix/`](docs/matrix/). The matrix is the chronological log of this repo.
- **Need the weekly update?** Use [`docs/weekly-human.md`](docs/weekly-human.md) for people and [`docs/weekly-agent.md`](docs/weekly-agent.md) for agent context.
- **What do we *know* about a topic?** Start at [`known/README.md`](known/README.md) — categories + nearness graph.
- **Reading list?** See [`docs/matrix/knowledge-sources.md`](docs/matrix/knowledge-sources.md) — the "knowledge mother nest" (chronological intake log).
- **What might we actually build?** See [`notes/plans/`](notes/plans/) for the 4 plans currently in draft.
- **How is this repo kept small enough to read?** See [`docs/maintenance.md`](docs/maintenance.md) — tier system, size caps, pruning, archive paths, hygiene checklist.

---

## Active topics

- **Inference-time training** — the $X \leftrightarrow W$ exchange. See the [2026-05-26 session](docs/matrix/2026-05-26-inference-time-training.md) for the starting point. Plans 01 / 03 / 08.
- **Long-context inference + catastrophic forgetting** (general method; RCA as application) — the *shared-substrate* thesis: the intrinsic sites that carry long context are the same sites perturbed by fine-tuning. Design rules + audit in [`notes/ideas/rca-transformer-intrinsic-2026-06-03.md`](notes/ideas/rca-transformer-intrinsic-2026-06-03.md); measure-first [Plan 09](notes/plans/09-intrinsic-site-protection/).

---

*Living document. Things here are speculative / WIP unless tagged otherwise.*
