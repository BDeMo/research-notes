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
│   └── long-context/
└── notes/
    ├── ideas/                                       # idea catalog
    │   ├── README.md                                # index (TOC with meta per idea)
    │   └── inference-time-training.md               # 50-idea brainstorm (X-W framing)
    └── plans/                                       # detailed project plans
        ├── README.md                                # index (TOC with meta per plan)
        ├── 01-x-saturation-curve/                   # data curation by inference-time difficulty
        ├── 03-w-space-best-of-n/                    # test-time search along the weights axis
        └── 08-model-outputs-delta-w/                # self-modifying LLMs
```

---

## Where to start

- **New here?** Read [`memory/README.md`](memory/README.md) **first** — it lists the standing rules. Then [`docs/workflow.md`](docs/workflow.md) for the methodology.
- **Want to know what we've done?** Skim [`docs/matrix/`](docs/matrix/). The matrix is the chronological log of this repo.
- **What do we *know* about a topic?** Start at [`known/README.md`](known/README.md) — categories + nearness graph.
- **Reading list?** See [`docs/matrix/knowledge-sources.md`](docs/matrix/knowledge-sources.md) — the "knowledge mother nest" (chronological intake log).
- **What might we actually build?** See [`notes/plans/`](notes/plans/) for the 3 plans currently in draft.

---

## Active topics

- **Inference-time training** — the $X \leftrightarrow W$ exchange. See the [2026-05-26 session](docs/matrix/2026-05-26-inference-time-training.md) for the starting point.

---

*Living document. Things here are speculative / WIP unless tagged otherwise.*
