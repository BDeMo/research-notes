# Standing instructions

> Explicit user instructions to carry across all sessions.
> Append-only. Supersede old entries; never silently delete.

---

### 2026-05-26 · 维护 memory/ 文件夹
**Source**: user instruction in 2026-05-26 session.
**Verbatim**: "所有要你记住的、和本repo相关的东西，写到本repo，都维护在memory文件夹里面"
**Rule**: anything the user asks to be remembered, that is related to this repo, must be written into the `memory/` folder of this repo. Not into chat history, not into the matrix entry, not into a plan — into `memory/`.
**How to apply**:
- When the user uses 记住 / remember / "from now on" / "ongoing rule" → write to `memory/`.
- New explicit instructions → append a dated entry to `memory/instructions.md`.
- New conventions → update `memory/conventions.md`.
- New stable user/project context → update `memory/context.md`.
- All sessions must read `memory/README.md` first.

### 2026-05-26 · Plans are private, structured, and budget-real
**Source**: user instruction in 2026-05-26 session.
**Verbatim**: "写得详细一点，我想要知道怎么验证，有哪些验证渠道，开销是多少"
**Rule**: every plan must include — in dedicated files — a validation protocol, validation channels (concrete benchmarks/datasets/baselines), and a budget (GPU-hours, $, time, headcount, decision points). One-line "we will validate on X" is not acceptable.
**How to apply**: enforce the plan template defined in [`docs/workflow.md §2`](../docs/workflow.md) for every new plan.

### 2026-05-26 · Maintain history matrix
**Source**: user instruction in 2026-05-26 session.
**Verbatim**: "再docs维护一个matrix文件夹，matrix就是我们的历史记录，记录这个repo干过什么事情，有哪些知识的母巢"
**Rule**: every working session must produce a dated entry in `docs/matrix/`, and any new papers / blogs / conversations referenced must be appended to `docs/matrix/knowledge-sources.md` (the "knowledge mother nest").
**How to apply**: do this at the *end* of every session, before closing.

### 2026-05-26 · Symbol system for ideas and plans
**Source**: user instruction in 2026-05-26 session.
**Verbatim**: "做一套符号，来记录idea验证进度、项目进度、有没有二期，开发模式等"
**Rule**: every idea/plan in any TOC must carry the four-axis symbol set defined in [`memory/symbols.md`](symbols.md): Status `S`, Priority `★`, Phase `φ`, Mode `M`.
**How to apply**:
- Update the `S` column on every meaningful state transition (even to `B` blocked).
- Use `φ` to record sequels: `1/N`, `↪#NN`, `←#NN`, `≈#NN`.
- Use `M` to record dev mode: `paper,solo` / `thesis,collab` / etc.
- The TOCs in `notes/ideas/README.md` and `notes/plans/README.md` are the single source of truth for these symbols.
- Do not invent new codes silently; if a new axis or code is needed, update `memory/symbols.md` first.

---

*If an instruction is later replaced, do NOT delete it. Add a `~~strikethrough~~ (superseded YYYY-MM-DD by …)` marker and the new entry below.*
