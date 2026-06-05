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

### 2026-05-26 · Maintain `known/` public knowledge base
**Source**: user instruction in 2026-05-26 session.
**Verbatim**: "维护一个公共知识库，作为一个文件夹，known，其需要按类别分类进不同的知识库，名字就是类别本身。然后里面要维护包含关系，以及最相近的知识库。维护这个"相近"关系的是known本身的目录"
**Rule**: a `known/` folder at the repo root contains one subfolder per category (folder name = category name).
- Each category's `README.md` maintains **containment** relations (Contained-by / Contains).
- The top-level `known/README.md` maintains the **nearness** relations across categories (Mermaid graph + adjacency list).
**How to apply**:
- When a new topic is touched in any session, decide whether it fits an existing category or warrants a new one.
- New category → create `known/<slug>/README.md` from the template, then update `known/README.md` (categories table + graph + adjacency list).
- Cross-reference: cite knowledge-sources `[id]`s from any `known/<cat>/` that contains them.
- This is *complementary* to `docs/matrix/knowledge-sources.md`: that is the chronological intake log; `known/` is the curated paradigm-level reference.

### 2026-05-26 · Maintenance plan to prevent context bloat
**Source**: user instruction in 2026-05-26 session.
**Verbatim**: "在关键的地方写好这套维护方法，为了避免累积造成context过大，我们必须做好这个知识库、idea库的管理文件以及方案"
**Rule**: the repo must carry an explicit, written maintenance plan that bounds session-start context. The plan is centralized in [`docs/maintenance.md`](../docs/maintenance.md). Every key entry point (the four T1 TOC files and `memory/conventions.md`) carries a local `## Maintenance` section linking back.
**How to apply**:
- Treat `docs/maintenance.md` as authoritative for tier definitions, file size caps, pruning triggers, archive paths, and the per-session hygiene checklist.
- T0 = `memory/*` + `docs/matrix/README.md` + latest matrix entry. T1 = the three TOC files. Combined budget ≤ ~1200 lines ≈ 15K tokens.
- When *any* T0/T1 file approaches its hard cap, prune (summarize + archive) before doing new work.
- Archive paths use `<original-parent>/_archive/...`. `git mv` only.
- Run the [hygiene checklist](../docs/maintenance.md#5-hygiene-checklist-per-session) at end of every session.

### 2026-06-04 · Result cells must trace to settings
**Source**: user instruction in 2026-06-04 session.
**Verbatim**: "记住，你要做好setting管理，每个有结果的文件，每个结果来自什么setting都要写清楚，但是不是每个都写全部，你可以统一写一个setting文件，然后link过去。把所有文档过一遍，每个cell来自于哪个settings，都写清楚，包括外部资源来源的link以及内部设置（比如：超参数等）。"
**Rule**: every file that reports results, tables, figures, or benchmark cells must state which setting produced each result. Do not duplicate full hyperparameters in every cell; use stable setting IDs that link to a centralized setting/provenance file for the plan or project.
**How to apply**:
- Create or maintain a central `settings.md` / `provenance.md` per active plan when results appear.
- Each result table/cell must include either a `setting` column, a caption-level setting ID, or nearby prose linking to the relevant setting.
- The setting entry must include: code/data source links, external resource links, base model, wrapper/model variant, datasets, benchmark protocol, seeds, key hyperparameters, compute/runtime context, artifact locations, and known caveats.
- For slides and short human reports, include only the setting ID and link; detailed numbers and facts live in the linked markdown/PDF/CSV artifacts.

### 2026-06-04 · Slide decks append inputs and avoid repeated covers
**Source**: user instruction in 2026-06-04 session.
**Verbatim**: "每次更新main，不要搞那么多封面，然后之前week的input也不要删，就往下继续append input就好了。上一层的readme要有下一层readme和文件夹的说明，以及怎么组织文件的方法，都更新一下。"
**Rule**: when updating a slide `main.tex`, keep previous weekly `\input{...}` lines and append the new week below them. Do not replace/delete older week inputs. Avoid repeated title/cover slides: the main deck owns the single deck-level cover, and weekly input files should not each call `\titlepage`.
**How to apply**:
- Append new weekly inputs in chronological order.
- Existing weekly inputs are historical record and remain in `main.tex` unless the user explicitly requests an archive/split.
- Parent README files must describe child folders, link child `README.md` files, and explain the organization/update method for the next layer.
- Child folders with nontrivial structure should carry their own `README.md`.

### 2026-06-05 · Weekly slides must recall context and define terms
**Source**: user instruction in 2026-06-05 session.
**Verbatim**: "每个词都要有解释，不要就wrapper上来，一周一次，大家都忙，没有上下文的，你要注意这一点。同样slides也是，你检查一下，这次汇报要用到的概念，不管前面出没出现过，都要分别有recall和preliminary。"
**Rule**: weekly presentation slides are for busy collaborators who may have no active context. Do not assume they remember prior discussions or project jargon. Every deck/update must first recall the background and define the concepts needed for that meeting before showing results.
**How to apply**:
- Start each weekly update with a short `Recall` / `Background` section: what problem is being studied, why it matters, and how the motivating benchmark/scenario connects to the paper question.
- Add a `Preliminaries` / vocabulary slide before using project-specific terms, even if those terms appeared in previous weeks.
- Define all key terms in plain language on first use: e.g. long-context inference, catastrophic forgetting, frozen base model, adaptation module, learned memory module/wrapper, soft memory/soft prompt, gate, do-no-harm, in-distribution, out-of-distribution, intrinsic signal, model-agnostic, AUROC.
- For weekly meetings, organize the story as: why are we doing this; what is it useful for; what are we doing; what is the goal; how is it going; what is good news; what is bad news/risk; what is next.
- Avoid starting directly from implementation names such as `wrapper`, `gate`, or setting IDs. Introduce the concept first, then give the local implementation name.
- If a detailed/appendix section uses extra technical terms, add a second preliminary slide local to that section.

---

*If an instruction is later replaced, do NOT delete it. Add a `~~strikethrough~~ (superseded YYYY-MM-DD by …)` marker and the new entry below.*
