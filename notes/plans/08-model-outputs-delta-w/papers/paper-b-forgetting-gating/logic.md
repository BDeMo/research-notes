# Paper B — Logic (reasoning from scratch, v0 · crux OPEN)

> Built from your 4 selling points (2026-06-05). This is a **shared working doc** — the central
> crux (§3) is deliberately left open until we agree. Honesty-first: every ⚠️ is a real
> tension with our evidence, not a detail.

## 1. Your 4 selling points → one system
The 4 points describe one module with four properties:

| # | your point | property |
|---|---|---|
| SP1 | detachable, model-agnostic (LoRA-like weight-fusion = fancy/v2) | **FORM** — a portable, removable module |
| SP2 | lightweight params that fast-ingest data = store new knowledge | **FUNCTION** — fast knowledge acquisition |
| SP3 | a robust signal gates *when* to apply ⇒ no interference | **SAFETY** — do-no-harm via one signal |
| SP4 | works across models, fast-extend, task-agnostic; target in-domain, floor = in-task/in-dataset no-harm | **SCOPE** — honest competence envelope |

**Candidate one-thesis:** *A portable, model-agnostic **do-no-harm knowledge module** — a lightweight,
detachable adapter on a frozen base that rapidly learns new data, helps in-domain, and is gated by a
robust intrinsic signal so it **provably never degrades** the base model — across model families and tasks.*
**Value prop:** *teach any model new data without risking what it already knows.*

## 2. Honest evidence map (✅ solid · ⚠️ tension · ❌ gap)
**SP1 — detachable + model-agnostic**
- ✅ detachable: frozen base, remove module ⇒ **exact base** (do-no-harm by construction). This is the *contrast* to LoRA/SFT, which mutate weights ⇒ forget (§7c).
- ✅ transformers-agnostic FORM: attaches via embeddings/prefix/x-attn; ran on 7 families.
- ⚠️ weight-fusion (LoRA-merge): our module is an **input-conditioned compressor, not a weight delta** → not fusable as-is. *Also: fusing into weights is in tension with a per-input gate (SP3) — you can't gate merged weights per input.* **Detachable+gated and weight-fused are ~mutually exclusive; our coherent story is detachable+gated.** (Park fusion → v2.)

**SP2 — fast-ingest data → store new knowledge** ⟵ **THE CRUX (§3)**
- ⚠️ Two readings: (lit) params *store the data* so no context needed at inference (parametric fact store); (ours) params learn *how to compress this distribution's context* so you *do* need context at inference.
- ❌ Our system is the latter. Trained-on-D helps on D **only when given D-context** (§8). It does **not** memorize facts for context-free recall — and exactly where it would (store→recall facts) it hits the **capacity wall** (RULER-NIAH→0, §7 / **Paper A**). **SP2's literal reading collides head-on with Paper A's negative result.**
- → reconciliation: what it ingests is **task/distribution competence**, not facts. "fast/lightweight" needs a **few-step-adaptation** experiment to back it.

**SP3 — robust signal gates ⇒ do-no-harm**
- ✅ exactly our core (gate = when to apply; do-no-harm by construction).
- ⚠️ "ONE robust signal (= our research)": **read baselines §7d show TARG (base-uncertainty) ≥ our `delta_last`** on the 3 families we still have data for. So the honest claim = "**a** robust intrinsic gate (base-uncertainty and/or memory-write, maybe combined) suffices," and **the specific signal is an ablation**, not the headline. (7-family head-to-head to settle it is **partly blocked**: ray/test pods deleted; can re-run Qwen3-14B cheaply, Phi/Mistral/Qwen2.5 need re-download.)

**SP4 — cross-model, fast-extend, task-agnostic; in-domain target, in-task/in-dataset no-harm floor**
- ✅✅✅ **Strongest fit.** §8 *is* exactly this: in-dist +2..+9pt, degrades at same-task, harmful cross. You explicitly adopt our honest limit as the **spec** ("泛化差没关系，追求in-domain，保底in-task/in-dataset不掉点"). The gate gives the **no-harm floor** when generalization fails.
- ✅ cross-model FORM (7 families) + gate transfers (TARG caveat). ✅ task-agnostic eval (9 benches).
- ⚠️ "fast-extend" needs the few-step-adaptation curve.

## 3. ★ THE CRUX (open) — what *is* the memory module?
Everything (claims, baselines, eval, collision with Paper A) hinges on this:

| option | what it is | needs context at inference? | our evidence | risk |
|---|---|---|---|---|
| **A** | inference-time **context compressor** | yes | ✅ matches | "just compression"; SP2 unsupported |
| **B** | **parametric fact store** (cartridge) | no | ❌ **capacity wall (Paper A)** | collides with our own negative result |
| **C** | **per-distribution adapter** that compresses inference-time context = *task/distribution memory* | yes | ✅ closest | must **cede fact-storage to Paper A** |

**My recommendation: C** — frame the "knowledge" as **task/distribution competence**, *explicitly* cede
context-free fact storage to Paper A's wall. SP2 becomes "fast adaptation to a new data *distribution*,"
not "stores the data." (B is the fancy/v2 goal; honest now = C.)

## 4. Logical spine (conditional on crux = C)
- **Problem:** adapting a pretrained model hurts what it knows — fine-tune ⇒ forgetting (§7c); always-on augmentation ⇒ negative transfer (§8).
- **Claim:** put new-data competence in a **detachable, model-agnostic adapter on a frozen base**, **gated by a robust intrinsic signal** ⇒ **in-domain lift + guaranteed no-harm floor**, portable across families/tasks.
- **Contributions:** C1 quantify both forgetting modes · C2 detachable transformers-agnostic form, do-no-harm by construction, 7 families · C3 in-domain lift + the *sharp characterized boundary* (§8, honest scope) · C4 robust intrinsic gate recovers most negative transfer cross-model, zero tuning (signal = ablation, TARG competitive) · C5 task-agnostic eval philosophy (no generalization claim; in-domain lift + no-harm everywhere).
- **Scope (honest):** in-domain 提点; floor = in-task/in-dataset no-harm; cross = gate closes. **Not** fact storage (→ Paper A).
- **Parked (v2/fancy):** weight-fusion, parametric fact cartridge (B), online gate, agentic multi-tool routing.

## 5. Experiments
- **Have:** §7c (SFT forgets) · §8 (boundary + mix) · §7/§7b (gate, locking) · §7d (read baselines) · §2 (cross-model signal).
- **Running:** `abl` significance (27/38).
- **Needed:** (a) **few-step adaptation curve** (back "fast-ingest") · (b) **relevance/agent eval** (mixed relevant/irrelevant context — earn the framing) · (c) head-to-head **+Qwen3-14B** (cheap re-run) [full 7 blocked].
