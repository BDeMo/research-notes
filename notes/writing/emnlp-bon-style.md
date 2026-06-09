# Writing style — house style for my papers

> Reusable style reference + checklist for all my papers. Seeded from the EMNLP dLLM-BoN
> paper (`workspace/test_env/EMNLP-dllm-BoN/acl_latex.tex`, "Where Should a Selection
> Verifier Read for Best-of-$N$ Sampling in Diffusion Language Model Reasoning?") and
> refined across later papers (ICLR-2027 do-no-harm / v1.6). Moves A-M came from the BoN
> paper; N-S were reinforced while drafting later papers. Use it as a checklist when
> drafting/editing. Maintained doc, paper-specific notes go in the per-paper log (last section).

## 0. One-line character
**Precise, mechanism-first, anti-overclaim empirical writing.** Short declarative
sentences; one coined concept as the spine; every claim explicitly scoped to what
the controlled evidence supports; "boundary" not "victory" framing. No hype.

## 1. Signature moves (the tells)

**A. Coin and define one precise concept, then use it as the spine.**
> "...a structural property we call *confluence*: under causal masking, that final
> hidden state is the single position whose receptive field contains the whole
> prompt and solution."

Name the idea (`confluence`, `readout`, `output-span readout`, `selection space`),
give it a one-sentence formal meaning, and route the whole argument through it.

**B. The "not X; it is Y" scoping construction (the most frequent tic).**
Used to pre-empt overclaiming on almost every claim:
> "This is not a claim that averaging more positions is always better. It is a
> targeted consequence of the missing-confluence diagnosis..."
> · "a hypothesis to test, not a default to inherit." · "informative rather than
> contradictory" · "a diagnostic rather than a law." · "The point is not that one
> readout dominates across tasks."

When you state a result, immediately say what it does **not** mean.

**C. Mechanism before convention — justify, never assert.**
> "This is not an arbitrary convention. Under causal masking, every position sees
> only earlier tokens, so the final position is the only single position whose
> receptive field contains the whole prompt and solution."

Explain *why* something works, then show the mechanism does/doesn't transfer.

**D. Controlled-comparison vocabulary; isolate one variable.**
> "we keep the sampler, labels, and verifier head fixed and isolate the readout."
> · "the candidate set, frozen dLLM, labels, and scoring form are shared. The main
> comparison changes only which hidden states are exposed..."

State explicitly what is held fixed and the single thing that varies.

**E. Question-driven framing.** Title is a question; section/paragraph heads are
questions answered in the first sentence:
> "We therefore focus on a central question: where should a selection verifier
> read...?" · **"Where does last-position pooling stop applying?"** ·
> **"When does a learned verifier help?"**

**F. Main vs. diagnostic discipline.** Constantly separate the central comparison
from supporting checks, and say why:
> "Because it changes the scored sequence and compute budget, we keep it as a
> diagnostic comparison rather than a main selector." · "The MLP row is a
> separability diagnostic, not a separate method family."

**G. Tell the reader how to read the evidence.**
> "This gap is the safest way to read Table 1: it separates a selector's mistake
> from the harder case where none of the four candidates is correct." · "Two
> examples illustrate how the table should be read."

**H. Honest bounds woven into the results (not just Limitations).**
> "These bounds prevent us from attributing every error to the readout when the
> candidate set itself may lack a correct answer." · "remaining selector gaps
> should not be read as purely readout failures."

**I. Contributions as design insights ("X should Y"), not a trophy.**
> "A dLLM selector should make the readout explicit, match it to the model's
> generation geometry, check whether the resulting representation is linearly
> separable, and distinguish readout errors from candidate-set limits."

**J. Tiny concrete worked examples** (in a boxed `exbox`) to ground an abstract claim:
> "Consider the prompt `Q: What is 3+4? A:` given to (a) an autoregressive decoder
> and (b) LLaDA-8B-Instruct..."

**K. Hedge the statistics, assert the pattern.**
> "we report point estimates for readability... Small rounded differences should
> not be over-interpreted, but the large readout swings and the repeated transfer
> patterns are the evidence the paper relies on."

**L. Boundary framing — find where the rule breaks, don't claim universal wins.**
> "The rest of the paper tests this boundary by changing only what the head reads."
> · "This reinforces the boundary claim rather than replacing it with a single
> universal rule."

**M. Parallelism / antithesis for contrast.**
> "Math often has a short final answer, so self-consistency can work well. Code is
> harder for voting because two correct programs can look different while passing
> the same tests."

**N. Motivate before mechanism: lead from significance, then ramp in.** Do not open on
how the system works. Order the reader's path: the world / why it matters -> the concrete
problem -> the single question -> only then the method and its mechanism. (Move C,
"mechanism-first", governs *how you explain a step*, not *what you open the paper with*.)
The abstract and intro should both start from the setting and its stakes, not the apparatus.

**O. The turn: set up the good, then pivot to the problem.** Establish the positive state
(the recent progress, the practice that works), then pivot with one contrast sentence that
exposes the problem it creates:
> "...has driven much of the recent progress in what agents can do. That reliance is also
> a vulnerability."
A single "but / yet / that very X is the Y" carries the transition; do not list problems
before you have set up why the reader should care.

**P. Self-contained: define or exemplify every term on first use.** No term appears before
it is defined, and no jargon arrives unannounced. The first time the reader meets a concept,
a symbol, or a metric, give a one-clause definition or a tiny example, including inside
table and figure captions (define every symbol there too). For an abstract term, add a
concrete example. If you coin a name, define it in the same sentence (this is move A).

**Q. Defensible hedging: no absolute you cannot defend.** Avoid *impossible, cannot, always,
never, any, proves, guarantees*, unless the statement is true by construction (then write
"by construction" and keep it exact). Prefer *tends to, in general, in our experiments,
often, usually, suggests, indicates*. State a result as "X does not match Y in our
experiments", not "X cannot match Y". Leave a reviewer room to disagree about scope, not
about facts.

**R. Citation integrity: cite every nontrivial claim, and verify each citation is real.**
Every empirical or attributed claim carries a reference. Before citing, confirm the paper
exists with the correct authors, venue, and identifier; never fabricate or guess metadata,
and never leave a placeholder author. A real, slightly-less-perfect citation beats an
invented ideal one.

**S. Sharpen in revision: one idea per paragraph; merge and cut.** Merge paragraphs that
carry the same single idea (two setup paragraphs become one), cut throat-clearing and
filler, and prefer the shorter of two equivalent phrasings. Sharpness is a second pass, not
the first draft.

## 2. Structure templates

**Abstract (significance-first, 7 beats):** (1) the stakes / why the setting matters
(ramp in, not the apparatus) → (2) the standard practice → (3) coin the property / why
the standard works → (4) **the turn**: the gap or risk it creates ("but ...") →
(5) the central **question**, stated outright → (6) the simple, targeted change →
(7) honest finding = a *set of checks / a boundary*, "no universal rule." (Moves N, O.)

**Intro (ramp + turn):** world/why-it-matters (define terms as they appear) → the
standard answer and **mechanism** for why it works → **the turn**: "that same reliance /
this very practice" creates the problem → the concrete problem, then the single question
→ targeted response (with a "not a claim that..." scope) → contributions as design
insights → forward-pointer to a figure. Open from significance, never from the apparatus.

**Results paragraph:** name the table → tell the reader the safest column to read →
the one key controlled comparison → the boundary (where preference flips) → an
explicit "this is not contradictory / not the headline" caveat.

## 3. Vocabulary

**Use:** isolate, controlled, hold fixed, geometry, boundary, diagnostic, explicit,
receptive field, anchor, competitive, informative, targeted, "the practical message
is", "we therefore", "by construction". For hedging: "tends to", "in general", "in our
experiments", "often", "usually", "suggests", "indicates", "to a first approximation".

**Avoid (hype):** novel, significant(ly), state-of-the-art, we are the first, dramatically,
clearly superior, outperform (as a headline). Prefer "simple but effective", "simple",
"a targeted consequence", "reduces the gap from X to Y".

**Avoid (indefensible absolutes):** impossible, cannot, always, never, any, all, proves,
guarantees (unless literally true by construction, then say so). Swap: impossible ->
"hard in general"; cannot -> "does not" / "tends not to"; always/never -> "often" /
"rarely"; "works on any X" -> "works across the X we test"; proves -> "shows" / "suggests".

**Sentence rhythm:** mostly short and declarative ("Selection determines whether
extra samples become extra accuracy."), punctuated by one longer precisely-qualified
sentence. One idea per sentence. Minimal adjectives.

**Punctuation (house rule): avoid em-dashes (破折号).** Do not use `---` to set off
clauses. Use a comma, colon, semicolon, parentheses, or a new sentence instead.
(Keep en-dash `--` for number ranges like `7--38\%`, `0.59--0.80`, `Tables 2--4`;
those are ranges, not dashes.) For empty table cells use `n/a`, not `---`.
Rewrites of the "not X --- it is Y" tic become "not X; it is Y" or "not X. It is Y".

## 4. Drafting / editing checklist
- [ ] Is there **one named concept** carrying the argument, defined in a sentence?
- [ ] Is the title/abstract built around a **question**?
- [ ] For each claim, did I add the **"not X; it is Y"** scope?
- [ ] Did I **justify by mechanism** before asserting a convention?
- [ ] Did I state what is **held fixed** and the **single variable** that changes?
- [ ] Are **main results vs. diagnostics** explicitly separated (and why)?
- [ ] Did I **tell the reader how to read** each table (safest column / metric)?
- [ ] Are **bounds/ceilings** used to prevent over-attribution of errors?
- [ ] Are contributions phrased as **design insights** ("X should Y"), not a trophy?
- [ ] Did I **hedge small numbers** but assert the large, repeated pattern?
- [ ] Is the framing a **boundary** ("where it holds vs. breaks"), not a universal win?
- [ ] Does the **abstract and intro open from significance** (the stakes), not the apparatus, and use one **turn** to pivot into the problem?
- [ ] Is every **term, symbol, and metric defined or exemplified on first use** (incl. in captions), and never used before it is defined?
- [ ] Did I remove **indefensible absolutes** (impossible/cannot/always/never/any/proves), keeping only construction-true claims (marked "by construction")?
- [ ] Does **every nontrivial claim have a citation**, and is **each citation verified real** (correct authors/venue/id, no fabrication, no placeholder authors)?
- [ ] Did I **sharpen** (merge one-idea paragraphs, cut filler, shorter of two phrasings)?
- [ ] Removed all hype words?
- [ ] **No em-dashes** (`---`)? Replaced with comma / colon / semicolon / parens; `n/a` in empty table cells. (En-dash `--` ranges are fine.)

## 5. Per-paper application log
Append a short entry per paper recording how the house style was applied (and any new move
that earned its place in section 1).

### ICLR-2027 do-no-harm / v1.6 (latent context compression for agentic tool use)
- Spine concept: **"do-no-harm floor"** (defined once: worst case is the bare base, recovered
  by detaching); the **circuit breaker / gate** is the mechanism on top.
- Open from significance (agent tool-use stakes -> long-context cost -> compress), turn into
  the problem ("that reliance is also a vulnerability"), then the question (moves N, O).
- Every result framed as a boundary: *where* compression helps vs. hurts (Mistral collapse),
  *where* the gate holds do-no-harm, *where* compression suffices vs. defers.
- Honesty as a feature: per-input breakage is hard to predict -> sell the floor + cheap gate,
  not a perfect detector.
- Hedged the absolutes (live-or-die -> depend heavily; impossible -> hard in general;
  cannot -> does not / tends to), keeping the detach identity exact "by construction" (move Q).
- Defined/exemplified every term and symbol incl. `delta_last`, margin, entropy, oracle, "-lite"
  (move P).
- Verified every citation on the web and corrected placeholder/wrong authors; replaced an
  unverifiable entry (OPLoRA) with the real O-LoRA (move R).
