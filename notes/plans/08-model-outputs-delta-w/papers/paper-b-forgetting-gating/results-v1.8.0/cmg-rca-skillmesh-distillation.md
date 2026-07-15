# CMG-RCA → SkillMesh Distillation — Working / Recall Doc

_Last updated: 2026-06-21. Purpose: state capture so work can be recalled and resumed._

## Where we are (one paragraph)
The cmg_rca MC benchmark ("pick the root-cause module letter from compressed telemetry") is **near-ill-posed** and caps low. We have pivoted to **distilling the real SkillMesh RCA skill OUTPUT** (a structured report), where the thing M compresses is the **evidence bundle** (logs/counters/stack-traces/analysis), not the generic perf dump. Distillation targets are **hand-authored by Opus 4.8** (me), not rule-templated. Hard process rule from user: **every generation must pass a case-study gate (no collapse + data valid) BEFORE any downstream training.**

## Key findings (evidence-backed)
- **cmg MC ceiling ≈ 0.32** even training AND testing on the SAME 38 cases (full overfit, recurrent). Parallel=0.16, K128=0.26. → architecture/task bottleneck, NOT data volume.
- **Compressed-M generation COLLAPSES**: MC-trained cmg adapters emit letter-spam `D B C D E F...`; best openrca adapter (r_rd_dfull, 0.391) emits the SAME `'telemetry module'` sentence repeated for every input. Input-independent + repetitive.
- **Full-context (uncompressed) base is genuine but says "system healthy / no incident"** — the perf-dump telemetry genuinely lacks failure evidence. The answer is NOT in the telemetry.
- The "collapse cosine metric" I used was an **artifact** (healthy RCA also reads ~1.0 due to hard-norm); valid signals = compress-vs-no_ctx accuracy + generation diversity.
- **Where compression DOES work** (for the paper): BFCL tool-use compress **0.65–0.70** (no_ctx 0.02), across 5 bases — clean strong result. rca_openrca **0.30–0.39** (no_ctx ~0.15), recon most useful, deeper encoder helps. cmg/squad/quality/toolace ≈ chance.

## Real SkillMesh RCA (how it's actually done) — `~/workspace/cmg-rca`
- Pipeline: AD agent + **RCA agent** (agentic): fetch DTS → similar-DTS → analyze logs/crash/ATS/stack-trace → module detect → trace to **file:function:line** → structured report → fix.
- Report sections (real, 15–20KB): Executive Summary (Defect ID/Root Cause/Confidence/Severity/Impact) · Issue Analysis (Symptoms w/ counters, Root Cause Identification w/ file:fn:line, Execution Flow, Code Analysis, Stack Trace, ATS/TS Findings, Log Correlation) · Impact · Fix Recommendations · Verification.
- Scored by **LLM-judge + SME on rubric** (binary Scope/Completeness, Likert Evidence-Grounding), NOT MC accuracy. Files: `nf_evals/output_rubric/{rubric_config.py, llm_judge/prompts.py, ui/RCA_output_sample.md}`.

## Data assets (local, real)
- `nf_data/dts_for_evaluation/<MODULE>/DTS*` = **47 eval DTS** (jh18/acctmgr11/gtp7/gxl7/nserv2/pfcp2). Raw telemetry is **git-LFS-broken locally** → only **38** have real telemetry (the cmg_rca test set). 47 needs Nokia-access re-pull (user can't either → stay at 38).
- `nf_data/bug_history/<MODULE>/*_dts_fetched.json` = real historical DTS bugs (jh391/acctmgr329/nserv239/gtp166/gxl147/pfcp62), descriptions w/ symptoms+counters+analysis. **Clean training source (non-eval, 41 overlaps excluded).**
- `*_enriched.json` = primary_file/primary_function/files_changed per bug (root-cause location).
- `nf_analysis/lte-*-{architecture,troubleshooting,bugs}-chunked.json` = per-module knowledge.

## Compression reframing (CONFIRMED by user)
- **Compress the EVIDENCE BUNDLE** (symptom logs/counters/stack-trace/analysis), not the perf dump.
- **Gold = SkillMesh RCA report format**, hand-authored by Opus 4.8.

## Distillation dataset construction
- Input (context, what M compresses) = bug description / evidence, **module names MASKED** (`<MOD>`), 0 leak verified.
- Gold = SkillMesh-format report (Executive Summary / Symptoms Observed / Root Cause Identification w/ Location / Analysis / Confidence), **module named in gold (it's the answer)**.
- Balanced across the 6 modules. NON-eval DTS only (no leakage with the 38 test).
- v1 (rule-templated, 735): `cmg_skill_cases.jsonl` — Symptoms section was weak (copied first sentence). Superseded.
- **v2 (Opus + grounded assembly, CLEANED): DONE.** **516 balanced** (acctmgr/jh/nserv 100, gxl 94, gtp 89, pfcp 33) + **797 full pool**. Deployed all pods `/mnt/persist/datasets/cmg_skill/cases.jsonl` (+`cases_full.jsonl`); workspace backup `cmg-rca/cmg_skill_opus_{516clean,797full}.jsonl`; assembler `cmg-rca/assemble_bulk.py`.
  - **53 hand-authored premium** (read evidence + genuine multi-sentence root-cause + verified file::function/fix) in `/tmp/gcmedit/cmg_skill_opus_reports{,,2}.json`.
  - **463 bulk** = grounded assembly from REAL fields only (extracted symptom/error/counter/crash-log lines + real summary). NO hallucinated location/fix (dts_fetched<->enriched join unreliable -> omitted). Coverity special-cased (parse Defect/function/file/line). Auto multi-analysis was UNRELIABLE (copied log lines / [3LS] escalation-template / chatter) -> **dropped**; bulk keeps Symptoms + clean Summary only.
  - **Cleaning gates (verified 0)**: dropped template-only boilerplate, FEATURE/dev-tasks (not RCA), customer-escalation [3LS]/[4LS] template leaks, URL/email/chatter; require a real defect signal; dedupe by summary; mask context. **Final: 0 text-junk, 0 context leaks, 516/516 distinct.** Crash-log hex/sig- symptoms KEPT (valid evidence).
  - Honest quality note: 53 hand-authored are rich multi-point; 463 bulk are clean but lighter (symptom + summary + module). For deeper analysis at scale -> hand-author more or use a strong-teacher API (none local).
- **v3 (QUALITY-FIRST, hand-authored only): CURRENT.** User directive: every record (prompt + evidence + report) must be comprehensive & high quality; build incrementally across turns; drop shallow bulk.
  - **49-record HQ core** = `/tmp/gcmedit/cmg_skill_hq.jsonl` (workspace `cmg-rca/cmg_skill_hq_49.jsonl`), deployed as the active `cmg_skill` cases.jsonl. Bulk-cleaned 516/797 kept only as `cases_full.jsonl` (NOT used for HQ training).
  - Each HQ record: **FULL untruncated evidence** (612-2358 chars, masked, 0 leaks) + **comprehensive prompt** (now instructs the full SkillMesh section structure - updated in generate_cmg_skill on all pods) + **comprehensive hand-authored report** (865-1470 chars, all have Root Cause Identification).
  - **Correctness pass (audit-driven)**: 4 risky inferred-mechanism reports CORRECTED to honest hypotheses w/ confidence (DTS442722 crash, DTS442580 bitmap, DTS443940 leak, DTS443411 PCF-FH); 7 feature/dev tasks DROPPED (not RCA). Audit baseline: of 56 hand, 5 Coverity-correct + 16 evidence-grounded + 35 inferred (now softened/dropped where risky).
  - Balance (small, will grow): acctmgr7/gtp10/gxl8/jh9/nserv9/pfcp6. **TODO: keep hand-authoring batches across turns to grow the HQ set while keeping every record comprehensive + grounded.**
  - Corrections file: `/tmp/gcmedit/cmg_skill_corrections.json`; full evidence: `/tmp/gcmedit/hand_full_evidence.json`.

## Infra / how to run
- Pods (per-pod `/mnt/persist`, NOT shared): sam-dev-{d1525(4gpu),d1530(2),d1420(2),test(1)}. free1-6 = other users (don't touch).
- Bench registration in `mem_embedding/gcm/data.py` (EVAL_GENS + MC_BENCHES + PRIMARY). Registered: cmg_rca, cmg_aug (MC), cmg_all (overfit, split=all), cmg_skill (generation, gold=report).
- Runner: `qrun10.sh <gpu> <jobfile>`; job line `tag|model|bench|extra_env`; COMMON sets K=64 etc; **overrides must come LAST** in extra_env. Results: `/mnt/persist/grid10/<tag>.log` (RECIPE_EVAL line) + adapters `/mnt/persist/gcm/out/Qwen3.5-9B/<tag>_adapters.pt`.
- Key env: GCM_CHUNK=4096 (chunked encode, fits long ctx single-GPU), GCM_RECUR=1 (AutoCompressor-style recurrent, best for long ctx), GCM_XCHUNK=N (cross-chunk refine), GCM_DEPTH, GCM_RECON/DEV, GCM_DISTILL=0 for cmg (teacher OOM on long ctx), GCM_EVAL_SKIP_FULL=1, GCM_GOLD_MAX/GEN_MAX (220 for reports).
- Diagnostics: `eval_cmg_all.py` (all-38 per-module acc), `gen_probe.py` (print generated text — collapse check), `collapse_check.py`.

## v3 DATA DESIGN: logic-chain, no truncation (CURRENT, per user 2026-06-21 night)
User directives: (1) **no truncation of inputs OR outputs** in authored data; (2) every record (report + prompt + evidence) **comprehensive**, quality > volume; (3) OK to author across many turns (not one shot); (4) **expose every link of the RCA logic chain (each step's input->output)**, not just end-to-end evidence->report, to fit GCM (compress carried state at each link) and plug into the SkillMesh skill.
- **5-link chain per DTS** (mirrors cmg-rca skill workflow): 1 triage (evidence->observations) · 2 localize (->module+rationale) · 3 rootcause (->symptom/proximate/root chain + confidence + file::function) · 4 report (->full SkillMesh report) · 5 fix (->minimal fix + files + verification).
- Each step record: `{chain_id, module, step, step_name, prompt, input, output}`. Comprehensive per-step prompt. input = carried reasoning state (GCM compresses it). NO caps/truncation.
- GCM training: compress `input` -> M, generate `output`; chain links connect via outputs -> trains compression along the whole reasoning chain.
- Authoring (hand, Opus): `/tmp/gcmedit/cmg_chains_authored.json` (per-DTS: evidence + triage/localize/rootcause/report/fix). Build: `build_chains.py` -> `cmg_skill_chains.jsonl`. **DONE so far: 2 chains (DTS443911 acctmgr, DTS442780 pfcp) = 10 step-records.** Continue authoring more chains (prioritize the 16 evidence-GROUNDED + 5 Coverity cases; soften/verify the 35 inferred before charting).
- SUPERSEDES the 516/797 single-shot bulk dataset (too shallow + truncated + unverified mechanisms).

## v4 cmg-distill-real: decomposed from REAL repo reports (CURRENT BEST, per user 2026-06-21 late)
User: use the ORIGINAL cmg repo samples + real cmg reports; each case 30 analyses+plans + final result.
- Source = the repo's REAL SkillMesh outputs: `nf_evals/output_rubric/ui/agent_outputs/<DTS>[ _nokb]/{AD_output, RCA_output}.md`. 56 non-empty real RCA reports (5.8-20KB, median 11.8KB), 38 unique DTS (kb+nokb variants), true root cause + file:function + fix + verification + evidence (cites similar bugs).
- Decomposed each real RCA report into links: 20 analysis (Exec Summary / Symptoms / Root Cause / Execution Flow / Code Analysis / Stack Trace / Log Correlation / Impact) + 10 plan (Fix / Alternatives / Files / Verification / Checklist) + 1 final_report. Input = the AD (anomaly) report = evidence. **NO fabrication — all content is the real agent's report.** No truncation.
- Output: **`/tmp/gcmedit/cmg_distill_real.jsonl` = 1705 samples** (55 cases x ~31). Workspace: `cmg-rca/skillmesh_distill_data/cmg-distill-real/`. Builder: `cmg-rca/extract_real_chains.py`. Every case reaches >=30 links.
- LEAKAGE NOTE: these are the 47 EVAL DTS -> overlaps the 38-case cmg test. This is the deliberate "train on real test cases + real reports" (强行学测试集) approach for demonstrating the compressor reproduces real RCA; NOT a clean held-out generalization claim. Keep a held-out split if a generalization number is needed.
- This SUPERSEDES the templated cmg-distill (15,480, low quality) and the single-shot 516/797. Recommended training data = cmg-distill-real.

## Process RED LINE (user)
**每次生成出来的东西都要先做 case study：确认不 collapse + 数据有效，再做之后的事情。** Don't launch big training before a small case-study gate passes.

## Next steps
1. Finish Opus-authored RCA reports for the 30 evidences (+ expand toward ~120 balanced if quality holds).
2. Build `cmg_skill_opus.jsonl` (input=masked evidence, gold=my report).
3. **Case-study gate**: short distill → generate from M → verify non-collapse + valid. THEN scale training.
4. Depth sweep (sk_d2/d4/d6) + compare to MC ceiling. Eval ideally via LLM-judge rubric, not just MC.
5. Open threads: ceil_d4 (depth=4 MC ceiling) result; mv1_par clean V1 all-38; TXL baseline (optional).
