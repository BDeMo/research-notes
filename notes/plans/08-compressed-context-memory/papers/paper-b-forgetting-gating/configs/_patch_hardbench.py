#!/usr/bin/env python3
"""Idempotently add the two HARDEST mainstream long-context benchmarks to the harness:
  - LongBench-v2 (THUDM/LongBench-v2): 503 hard MC over 8k-2M-word real contexts.
  - InfiniteBench longbook_choice_eng (>100k-token English book MC).
Both are MULTIPLE-CHOICE so scoring is clean accuracy (no F1 noise).
Edits the SHARED /mnt/persist copies of datasets.py and data.py. Safe to re-run."""
import re, io, sys

DS = "/mnt/persist/v17/llm-infra/src/llm_infra/datasets.py"
DATA = "/mnt/persist/v17/mem-embedding/src/mem_embedding/gcm/data.py"

LBV2 = '''

def generate_longbench_v2(*, config="all", n_items=100, n_chunks=8, seed=12345,
                          split="train", max_chars=8_000_000):
    """LongBench-v2 (Bai et al. 2025): 503 hard human-annotated MC questions over real long
    contexts (8k-2M words), 4-way choice. gold = answer letter (MC-scored). ``config`` filters
    by domain ('all' keeps everything). Real length; only a safety char cap, no task truncation."""
    import random, json
    from huggingface_hub import hf_hub_download
    fp = hf_hub_download(repo_id="THUDM/LongBench-v2", filename="data.json", repo_type="dataset")
    rows = json.load(open(fp, encoding="utf-8"))
    letters = ["A", "B", "C", "D"]
    idx = list(range(len(rows))); random.Random(seed).shuffle(idx)
    items = []
    for si in idx:
        if len(items) >= n_items:
            break
        row = rows[si]
        if config not in ("all", None, "") and str(row.get("domain", "")).lower() != str(config).lower():
            continue
        ctx = str(row.get("context", "")).strip()
        q = str(row.get("question", "")).strip()
        opts = [str(row.get("choice_" + L, "")).strip() for L in letters]
        ans = str(row.get("answer", "")).strip().upper()
        if not ctx or not q or any(not o for o in opts) or ans not in letters:
            continue
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars]
        chunks = _chunk_text_balanced(ctx, n_chunks)
        opt_lines = "\\n".join(f"{letters[k]}) {opts[k]}" for k in range(4))
        query = (f"{q}\\n\\nChoose the single best answer.\\n{opt_lines}\\n\\n"
                 f"Respond with exactly one letter from {{A, B, C, D}}.")
        items.append(LongContextItem(
            item_id="lbv2_" + str(row.get("_id", si)), chunks=chunks, query=query, gold=ans,
            needle_chunk=-1,
            meta={"src_idx": si, "options": opts, "dataset": "longbench_v2",
                  "domain": row.get("domain"), "difficulty": row.get("difficulty"),
                  "length_bucket": row.get("length"), "n_chunks": n_chunks,
                  "answer_choices": letters}))
    if not items:
        raise RuntimeError("generate_longbench_v2 produced 0 items (config=%r)" % config)
    return items


def generate_infinitebench(*, config="longbook_choice_eng", n_items=100, n_chunks=8, seed=12345,
                           split="test", max_chars=6_000_000):
    """InfiniteBench (Zhang et al. 2024), MC subset ``longbook_choice_eng`` by default: English
    book QA at >100k tokens, 4-way choice (MC-scored). Downloads the task jsonl directly (no
    load_dataset script). Real length; safety char cap only."""
    import random, json, gzip
    from huggingface_hub import hf_hub_download
    fp = None
    for fn in (config + ".jsonl", "data/" + config + ".jsonl", config + ".jsonl.gz",
               "longbook_choice_eng.jsonl"):
        try:
            fp = hf_hub_download(repo_id="xinrongzhang2022/InfiniteBench", filename=fn, repo_type="dataset")
            break
        except Exception:
            continue
    if fp is None:
        raise RuntimeError("generate_infinitebench: cannot fetch task %r" % config)
    opener = gzip.open if fp.endswith(".gz") else open
    rows = [json.loads(l) for l in opener(fp, "rt", encoding="utf-8") if l.strip()]
    letters = ["A", "B", "C", "D", "E", "F"]
    idx = list(range(len(rows))); random.Random(seed).shuffle(idx)
    items = []
    for si in idx:
        if len(items) >= n_items:
            break
        row = rows[si]
        ctx = str(row.get("context", "")).strip()
        q = str(row.get("input", "")).strip()
        opts = [str(o).strip() for o in (row.get("options") or [])]
        ans = row.get("answer")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        ans = str(ans).strip()
        if not ctx or not q or len(opts) < 2:
            continue
        if ans in letters[:len(opts)]:
            gold = ans
        else:
            try:
                gold = letters[[o.lower() for o in opts].index(ans.lower())]
            except ValueError:
                continue
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars]
        chunks = _chunk_text_balanced(ctx, n_chunks)
        opt_lines = "\\n".join(f"{letters[k]}) {opts[k]}" for k in range(len(opts)))
        query = (f"{q}\\n\\nChoose the single best answer.\\n{opt_lines}\\n\\n"
                 f"Respond with exactly one letter.")
        items.append(LongContextItem(
            item_id="inf_" + config + "_" + str(si), chunks=chunks, query=query, gold=gold,
            needle_chunk=-1,
            meta={"src_idx": si, "options": opts, "dataset": "infinitebench",
                  "task": config, "n_chunks": n_chunks, "answer_choices": letters[:len(opts)]}))
    if not items:
        raise RuntimeError("generate_infinitebench produced 0 items (task=%r)" % config)
    return items
'''

def patch_datasets():
    s = open(DS, encoding="utf-8").read()
    if "def generate_longbench_v2" in s:
        print("[datasets] already patched"); return
    s = s.rstrip() + "\n" + LBV2
    open(DS, "w", encoding="utf-8").write(s)
    print("[datasets] appended generate_longbench_v2 + generate_infinitebench")

def patch_data():
    s = open(DATA, encoding="utf-8").read()
    changed = False
    # 1) import
    if "generate_longbench_v2" not in s:
        s = s.replace(
            "from llm_infra.datasets import generate_longbench\n",
            "from llm_infra.datasets import generate_longbench\n"
            "from llm_infra.datasets import generate_longbench_v2, generate_infinitebench\n", 1)
        changed = True
    # 2) registry entries (insert right after the EVAL_GENS opening brace line)
    if '"longbench_v2"' not in s:
        anchor = 'EVAL_GENS: dict[str, Callable[..., list]] = {\n'
        add = (anchor
               + '    "longbench_v2": generate_longbench_v2,\n'
               + '    "infbench_choice": lambda **k: generate_infinitebench(config="longbook_choice_eng", **k),\n')
        s = s.replace(anchor, add, 1)
        changed = True
    # 3) MC_BENCHES
    if '"longbench_v2"' not in s.split("MC_BENCHES")[1].split("}")[0]:
        s = s.replace(
            'MC_BENCHES = {"quality", "quality_hard", "musr_mm", "rca_openrca", "rca_rcaeval"}',
            'MC_BENCHES = {"quality", "quality_hard", "musr_mm", "rca_openrca", "rca_rcaeval", "longbench_v2", "infbench_choice"}',
            1)
        changed = True
    # 4) DOMAIN entries
    if '"longbench_v2": "longreal"' not in s:
        s = s.replace(
            '    "ms_marco": "web",\n',
            '    "ms_marco": "web",\n    "longbench_v2": "longreal", "infbench_choice": "longreal",\n', 1)
        changed = True
    if changed:
        open(DATA, "w", encoding="utf-8").write(s)
        print("[data] registry/MC/DOMAIN updated")
    else:
        print("[data] already patched")

if __name__ == "__main__":
    patch_datasets()
    patch_data()
    print("PATCH_OK")
