#!/usr/bin/env python3
"""Re-register the `cmg_rca` bench that the cmg-rca-gcm demo needs (load_items('cmg_rca', ...)).
It maps to generate_rca reading the pre-built Nokia CMG cases at /mnt/persist/datasets/cmg_rca_built/cases.jsonl.
Idempotent; edits the SHARED data.py only."""
DATA = "/mnt/persist/v17/mem-embedding/src/mem_embedding/gcm/data.py"
CMG_PATH = "/mnt/persist/datasets/cmg_rca_built/cases.jsonl"

s = open(DATA, encoding="utf-8").read()
changed = False

if '"cmg_rca"' not in s:
    anchor = 'EVAL_GENS: dict[str, Callable[..., list]] = {\n'
    add = anchor + '    "cmg_rca": lambda **k: generate_rca(source="%s", **k),\n' % CMG_PATH
    s = s.replace(anchor, add, 1)
    changed = True

# MC bench + primary metric + domain, mirroring rca_openrca
if '"cmg_rca"' not in s.split("MC_BENCHES")[1].split("}")[0]:
    s = s.replace(
        'MC_BENCHES = {"quality", "quality_hard", "musr_mm", "rca_openrca", "rca_rcaeval", "longbench_v2", "infbench_choice"}',
        'MC_BENCHES = {"quality", "quality_hard", "musr_mm", "rca_openrca", "rca_rcaeval", "longbench_v2", "infbench_choice", "cmg_rca"}',
        1)
    changed = True

if '"cmg_rca": "primary_service_match"' not in s:
    s = s.replace(
        '    "rca_openrca": "primary_service_match",\n',
        '    "rca_openrca": "primary_service_match",\n    "cmg_rca": "primary_service_match",\n', 1)
    changed = True

if '"cmg_rca": "ops"' not in s:
    s = s.replace(
        '    "rca_openrca": "ops", "rca_rcaeval": "ops",\n',
        '    "rca_openrca": "ops", "rca_rcaeval": "ops", "cmg_rca": "ops",\n', 1)
    changed = True

if changed:
    open(DATA, "w", encoding="utf-8").write(s)
    print("PATCH_OK cmg_rca registered ->", CMG_PATH)
else:
    print("already registered")
