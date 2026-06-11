#!/usr/bin/env python3
"""gpu-runq client — submit/inspect/control jobs on the file-system queue.

The queue ROOT is taken from --root or $GPU_RUNQ_ROOT.

  runq.py submit --id JID --cmd "python x.py ..." [--prio 100] [--gpus 1] [--cwd DIR] [--env K=V ...]
  runq.py ls                 # full board: workers + queue + running + recent done/failed
  runq.py status             # one-line per GPU + counts
  runq.py log JID [-n 40]    # tail a job log
  runq.py prio JID 0         # queue-jump (lower = sooner)
  runq.py cancel JID         # remove from queue (or stop if running)
  runq.py stop JID           # stop a running job
  runq.py pause | resume | stopall | shutdown
  runq.py clean [--done] [--failed]   # clear finished records
"""
import os, sys, json, time, glob, argparse

def root_of(a):
    r = a.root or os.environ.get("GPU_RUNQ_ROOT")
    if not r: sys.exit("set --root or $GPU_RUNQ_ROOT")
    return os.path.abspath(r)

def P(root, *a): return os.path.join(root, *a)
def load(f):
    try: return json.load(open(f))
    except Exception: return {}
def jobs_in(root, d): return [load(f) for f in sorted(glob.glob(P(root, d, "*.json")))]

def ensure(root):
    for d in ["queue","running","done","failed","cancelled","logs","control","workers"]:
        os.makedirs(P(root, d), exist_ok=True)

def cmd_submit(a):
    root = root_of(a); ensure(root)
    jid = a.id or ("job_%d" % int(time.time()*1000))
    env = {}
    for kv in a.env or []:
        k, _, v = kv.partition("="); env[k] = v
    spec = dict(id=jid, cmd=a.cmd, priority=a.prio, gpus=a.gpus,
                cwd=a.cwd or "", env=env, ts=time.time())
    # don't silently overwrite a running/queued job
    for d in ["queue", "running"]:
        if os.path.exists(P(root, d, jid + ".json")):
            sys.exit(f"job '{jid}' already in {d}; pick another --id or cancel it")
    json.dump(spec, open(P(root, "queue", jid + ".json"), "w"), indent=2)
    print(f"queued {jid} (prio {a.prio}, gpus {a.gpus})")

def _fmt_age(ts):
    if not ts: return "?"
    s = int(time.time() - ts)
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"

def cmd_ls(a):
    root = root_of(a); ensure(root)
    hbs = [load(f) for f in glob.glob(P(root, "workers", "*_gpu*.json"))]
    print("== GPUs ==")
    for h in sorted(hbs, key=lambda x: (x.get("host",""), x.get("gpu",0))):
        fresh = "ok" if time.time() - h.get("ts", 0) < 30 else "STALE"
        print(f"  {h.get('host'):16s} gpu{h.get('gpu')} util={str(h.get('util')):>5}% "
              f"mem={str(int(h.get('mem') or 0)):>6}MB job={h.get('job')} [{fresh}]")
    q = sorted(jobs_in(root, "queue"), key=lambda s: (s.get("priority",100), s.get("ts",0)))
    print(f"\n== QUEUE ({len(q)}) ==")
    for s in q[:40]:
        print(f"  [{s.get('priority',100):>3}] {s['id']:32s} gpus={s.get('gpus',1)} cmd={s.get('cmd','')[:60]}")
    run = jobs_in(root, "running")
    print(f"\n== RUNNING ({len(run)}) ==")
    for r in run:
        print(f"  {r.get('id'):32s} gpu={r.get('gpus')} pid={r.get('pid')} up={_fmt_age(r.get('start'))}")
    done, failed = jobs_in(root, "done"), jobs_in(root, "failed")
    print(f"\n== DONE {len(done)} | FAILED {len(failed)} ==")
    for r in sorted(failed, key=lambda x: x.get('end',0), reverse=True)[:8]:
        print(f"  FAIL {r.get('id'):32s} exit={r.get('exit')} ({r.get('status')})")

def cmd_status(a):
    root = root_of(a)
    hbs = [load(f) for f in glob.glob(P(root, "workers", "*_gpu*.json"))]
    busy = sum(1 for h in hbs if h.get("job"))
    print(f"workers={len(hbs)} busy={busy} | queue={len(glob.glob(P(root,'queue','*.json')))} "
          f"running={len(glob.glob(P(root,'running','*.json')))} "
          f"done={len(glob.glob(P(root,'done','*.json')))} "
          f"failed={len(glob.glob(P(root,'failed','*.json')))}")

def cmd_log(a):
    root = root_of(a); f = P(root, "logs", a.id + ".log")
    if not os.path.exists(f): sys.exit(f"no log for {a.id}")
    lines = open(f, errors="replace").read().splitlines()
    print("\n".join(lines[-a.n:]))

def cmd_prio(a):
    root = root_of(a); f = P(root, "queue", a.id + ".json")
    if not os.path.exists(f): sys.exit(f"{a.id} not in queue (already running/done?)")
    s = load(f); s["priority"] = a.value; json.dump(s, open(f, "w"), indent=2)
    print(f"{a.id} priority -> {a.value}")

def cmd_cancel(a):
    root = root_of(a); qf = P(root, "queue", a.id + ".json")
    if os.path.exists(qf):
        s = load(qf); s["status"] = "cancelled"
        json.dump(s, open(P(root, "cancelled", a.id + ".json"), "w"), indent=2); os.remove(qf)
        print(f"cancelled queued {a.id}")
    elif os.path.exists(P(root, "running", a.id + ".json")):
        open(P(root, "control", "stop_" + a.id), "w").close()
        print(f"stop signal sent to running {a.id}")
    else:
        print(f"{a.id} not found in queue/running")

def cmd_stop(a):
    root = root_of(a); open(P(root, "control", "stop_" + a.id), "w").close()
    print(f"stop signal -> {a.id}")

def _touch(root, name, on):
    ensure(root); f = P(root, "control", name)
    if on: open(f, "w").close(); print(f"{name} ON")
    else:
        try: os.remove(f)
        except Exception: pass
        print(f"{name} OFF")

def cmd_clean(a):
    root = root_of(a); n = 0
    for d in (["done"] if a.done else []) + (["failed"] if a.failed else []):
        for f in glob.glob(P(root, d, "*.json")): os.remove(f); n += 1
    print(f"removed {n} records")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="c", required=True)
    s = sub.add_parser("submit"); s.add_argument("--id"); s.add_argument("--cmd", required=True)
    s.add_argument("--prio", type=int, default=100); s.add_argument("--gpus", type=int, default=1)
    s.add_argument("--cwd"); s.add_argument("--env", nargs="*"); s.set_defaults(fn=cmd_submit)
    for name, fn in [("ls", cmd_ls), ("status", cmd_status)]:
        sub.add_parser(name).set_defaults(fn=fn)
    s = sub.add_parser("log"); s.add_argument("id"); s.add_argument("-n", type=int, default=40); s.set_defaults(fn=cmd_log)
    s = sub.add_parser("prio"); s.add_argument("id"); s.add_argument("value", type=int); s.set_defaults(fn=cmd_prio)
    s = sub.add_parser("cancel"); s.add_argument("id"); s.set_defaults(fn=cmd_cancel)
    s = sub.add_parser("stop"); s.add_argument("id"); s.set_defaults(fn=cmd_stop)
    sub.add_parser("pause").set_defaults(fn=lambda a: _touch(root_of(a), "pause", True))
    sub.add_parser("resume").set_defaults(fn=lambda a: _touch(root_of(a), "pause", False))
    sub.add_parser("stopall").set_defaults(fn=lambda a: _touch(root_of(a), "stop_all", True))
    sub.add_parser("shutdown").set_defaults(fn=lambda a: _touch(root_of(a), "shutdown", True))
    s = sub.add_parser("clean"); s.add_argument("--done", action="store_true"); s.add_argument("--failed", action="store_true"); s.set_defaults(fn=cmd_clean)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
