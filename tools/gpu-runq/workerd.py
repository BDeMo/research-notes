#!/usr/bin/env python3
"""gpu-runq — file-system GPU job worker daemon.

One daemon per machine; it manages the local GPUs, pulls jobs from <root>/queue,
runs each on a free GPU, and records all state on disk. No external polling needed:
read <root>/{workers,running,done,failed} to see everything.

It is COOPERATIVE: a GPU is "free" only if its memory.used <= --free-mem MB, so it
never stomps on jobs started outside the queue (e.g. a manual sweep on other GPUs).

State dirs under <root>:
  queue/<id>.json      pending jobs (sorted by priority asc, then submit ts)
  running/<id>.json    in-flight (gpu, pid, start)   + <id>.rc (exit code)
  done/<id>.json       exit 0
  failed/<id>.json     exit != 0 / stopped / launch error
  cancelled/<id>.json  cancelled while queued
  logs/<id>.log        stdout+stderr
  workers/<host>_gpu<N>.json   per-GPU heartbeat (util/mem/job/ts)
  workers/<host>.daemon.log    daemon event log
  control/             touch-files (see below)

Control (create files under <root>/control; the daemon consumes them):
  pause            stop launching new jobs (running continue)   [resume: delete it]
  stop_all         kill ALL running jobs on this host + auto-pause
  shutdown         stop the daemon loop (running jobs keep running, detached)
  stop_<id>        kill that one running job

Queue jumping: lower 'priority' runs first (default 100). runq.py prio <id> N rewrites it.

usage: workerd.py --root <dir> [--gpus 0,1,2,3] [--poll 5] [--free-mem 2000] [--fake-gpus N]
"""
import os, sys, json, time, glob, socket, shlex, subprocess, argparse

DIRS = ["queue", "running", "done", "failed", "cancelled", "logs", "control", "workers"]

def smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        d = {}
        for ln in out.strip().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) < 4: continue
            i, u, m, t = parts[:4]
            d[int(i)] = dict(util=float(u), mem=float(m), total=float(t))
        return d
    except Exception:
        return {}

def alive(pid):
    """True only if the pid exists AND is not a zombie (defunct == effectively dead)."""
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    try:                                   # Linux: /proc state field
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except Exception:
        pass
    try:                                   # portable fallback (macOS)
        st = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                            capture_output=True, text=True, timeout=5).stdout.strip()
        return not st.startswith("Z")
    except Exception:
        return True

def set_subreaper():
    """Become a child-subreaper (Linux) so orphaned grandchildren reparent to us and can be
    reaped here instead of lingering as zombies under a non-reaping container PID 1."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except Exception:
        pass

def reap_children():
    """Reap any dead children the daemon owns so they don't linger as zombies."""
    try:
        while True:
            wpid, _ = os.waitpid(-1, os.WNOHANG)
            if wpid == 0: break
    except ChildProcessError:
        pass
    except Exception:
        pass

def _pgid_members(pgid):
    """All pids whose process-group == pgid (portable: Linux + macOS `ps`)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True,
                             text=True, timeout=10).stdout
        pids = []
        for ln in out.splitlines():
            a = ln.split()
            if len(a) >= 2 and a[1] == str(pgid):
                pids.append(int(a[0]))
        return pids
    except Exception:
        return []

class Worker:
    def __init__(self, root, host, gpus, poll, free_mem, fake, max_per_gpu=1):
        self.root = os.path.abspath(root); self.host = host
        self.poll = poll; self.free_mem = free_mem; self.fake = fake
        self.gpus = gpus; self.max_per_gpu = max_per_gpu
        for d in DIRS:
            os.makedirs(self.p(d), exist_ok=True)
        self.jobs = {}            # id -> running record (gpus, pid, ...)
        self.procs = {}           # id -> Popen handle (this session only; for zombie reaping)
        self.recover()
        self.log(f"daemon up host={host} gpus={gpus} root={self.root} "
                 f"free_mem<= {free_mem}MB fake={fake}")

    def p(self, *a): return os.path.join(self.root, *a)

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        try:
            with open(self.p("workers", f"{self.host}.daemon.log"), "a") as f:
                f.write(line + "\n")
        except Exception: pass
        print(line, flush=True)

    def recover(self):
        """Re-adopt jobs launched by a previous daemon instance (tmux restart safe)."""
        for f in glob.glob(self.p("running", "*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if r.get("host") != self.host:
                continue
            if alive(r.get("pid", -1)):
                self.jobs[r["id"]] = r
                self.log(f"recovered running {r['id']} pid {r['pid']} gpu{r['gpus']}")
            # dead ones are handled by the first reap()

    # ---------------------------------------------------------------- assignment
    def free_gpus(self, smid):
        # count our jobs per GPU; allow up to max_per_gpu (packing), one new slot offered per loop
        # so memory is re-checked between launches. free_mem acts as a safety ceiling for stacking.
        used = {}
        for r in self.jobs.values():
            for g in r["gpus"]:
                used[g] = used.get(g, 0) + 1
        free = []
        for g in self.gpus:
            if used.get(g, 0) >= self.max_per_gpu: continue
            if self.fake:
                free.append(g); continue
            info = smid.get(g)
            if info is not None and info["mem"] <= self.free_mem:
                free.append(g)
        return free

    def claim(self, ngpu_free):
        """Atomically claim the highest-priority queued job that fits in ngpu_free."""
        cands = []
        for f in glob.glob(self.p("queue", "*.json")):
            try:
                spec = json.load(open(f)); spec["_file"] = f; cands.append(spec)
            except Exception:
                pass
        cands.sort(key=lambda s: (s.get("priority", 100), s.get("ts", 0)))
        for spec in cands:
            if int(spec.get("gpus", 1)) > ngpu_free:
                continue
            claimed = self.p("running", f".claim_{self.host}_{spec['id']}.tmp")
            try:
                os.rename(spec["_file"], claimed)      # atomic; loser gets FileNotFoundError
            except FileNotFoundError:
                continue
            spec.pop("_file", None)
            return spec, claimed
        return None, None

    def launch(self, spec, gpus, claimed):
        jid = spec["id"]
        log = self.p("logs", jid + ".log"); rc = self.p("running", jid + ".rc")
        env = dict(os.environ); env.update({k: str(v) for k, v in spec.get("env", {}).items()})
        gpustr = ",".join(str(g) for g in gpus)
        env["CUDA_VISIBLE_DEVICES"] = gpustr
        cmd = spec["cmd"].replace("{gpu}", gpustr)
        cwd = spec.get("cwd") or self.root
        wrapped = f"({cmd}); echo $? > {shlex.quote(rc)}"
        fh = open(log, "w")
        p = subprocess.Popen(["bash", "-lc", wrapped], cwd=cwd, env=env,
                             stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        rec = dict(id=jid, gpus=gpus, pid=p.pid, host=self.host, start=time.time(),
                   log=log, rc=rc, cmd=cmd, cwd=cwd, priority=spec.get("priority", 100))
        json.dump(rec, open(self.p("running", jid + ".json"), "w"), indent=2)
        try: os.remove(claimed)
        except Exception: pass
        self.jobs[jid] = rec; self.procs[jid] = p
        self.log(f"START {jid} gpu={gpustr} pid={p.pid}")

    # ---------------------------------------------------------------- lifecycle
    def killjob(self, rec):
        """SIGTERM the process group, grace, then SIGKILL — both via killpg AND a portable
        ps-based sweep of every pid sharing the group, so no descendant is orphaned."""
        import signal as _sig
        try: pgid = os.getpgid(rec["pid"])
        except Exception: pgid = None
        def sweep(sg):
            try:
                if pgid is not None: os.killpg(pgid, sg)
                else: os.kill(rec["pid"], sg)
            except Exception: pass
            for pid in (_pgid_members(pgid) if pgid is not None else [rec["pid"]]):
                try: os.kill(pid, sg)
                except Exception: pass
        sweep(_sig.SIGTERM); time.sleep(2); sweep(_sig.SIGKILL)
        reap_children()

    def finish(self, jid, rec, status, code):
        rec = dict(rec); rec["end"] = time.time(); rec["status"] = status; rec["exit"] = code
        rec["dur_s"] = round(rec["end"] - rec["start"], 1)
        dest = "done" if status == "done" else "failed"
        json.dump(rec, open(self.p(dest, jid + ".json"), "w"), indent=2)
        for pth in [self.p("running", jid + ".json"), rec.get("rc", "")]:
            try: os.remove(pth)
            except Exception: pass
        self.jobs.pop(jid, None); self.procs.pop(jid, None)
        self.log(f"{status.upper()} {jid} exit={code} dur={rec['dur_s']}s")

    def reap(self):
        # completion signal = the rc file (written by the wrapper before it exits) -> restart-safe.
        # poll() the handle each loop so finished children don't linger as zombies.
        for jid, rec in list(self.jobs.items()):
            p = self.procs.get(jid)
            if p is not None:
                try: p.poll()
                except Exception: pass
            stopf = self.p("control", "stop_" + jid)
            if os.path.exists(stopf):
                self.killjob(rec); self.finish(jid, rec, "stopped", None)
                try: os.remove(stopf)
                except Exception: pass
                continue
            if os.path.exists(rec["rc"]):                      # finished; rc has authoritative code
                code = None
                try: code = int(open(rec["rc"]).read().strip())
                except Exception: pass
                self.finish(jid, rec, "done" if code == 0 else "failed", code)
            elif p is not None and p.poll() is not None:       # exited; trust Popen returncode
                code = p.returncode                            # (0 => done even if rc not flushed yet)
                self.finish(jid, rec, "done" if code == 0 else "failed", code)
            elif p is None and not alive(rec["pid"]):          # recovered job, gone, no rc -> unknown
                self.finish(jid, rec, "failed", None)

    def heartbeat(self, smid):
        for g in self.gpus:
            info = smid.get(g, {})
            on = [jid for jid, r in self.jobs.items() if g in r["gpus"]]
            job = ",".join(on) if on else None
            try:
                json.dump(dict(host=self.host, gpu=g, util=info.get("util"),
                               mem=info.get("mem"), total=info.get("total"),
                               job=job, ts=time.time()),
                          open(self.p("workers", f"{self.host}_gpu{g}.json"), "w"))
            except Exception: pass

    def run(self):
        while True:
            reap_children()                 # clear zombies the daemon owns
            smid = {} if self.fake else smi()
            self.reap()
            if os.path.exists(self.p("control", "stop_all")):
                self.log("stop_all -> killing running + pausing")
                for jid, rec in list(self.jobs.items()):
                    self.killjob(rec); self.finish(jid, rec, "stopped", None)
                open(self.p("control", "pause"), "w").close()
                try: os.remove(self.p("control", "stop_all"))
                except Exception: pass
            self.heartbeat(smid)
            if os.path.exists(self.p("control", "shutdown")):
                self.log("shutdown requested -> exiting loop (running jobs stay detached)")
                try: os.remove(self.p("control", "shutdown"))
                except Exception: pass
                break
            if not os.path.exists(self.p("control", "pause")):
                free = self.free_gpus(smid)
                while free:
                    spec, claimed = self.claim(len(free))
                    if not spec: break
                    need = int(spec.get("gpus", 1))
                    use, free = free[:need], free[need:]
                    try:
                        self.launch(spec, use, claimed)
                    except Exception as e:
                        self.log(f"launch FAIL {spec.get('id')}: {e}")
                        try:
                            spec["error"] = str(e); spec["status"] = "launch_error"
                            json.dump(spec, open(self.p("failed", spec["id"] + ".json"), "w"), indent=2)
                            os.remove(claimed)
                        except Exception: pass
            time.sleep(self.poll)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpus", default="", help="comma indices to manage; default = all visible")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--free-mem", type=float, default=2000.0, help="GPU free if memory.used<=this (MB)")
    ap.add_argument("--max-per-gpu", type=int, default=1, help="max concurrent jobs to PACK per GPU")
    ap.add_argument("--fake-gpus", type=int, default=0, help="TEST: present N virtual always-free GPUs")
    ap.add_argument("--host", default=socket.gethostname())
    a = ap.parse_args()
    set_subreaper()
    if a.fake_gpus > 0:
        gpus = list(range(a.fake_gpus)); fake = True
    else:
        fake = False
        gpus = [int(x) for x in a.gpus.split(",") if x.strip()] if a.gpus else sorted(smi().keys())
        if not gpus:
            print("no GPUs found; use --gpus or --fake-gpus", file=sys.stderr); sys.exit(1)
    Worker(a.root, a.host, gpus, a.poll, a.free_mem, fake, a.max_per_gpu).run()

if __name__ == "__main__":
    main()
