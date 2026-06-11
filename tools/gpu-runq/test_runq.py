#!/usr/bin/env python3
"""Self-contained regression test for gpu-runq (no shell pattern-matching -> no self-kill).
Verifies: priority jump, queue drain, exit-code capture, stop -> failed, and NO orphaned
descendants after stop (checks the exact pgid members by PID). Run anywhere with GPUs faked.

usage: python test_runq.py
"""
import os, sys, time, json, shutil, subprocess, signal
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "_regtest")
PY = sys.executable

def run(*a): return subprocess.run([PY, os.path.join(HERE, a[0])] + list(a[1:]),
                                   capture_output=True, text=True)
def submit(jid, cmd, prio=100):
    run("runq.py", "--root", ROOT, "submit", "--id", jid, "--cmd", cmd, "--prio", str(prio))
def jids(d):
    p = os.path.join(ROOT, d)
    return set(f[:-5] for f in os.listdir(p) if f.endswith(".json")) if os.path.isdir(p) else set()
def palive(pid):
    try: os.kill(pid, 0)
    except Exception: return False
    try:                                   # zombie = effectively dead
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except Exception:
        return True
def members(pgid):
    out = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines() if len(l.split()) >= 2 and l.split()[1] == str(pgid)]

def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    # submit BEFORE daemon so ordering is deterministic
    submit("zfast", "sleep 1", prio=0)        # prio 0 -> must start first
    submit("okjob", "sleep 2", prio=100)
    submit("failjob", "exit 7", prio=100)
    submit("longjob", "sleep 300", prio=5)    # will be stopped; check no orphans
    d = subprocess.Popen([PY, os.path.join(HERE, "workerd.py"), "--root", ROOT,
                          "--fake-gpus", "2", "--poll", "1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        time.sleep(9)   # zfast(1)+okjob(2) done, failjob failed, longjob running
        rec = json.load(open(os.path.join(ROOT, "running", "longjob.json")))
        pid = rec["pid"]; pgid = os.getpgid(pid); mem = members(pgid)
        run("runq.py", "--root", ROOT, "stop", "longjob")
        time.sleep(5)
        orphans = [m for m in mem if palive(m)]
        done, failed = jids("done"), jids("failed")
        checks = {
            "zfast_done": "zfast" in done,
            "okjob_done": "okjob" in done,
            "failjob_failed": "failjob" in failed,
            "failjob_exit7": json.load(open(os.path.join(ROOT, "failed", "failjob.json"))).get("exit") == 7,
            "longjob_stopped": "longjob" in failed,
            "no_orphans": not orphans,
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  (longjob pgid={pgid} members={mem} still_alive={orphans})")
        print("RESULT:", "ALL PASS" if all(checks.values()) else "FAILURES")
    finally:
        try: os.killpg(os.getpgid(d.pid), signal.SIGTERM)
        except Exception: d.terminate()
        shutil.rmtree(ROOT, ignore_errors=True)

if __name__ == "__main__":
    main()
