# gpu-runq — file-system GPU job queue + per-machine worker

> **ARCHIVE / reuse snapshot.** Canonical source + active development live in the workspace
> at `gpu-runq/` (pip-installable: `pip install -e gpu-runq` → `gpu-workerd`, `runq`). This
> copy is kept for archival/reuse only — **do not edit here**; sync snapshots in. The daemon
> runs on each pod in a tmux session (`gpuworker`), deployed at `/root/gpu-runq`.

A tiny, dependency-free (pure stdlib) scheduler so **GPUs stay saturated** without anyone
hand-watching `nvidia-smi`. One **daemon per machine** monitors local GPUs and runs jobs from a
**file-system queue**; you submit/stop/reprioritize jobs by writing files. Status = read files.

## Why
- The agent should **not** poll GPUs. The daemon writes per-GPU heartbeats; reading a few small
  files tells you the whole picture.
- All run state (todo / running / done / failed) lives on the file system → restart-safe,
  inspectable, and controllable by touching files.
- **Cooperative**: a GPU is "free" only if `memory.used <= --free-mem` MB, so the daemon never
  stomps on jobs launched outside the queue.

## Files
- `workerd.py` — the daemon (claim → run → reap; control files; heartbeats; multi-GPU jobs).
- `runq.py` — client: submit / ls / status / log / prio / cancel / stop / pause / resume / stopall / shutdown.
- `start_worker.sh` — launch the daemon in a detached tmux session `gpuworker`.

## Layout under `<root>`
```
queue/      pending  <id>.json        running/  <id>.json (+<id>.rc)
done/       exit 0                     failed/   exit!=0 / stopped / launch_error
cancelled/  cancelled while queued     logs/     <id>.log
workers/    <host>_gpu<N>.json heartbeat, <host>.daemon.log
control/    pause | stop_all | shutdown | stop_<id>
```

## Quickstart
```bash
# 1) start the daemon on a machine (manages all visible GPUs; only idle ones get used)
export GPU_RUNQ_ROOT=/root/runq-root
bash start_worker.sh $GPU_RUNQ_ROOT /root/q35iso/bin/python --free-mem 3000
#   or pin GPUs:  ... --gpus 0,1,2,3

# 2) submit jobs (cmd is GPU-agnostic: the daemon sets CUDA_VISIBLE_DEVICES for you)
python runq.py submit --id gate_qwen3-8b \
  --cmd "/root/q35iso/bin/python gate_study.py --model qwen3-8b --datasets squad,niah --n 60" \
  --cwd /root/janus-methods --prio 100

# 3) look (no nvidia-smi needed)
python runq.py ls
python runq.py status
python runq.py log gate_qwen3-8b -n 30

# 4) control
python runq.py prio gate_qwen3-8b 0     # jump the queue
python runq.py cancel gate_qwen3-8b     # drop from queue, or stop if running
python runq.py pause / resume           # stop / resume launching new jobs
python runq.py stopall                  # kill all running on the host + pause
python runq.py shutdown                 # stop the daemon loop (running jobs detach & continue)
```

## Notes
- **Queue jumping** = lower `priority` (default 100; urgent = 0). `prio` rewrites it in place.
- **Multi-GPU** job: `--gpus 2`; the daemon assigns when ≥2 local GPUs are free and sets
  `CUDA_VISIBLE_DEVICES=g1,g2`. (Big jobs can wait behind small ones — acceptable; no strict
  reservation.)
- Restart-safe: on start the daemon re-adopts its still-running jobs (records in `running/`).
- One daemon per host. Claims use atomic `rename`, so a shared-FS double-start won't double-run.
