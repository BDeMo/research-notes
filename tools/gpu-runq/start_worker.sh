#!/usr/bin/env bash
# Start (or restart) the gpu-runq daemon in a detached tmux session on this machine.
# usage: start_worker.sh <root> [python] [extra workerd args...]
#   e.g. start_worker.sh /root/runq-root /root/q35iso/bin/python --free-mem 3000
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:?usage: start_worker.sh <root> [python] [args...]}"; shift || true
PY="${1:-python3}"; [ $# -gt 0 ] && shift || true
SESSION="gpuworker"
mkdir -p "$ROOT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' exists; killing + restarting"; tmux kill-session -t "$SESSION"
fi
tmux new-session -d -s "$SESSION" \
  "$PY $HERE/workerd.py --root $ROOT $* 2>&1 | tee -a $ROOT/workers/daemon.console.log"
sleep 2
tmux ls 2>/dev/null | grep "$SESSION" && echo "started '$SESSION' (root=$ROOT, py=$PY, args=$*)"
echo "tail:  tmux attach -t $SESSION   (detach: Ctrl-b d)"
