#!/usr/bin/env bash
# Janus FACTS — ray (GPU 0,1,2,3). Qwen3 ladder. Forward-only broad metrics.
set -u
JHOME=/root/janus; PY=$JHOME/janus_run.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/mnt/persist/hf-cache PYTHONUNBUFFERED=1
LOG=$JHOME/logs; mkdir -p "$LOG"; cd $JHOME
run () { local gpu="$1" m="$2"
  echo "[facts $(date +%H:%M:%S)] START $m gpu$gpu" | tee -a "$LOG/facts.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" facts --model "$m" > "$LOG/f_$m.log" 2>&1
  echo "[facts $(date +%H:%M:%S)] END   $m rc=$?" | tee -a "$LOG/facts.log"; }
( run 0 qwen3-0.6b; run 0 qwen3-8b ) &
( run 1 qwen3-1.7b; run 1 qwen3-14b ) &
( run 2 qwen3-4b ) &
wait
echo "[facts $(date +%H:%M:%S)] RAY FACTS DONE" | tee -a "$LOG/facts.log"; touch $JANUS_OUT/_FACTS_DONE
