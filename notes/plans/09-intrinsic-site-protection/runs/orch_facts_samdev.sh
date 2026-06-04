#!/usr/bin/env bash
# Janus FACTS — sam-dev (GPU 0,1,3; GPU2 reserved for 32B intrinsic). Qwen2.5 + GLM-4.
set -u
JHOME=/home/devuser/janus; PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/home/devuser/.cache/huggingface PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$JHOME/logs; mkdir -p "$LOG"; cd $JHOME
run () { local gpu="$1" m="$2"
  echo "[facts $(date +%H:%M:%S)] START $m gpu$gpu" | tee -a "$LOG/facts.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" facts --model "$m" > "$LOG/f_$m.log" 2>&1
  echo "[facts $(date +%H:%M:%S)] END   $m rc=$?" | tee -a "$LOG/facts.log"; }
( run 0 qwen2.5-1.5b; run 0 qwen2.5-1.5b-instruct; run 0 qwen3-8b ) &
( run 1 qwen2.5-7b-instruct; run 1 glm4-9b; run 1 qwen3-14b ) &
( run 3 glm4-32b ) &
wait
python3 "$VIZ" >> "$LOG/viz.log" 2>&1
echo "[facts $(date +%H:%M:%S)] SAMDEV FACTS DONE" | tee -a "$LOG/facts.log"; touch $JANUS_OUT/_FACTS_DONE
