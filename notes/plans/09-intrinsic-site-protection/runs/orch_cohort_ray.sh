#!/usr/bin/env bash
# Janus 7-9B COHORT (native bf16, no quant) in the iso venv (torch2.11/tf5.10) on ray.
# One model per GPU; intrinsic THEN facts (so facts cross-links intrinsic.npz).
set -u
JHOME=/root/janus; PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/mnt/persist/hf-cache PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYBIN=/root/q35iso/bin/python
LOG=$JHOME/logs; mkdir -p "$LOG"; cd $JHOME
run () { local gpu="$1" name="$2"; shift 2
  echo "[cohort $(date +%H:%M:%S)] START $name gpu$gpu: $*" | tee -a "$LOG/cohort.log"
  CUDA_VISIBLE_DEVICES="$gpu" $PYBIN "$PY" "$@" > "$LOG/c_$name.log" 2>&1
  echo "[cohort $(date +%H:%M:%S)] END   $name rc=$?" | tee -a "$LOG/cohort.log"; }
lane () { local gpu="$1" m="$2"
  run $gpu ${m}_intr intrinsic --model "$m" --domain trivia --sft_steps 150 --fisher_batches 12
  run $gpu ${m}_facts facts --model "$m"
  touch $JANUS_OUT/_COH_${m}_DONE
}
lane 0 qwen3.5-9b &
lane 1 glm4-9b &
lane 2 qwen3-8b &
lane 3 qwen2.5-7b-instruct &
wait
$PYBIN "$VIZ" >> "$LOG/viz.log" 2>&1
echo "[cohort $(date +%H:%M:%S)] COHORT DONE" | tee -a "$LOG/cohort.log"; touch $JANUS_OUT/_COHORT_DONE
