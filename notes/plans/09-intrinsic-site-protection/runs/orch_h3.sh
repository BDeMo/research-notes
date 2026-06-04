#!/usr/bin/env bash
# Janus H3 causal test — one model, 5 protection variants across 4 GPUs.
# Usage: orch_h3.sh <pybin> <janus_out> <hf_home> <model> <domain> <steps> <k>
set -u
PYBIN="$1"; OUT="$2"; HF="$3"; MODEL="$4"; DOM="$5"; STEPS="$6"; K="$7"
JHOME="$(dirname "$OUT")"; PY="$JHOME/janus_run.py"
export JANUS_OUT="$OUT" HF_HOME="$HF" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$JHOME/logs"; mkdir -p "$LOG" "$OUT"; cd "$JHOME"
cell () { local gpu="$1" v="$2"
  echo "[h3 $(date +%H:%M:%S)] START $MODEL/$v gpu$gpu" | tee -a "$LOG/h3.log"
  CUDA_VISIBLE_DEVICES="$gpu" $PYBIN "$PY" h3 --model "$MODEL" --protect "$v" --protect_k "$K" \
     --domain "$DOM" --steps "$STEPS" --n_eval 120 > "$LOG/h3_${MODEL}_${v}.log" 2>&1
  echo "[h3 $(date +%H:%M:%S)] END   $MODEL/$v rc=$?" | tee -a "$LOG/h3.log"; }
# 5 variants on 4 GPUs: none+deltaw share GPU0
( cell 0 none; cell 0 deltaw ) &
( cell 1 lc ) &
( cell 2 retrieval ) &
( cell 3 random ) &
wait
echo "[h3 $(date +%H:%M:%S)] H3 NODE DONE ($MODEL)" | tee -a "$LOG/h3.log"; touch "$OUT/_H3_${MODEL}_DONE"
