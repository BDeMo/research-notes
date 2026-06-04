#!/usr/bin/env bash
# Janus GRID — unified metric harness over (models x 12 datasets), distributed across 4 GPUs.
# Usage: orch_grid.sh <pybin> <janus_out> <hf_home> <ctxlen> <model1> [model2 ...]
set -u
PYBIN="$1"; OUT="$2"; HF="$3"; CTX="$4"; shift 4
MODELS=("$@")
JHOME="$(dirname "$OUT")"; PY="$JHOME/janus_run.py"; VIZ="$JHOME/janus_viz.py"
export JANUS_OUT="$OUT" HF_HOME="$HF" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$JHOME/logs"; mkdir -p "$LOG" "$OUT"; cd "$JHOME"
DATASETS=(wikitext mmlu math gsm8k triviaqa bbh squad hotpotqa quality narrativeqa musr msmarco)

cell () { local gpu="$1" m="$2" d="$3"
  echo "[grid $(date +%H:%M:%S)] START $m/$d gpu$gpu" | tee -a "$LOG/grid.log"
  CUDA_VISIBLE_DEVICES="$gpu" $PYBIN "$PY" grid --model "$m" --dataset "$d" --ctxlen "$CTX" \
     --n_texts 12 --fisher_batches 8 --sft_steps 80 > "$LOG/g_${m}_${d}.log" 2>&1
  echo "[grid $(date +%H:%M:%S)] END   $m/$d rc=$?" | tee -a "$LOG/grid.log"; }

# Build the full cell list (model x dataset), distribute round-robin to 4 GPUs.
CELLS=(); for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do CELLS+=("$m|$d"); done; done
# precompute spectra once per model (serial, fast) to avoid first-cell duplication
for m in "${MODELS[@]}"; do cell 0 "$m" wikitext; done   # wikitext cell also seeds spectra cache

gpu_lane () { local gpu="$1"; shift; for c in "$@"; do m="${c%%|*}"; d="${c##*|}"; cell "$gpu" "$m" "$d"; done; touch "$OUT/_GRIDLANE_${gpu}_DONE"; }
L0=(); L1=(); L2=(); L3=(); i=0
for c in "${CELLS[@]}"; do
  d="${c##*|}"; [ "$d" = "wikitext" ] && continue   # already done in seed
  case $((i % 4)) in 0) L0+=("$c");; 1) L1+=("$c");; 2) L2+=("$c");; 3) L3+=("$c");; esac; i=$((i+1))
done
gpu_lane 0 "${L0[@]}" & P0=$!
gpu_lane 1 "${L1[@]}" & P1=$!
gpu_lane 2 "${L2[@]}" & P2=$!
gpu_lane 3 "${L3[@]}" & P3=$!
wait $P0 $P1 $P2 $P3
echo "[grid $(date +%H:%M:%S)] GRID NODE DONE" | tee -a "$LOG/grid.log"; touch "$OUT/_GRID_DONE"
