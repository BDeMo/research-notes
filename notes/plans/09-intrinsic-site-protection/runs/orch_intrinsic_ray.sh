#!/usr/bin/env bash
# Janus intrinsic-metrics study — ray (GPU 0,1,2,3). Full Qwen3 ladder, domain=trivia
# (ray has no math/mmlu cached). Gives the scale-axis of the coupling map.
set -u
JHOME=/root/janus
PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/mnt/persist/hf-cache PYTHONUNBUFFERED=1
LOG=$JHOME/logs; mkdir -p "$JANUS_OUT" "$LOG"; cd $JHOME

run () { local gpu="$1" name="$2"; shift 2
  echo "[intr $(date +%H:%M:%S)] START $name gpu$gpu: $*" | tee -a "$LOG/intr.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[intr $(date +%H:%M:%S)] END   $name rc=$?" | tee -a "$LOG/intr.log"; }

intr () { local gpu="$1" model="$2"
  run $gpu i_${model}_trivia intrinsic --model $model --domain trivia --sft_steps 150 --fisher_batches 12 --lr 2e-5
  cp $JANUS_OUT/$model/intrinsic.json $JANUS_OUT/$model/intrinsic_trivia.json 2>/dev/null
  cp $JANUS_OUT/$model/intrinsic.npz  $JANUS_OUT/$model/intrinsic_trivia.npz  2>/dev/null
}

laneR0 () { intr 0 qwen3-0.6b; touch $JANUS_OUT/_IR0_DONE; }
laneR1 () { intr 1 qwen3-1.7b; touch $JANUS_OUT/_IR1_DONE; }
laneR2 () { intr 2 qwen3-4b;   touch $JANUS_OUT/_IR2_DONE; }
laneR3 () { intr 3 qwen3-8b; intr 3 qwen3-14b; touch $JANUS_OUT/_IR3_DONE; }

laneR0 & P0=$!; laneR1 & P1=$!; laneR2 & P2=$!; laneR3 & P3=$!
wait $P0 $P1 $P2 $P3
echo "[intr $(date +%H:%M:%S)] RAY INTRINSIC DONE" | tee -a "$LOG/intr.log"; touch $JANUS_OUT/_INTR_DONE
