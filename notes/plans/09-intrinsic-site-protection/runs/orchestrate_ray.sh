#!/usr/bin/env bash
# Janus ray orchestrator — 4 lanes (GPU 0,1,2,3). Data-LIGHT: detect + NIAH(+ablations)
# across the Qwen3 scaling ladder + trivia full-FT coupling (no MMLU/MATH/peft needed).
set -u
JHOME=/root/janus
PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/mnt/persist/hf-cache PYTHONUNBUFFERED=1
LOG=$JHOME/logs; mkdir -p "$JANUS_OUT" "$LOG"; cd $JHOME

run () { local gpu="$1" name="$2"; shift 2
  echo "[orch $(date +%H:%M:%S)] START $name gpu$gpu: $*" | tee -a "$LOG/orch.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[orch $(date +%H:%M:%S)] END   $name rc=$?" | tee -a "$LOG/orch.log"; }

ladder () {  # <gpu> <model> [sft?]  detect + niah base/retr/sink (+ optional trivia coupling)
  local G=$1 M=$2 D=$JANUS_OUT/$2
  run $G ${M}_det  detect --model $M
  run $G ${M}_nb   niah   --model $M --lengths 1500,4000 --samples 5
  run $G ${M}_nr   niah   --model $M --lengths 1500,4000 --samples 5 --ablate --ablate_site retr --ablate_k 20
  run $G ${M}_ns   niah   --model $M --lengths 1500,4000 --samples 5 --ablate --ablate_site sink --ablate_k 20
  if [ "${3:-}" = "sft" ]; then
    run $G ${M}_cb  capeval --model $M --n 150
    run $G ${M}_sft sft     --model $M --domain trivia --mode full --protect none --steps 300 --lr 2e-5 --bs 2
    run $G ${M}_cn  capeval --model $M --n 150 --ckpt $D/ckpt_trivia_full_none
    run $G ${M}_pr  sft     --model $M --domain trivia --mode full --protect retrieval --steps 300 --lr 2e-5 --bs 2 --protect_k 16
    run $G ${M}_cpr capeval --model $M --n 150 --ckpt $D/ckpt_trivia_full_retrieval
  fi
}

laneR0 () { ladder 0 qwen3-0.6b sft; python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_R0_DONE; }
laneR1 () { ladder 1 qwen3-1.7b sft; python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_R1_DONE; }
laneR2 () { ladder 2 qwen3-4b   sft; python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_R2_DONE; }
laneR3 () { ladder 3 qwen3-8b; ladder 3 qwen3-14b; python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_R3_DONE; }

vizloop () { for i in $(seq 1 60); do sleep 900; python3 "$VIZ" >> "$LOG/viz.log" 2>&1
  [ -f $JANUS_OUT/_R0_DONE ] && [ -f $JANUS_OUT/_R1_DONE ] && [ -f $JANUS_OUT/_R2_DONE ] && [ -f $JANUS_OUT/_R3_DONE ] && break; done; }

laneR0 & P0=$!; laneR1 & P1=$!; laneR2 & P2=$!; laneR3 & P3=$!; vizloop & PV=$!
wait $P0 $P1 $P2 $P3
python3 "$VIZ" >> "$LOG/viz.log" 2>&1; kill $PV 2>/dev/null
echo "[orch $(date +%H:%M:%S)] RAY ALL DONE" | tee -a "$LOG/orch.log"; touch $JANUS_OUT/_ALL_DONE
