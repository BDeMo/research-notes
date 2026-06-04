#!/usr/bin/env bash
# Janus intrinsic-metrics study — sam-dev (GPU 0,2,3). Cross-domain (math+trivia).
set -u
JHOME=/home/devuser/janus
PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/home/devuser/.cache/huggingface PYTHONUNBUFFERED=1
LOG=$JHOME/logs; mkdir -p "$JANUS_OUT" "$LOG"; cd $JHOME

run () { local gpu="$1" name="$2"; shift 2
  echo "[intr $(date +%H:%M:%S)] START $name gpu$gpu: $*" | tee -a "$LOG/intr.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[intr $(date +%H:%M:%S)] END   $name rc=$?" | tee -a "$LOG/intr.log"; }

# intrinsic writes intrinsic.json/.npz keyed per-model; to keep math vs trivia
# separate we copy the artifact after each domain.
intr () { local gpu="$1" model="$2" domain="$3"
  run $gpu i_${model}_${domain} intrinsic --model $model --domain $domain --sft_steps 150 --fisher_batches 12 --lr 2e-5
  cp $JANUS_OUT/$model/intrinsic.json $JANUS_OUT/$model/intrinsic_${domain}.json 2>/dev/null
  cp $JANUS_OUT/$model/intrinsic.npz  $JANUS_OUT/$model/intrinsic_${domain}.npz  2>/dev/null
}

laneA () { intr 2 qwen2.5-1.5b math; intr 2 qwen2.5-1.5b trivia; intr 2 qwen2.5-1.5b-instruct math; python3 "$VIZ" >>"$LOG/viz.log" 2>&1; touch $JANUS_OUT/_IA_DONE; }
laneB () { intr 3 qwen3-8b math; intr 3 qwen3-8b trivia; python3 "$VIZ" >>"$LOG/viz.log" 2>&1; touch $JANUS_OUT/_IB_DONE; }
laneC () { intr 0 qwen3-14b math; intr 0 qwen2.5-7b-instruct math; intr 0 qwen2.5-7b-instruct trivia; python3 "$VIZ" >>"$LOG/viz.log" 2>&1; touch $JANUS_OUT/_IC_DONE; }

vizloop () { for i in $(seq 1 40); do sleep 600; python3 "$VIZ" >>"$LOG/viz.log" 2>&1
  [ -f $JANUS_OUT/_IA_DONE ] && [ -f $JANUS_OUT/_IB_DONE ] && [ -f $JANUS_OUT/_IC_DONE ] && break; done; }

laneA & PA=$!; laneB & PB=$!; laneC & PC=$!; vizloop & PV=$!
wait $PA $PB $PC; python3 "$VIZ" >>"$LOG/viz.log" 2>&1; kill $PV 2>/dev/null
echo "[intr $(date +%H:%M:%S)] SAMDEV INTRINSIC DONE" | tee -a "$LOG/intr.log"; touch $JANUS_OUT/_INTR_DONE
