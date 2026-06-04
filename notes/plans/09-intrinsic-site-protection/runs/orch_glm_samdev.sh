#!/usr/bin/env bash
# Janus intrinsic — GLM-4 cross-family (sam-dev GPU 0,1,2,3). math+trivia.
set -u
JHOME=/home/devuser/janus
PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/home/devuser/.cache/huggingface PYTHONUNBUFFERED=1
LOG=$JHOME/logs; mkdir -p "$JANUS_OUT" "$LOG"; cd $JHOME

run () { local gpu="$1" name="$2"; shift 2
  echo "[glm $(date +%H:%M:%S)] START $name gpu$gpu: $*" | tee -a "$LOG/glm.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[glm $(date +%H:%M:%S)] END   $name rc=$?" | tee -a "$LOG/glm.log"; }

intr () { local gpu="$1" model="$2" domain="$3"
  run $gpu g_${model}_${domain} intrinsic --model $model --domain $domain --sft_steps 150 --fisher_batches 12 --lr 2e-5
  cp $JANUS_OUT/$model/intrinsic.json $JANUS_OUT/$model/intrinsic_${domain}.json 2>/dev/null
  cp $JANUS_OUT/$model/intrinsic.npz  $JANUS_OUT/$model/intrinsic_${domain}.npz  2>/dev/null
}

l0 () { intr 0 glm4-9b math; touch $JANUS_OUT/_G0_DONE; }
l1 () { intr 1 glm4-9b trivia; touch $JANUS_OUT/_G1_DONE; }
l2 () { intr 2 glm4-32b math; touch $JANUS_OUT/_G2_DONE; }
l3 () { intr 3 glm4-32b trivia; touch $JANUS_OUT/_G3_DONE; }

l0 & P0=$!; l1 & P1=$!; l2 & P2=$!; l3 & P3=$!
wait $P0 $P1 $P2 $P3
python3 "$VIZ" >>"$LOG/viz.log" 2>&1
echo "[glm $(date +%H:%M:%S)] GLM DONE" | tee -a "$LOG/glm.log"; touch $JANUS_OUT/_GLM_DONE
