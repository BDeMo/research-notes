#!/bin/bash
GPU=$1; Q=$2
cd /mnt/persist/gcm
COMMON='PYTHONPATH=/mnt/persist/gcm/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src HF_HOME=/mnt/persist/hf-cache MEM_RCA_OPENRCA=/mnt/persist/datasets/openrca_built/cases.jsonl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton WANDB_MODE=disabled GCM_GOLD_MAX=64 GCM_GEN_MAX=64 GCM_NVAL=128 GCM_DEPTH=half GCM_RECUR=1 GCM_CHUNK=1024 GCM_DEV=0.05 GCM_SHUFFLE=1 GCM_ACCUM=8 GCM_LR=2e-4 GCM_NTRAIN=3000 GCM_LORA=64 GCM_K=256'
mkdir -p /mnt/persist/grid_main
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  ENVKV="${line%%:::*}"; MODEL="$(echo "${line##*:::}" | xargs)"
  TAG=$(echo "$ENVKV" | grep -oE 'GCM_TAG=[^ ]+' | cut -d= -f2)
  [ -f "/mnt/persist/grid_main/$TAG.json.done" ] && { echo "[g$GPU] skip $TAG (done)"; continue; }
  if echo "$ENVKV" | grep -q GCM_KVPRESS; then SCRIPT=run_kvpress.py; elif echo "$ENVKV" | grep -q GCM_BASELINE; then SCRIPT=run_baseline.py; else SCRIPT=run_recipe.py; fi
  while [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" -gt 3000 ]; do sleep 30; done
  echo "[g$GPU] $(date +%H:%M) start $TAG ($SCRIPT)"
  env CUDA_VISIBLE_DEVICES=$GPU $COMMON $ENVKV /mnt/persist/venv/bin/python experiments/$SCRIPT "$MODEL" > /mnt/persist/grid_main/$TAG.log 2>&1
  grep -aq RECIPE_EVAL "/mnt/persist/grid_main/$TAG.log" && touch "/mnt/persist/grid_main/$TAG.json.done"
  echo "[g$GPU] $(date +%H:%M) end   $TAG rc=$?"
done < "$Q"
echo "[g$GPU] queue drained $(date +%H:%M)"
