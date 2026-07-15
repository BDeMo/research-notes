#!/bin/bash
# explore_gen.sh — multi-model generality sweep (patch-free benches so it runs on ANY pod disk).
# Env: MODELS=";"-sep list of "tag|hf_path|arch(quad|linear)"; GPUS="0 1.."; OUTDIR (default grid_generality);
#      NVAL (default 100). Methods: window,rag,ll2,tome,imp (+kvzip,knorm for quad). Runs models sequentially,
#      each spread across GPUS. Skips cells already having RECIPE_EVAL. Output g_<tag>_<method>_<bench>.log
set -u
: "${MODELS:?set MODELS}"; : "${GPUS:=0 1}"; : "${OUTDIR:=/mnt/persist/grid_generality}"; : "${NVAL:=100}"
ROOT=/mnt/persist/gcm; mkdir -p "$OUTDIR"
PP="$ROOT/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src"
COMMON="HF_HOME=/mnt/persist/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton GCM_NVAL=$NVAL"
cd "$ROOT"
# patch-free benches: bench|nchunks|maxctx|timeout
BENCHES=("ruler_niah|88|16384|4200" "hotpot_qa|8|4096|2400" "squad_v2|8|4096|2400" "quality_hard|8|8192|3600" "trivia_qa|8|4096|2400")
METH_BASE=(
 "window|run_baseline.py|GCM_BASELINE=txl GCM_TXL_WINDOW=1024"
 "rag|run_baseline.py|GCM_BASELINE=rag GCM_RAG_BUDGET=2048"
 "ll2|run_baseline.py|GCM_BASELINE=llmlingua GCM_LL_RATE=0.5"
 "tome|run_baseline.py|GCM_BASELINE=tome GCM_TOME_RATIO=0.5"
 "imp|run_baseline.py|GCM_BASELINE=imp GCM_IMP_SPAN=32 GCM_LL_RATE=0.5")
METH_KV=(
 "kvzip|run_kvpress.py|GCM_KVPRESS=kvzip GCM_KV_RATIO=0.5"
 "knorm|run_kvpress.py|GCM_KVPRESS=knorm GCM_KV_RATIO=0.5")

IFS=';' read -ra MODELLIST <<< "$MODELS"
for entry in "${MODELLIST[@]}"; do
  [ -z "$entry" ] && continue
  IFS='|' read -r mtag mpath march <<< "$entry"
  METHODS=("${METH_BASE[@]}"); [ "$march" = "quad" ] && METHODS+=("${METH_KV[@]}")
  CELLS=()
  for m in "${METHODS[@]}"; do IFS='|' read -r mn sc me <<< "$m"
    for b in "${BENCHES[@]}"; do IFS='|' read -r ev nc mx to <<< "$b"
      CELLS+=("g_${mtag}_${mn}_${ev}|$sc|$me|$ev|$nc|$mx|$to"); done; done
  echo "[$mtag/$march] ${#CELLS[@]} cells @ $(date +%T)"
  GA=($GPUS); NG=${#GA[@]}
  for gi in "${!GA[@]}"; do
  ( g=${GA[$gi]}; i=$gi
    while [ $i -lt ${#CELLS[@]} ]; do
      IFS='|' read -r tag sc me ev nc mx to <<< "${CELLS[$i]}"; log="$OUTDIR/${tag}.log"
      grep -qa RECIPE_EVAL "$log" 2>/dev/null && { i=$((i+NG)); continue; }
      echo "[$mtag $g] START $tag @ $(date +%T)"
      env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP $COMMON $me GCM_EVAL=$ev GCM_NCHUNKS=$nc GCM_MAXCTX=$mx GCM_TAG=$tag \
          timeout $to /mnt/persist/venv/bin/python experiments/$sc "$mpath" > "$log" 2>&1 </dev/null \
          || echo "CELL_FAIL $tag rc=$? @ $(date +%T)" >> "$log"
      i=$((i+NG))
    done ) &
  done
  wait
  echo "[$mtag] MODEL DONE @ $(date +%T)"
done
echo "EXPLORE_GEN_ALL_DONE @ $(date +%T)"
