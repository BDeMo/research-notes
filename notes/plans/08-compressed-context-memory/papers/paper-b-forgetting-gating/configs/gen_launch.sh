#!/bin/bash
# gen_launch.sh — Paper-B GENERALITY table (cross-architecture / cross-family / size).
# One model per invocation. Representative bench subset at DISCLOSED N=100 (linear GDN uses torch fallback,
# so full-set is infeasible; the headline FULL table stays Qwen3-8B). Text-side methods work on ANY causal LM;
# KV-eviction (kvzip/knorm) only applies to quadratic-attention models (skipped when ARCH=linear).
# Env: MODEL (hf id), MTAG (short tag), ARCH (quad|linear), GPUS (space-sep). Output: /mnt/persist/grid_generality.
set -u
: "${MODEL:?set MODEL}"; : "${MTAG:?set MTAG}"; : "${ARCH:=quad}"; : "${GPUS:=0 1 2 3}"
ROOT=/mnt/persist/gcm; OUT=/mnt/persist/grid_generality; mkdir -p "$OUT"
PP="$ROOT/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src"
COMMON="HF_HOME=/mnt/persist/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton GCM_NVAL=100"
cd "$ROOT"

METH_BASE=(
 "window|run_baseline.py|GCM_BASELINE=txl GCM_TXL_WINDOW=1024"
 "rag|run_baseline.py|GCM_BASELINE=rag GCM_RAG_BUDGET=2048"
 "ll2|run_baseline.py|GCM_BASELINE=llmlingua GCM_LL_RATE=0.5"
 "tome|run_baseline.py|GCM_BASELINE=tome GCM_TOME_RATIO=0.5"
 "imp|run_baseline.py|GCM_BASELINE=imp GCM_IMP_SPAN=32 GCM_LL_RATE=0.5"
)
METH_KV=(
 "kvzip|run_kvpress.py|GCM_KVPRESS=kvzip GCM_KV_RATIO=0.5"
 "knorm|run_kvpress.py|GCM_KVPRESS=knorm GCM_KV_RATIO=0.5"
)
METHODS=("${METH_BASE[@]}")
[ "$ARCH" = "quad" ] && METHODS+=("${METH_KV[@]}")

# representative subset: retrieval / hardest-MC / multi-hop / literary-MC(full-hurts) / extractive
BENCHES=(
 "ruler_niah|88|16384|3600"
 "longbench_v2|12|16384|5400"
 "lb_hotpotqa|12|16384|5400"
 "quality_hard|8|8192|3600"
 "squad_v2|8|4096|2400"
)
CELLS=()
for m in "${METHODS[@]}"; do IFS='|' read -r mn sc me <<< "$m"
  for b in "${BENCHES[@]}"; do IFS='|' read -r ev nc mx to <<< "$b"
    CELLS+=("g_${MTAG}_${mn}_${ev}|$sc|$me|$ev|$nc|$mx|$to"); done; done
echo "[$MTAG/$ARCH] generality cells: ${#CELLS[@]} on GPUs: $GPUS"
GA=($GPUS); NG=${#GA[@]}
for gi in "${!GA[@]}"; do
( g=${GA[$gi]}; i=$gi
  while [ $i -lt ${#CELLS[@]} ]; do
    IFS='|' read -r tag sc me ev nc mx to <<< "${CELLS[$i]}"; log="$OUT/${tag}.log"
    grep -qa RECIPE_EVAL "$log" 2>/dev/null && { i=$((i+NG)); continue; }
    echo "[$g] START $tag @ $(date +%T)"
    env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP $COMMON $me GCM_EVAL=$ev GCM_NCHUNKS=$nc GCM_MAXCTX=$mx GCM_TAG=$tag \
        timeout $to /mnt/persist/venv/bin/python experiments/$sc "$MODEL" > "$log" 2>&1 </dev/null \
        || echo "CELL_FAIL $tag rc=$? @ $(date +%T)" >> "$log"
    grep -qa RECIPE_EVAL "$log" && echo "[$g] DONE $tag" || echo "[$g] NOEVAL $tag"
    i=$((i+NG))
  done; echo "[$g] WORKER DONE @ $(date +%T)" ) &
done
wait; echo "GEN_DONE $MTAG @ $(date +%T)"
