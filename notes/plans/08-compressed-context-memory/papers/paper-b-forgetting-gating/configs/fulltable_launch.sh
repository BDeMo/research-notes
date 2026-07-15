#!/bin/bash
# fulltable_launch.sh — PAPER-GRADE main table on the HARDEST + core long-context benches.
# FULL test split for headline long-context benches; short sanity benches disclosed-subsampled to N=500.
# Runs on free1 GPUs 0-3 (clean idle 96GB cards); writes to /mnt/persist/grid_fulltable.
# Each run_baseline cell ALSO emits the no_ctx (blind) and full (uncompressed) columns.
set -u
ROOT=/mnt/persist/gcm
OUT=/mnt/persist/grid_fulltable; mkdir -p "$OUT"
MODEL=Qwen/Qwen3-8B
PP="$ROOT/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src"
COMMON="HF_HOME=/mnt/persist/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton"
cd "$ROOT"

# method families -> env (run_baseline gives no_ctx+full+method; run_kvpress gives method only)
# tag | script | env
METHODS=(
 "window|run_baseline.py|GCM_BASELINE=txl GCM_TXL_WINDOW=1024"
 "rag|run_baseline.py|GCM_BASELINE=rag GCM_RAG_BUDGET=2048"
 "ll2|run_baseline.py|GCM_BASELINE=llmlingua GCM_LL_RATE=0.5"
 "tome|run_baseline.py|GCM_BASELINE=tome GCM_TOME_RATIO=0.5"
 "imp|run_baseline.py|GCM_BASELINE=imp GCM_IMP_SPAN=32 GCM_LL_RATE=0.5"
 "kvzip|run_kvpress.py|GCM_KVPRESS=kvzip GCM_KV_RATIO=0.5"
 "knorm|run_kvpress.py|GCM_KVPRESS=knorm GCM_KV_RATIO=0.5"
)

# bench | nchunks | maxctx | nval | timeout(s)   -- nval=100000 => FULL split (generator caps to dataset size)
BENCHES=(
 # --- HARDEST real long-context (FULL) ---
 "longbench_v2|16|32768|100000|14400"
 "infbench_choice|16|32768|100000|14400"
 # --- LongBench-v1 real long-context (FULL) ---
 "lb_2wikimqa|12|16384|100000|7200"
 "lb_hotpotqa|12|16384|100000|7200"
 "lb_musique|12|16384|100000|7200"
 "lb_multifieldqa|12|16384|100000|7200"
 "lb_qasper|12|16384|100000|7200"
 "lb_narrativeqa|12|16384|100000|7200"
 # --- synthetic retrieval at 32k (FULL fixed synthetic N=500) ---
 "ruler_niah|176|32768|500|10800"
 # --- literary/reasoning MC (FULL) ---
 "quality|8|8192|100000|7200"
 "quality_hard|8|8192|100000|7200"
 "musr_mm|8|8192|100000|5400"
 # --- short-context sanity (DISCLOSED subset N=500) ---
 "squad_v2|8|4096|500|3600"
 "hotpot_qa|8|4096|500|3600"
 "trivia_qa|8|4096|500|3600"
 "ms_marco|8|4096|500|3600"
)

CELLS=()
for m in "${METHODS[@]}"; do
  IFS='|' read -r mn script menv <<< "$m"
  for b in "${BENCHES[@]}"; do
    IFS='|' read -r ev nc mx nv to <<< "$b"
    CELLS+=("ft_${mn}_${ev}|$script|$menv|$ev|$nc|$mx|$nv|$to")
  done
done
echo "full-table cells: ${#CELLS[@]}"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3}); NG=${#GPUS[@]}
for gi in "${!GPUS[@]}"; do
( g=${GPUS[$gi]}; i=$gi
  while [ $i -lt ${#CELLS[@]} ]; do
    IFS='|' read -r tag script menv ev nc mx nv to <<< "${CELLS[$i]}"
    log="$OUT/${tag}.log"
    if grep -qa RECIPE_EVAL "$log" 2>/dev/null; then i=$((i+NG)); continue; fi
    echo "[$g] START $tag @ $(date +%T)  (nc=$nc mx=$mx nval=$nv to=$to)"
    env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP $COMMON $menv \
        GCM_EVAL=$ev GCM_NCHUNKS=$nc GCM_MAXCTX=$mx GCM_NVAL=$nv GCM_TAG=$tag \
        timeout $to /mnt/persist/venv/bin/python experiments/$script "$MODEL" > "$log" 2>&1 < /dev/null \
        || echo "CELL_FAIL $tag rc=$? @ $(date +%T)" >> "$log"
    grep -qa RECIPE_EVAL "$log" && echo "[$g] DONE $tag @ $(date +%T)" || echo "[$g] NOEVAL $tag @ $(date +%T)"
    i=$((i+NG))
  done
  echo "[$g] WORKER COMPLETE @ $(date +%T)"
) &
done
wait
echo "FULLTABLE_DONE @ $(date +%T)"
