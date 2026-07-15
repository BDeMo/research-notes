#!/bin/bash
# fulltable_retry.sh — repair the OOM cells from the full main table.
# RULER at 32k generation OOMs the uncompressed `full` column on a 96GB card (memory wall).
# Re-run RULER at 16k (nchunks=88) => tag ft_<m>_ruler16k. Also retry the borderline rag_lb_musique.
# Runs on d1530 GPUs 0-1 (idle); shared /mnt/persist, same output dir grid_fulltable.
set -u
ROOT=/mnt/persist/gcm
OUT=/mnt/persist/grid_fulltable; mkdir -p "$OUT"
MODEL=Qwen/Qwen3-8B
PP="$ROOT/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src"
COMMON="HF_HOME=/mnt/persist/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton"
cd "$ROOT"

# tag | script | env | ev | nc | mx | nval | timeout
CELLS=(
 "ft_window_ruler16k|run_baseline.py|GCM_BASELINE=txl GCM_TXL_WINDOW=1024|ruler_niah|88|16384|500|5400"
 "ft_rag_ruler16k|run_baseline.py|GCM_BASELINE=rag GCM_RAG_BUDGET=2048|ruler_niah|88|16384|500|5400"
 "ft_ll2_ruler16k|run_baseline.py|GCM_BASELINE=llmlingua GCM_LL_RATE=0.5|ruler_niah|88|16384|500|5400"
 "ft_tome_ruler16k|run_baseline.py|GCM_BASELINE=tome GCM_TOME_RATIO=0.5|ruler_niah|88|16384|500|5400"
 "ft_imp_ruler16k|run_baseline.py|GCM_BASELINE=imp GCM_IMP_SPAN=32 GCM_LL_RATE=0.5|ruler_niah|88|16384|500|5400"
 "ft_kvzip_ruler16k|run_kvpress.py|GCM_KVPRESS=kvzip GCM_KV_RATIO=0.5|ruler_niah|88|16384|500|5400"
 "ft_knorm_ruler16k|run_kvpress.py|GCM_KVPRESS=knorm GCM_KV_RATIO=0.5|ruler_niah|88|16384|500|5400"
 "ft_rag_lb_musique|run_baseline.py|GCM_BASELINE=rag GCM_RAG_BUDGET=2048|lb_musique|12|16384|100000|7200"
)
echo "retry cells: ${#CELLS[@]}"
GPUS=(0 1 2 3); NG=${#GPUS[@]}
for gi in "${!GPUS[@]}"; do
( g=${GPUS[$gi]}; i=$gi
  while [ $i -lt ${#CELLS[@]} ]; do
    IFS='|' read -r tag script menv ev nc mx nv to <<< "${CELLS[$i]}"
    log="$OUT/${tag}.log"
    if grep -qa RECIPE_EVAL "$log" 2>/dev/null; then i=$((i+NG)); continue; fi
    echo "[$g] START $tag @ $(date +%T)"
    env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP $COMMON $menv \
        GCM_EVAL=$ev GCM_NCHUNKS=$nc GCM_MAXCTX=$mx GCM_NVAL=$nv GCM_TAG=$tag \
        timeout $to /mnt/persist/venv/bin/python experiments/$script "$MODEL" > "$log" 2>&1 < /dev/null \
        || echo "CELL_FAIL $tag rc=$? @ $(date +%T)" >> "$log"
    grep -qa RECIPE_EVAL "$log" && echo "[$g] DONE $tag @ $(date +%T)" || echo "[$g] NOEVAL $tag @ $(date +%T)"
    i=$((i+NG))
  done
  echo "[$g] RETRY WORKER COMPLETE @ $(date +%T)"
) &
done
wait
echo "RETRY_DONE @ $(date +%T)"
