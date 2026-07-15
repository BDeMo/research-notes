#!/bin/bash
# keep_ablation.sh — Paper-B KEEP-RATE ablation for ALL methods on one axis.
# One shared "keep fraction" k in {0.1,0.25,0.5,0.75} mapped to each method's native knob, so every
# method is compared at the SAME retained budget. Base Qwen3-8B; N=100 (disclosed ablation subset);
# representative 4-bench set. Output /mnt/persist/grid_keepabl/ka_<method>_<bench>_k<k>.log
# knob mapping (k = fraction of context KEPT):
#   window  GCM_TXL_WINDOW = round(k*maxctx)     rag  GCM_RAG_BUDGET = round(k*maxctx)
#   ll2     GCM_LL_RATE    = k                   tome GCM_TOME_RATIO = k        imp GCM_LL_RATE = k (span32)
#   kvzip/knorm GCM_KV_RATIO = 1-k  (KV ratio = fraction DROPPED)
set -u
ROOT=/mnt/persist/gcm; OUT=/mnt/persist/grid_keepabl; mkdir -p "$OUT"
MODEL=Qwen/Qwen3-8B
PP="$ROOT/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src"
COMMON="HF_HOME=/mnt/persist/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton GCM_NVAL=100"
cd "$ROOT"
KEEPS=(0.1 0.25 0.5 0.75)
METHODS=(window rag ll2 tome imp kvzip knorm)
# bench | nchunks | maxctx | timeout
BENCHES=("ruler_niah|88|16384|4200" "lb_hotpotqa|12|16384|5400" "squad_v2|8|4096|2400" "quality_hard|8|8192|3600")

mk_env() {  # $1=method $2=k $3=maxctx  -> prints "SCRIPT|ENV"
  local m=$1 k=$2 mx=$3 tok; tok=$(python3 -c "print(int(round($k*$mx)))")
  case $m in
    window) echo "run_baseline.py|GCM_BASELINE=txl GCM_TXL_WINDOW=$tok";;
    rag)    echo "run_baseline.py|GCM_BASELINE=rag GCM_RAG_BUDGET=$tok";;
    ll2)    echo "run_baseline.py|GCM_BASELINE=llmlingua GCM_LL_RATE=$k";;
    tome)   echo "run_baseline.py|GCM_BASELINE=tome GCM_TOME_RATIO=$k";;
    imp)    echo "run_baseline.py|GCM_BASELINE=imp GCM_IMP_SPAN=32 GCM_LL_RATE=$k";;
    kvzip)  echo "run_kvpress.py|GCM_KVPRESS=kvzip GCM_KV_RATIO=$(python3 -c "print(round(1-$k,3))")";;
    knorm)  echo "run_kvpress.py|GCM_KVPRESS=knorm GCM_KV_RATIO=$(python3 -c "print(round(1-$k,3))")";;
  esac
}

CELLS=()
for m in "${METHODS[@]}"; do for b in "${BENCHES[@]}"; do IFS='|' read -r ev nc mx to <<< "$b"
  for k in "${KEEPS[@]}"; do se=$(mk_env "$m" "$k" "$mx"); sc="${se%%|*}"; me="${se#*|}"
    CELLS+=("ka_${m}_${ev}_k${k}|$sc|$me|$ev|$nc|$mx|$to"); done; done; done
echo "keep-ablation cells: ${#CELLS[@]}  (methods=${#METHODS[@]} × benches=${#BENCHES[@]} × keeps=${#KEEPS[@]})"
GPUS=(${GPUS_OVERRIDE:-0 1 2 3}); NG=${#GPUS[@]}
for gi in "${!GPUS[@]}"; do
( g=${GPUS[$gi]}; i=$gi
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
wait; echo "KEEPABL_DONE @ $(date +%T)"
