#!/bin/bash
# explore_imp_abl.sh — IMP design-ceiling ablation on a fixed base (Qwen3-8B).
# Grid: signal {query, surprisal, both} × span {8,16,32,64,128} on benches where IMP is weak/strong.
# Question: can a better signal/span combo close IMP's gap to RAG? (F30). N=100. Patch-free benches.
# Output /mnt/persist/grid_impabl/ia_<sig>_s<span>_<bench>.log
set -u
: "${GPUS:=0 1}"; MODEL=Qwen/Qwen3-8B; OUTDIR=/mnt/persist/grid_impabl; mkdir -p "$OUTDIR"
ROOT=/mnt/persist/gcm
PP="$ROOT/src:/mnt/persist/v17/svc/src:/mnt/persist/v17/mem-embedding/src:/mnt/persist/v17/llm-infra/src"
COMMON="HF_HOME=/mnt/persist/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/mnt/persist/.triton GCM_NVAL=100 GCM_LL_RATE=0.5"
cd "$ROOT"
SIGS=(query surprisal both); SPANS=(8 16 32 64 128)
BENCHES=("ruler_niah|88|16384|4200" "squad_v2|8|4096|2400" "hotpot_qa|8|4096|2400" "quality_hard|8|8192|3600")
CELLS=()
for sig in "${SIGS[@]}"; do for sp in "${SPANS[@]}"; do for b in "${BENCHES[@]}"; do
  IFS='|' read -r ev nc mx to <<< "$b"; CELLS+=("ia_${sig}_s${sp}_${ev}|$sig|$sp|$ev|$nc|$mx|$to"); done; done; done
echo "imp-ablation cells: ${#CELLS[@]}"
GA=($GPUS); NG=${#GA[@]}
for gi in "${!GA[@]}"; do
( g=${GA[$gi]}; i=$gi
  while [ $i -lt ${#CELLS[@]} ]; do
    IFS='|' read -r tag sig sp ev nc mx to <<< "${CELLS[$i]}"; log="$OUTDIR/${tag}.log"
    grep -qa RECIPE_EVAL "$log" 2>/dev/null && { i=$((i+NG)); continue; }
    echo "[$g] START $tag @ $(date +%T)"
    env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP $COMMON GCM_BASELINE=imp GCM_IMP_SIGNAL=$sig GCM_IMP_SPAN=$sp \
        GCM_EVAL=$ev GCM_NCHUNKS=$nc GCM_MAXCTX=$mx GCM_TAG=$tag \
        timeout $to /mnt/persist/venv/bin/python experiments/run_baseline.py "$MODEL" > "$log" 2>&1 </dev/null \
        || echo "CELL_FAIL $tag rc=$? @ $(date +%T)" >> "$log"
    i=$((i+NG))
  done; echo "[$g] WORKER DONE @ $(date +%T)" ) &
done
wait; echo "IMPABL_DONE @ $(date +%T)"
