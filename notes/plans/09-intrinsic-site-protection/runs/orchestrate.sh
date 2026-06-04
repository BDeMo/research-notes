#!/usr/bin/env bash
# Janus overnight orchestrator. Two GPU lanes (2,3) + periodic viz.
# Robust: every step wrapped; failures logged and skipped.
set -u
PY=/home/devuser/janus/janus_run.py
VIZ=/home/devuser/janus/janus_viz.py
export JANUS_OUT=/home/devuser/janus/artifacts
export HF_HOME=/home/devuser/.cache/huggingface
export PYTHONUNBUFFERED=1
LOG=/home/devuser/janus/logs
mkdir -p "$JANUS_OUT" "$LOG"
cd /home/devuser/janus

run () {  # run <gpu> <logname> <args...>
  local gpu="$1"; local name="$2"; shift 2
  echo "[orch $(date +%H:%M:%S)] START $name (gpu $gpu): $*" | tee -a "$LOG/orch.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[orch $(date +%H:%M:%S)] END   $name rc=$rc" | tee -a "$LOG/orch.log"
}

lane_small () {  # GPU 2 — qwen2.5-1.5b full pipeline (centerpiece: H2 + H3)
  local G=2 M=qwen2.5-1.5b
  run $G s_detect      detect  --model $M
  run $G s_cap_base    capeval --model $M --n 200
  run $G s_niah_base   niah    --model $M --lengths 1500,4000 --samples 6
  run $G s_niah_ablate niah    --model $M --lengths 1500,4000 --samples 6 --ablate --ablate_k 24
  # vanilla full-FT (defines deltaw set) + coupling
  run $G s_sft_none    sft     --model $M --domain math --mode full --protect none     --steps 400 --lr 2e-5 --bs 2
  run $G s_cap_none    capeval --model $M --n 200 --ckpt $JANUS_OUT/$M/ckpt_math_full_none
  run $G s_niah_none   niah    --model $M --lengths 1500,4000 --samples 6 --ckpt $JANUS_OUT/$M/ckpt_math_full_none
  # protection variants (H3): retrieval vs random vs deltaw
  run $G s_sft_retr    sft     --model $M --domain math --mode full --protect retrieval --steps 400 --lr 2e-5 --bs 2 --protect_k 24
  run $G s_cap_retr    capeval --model $M --n 200 --ckpt $JANUS_OUT/$M/ckpt_math_full_retrieval
  run $G s_sft_rand    sft     --model $M --domain math --mode full --protect random    --steps 400 --lr 2e-5 --bs 2 --protect_k 24
  run $G s_cap_rand    capeval --model $M --n 200 --ckpt $JANUS_OUT/$M/ckpt_math_full_random
  run $G s_sft_dw      sft     --model $M --domain math --mode full --protect deltaw    --steps 400 --lr 2e-5 --bs 2 --protect_k 24
  run $G s_cap_dw      capeval --model $M --n 200 --ckpt $JANUS_OUT/$M/ckpt_math_full_deltaw
  # cross-domain coupling
  run $G s_sft_triv    sft     --model $M --domain trivia --mode full --protect none --steps 300 --lr 2e-5 --bs 2
  run $G s_cap_triv    capeval --model $M --n 200 --ckpt $JANUS_OUT/$M/ckpt_trivia_full_none
  python3 "$VIZ" >> "$LOG/viz.log" 2>&1
  touch $JANUS_OUT/_LANE_SMALL_DONE
}

lane_big () {  # GPU 3 — scale-up (8B/14B detect, 8B LoRA coupling + long-ctx)
  local G=3
  run $G b_detect_8b  detect  --model qwen3-8b
  run $G b_detect_14b detect  --model qwen3-14b
  run $G b_detect_7b  detect  --model qwen2.5-7b-instruct
  run $G b_cap_base   capeval --model qwen3-8b --n 150
  run $G b_niah_base  niah    --model qwen3-8b --lengths 2000,6000 --samples 5
  run $G b_niah_abl   niah    --model qwen3-8b --lengths 2000,6000 --samples 5 --ablate --ablate_k 24
  run $G b_sft_math   sft     --model qwen3-8b --domain math --mode lora --protect none --steps 300 --lr 1e-4 --bs 2
  run $G b_cap_math   capeval --model qwen3-8b --n 150 --ckpt $JANUS_OUT/qwen3-8b/ckpt_math_lora_none
  run $G b_niah_math  niah    --model qwen3-8b --lengths 2000,6000 --samples 5 --ckpt $JANUS_OUT/qwen3-8b/ckpt_math_lora_none
  run $G b_sft_triv   sft     --model qwen3-8b --domain trivia --mode lora --protect none --steps 250 --lr 1e-4 --bs 2
  run $G b_cap_triv   capeval --model qwen3-8b --n 150 --ckpt $JANUS_OUT/qwen3-8b/ckpt_trivia_lora_none
  python3 "$VIZ" >> "$LOG/viz.log" 2>&1
  touch $JANUS_OUT/_LANE_BIG_DONE
}

viz_loop () {
  for i in $(seq 1 40); do
    sleep 1200
    python3 "$VIZ" >> "$LOG/viz.log" 2>&1
    [ -f $JANUS_OUT/_LANE_SMALL_DONE ] && [ -f $JANUS_OUT/_LANE_BIG_DONE ] && break
  done
}

lane_small &
PID_S=$!
lane_big &
PID_B=$!
viz_loop &
PID_V=$!
wait $PID_S $PID_B
python3 "$VIZ" >> "$LOG/viz.log" 2>&1
kill $PID_V 2>/dev/null
echo "[orch $(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG/orch.log"
touch $JANUS_OUT/_ALL_DONE
