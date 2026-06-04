#!/usr/bin/env bash
# Janus sam-dev orchestrator — 3 lanes (GPU 0,2,3). Data-heavy: MATH+MMLU coupling/forgetting/H3.
set -u
JHOME=/home/devuser/janus
PY=$JHOME/janus_run.py; VIZ=$JHOME/janus_viz.py
export JANUS_OUT=$JHOME/artifacts HF_HOME=/home/devuser/.cache/huggingface PYTHONUNBUFFERED=1
LOG=$JHOME/logs; mkdir -p "$JANUS_OUT" "$LOG"; cd $JHOME

run () { local gpu="$1" name="$2"; shift 2
  echo "[orch $(date +%H:%M:%S)] START $name gpu$gpu: $*" | tee -a "$LOG/orch.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[orch $(date +%H:%M:%S)] END   $name rc=$?" | tee -a "$LOG/orch.log"; }

laneA () {  # GPU2: 1.5B MATH full-FT — H2 + H3 + k-sweep + dose + cross-domain
  local G=2 M=qwen2.5-1.5b O=$JANUS_OUT/qwen2.5-1.5b
  run $G A_det     detect  --model $M
  run $G A_capb    capeval --model $M --n 200
  run $G A_niahb   niah    --model $M --lengths 1500,4000 --samples 6
  run $G A_ablr    niah    --model $M --lengths 1500,4000 --samples 6 --ablate --ablate_site retr --ablate_k 24
  run $G A_abls    niah    --model $M --lengths 1500,4000 --samples 6 --ablate --ablate_site sink --ablate_k 24
  run $G A_none    sft     --model $M --domain math --mode full --protect none      --steps 500 --lr 2e-5 --bs 2
  run $G A_capn    capeval --model $M --n 200 --ckpt $O/ckpt_math_full_none
  run $G A_niahn   niah    --model $M --lengths 1500,4000 --samples 6 --ckpt $O/ckpt_math_full_none
  run $G A_retr    sft     --model $M --domain math --mode full --protect retrieval --steps 500 --lr 2e-5 --bs 2 --protect_k 24
  run $G A_capr    capeval --model $M --n 200 --ckpt $O/ckpt_math_full_retrieval
  run $G A_rand    sft     --model $M --domain math --mode full --protect random    --steps 500 --lr 2e-5 --bs 2 --protect_k 24
  run $G A_caprd   capeval --model $M --n 200 --ckpt $O/ckpt_math_full_random
  run $G A_dw      sft     --model $M --domain math --mode full --protect deltaw    --steps 500 --lr 2e-5 --bs 2 --protect_k 24
  run $G A_capdw   capeval --model $M --n 200 --ckpt $O/ckpt_math_full_deltaw
  run $G A_sink    sft     --model $M --domain math --mode full --protect sink      --steps 500 --lr 2e-5 --bs 2 --protect_k 24 --run_tag math_full_sink
  run $G A_capsk   capeval --model $M --n 200 --ckpt $O/ckpt_math_full_sink
  run $G A_k12     sft     --model $M --domain math --mode full --protect retrieval --steps 500 --lr 2e-5 --bs 2 --protect_k 12 --run_tag math_full_retrieval_k12
  run $G A_capk12  capeval --model $M --n 200 --ckpt $O/ckpt_math_full_retrieval_k12
  run $G A_k48     sft     --model $M --domain math --mode full --protect retrieval --steps 500 --lr 2e-5 --bs 2 --protect_k 48 --run_tag math_full_retrieval_k48
  run $G A_capk48  capeval --model $M --n 200 --ckpt $O/ckpt_math_full_retrieval_k48
  run $G A_d150    sft     --model $M --domain math --mode full --protect none --steps 150 --lr 2e-5 --bs 2 --run_tag math_full_none_d150
  run $G A_capd150 capeval --model $M --n 200 --ckpt $O/ckpt_math_full_none_d150
  run $G A_triv    sft     --model $M --domain trivia --mode full --protect none --steps 300 --lr 2e-5 --bs 2
  run $G A_capt    capeval --model $M --n 200 --ckpt $O/ckpt_trivia_full_none
  python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_A_DONE
}

laneB () {  # GPU3: 8B LoRA coupling + long-context causal
  local G=3 M=qwen3-8b O=$JANUS_OUT/qwen3-8b
  run $G B_det    detect  --model $M
  run $G B_capb   capeval --model $M --n 150
  run $G B_niahb  niah    --model $M --lengths 2000,6000 --samples 5
  run $G B_ablr   niah    --model $M --lengths 2000,6000 --samples 5 --ablate --ablate_site retr --ablate_k 24
  run $G B_abls   niah    --model $M --lengths 2000,6000 --samples 5 --ablate --ablate_site sink --ablate_k 24
  run $G B_none   sft     --model $M --domain math --mode lora --protect none --steps 350 --lr 1e-4 --bs 2
  run $G B_capn   capeval --model $M --n 150 --ckpt $O/ckpt_math_lora_none
  run $G B_niahn  niah    --model $M --lengths 2000,6000 --samples 5 --ckpt $O/ckpt_math_lora_none
  run $G B_triv   sft     --model $M --domain trivia --mode lora --protect none --steps 300 --lr 1e-4 --bs 2
  run $G B_capt   capeval --model $M --n 150 --ckpt $O/ckpt_trivia_lora_none
  python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_B_DONE
}

laneC () {  # GPU0: cross-family + base-vs-instruct + 14B long-ctx
  local G=0
  run $G C_det14   detect  --model qwen3-14b
  run $G C_niah14  niah    --model qwen3-14b --lengths 2000,6000 --samples 5
  run $G C_abl14r  niah    --model qwen3-14b --lengths 2000,6000 --samples 5 --ablate --ablate_site retr --ablate_k 24
  run $G C_abl14s  niah    --model qwen3-14b --lengths 2000,6000 --samples 5 --ablate --ablate_site sink --ablate_k 24
  run $G C_det7    detect  --model qwen2.5-7b-instruct
  run $G C_cap7b   capeval --model qwen2.5-7b-instruct --n 120
  run $G C_sft7    sft     --model qwen2.5-7b-instruct --domain math --mode lora --protect none --steps 300 --lr 1e-4 --bs 2
  run $G C_cap7n   capeval --model qwen2.5-7b-instruct --n 120 --ckpt $JANUS_OUT/qwen2.5-7b-instruct/ckpt_math_lora_none
  run $G C_deti    detect  --model qwen2.5-1.5b-instruct
  run $G C_capib   capeval --model qwen2.5-1.5b-instruct --n 200
  run $G C_sfti    sft     --model qwen2.5-1.5b-instruct --domain math --mode full --protect none --steps 500 --lr 2e-5 --bs 2
  run $G C_capin   capeval --model qwen2.5-1.5b-instruct --n 200 --ckpt $JANUS_OUT/qwen2.5-1.5b-instruct/ckpt_math_full_none
  python3 "$VIZ" >> "$LOG/viz.log" 2>&1; touch $JANUS_OUT/_C_DONE
}

vizloop () { for i in $(seq 1 60); do sleep 900; python3 "$VIZ" >> "$LOG/viz.log" 2>&1
  [ -f $JANUS_OUT/_A_DONE ] && [ -f $JANUS_OUT/_B_DONE ] && [ -f $JANUS_OUT/_C_DONE ] && break; done; }

laneA & PA=$!; laneB & PB=$!; laneC & PC=$!; vizloop & PV=$!
wait $PA $PB $PC
python3 "$VIZ" >> "$LOG/viz.log" 2>&1; kill $PV 2>/dev/null
echo "[orch $(date +%H:%M:%S)] SAMDEV ALL DONE" | tee -a "$LOG/orch.log"; touch $JANUS_OUT/_ALL_DONE
