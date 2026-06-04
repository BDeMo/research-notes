#!/usr/bin/env bash
cd /home/devuser/janus
export JANUS_OUT=/home/devuser/janus/artifacts HF_HOME=/home/devuser/.cache/huggingface PYTHONUNBUFFERED=1
cp -rn ray_stage/qwen3-0.6b ray_stage/qwen3-1.7b ray_stage/qwen3-4b artifacts/ 2>/dev/null
cp -n ray_stage/qwen3-8b/* artifacts/qwen3-8b/ 2>/dev/null
cp -n ray_stage/qwen3-14b/* artifacts/qwen3-14b/ 2>/dev/null
echo "MODELS:"; ls artifacts/ | grep qwen
echo "RENDER:"; python3 janus_viz.py 2>&1 | tail -8
echo "FIGS:"; ls artifacts/figs/ | wc -l
