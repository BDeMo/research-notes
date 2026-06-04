#!/usr/bin/env bash
# Local autopoll: pull Janus figure galleries from both pods every 15 min.
DST=/Users/s1shi/workspace/janus/gallery
mkdir -p "$DST/samdev" "$DST/ray"
while true; do
  KUBECONFIG=/Users/s1shi/.kube/company.yaml kubectl exec sam-dev -- \
    tar -C /home/devuser/janus/artifacts -cf - figs INDEX.md 2>/dev/null \
    | tar -C "$DST/samdev" -xf - 2>/dev/null && echo "[pull $(date +%H:%M)] samdev ok"
  KUBECONFIG=/Users/s1shi/.kube/aird-ray-dev.yaml kubectl exec sam-dev-ray -n aird-ray-dev -- \
    tar -C /root/janus/artifacts -cf - figs INDEX.md 2>/dev/null \
    | tar -C "$DST/ray" -xf - 2>/dev/null && echo "[pull $(date +%H:%M)] ray ok"
  sleep 900
done
