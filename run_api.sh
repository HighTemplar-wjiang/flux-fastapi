#!/usr/bin/env bash
set -euo pipefail

# Configuration (override by exporting before running or editing below)
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Model/settings
export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
export DEFAULT_GUIDANCE="${DEFAULT_GUIDANCE:-0.0}"
export DEFAULT_STEPS="${DEFAULT_STEPS:-4}"
export DEFAULT_MAX_SEQ_LEN="${DEFAULT_MAX_SEQ_LEN:-256}"
export DEFAULT_HEIGHT="${DEFAULT_HEIGHT:-1024}"
export DEFAULT_WIDTH="${DEFAULT_WIDTH:-1024}"
export ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-true}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"  # options: bfloat16|float16|float32

# GPU exposure: set which GPUs are visible to the server.
# Examples:
#   export CUDA_VISIBLE_DEVICES=0
#   export CUDA_VISIBLE_DEVICES=0,1
# If unset, all GPUs are visible (if present).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"

echo "Starting FLUX API on ${HOST}:${PORT}"
echo "MODEL_ID=${MODEL_ID}"
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  echo "CUDA_VISIBLE_DEVICES is not set (all available GPUs visible)"
fi

exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers "${WORKERS:-1}"

