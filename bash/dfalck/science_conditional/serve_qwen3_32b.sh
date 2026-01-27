#!/bin/bash
# Start vLLM server for Qwen3-32B with willcb chat template
# This script starts a vLLM server for eval_gen_* configs using 4 GPUs

MODEL="willcb/Qwen3-32B"
PORT=8000

echo "Starting vLLM server with model: $MODEL"
echo "Data parallel size: 8 GPUs"
echo "Server will be available at: http://localhost:$PORT/v1"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=0

vllm serve "$MODEL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --data-parallel-size 8 \
  --port $PORT

# Notes:
# - Press Ctrl+C to stop the server
# - Make sure port 8000 is available before running this script
# - Uses data parallelism (4 model copies) for higher throughput
# - The willcb model includes a chat template that preserves <think> blocks from previous turns
