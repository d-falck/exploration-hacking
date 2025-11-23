#!/bin/bash
# Start vLLM server for science_conditional evaluations
# This script starts a vLLM server with the model used by eval_gen_benign.yaml

MODEL="willcb/Qwen3-14B"
PORT=8000

echo "Starting vLLM server with model: $MODEL"
echo "Server will be available at: http://localhost:$PORT/v1"

vllm serve "$MODEL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --data-parallel-size 1 \
  --port $PORT

# Notes:
# - Press Ctrl+C to stop the server
# - Make sure port 8000 is available before running this script
# - The server uses the default vLLM arguments from the LocalBackend configuration
