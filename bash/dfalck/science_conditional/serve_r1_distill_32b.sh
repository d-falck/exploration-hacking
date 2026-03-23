#!/bin/bash
# Serve DeepSeek-R1-Distill-Qwen-32B for data generation
# This is the larger model used to generate SFT training data
# for the R1-Distill-14B pipeline.

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
PORT=8000

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3 \
  --tensor-parallel-size 2 \
  --port $PORT
