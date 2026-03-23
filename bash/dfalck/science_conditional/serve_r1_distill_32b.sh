#!/bin/bash
# Serve DeepSeek-R1-Distill-Qwen-32B for data generation
# This is the larger model used to generate SFT training data
# for the R1-Distill-14B pipeline.
#
# Uses a custom chat template that adds hermes-style tool calling
# support, since R1-Distill models don't have native tool call tokens.
# Think blocks are preserved in conversation history.

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
PORT=8000
TEMPLATE="etc/dfalck/science_conditional/final_r1_distill/chat_template.jinja"

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --chat-template "$TEMPLATE" \
  --tensor-parallel-size 2 \
  --port $PORT
