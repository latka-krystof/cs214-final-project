#!/bin/bash
MAIN_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
DRAFT_MODEL="Qwen/Qwen2.5-0.5B-Instruct-AWQ"

pkill -9 -f vllm
fuser -k 8001/tcp 2>/dev/null
fuser -k 8002/tcp 2>/dev/null
sleep 3

# INSTANCE A (Neural Speculative)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.45 \
    --max-model-len 1024 \
    --port 8001 \
    --enforce-eager \
    --disable-log-requests &

sleep 60

# INSTANCE B (Standard)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --gpu-memory-utilization 0.45 \
    --max-model-len 1024 \
    --port 8002 \
    --enforce-eager \
    --disable-log-requests &

wait
