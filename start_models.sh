#!/bin/bash

# --- CONFIGURATION ---
# Main Model: Qwen 2.5 3B (Quantized)
MAIN_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
# Draft Model: Qwen 2.5 0.5B (Standard)
DRAFT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Clean up any old processes
pkill -f vllm

echo ">>> Starting Instance A: SPECULATIVE (Fast but fragile) on Port 8001..."
# FIX: Use --speculative-config JSON instead of separate flags
# FIX: Use --uvicorn-log-level instead of --log-level
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.45 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &

# Wait for the first model to load
echo "Waiting 60s for Instance A to load..."
sleep 60

echo ">>> Starting Instance B: STANDARD (Slow but robust) on Port 8002..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --gpu-memory-utilization 0.45 \
    --port 8002 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &

echo ">>> Both models launching. Tail the logs or run 'stress_test.py' to see results."
wait
