#!/bin/bash

MAIN_MODEL="Qwen/Qwen2.5-3B-Instruct"
DRAFT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

echo ">>> Cleaning up old processes..."
pkill -9 -f vllm
sleep 3

# --- INSTANCE A: SPECULATIVE (Port 8001) ---
# Removed log suppression to expose the speculative acceptance rates.
echo ">>> Booting Speculative Engine (Instance A)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.45 \
    --max-model-len 1024 \
    --port 8001 \
    --enforce-eager \
    --disable-log-requests &

echo "Waiting 60s for Instance A to load weights into VRAM..."
sleep 60

# --- INSTANCE B: STANDARD (Port 8002) ---
echo ">>> Booting Standard Engine (Instance B)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --gpu-memory-utilization 0.45 \
    --max-model-len 1024 \
    --port 8002 \
    --enforce-eager \
    --disable-log-requests &

echo ">>> Both backends are online!"
wait
