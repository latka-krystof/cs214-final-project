#!/bin/bash

MAIN_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"

echo ">>> Cleaning up old processes..."
pkill -9 -f vllm
fuser -k 8001/tcp 2>/dev/null
fuser -k 8002/tcp 2>/dev/null
sleep 3

# --- INSTANCE A: SPECULATIVE (Port 8001) ---
# Swapped to Prompt Lookup (N-Gram) Decoding. Zero VRAM overhead.
echo ">>> Booting Speculative Engine (Instance A)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --speculative-config "{\"method\": \"ngram\", \"num_speculative_tokens\": 5, \"prompt_lookup_max\": 4}" \
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

echo ">>> Both backends are online and ready for the benchmark!"
wait
