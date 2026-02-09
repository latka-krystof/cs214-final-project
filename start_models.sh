#!/bin/bash

# Define Models (Using AWQ Quantization to save memory)
# Main Model: Llama-2-7B-Chat (4-bit quantized) -> Takes ~5GB VRAM
MAIN_MODEL="TheBloke/Llama-2-7b-Chat-AWQ"
# Draft Model: TinyLlama-1.1B (Standard) -> Takes ~2.2GB VRAM
DRAFT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

echo ">>> Starting Instance A: SPECULATIVE (Fast but fragile) on Port 8001..."
# We give it 60% of GPU memory to hold BOTH Main and Draft models + KV Cache
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --speculative-model $DRAFT_MODEL \
    --num-speculative-tokens 5 \
    --gpu-memory-utilization 0.6 \
    --port 8001 \
    --log-level info &

# Wait 30 seconds for the first one to settle
sleep 30

echo ">>> Starting Instance B: STANDARD (Slow but robust) on Port 8002..."
# We give it 30% of GPU memory (only needs Main model)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --gpu-memory-utilization 0.3 \
    --port 8002 \
    --log-level info &

echo ">>> Both models launching. Tail the logs to see when they are ready."
wait