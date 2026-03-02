


MAIN_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"

echo ">>> Starting Instance A: Main Model on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


echo ">>> Starting Instance B: SPECULATIVE (Fast but fragile) on Port 8003..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --max-model-len 2048 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.85 \
    --port 8003 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &

