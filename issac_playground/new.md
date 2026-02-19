

pip install vllm-router  

MAIN_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
DRAFT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"



echo ">>> Starting Instance A: Main Model on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --gpu-memory-utilization 0.45 \
    --port 8001 \
    --swap-space 2 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


echo ">>> Starting Instance B: Small Model on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $DRAFT_MODEL \
    --quantization awq \
    --gpu-memory-utilization 0.15 \
    --port 8002 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


echo ">>> Starting Instance C: SPECULATIVE (Fast but fragile) on Port 8003..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.6 \
    --port 8003 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &






vllm-router \
--worker-urls http://localhost:8001/v1/completions http://localhost:8002/v1/completions \
--policy cache_aware

vllm-router \
--worker-urls http://localhost:8003/v1/completions
--policy cache_aware