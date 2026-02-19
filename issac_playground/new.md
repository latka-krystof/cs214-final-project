

pip install vllm-router  

MAIN_MODEL="Qwen/Qwen3-4B-AWQ"
DRAFT_MODEL="Qwen/Qwen3-0.6B"



________________________________

<!-- current experiment 0.6 GPU Main vs Speculative -->

echo ">>> Starting Instance A: Main Model on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


echo ">>> Starting Instance C: SPECULATIVE (Fast but fragile) on Port 8003..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --quantization awq \
    --max-model-len 2048 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 2, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.85 \
    --port 8003 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &



________________________________

<!-- next experiment 0.6 GPU One big + one small vs Speculative -->


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




vllm-router \
--worker-urls http://localhost:8001/v1/completions http://localhost:8002/v1/completions \
--policy cache_aware

vllm-router \
--worker-urls http://localhost:8003/v1/completions
--policy cache_aware