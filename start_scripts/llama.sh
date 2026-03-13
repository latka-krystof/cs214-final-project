



MAIN_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"



# Main Experiment - One Speculative * One Regular

echo ">>> Starting Instance B: SPECULATIVE (Fast but fragile) on Port 8002..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --max-model-len 768 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.5 \
    --port 8002 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &

echo ">>> Starting Instance A: Main Model on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --gpu-memory-utilization 0.43 \
    --max-model-len 768 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


# 1 REGULAR * 1 REGULAR

echo ">>> Starting Instance A: Main Model on Port 8002..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --gpu-memory-utilization 0.5 \
    --max-model-len 768 \
    --port 8002 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &

echo ">>> Starting Instance A: Main Model on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --gpu-memory-utilization 0.43 \
    --max-model-len 768 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


# Speculative * Speculative - NOT ENOUGH GPU SPACE SKIP!

echo ">>> Starting Instance B: SPECULATIVE (Fast but fragile) on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --max-model-len 768 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.5 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &

echo ">>> Starting Instance B: SPECULATIVE (Fast but fragile) on Port 8002..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --max-model-len 768 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.43 \
    --port 8002 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &


# Just one big speculative

echo ">>> Starting Instance B: SPECULATIVE (Fast but fragile) on Port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MAIN_MODEL \
    --max-model-len 768 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5, \"method\": \"draft_model\"}" \
    --gpu-memory-utilization 0.93 \
    --port 8001 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --uvicorn-log-level error &