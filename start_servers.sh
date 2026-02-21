#!/bin/bash
# SmartSpec - Start vLLM backends on a single T4 GPU
# Instance A: Standard decoding (port 8001)
# Instance B: Speculative decoding (port 8003)
#
# NOTE: T4 has 16GB VRAM. We run ONE server at a time during benchmarking.
# Use START_MODE=standard or START_MODE=speculative to control which one starts.
# If you have multi-GPU, set CUDA_VISIBLE_DEVICES accordingly.

MAIN_MODEL="Qwen/Qwen3-0.6B"   # Lightweight enough to fit on T4 comfortably
DRAFT_MODEL="Qwen/Qwen3-0.6B"  # Same model used as draft (or swap for a smaller one)

# --- T4-safe settings ---
# T4 = 16GB VRAM. AWQ 8B model alone is ~5-6GB. Draft model adds ~1-2GB overhead.
# We use a smaller main model here so both can realistically serve requests.
# Swap MAIN_MODEL to Qwen3-8B-AWQ if you have enough headroom and test one server at a time.

START_MODE="${1:-both}"  # Options: standard | speculative | both (sequential, not parallel)

start_standard() {
    echo ""
    echo "=========================================="
    echo ">>> Starting STANDARD decoding on :8001"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model $MAIN_MODEL \
        --gpu-memory-utilization 0.88 \
        --max-model-len 2048 \
        --port 8001 \
        --max-num-seqs 64 \
        --disable-log-requests \
        --uvicorn-log-level error &

    STANDARD_PID=$!
    echo ">>> Standard server PID: $STANDARD_PID"
    echo $STANDARD_PID > /tmp/smartspec_standard.pid

    echo ">>> Waiting for standard server to be ready..."
    for i in $(seq 1 60); do
        sleep 3
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo ">>> Standard server is UP on port 8001"
            return 0
        fi
        echo "    ...still waiting ($((i*3))s)"
    done
    echo "ERROR: Standard server failed to start in 180s"
    return 1
}

start_speculative() {
    echo ""
    echo "=========================================="
    echo ">>> Starting SPECULATIVE decoding on :8003"
    echo "=========================================="

    SPEC_CONFIG=$(python3 -c "
import json
print(json.dumps({
    'model': '$DRAFT_MODEL',
    'num_speculative_tokens': 3,
    'method': 'draft_model'
}))
")

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model $MAIN_MODEL \
        --gpu-memory-utilization 0.88 \
        --max-model-len 2048 \
        --port 8003 \
        --max-num-seqs 64 \
        --speculative-config "$SPEC_CONFIG" \
        --disable-log-requests \
        --uvicorn-log-level error &

    SPEC_PID=$!
    echo ">>> Speculative server PID: $SPEC_PID"
    echo $SPEC_PID > /tmp/smartspec_speculative.pid

    echo ">>> Waiting for speculative server to be ready..."
    for i in $(seq 1 60); do
        sleep 3
        if curl -s http://localhost:8003/health > /dev/null 2>&1; then
            echo ">>> Speculative server is UP on port 8003"
            return 0
        fi
        echo "    ...still waiting ($((i*3))s)"
    done
    echo "ERROR: Speculative server failed to start in 180s"
    return 1
}

stop_server() {
    local PIDFILE=$1
    if [ -f "$PIDFILE" ]; then
        PID=$(cat $PIDFILE)
        echo ">>> Stopping PID $PID..."
        kill $PID 2>/dev/null
        rm -f $PIDFILE
    fi
}

# ---- Main logic ----

if [ "$START_MODE" = "standard" ]; then
    start_standard

elif [ "$START_MODE" = "speculative" ]; then
    start_speculative

elif [ "$START_MODE" = "both" ]; then
    # On a single T4 we benchmark sequentially: start standard, run benchmark, stop, start spec, run benchmark
    echo ""
    echo ">>> SEQUENTIAL mode: will start standard first, then speculative."
    echo ">>> Run your benchmark script against each port when prompted."
    echo ""

    start_standard
    echo ""
    echo "================================================================"
    echo "  Standard server running on :8001"
    echo "  Run:  python benchmark.py --mode standard"
    echo "  Then press ENTER here to stop it and start speculative server."
    echo "================================================================"
    read -r

    stop_server /tmp/smartspec_standard.pid
    sleep 5

    start_speculative
    echo ""
    echo "================================================================"
    echo "  Speculative server running on :8003"
    echo "  Run:  python benchmark.py --mode speculative"
    echo "  Then press ENTER here to stop it and exit."
    echo "================================================================"
    read -r

    stop_server /tmp/smartspec_speculative.pid

else
    echo "Usage: $0 [standard|speculative|both]"
    exit 1
fi
