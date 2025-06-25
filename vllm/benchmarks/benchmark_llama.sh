#!/bin/bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
PORT=8000
HEALTH_ENDPOINT="http://localhost:$PORT/health"
DEVICES="0"

VLLM_CMD="CUDA_VISIBLE_DEVICES=$DEVICES vllm serve $MODEL --disable-log-requests --port $PORT --enforce-eager &"

# Function to clean up if script is interrupted
cleanup() {
    echo "Stopping vLLM (PID=$VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

eval $VLLM_CMD
VLLM_PID=$!

# Wait for /health endpoint to be ready
echo "Waiting for vLLM to become healthy..."
until curl -sf "$HEALTH_ENDPOINT"; do
    if ! ps -p $VLLM_PID > /dev/null; then
        echo "vLLM process exited unexpectedly."
        exit 1
    fi
    sleep 2
done

echo "vLLM is up and healthy!"

# create sonnet-4x.txt
echo "" > benchmarks/sonnet_4x.txt
for _ in {1..4}
do
    cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
done

BM_LOG=~/vllm/unquantized.txt
# TODO move into justfile
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL \
    --dataset-name sonnet \
    --dataset-path benchmarks/sonnet_4x.txt \
    --sonnet-input-len 1800 \
    --sonnet-output-len 128 \
    --ignore-eos \
    --port $PORT | tee "$BM_LOG"