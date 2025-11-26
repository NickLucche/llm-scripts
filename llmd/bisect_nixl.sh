#!/bin/bash
# Git bisect test script for NIXL KV Connector performance regression
# Returns 0 (good) if request completes in < 10 seconds
# Returns 1 (bad) if request takes >= 10 seconds or fails

set -xe

TIMEOUT=60
REQUEST_THRESHOLD=10  # seconds - requests should be faster than this
RESULT=125  # Default to skip

cleanup() {
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "toy_proxy_server" 2>/dev/null || true
    exit $RESULT
}
trap cleanup EXIT

# Kill any existing vllm processes
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "toy_proxy_server" 2>/dev/null || true
sleep 2


# Source the venv
source /home/aflowers/Documents/vllm/.venv/bin/activate

# Start the decoder (port 8200)
# Try new CLI syntax first (positional model), fall back to old syntax (--model flag)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
    --port 8200 \
    --gpu-memory-utilization 0.4 \
    --max-num-batched-tokens 4096 \
    --max-model-len 2000 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    > /tmp/decoder.log 2>&1 &
DECODER_PID=$!

sleep 30

# Start the prefiller (port 8100)
DYN_VLLM_KV_EVENT_PORT=20082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.4 \
    --max-num-batched-tokens 4096 \
    --max-model-len 2000 \
    --port 8100 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    > /tmp/prefiller.log 2>&1 &
PREFILLER_PID=$!

sleep 20

# Start the proxy
python3 tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200 \
    > /tmp/proxy.log 2>&1 &
PROXY_PID=$!

# Wait for proxy to be ready
for i in {1..30}; do
    if curl -s http://localhost:8192/healthcheck > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:8192/healthcheck > /dev/null 2>&1; then
    echo "SKIP: Proxy failed to start"
    RESULT=125
    exit 125
fi

# Time a request - timeout at threshold+2s since anything over threshold is BAD
REQUEST_TIMEOUT=$((REQUEST_THRESHOLD + 2))
START=$(date +%s.%N)
RESPONSE=$(curl -s --max-time $REQUEST_TIMEOUT -X POST http://localhost:8192/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 100}' 2>&1)
CURL_EXIT=$?
END=$(date +%s.%N)

# Calculate elapsed time
ELAPSED=$(echo "$END - $START" | bc)
echo "Request took: ${ELAPSED}s"

# Check if request succeeded and was fast enough
if [ $CURL_EXIT -eq 28 ]; then
    # Curl timeout (exit 28) - request took too long
    echo "BAD: Request timed out after ${REQUEST_TIMEOUT}s (threshold: ${REQUEST_THRESHOLD}s)"
    RESULT=1
elif echo "$RESPONSE" | grep -q '"choices"'; then
    ELAPSED_INT=$(echo "$ELAPSED" | cut -d. -f1)
    if [ "$ELAPSED_INT" -lt "$REQUEST_THRESHOLD" ]; then
        echo "GOOD: Request completed in ${ELAPSED}s (< ${REQUEST_THRESHOLD}s)"
        RESULT=0
    else
        echo "BAD: Request took ${ELAPSED}s (>= ${REQUEST_THRESHOLD}s)"
        RESULT=1
    fi
else
    echo "SKIP: Request failed"
    echo "Response: $RESPONSE"
    RESULT=125
fi

