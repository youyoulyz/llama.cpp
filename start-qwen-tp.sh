#!/usr/bin/env bash
#
# One-click start: Qwen3.8-27B TP service (2x RTX 3080 20G)
#   - tensor parallel:  -sm tensor (NCCL)
#   - max context:      default 262144, total context split among slots
#   - flash attention:  -fa on
#   - KV cache quant:   -ctk q8_0 -ctv q8_0 (f16 → q8, half KV memory)
#   - speculation:      default draft-mtp; set SPEC_TYPE=draft-dflash with SPEC_DRAFT_MODEL
#                       to use the DFlash2 drafter (works with -sm tensor since the draft
#                       is pinned to a single GPU and owns local copies of tok_embd/output)
#   - vision:           mmproj attached automatically, --image-min-tokens 1024
#
# Usage: ./start-qwen-tp.sh
# Override via env: CTX, PARALLEL, PORT, HOST, MTP_N_MAX, MODEL, MMPROJ, LOG,
#                   FLASH_ATTN, CACHE_TYPE_K, CACHE_TYPE_V,
#                   SPEC_TYPE, SPEC_DRAFT_MODEL, SPEC_DRAFT_N_MAX
#
# Note: -c is the TOTAL context, divided equally among slots.
#       With Q8 KV cache, each slot uses ~half KV memory vs f16.

set -euo pipefail

# ---- config ----
MODEL=${MODEL:-/mnt/public/GGUF_models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-/mnt/public/GGUF_models/Qwen3.8-27B-GGUF/mmproj-BF16.gguf}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$SCRIPT_DIR/build/bin/llama-server}
CTX=${CTX:-262144}
PARALLEL=${PARALLEL:-2}
MTP_N_MAX=${MTP_N_MAX:-2}
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
LOG=${LOG:-$SCRIPT_DIR/llama-server.log}
FLASH_ATTN=${FLASH_ATTN:-on}
CACHE_TYPE_K=${CACHE_TYPE_K:-q8_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-q8_0}
SPEC_TYPE=${SPEC_TYPE:-draft-mtp}
SPEC_DRAFT_MODEL=${SPEC_DRAFT_MODEL:-/mnt/public/GGUF_models/Qwen3.8-27B-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}
SPEC_DRAFT_N_MAX=${SPEC_DRAFT_N_MAX:-2}

# ---- sanity checks ----
for f in "$MODEL" "$MMPROJ" "$BIN"; do
    if [ ! -f "$f" ]; then
        echo "error: file not found: $f" >&2
        exit 1
    fi
done

if [ "$PARALLEL" -lt 1 ]; then
    echo "error: PARALLEL must be >= 1" >&2
    exit 1
fi

SPEC_ARGS=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_DRAFT_N_MAX")
if [ "$SPEC_TYPE" != "draft-mtp" ] && [ "$SPEC_TYPE" != "ngram" ]; then
    SPEC_ARGS+=(-md "$SPEC_DRAFT_MODEL")
fi

echo "starting Qwen3.8-27B TP service:"
echo "  model     : $MODEL"
echo "  context   : $CTX total (per slot: $((CTX / PARALLEL))), $PARALLEL slots"
echo "  flash attn: $FLASH_ATTN"
echo "  KV cache  : K=$CACHE_TYPE_K, V=$CACHE_TYPE_V"
echo "  spec      : $SPEC_TYPE n-max=$SPEC_DRAFT_N_MAX"
echo "  listen    : $HOST:$PORT"
echo "  log       : $LOG"

"$BIN" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    -ngl 99 \
    -sm tensor \
    -c "$CTX" \
    --parallel "$PARALLEL" \
    "${SPEC_ARGS[@]}" \
    -fa "$FLASH_ATTN" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --image-min-tokens 1024 \
    --kv-unified \
    --host "$HOST" \
    --port "$PORT" \
    2>&1 | tee "$LOG" &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup INT TERM

echo "waiting for model load and health check..."
READY=0
for i in $(seq 1 200); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        echo "ready: http://$HOST:$PORT (OpenAI API: /v1/chat/completions)"
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: server exited, see $LOG" >&2
        exit 1
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "error: server not ready after 1000s, see $LOG" >&2
    exit 1
fi

wait "$SERVER_PID"
