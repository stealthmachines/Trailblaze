#!/usr/bin/env bash
# scripts/run.sh — Start the TRAILBLAZE inference server
# Usage: bash scripts/run.sh [--model <path>] [--port 11434] [--hdgl] [--test-only]
#
# Defaults to the smallest Qwen3 model (4B, ~2GB) for quick testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PORT=11434
HDGL_FLAGS=""
MODEL_PATH=""
TEST_ONLY=0

# ------ Qwen3 model catalogue ------
OLLAMA_BLOBS="$HOME/.ollama/models/blobs"
MODEL_4B="$OLLAMA_BLOBS/ermiaazarkhalili/Qwen3.5-4B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF/qwen3.5-4b-sft-claude-opus-reasoning-unsloth.q3_k_m.gguf"
MODEL_9B_Q3="$OLLAMA_BLOBS/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q3_K_XL.gguf"
MODEL_9B_Q6="$OLLAMA_BLOBS/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q6_K_XL.gguf"
MODEL_9B_Q8="$OLLAMA_BLOBS/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q8_K_XL.gguf"
MODEL_27B="$OLLAMA_BLOBS/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf"
MODEL_35B="$OLLAMA_BLOBS/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL_PATH="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --hdgl)      HDGL_FLAGS="--hdgl --hdgl-alpha 0.2"; shift ;;
        --test-only) TEST_ONLY=1; shift ;;
        --4b)        MODEL_PATH="$MODEL_4B"; shift ;;
        --9b)        MODEL_PATH="$MODEL_9B_Q3"; shift ;;
        --9b-q6)     MODEL_PATH="$MODEL_9B_Q6"; shift ;;
        --9b-q8)     MODEL_PATH="$MODEL_9B_Q8"; shift ;;
        --27b)       MODEL_PATH="$MODEL_27B"; shift ;;
        --35b)       MODEL_PATH="$MODEL_35B"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-select smallest available model if none specified
if [ -z "$MODEL_PATH" ]; then
    for m in "$MODEL_4B" "$MODEL_9B_Q3" "$MODEL_9B_Q6" "$MODEL_27B" "$MODEL_35B"; do
        if [ -f "$m" ]; then
            MODEL_PATH="$m"
            echo "  Auto-selected: $m"
            break
        fi
    done
fi

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No model found. Specify --model <path.gguf> or place Qwen3 GGUFs at:"
    echo "  $OLLAMA_BLOBS/..."
    exit 1
fi

if [ ! -f "./bin/tb_infer" ]; then
    echo "Binary not found — building first..."
    bash "$SCRIPT_DIR/build.sh"
fi

echo "=== TRAILBLAZE v0.3 Inference Server ==="
echo "  Model:  $MODEL_PATH"
echo "  Port:   $PORT"
echo "  HDGL:   ${HDGL_FLAGS:-disabled}"
echo ""

if [ $TEST_ONLY -eq 1 ]; then
    ./bin/tb_infer --model "$MODEL_PATH" --test
else
    ./bin/tb_infer --model "$MODEL_PATH" --serve --port "$PORT" $HDGL_FLAGS
fi
