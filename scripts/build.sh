#!/usr/bin/env bash
# scripts/build.sh — Build the TRAILBLAZE inference runtime in WSL/Linux
# Usage: bash scripts/build.sh [--debug] [--test]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

DEBUG=0
RUN_TEST=0
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG=1 ;;
        --test)  RUN_TEST=1 ;;
    esac
done

echo "=== TRAILBLAZE v0.3 Build ==="
echo "  Root: $ROOT"

# Ensure bin/ exists
mkdir -p bin

# Base flags
CC="gcc"
CFLAGS="-O3 -march=native -std=c11 -Wall -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-stringop-overflow -Wno-array-bounds"
if [ $DEBUG -eq 1 ]; then
    CFLAGS="-g -O0 -fsanitize=address,undefined -std=c11 -Wall"
    echo "  Mode: DEBUG (ASan+UBSan)"
else
    echo "  Mode: RELEASE (-O3 -march=native)"
fi

INC="-Ilayer0 -Ilayer1 -Ilayer2 -Ilayer3 -Ilayer4 -Ilayer5 -Iinclude -Isrc"
LDFLAGS="-lm -lpthread"

TB_CORE="layer0/tb_phi_lattice.c layer1/tb_tensor.c layer2/tb_graph.c"

TB_INFER_SRC="layer4/tb_infer.c layer4/tb_gguf.c layer4/tb_tokenizer.c \
    layer3/tb_orchestration.c \
    layer5/tb_semantic_os.c \
    $TB_CORE \
    src/sha256_minimal.c \
    src/hdgl_bootloaderz.c \
    src/hdgl_router.c \
    src/vector_container.c \
    src/analog_engine.c"

echo "  Compiling tb_infer ..."
$CC $CFLAGS -DTB_INFER_TEST $INC $TB_INFER_SRC $LDFLAGS -o bin/tb_infer
echo "  Done: bin/tb_infer"

if [ $RUN_TEST -eq 1 ]; then
    echo ""
    echo "=== Synthetic test (no model required) ==="
    ./bin/tb_infer --synthetic-test
fi

echo ""
echo "=== Build complete ==="
echo "  ./bin/tb_infer --model <path.gguf> [--serve] [--port 11434]"
