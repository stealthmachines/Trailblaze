# TRAILBLAZE — Post-CUDA Recursive Cognition Runtime
# Master Makefile
#
# Derived from: NGINX-HDGL-0.6-c/Makefile (extended)
# Targets:
#   make all        — build integration test + layer tests
#   make test       — run full integration test
#   make bench      — benchmark suite
#   make layers     — build + test each layer independently
#   make daemon     — build the zchg MCP daemon (Layer 4 HTTP server)
#   make clean

CC       = gcc
CFLAGS   = -O3 -march=native -std=c11 -Wall -Wextra \
           -Wno-unused-result -Wno-unused-parameter \
           -Wno-stringop-overflow -Wno-array-bounds \
           -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
LDFLAGS  = -lm -lpthread
DBGFLAGS = -g -O0 -fsanitize=address,undefined

SRCDIR   = src
INCDIR   = include
BINDIR   = bin
OBJDIR   = obj

UPSTREAM_SRC = \
    $(SRCDIR)/zchg_lattice.c \
    $(SRCDIR)/zchg_frame.c \
    $(SRCDIR)/zchg_gossip.c \
    $(SRCDIR)/zchg_transport.c \
    $(SRCDIR)/zchg_http.c \
    $(SRCDIR)/zchg_fileswap.c \
    $(SRCDIR)/zchg_store.c

TB_CORE = \
    layer0/tb_phi_lattice.c \
    layer1/tb_tensor.c \
    layer2/tb_graph.c

# Source list for tb_infer (Ollama/CUDA replacement, all 6 layers)
TB_INFER_SRC = \
    layer4/tb_infer.c \
    layer4/tb_gguf.c \
    layer4/tb_tokenizer.c \
    layer3/tb_orchestration.c \
    layer5/tb_semantic_os.c \
    $(TB_CORE) \
    src/hdgl_bootloaderz.c \
    src/hdgl_router.c \
    src/analog_engine.c \
    src/vector_container.c \
    src/sha256_minimal.c

.PHONY: all test bench layers daemon clean debug help

all: $(BINDIR) $(BINDIR)/tb_integration_test

$(BINDIR):
	@mkdir -p $(BINDIR)

# ── Integration test (all layers) ──────────────────────────────────────────
$(BINDIR)/tb_integration_test: tb_integration_test.c $(TB_CORE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built: $@"

test: $(BINDIR)/tb_integration_test
	@echo ""
	@echo "Running TRAILBLAZE C integration test..."
	@echo ""
	@$(BINDIR)/tb_integration_test

# ── Individual layer tests ─────────────────────────────────────────────────
$(BINDIR)/tb_l0_test: layer0/tb_phi_lattice.c
	$(CC) $(CFLAGS) -DTB_L0_TEST -o $@ $^ $(LDFLAGS)
	@echo "✓ Built: $@"

$(BINDIR)/tb_l1_test: layer1/tb_tensor.c layer0/tb_phi_lattice.c
	$(CC) $(CFLAGS) -DTB_L1_TEST -o $@ $^ $(LDFLAGS)
	@echo "✓ Built: $@"

$(BINDIR)/tb_l23_test: layer2/tb_graph.c layer1/tb_tensor.c layer0/tb_phi_lattice.c
	$(CC) $(CFLAGS) -DTB_L23_TEST -o $@ $^ $(LDFLAGS)
	@echo "✓ Built: $@"

layers: $(BINDIR) \
    $(BINDIR)/tb_l0_test \
    $(BINDIR)/tb_l1_test \
    $(BINDIR)/tb_l23_test
	@echo ""
	@echo "=== Layer 0 ==="
	@$(BINDIR)/tb_l0_test
	@echo ""
	@echo "=== Layer 1 ==="
	@$(BINDIR)/tb_l1_test
	@echo ""
	@echo "=== Layers 2+3 ==="
	@$(BINDIR)/tb_l23_test

# ── Benchmark ──────────────────────────────────────────────────────────────
bench: $(BINDIR)/tb_integration_test
	@echo "=== TRAILBLAZE Benchmark ==="
	@$(BINDIR)/tb_integration_test 2>&1 | grep -E "^\s+✓.*(ms|GFLOPS|/s)"

# ── Inference runtime (tb_infer — Ollama/CUDA replacement) ────────────────
# All 6 layers: phi-lattice + tensor + graph + orchestration + GGUF + semantic
# Usage: make infer && bin/tb_infer --model path/to/model.gguf --serve
infer: $(BINDIR)
	$(CC) $(CFLAGS) \
	    -Ilayer3 -Ilayer4 -Ilayer5 -I$(INCDIR) -Isrc \
	    $(TB_INFER_SRC) \
	    -DTB_INFER_MAIN \
	    -o $(BINDIR)/tb_infer \
	    $(LDFLAGS)
	@echo "✓ Built: $(BINDIR)/tb_infer (6-layer Ollama replacement)"
	@echo "  Run:   bin/tb_infer --model model.gguf --serve --port 11434"
	@echo "  Test:  bin/tb_infer --model model.gguf --prompt 'Hello'"

# ── MCP daemon (Layer 4: zchg HTTP server with TRAILBLAZE backend) ─────────
# Requires: libssl-dev (apt install libssl-dev)
daemon: $(BINDIR) $(UPSTREAM_SRC) $(SRCDIR)/zchg_main.c $(TB_CORE)
	@echo "Building TRAILBLAZE MCP daemon (requires libssl)..."
	$(CC) $(CFLAGS) -I$(INCDIR) \
	    $(SRCDIR)/zchg_main.c \
	    $(SRCDIR)/zchg_lattice.c \
	    $(SRCDIR)/zchg_frame.c \
	    $(SRCDIR)/zchg_gossip.c \
	    $(SRCDIR)/zchg_transport.c \
	    $(SRCDIR)/zchg_http.c \
	    $(SRCDIR)/zchg_fileswap.c \
	    $(TB_CORE) \
	    -o $(BINDIR)/trailblaze_daemon \
	    $(LDFLAGS) -lssl -lcrypto
	@echo "✓ Built: $(BINDIR)/trailblaze_daemon"
	@echo "  Start: LN_LOCAL_NODE=127.0.0.1 LN_CLUSTER_SECRET=secret $(BINDIR)/trailblaze_daemon"

# ── Debug build with ASan/UBSan ────────────────────────────────────────────
debug: $(BINDIR)
	$(CC) $(CFLAGS) $(DBGFLAGS) \
	    tb_integration_test.c $(TB_CORE) \
	    -o $(BINDIR)/tb_integration_debug $(LDFLAGS)
	@echo "✓ Debug build: $(BINDIR)/tb_integration_debug"
	@$(BINDIR)/tb_integration_debug

# ── CUDA path (Phase 2 — requires nvcc + CUDA toolkit) ────────────────────
cuda:
	@echo "CUDA kernels (Phase 2 — not yet implemented)"
	@echo "Stubs: src/hdgl_analog_v31.c (ll_analog backend)"
	@echo "CUDA:  src/hdgl_analog_v35.cu (once CUDA env available)"
	@echo ""
	@echo "To build with CUDA when available:"
	@echo "  nvcc -O3 -arch=sm_80 src/hdgl_analog_v35.cu layer0/tb_phi_lattice.c -lm -o bin/tb_cuda"

# ── Store test (HDGL-SQL strand-native persistence) ────────────────────────
store: $(BINDIR) $(SRCDIR)/zchg_store.c $(SRCDIR)/zchg_lattice.c layer0/tb_phi_lattice.c
	$(CC) $(CFLAGS) -I$(INCDIR) -DTB_STORE_TEST \
	    $(SRCDIR)/zchg_store.c \
	    $(SRCDIR)/zchg_lattice.c \
	    layer0/tb_phi_lattice.c \
	    -o $(BINDIR)/tb_store_test $(LDFLAGS)
	@echo "✓ Built store test"
	@$(BINDIR)/tb_store_test

clean:
	rm -rf $(BINDIR) $(OBJDIR)
	@echo "✓ Cleaned"

help:
	@echo "TRAILBLAZE C Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make all     — build integration test (default)"
	@echo "  make test    — run full integration test (all layers)"
	@echo "  make layers  — build + test each layer independently"
	@echo "  make bench   — benchmark suite"
	@echo "  make infer   — build tb_infer (Ollama/CUDA replacement, all 6 layers)"
	@echo "  make daemon  — build MCP HTTP daemon (needs libssl)"
	@echo "  make debug   — ASan+UBSan debug build"
	@echo "  make clean   — remove build artifacts"
	@echo ""
	@echo "Architecture (6 layers):"
	@echo "  Layer 0: layer0/tb_phi_lattice.{h,c}   — Phi-lattice, Kuramoto, AEAD"
	@echo "  Layer 1: layer1/tb_tensor.{h,c}         — Tensors, KV cache, Hopfield"
	@echo "  Layer 2: layer2/tb_graph.{h,c}          — Graph engine, HDGL router"
	@echo "  Layer 3: layer3/tb_orchestration.{h,c}  — Agent fabric, tool registry, unfold"
	@echo "  Layer 4: layer4/tb_infer.{h,c}          — GGUF inference, HTTP server"
	@echo "  Layer 5: layer5/tb_semantic_os.{h,c}    — Semantic OS, WuWei codec, cap tokens"
	@echo ""
	@echo "Upstream C sources (from uploaded repos):"
	@echo "  src/ll_analog.c         — 8D Kuramoto oscillator (from analog-prime)"
	@echo "  src/hdgl_analog_v31.c   — GRA closed-form (from analog-prime)"
	@echo "  src/zchg_*.c            — HDGL daemon (from NGINX-HDGL-0.6-c)"
	@echo "  src/zchg_store.c        — Strand-native persistence (from HDGL-SQL)"
	@echo ""
	@echo "Performance targets (v0.1 C baseline):"
	@echo "  phi_fold32:   ~15K/s"
	@echo "  slot routing: ~200K/s"
	@echo "  SGEMM 64x64:  auto-vectorised"
	@echo "  ERL append:   ~50K/s"

# ── Inference runtime (Ollama replacement) ──────────────────────────────────
# Build: make infer
# Test:  make infer-test
# Serve: ./bin/tb_infer --model ./models/mistral-7b-q4.gguf --serve --port 11434
# Deps:  sha256_minimal.c, zchg_lattice.c, zchg_store.c, hdgl_router.c, vector_container.c

TB_INFER_SRC = \
    layer4/tb_infer.c \
    layer4/tb_gguf.c \
    layer4/tb_tokenizer.c \
    $(TB_CORE) \
    $(SRCDIR)/sha256_minimal.c \
    $(SRCDIR)/zchg_lattice.c \
    $(SRCDIR)/zchg_store.c \
    $(SRCDIR)/hdgl_router.c \
    $(SRCDIR)/vector_container.c \
    $(SRCDIR)/analog_engine.c

infer: $(BINDIR) $(TB_INFER_SRC)
	$(CC) $(CFLAGS) -DTB_INFER_TEST \
	    -Ilayer0 -Ilayer1 -Ilayer2 -Ilayer4 -I$(INCDIR) -I$(SRCDIR) \
	    $(TB_INFER_SRC) \
	    -lm -lpthread -o $(BINDIR)/tb_infer
	@echo "✓ Built: $(BINDIR)/tb_infer"
	@echo "  Usage:  $(BINDIR)/tb_infer --model <path.gguf> [--serve --port 11434]"
	@echo "  HDGL:   $(BINDIR)/tb_infer --model <path.gguf> --serve --hdgl --hdgl-alpha 0.2"

# ── Qwen3 model paths (all in ~/.ollama/models/blobs/) ─────────────────────
QWEN3_4B  = $(HOME)/.ollama/models/blobs/ermiaazarkhalili/Qwen3.5-4B-SFT-Claude-Opus-Reasoning-Unsloth-GGUF/qwen3.5-4b-sft-claude-opus-reasoning-unsloth.q3_k_m.gguf
QWEN3_9B_Q3  = $(HOME)/.ollama/models/blobs/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q3_K_XL.gguf
QWEN3_9B_Q6  = $(HOME)/.ollama/models/blobs/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q6_K_XL.gguf
QWEN3_9B_Q8  = $(HOME)/.ollama/models/blobs/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q8_K_XL.gguf
QWEN3_27B    = $(HOME)/.ollama/models/blobs/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf
QWEN3_35B    = $(HOME)/.ollama/models/blobs/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

PORT     = 11434
HDGL_FLAGS = --hdgl --hdgl-alpha 0.2

detect-models:
	@echo "=== Available Qwen3 models (from ~/.ollama/models) ==="
	@for m in "$(QWEN3_4B)" "$(QWEN3_9B_Q3)" "$(QWEN3_9B_Q6)" "$(QWEN3_9B_Q8)" "$(QWEN3_27B)" "$(QWEN3_35B)"; do \
	    if [ -f "$$m" ]; then echo "  FOUND: $$m"; \
	    else echo "  MISSING: $$m"; fi; done

# Quick test — load model, print config, run synthetic prompt, exit
infer-test: infer
	$(BINDIR)/tb_infer --model $(QWEN3_4B) --test

# Start HTTP server bound to Ollama-compatible port
serve-4b: infer
	$(BINDIR)/tb_infer --model $(QWEN3_4B) --serve --port $(PORT) $(HDGL_FLAGS)

serve-9b: infer
	$(BINDIR)/tb_infer --model $(QWEN3_9B_Q3) --serve --port $(PORT) $(HDGL_FLAGS)

serve-27b: infer
	$(BINDIR)/tb_infer --model $(QWEN3_27B) --serve --port $(PORT) $(HDGL_FLAGS)

serve-35b: infer
	$(BINDIR)/tb_infer --model $(QWEN3_35B) --serve --port $(PORT) $(HDGL_FLAGS)

# ── Full (integration test + inference + store + daemon) ───────────────────
full: test infer store
	@echo ""
	@echo "=== TRAILBLAZE FULL BUILD COMPLETE ==="
	@echo "  Test:  $(BINDIR)/tb_infer --benchmark"

infer-test: $(BINDIR)/tb_infer
	@$(BINDIR)/tb_infer --benchmark

# Diff targets (rapid development aid)
diff-hdgl-sql: ## Diff HDGL-SQL v0.1 → v0.2
	@diff -ruN \
	    /tmp/tb_new/HDGL-SQL-main/HDGL-SQL-main/src/ \
	    /tmp/hdgl_sql_02/HDGL-SQL-0.2/src/ \
	    > ../diffs/hdgl_sql_v01_v02.patch 2>&1 || true
	@echo "✓ Patch: ../diffs/hdgl_sql_v01_v02.patch"

diff-infer: ## Diff conscious nonmetal_infer → tb_infer
	@diff -u \
	    /tmp/conscious/conscious-128-bit-floor/metal_infer_for_primes/nonmetal_infer.c \
	    layer4/tb_infer.c \
	    > ../diffs/nonmetal_vs_tbinfer.patch 2>&1 || true
	@echo "✓ Patch: ../diffs/nonmetal_vs_tbinfer.patch"

# ── Tokenizer + Long-Term Cognition ────────────────────────────────────────
tokenizer: $(BINDIR)
	$(CC) $(CFLAGS) -DTB_TOK_TEST -Ilayer4 \
	    layer4/tb_tokenizer.c $(TB_CORE) \
	    -lm -o $(BINDIR)/tb_tokenizer
	@echo "✓ Built: $(BINDIR)/tb_tokenizer"

tokenizer-test: $(BINDIR)/tb_tokenizer
	@$(BINDIR)/tb_tokenizer

# ── Full stack: infer with tokenizer + LTC ─────────────────────────────────
full: $(BINDIR)
	$(CC) $(CFLAGS) -DTB_INFER_TEST -Ilayer4 \
	    layer4/tb_infer.c layer4/tb_tokenizer.c \
	    $(TB_CORE) -lm -lpthread -o $(BINDIR)/trailblaze
	@echo "✓ Built: $(BINDIR)/trailblaze (full Ollama replacement)"
	@echo "  Run:  $(BINDIR)/trailblaze --model <path> --serve --port 11434"

# ── Conscious diff (rapid development) ────────────────────────────────────
diff-conscious: ## Diff conscious nonmetal_infer → tb_infer + tb_tokenizer
	@diff -u \
	    /tmp/conscious/conscious-128-bit-floor/metal_infer_for_primes/nonmetal_infer.c \
	    layer4/tb_infer.c \
	    > ../diffs/nonmetal_vs_tbinfer.patch 2>&1 || true
	@diff -u \
	    /tmp/conscious/conscious-128-bit-floor/metal_infer_for_primes/tokenizer.h \
	    layer4/tb_tokenizer.h \
	    > ../diffs/tokenizer_delta.patch 2>&1 || true
	@diff -u \
	    /tmp/conscious/conscious-128-bit-floor/metal_infer_for_primes/analog_engine.h \
	    include/analog_engine.h \
	    > ../diffs/analog_engine_delta.patch 2>&1 || true
	@echo "✓ Patches written to ../diffs/"
	@echo "  hdgl_sql_v01_v02.patch  — store scaling changes"
	@echo "  nonmetal_vs_tbinfer.patch — inference runtime delta"
	@echo "  tokenizer_delta.patch   — BPE tokenizer extensions"
	@echo "  analog_engine_delta.patch — Kuramoto oscillator"
