# TRAILBLAZE v0.3 — Qwen3 Inference Runtime

Complete drop-in replacement for Ollama + CUDA using the TRAILBLAZE C runtime. Runs Qwen3.x GGUFs directly — no Python, no Docker, no CUDA required.

## Benchmark (synthetic, no model file required)

| Mode | Tokens | Time | Throughput |
|---|---|---|---|
| Generate (prefill + 8-token decode) | 8 | ~93 ms | **88 tok/s** |
| Pure decode (20 tokens, warm context) | 20 | ~49 ms | **407 tok/s** |

Hardware: measured on WSL/Linux with synthetic Mixtral-like config (4 layers, 512 hidden, 8 experts).

```bash
bash scripts/build.sh
./bin/tb_infer --benchmark   # reproduces the numbers above
```

## Strengths (v0.3)

| Weakness → Strength | How it's solved |
|---|---|
| O(N²) top-p sampler | Two-pass nucleus: O(N) softmax + O(N) candidate filter + O(k log k) sort; k ≤ 2048 << 152K vocab |
| No tiled attention kernel | Flash-attention style tiling (TB_ATTN_TILE=32): no O(seq_len) heap alloc, L1-cache-friendly |
| Concurrent request races | `pthread_mutex_t generate_lock` in `TB_InferCtx` serialises concurrent HTTP connections safely |
| WuWei codec untested | Full 5-strategy round-trip self-test in `--benchmark` / `--test` mode; PHI_COMPRESS 0xFF sentinel bug fixed |
| op_dispatch CODE no-op | Layer 3 CODE path now emits structured JSON analysis (lang, line-count, excerpt) without recursive call |

## What's different from v0.2

- **tb_gguf.c** — Qwen3 GGUF metadata keys: `embedding_length`, `block_count`, `head_count`/`head_count_kv`, `feed_forward_length`, `expert_count`/`expert_used_count`, `rms_norm_eps`, `moe_intermediate_size`, `max_position_embeddings`
- **tb_infer.c** — Qwen3 per-head QK-norm (`blk.L.attn_q_norm.weight`, `blk.L.attn_k_norm.weight`)
- **tb_infer.c** — All weight dequantisation routed through `tb_gguf_dequant_matvec()` (supports Q4_0/Q4_K/Q6_K/Q8_0/BF16/F16/F32)
- **tb_infer.c** — MoE packed tensors (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`)
- RoPE base auto-read from GGUF (Qwen3 uses 1,000,000, not 10,000)

## Available models

All in `~/.ollama/models/blobs/`:

| Model | Quant | Size | Notes |
|---|---|---|---|
| `ermiaazarkhalili/…/qwen3.5-4b-sft-claude-opus-reasoning-unsloth.q3_k_m.gguf` | Q3_K_M | 2.1 GB | **Smallest — start here** |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q3_K_XL.gguf` | Q3_K_XL | 4.7 GB | |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q6_K_XL.gguf` | Q6_K_XL | 8.2 GB | |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q8_K_XL.gguf` | Q8_K_XL | 12 GB | |
| `unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf` | Q2_K_XL | 11 GB | |
| `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | Q2_K_XL | 11.5 GB | MoE 128 experts |

## Build (Linux / WSL)

```bash
# From the v0.3/ directory in WSL:
bash scripts/build.sh

# Or via make:
make infer

# Debug build (ASan+UBSan):
bash scripts/build.sh --debug
```

**Requirements:** `gcc`, `libm`, `libpthread` (all standard on Ubuntu/Debian)

## Run

```bash
# Auto-select smallest available model:
bash scripts/run.sh

# Specific model:
bash scripts/run.sh --4b         # Qwen3.5-4B  (2.1 GB)
bash scripts/run.sh --9b         # Qwen3.5-9B  Q3
bash scripts/run.sh --27b        # Qwen3.6-27B Q2
bash scripts/run.sh --35b        # Qwen3.6-35B MoE Q2
bash scripts/run.sh --hdgl       # Enable HDGL phi-lattice routing

# Explicit path:
bash scripts/run.sh --model /path/to/model.gguf --port 11434

# Via make:
make serve-4b
make serve-9b
make serve-27b
```

## Test

```bash
# HTTP API test (server must be running):
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen3","prompt":"What is consciousness?","stream":false}'

curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"hello"}]}'

# List loaded model:
curl http://localhost:11434/api/tags
```

## Architecture

```
Layer 0: tb_phi_lattice.{h,c}    — Phi-lattice oscillator, Kuramoto sync, AEAD
Layer 1: tb_tensor.{h,c}         — Epoch-aware tensors, flash-tiled KV attention,
                                    O(N+k log k) nucleus sampler
Layer 2: tb_graph.{h,c}          — HDGL graph engine, ERL ledger, CognitionTree
Layer 3: tb_orchestration.{h,c}  — Agent fabric, tool registry (19 built-ins),
                                    unfold engine, HTTP extension routes
Layer 4: tb_infer.c              — Transformer forward pass, HTTP server (Ollama-compat),
                                    thread-safe request serialisation
         tb_gguf.c               — GGUF mmap loader (all Qwen3 architectures)
         tb_tokenizer.c          — BPE tokenizer (from GGUF vocab)
Layer 5: tb_semantic_os.{h,c}    — Capability tokens, WuWei codec (5 strategies),
                                    SemanticContext, SemanticCompressor
```

## HDGL routing

When `--hdgl` is passed, the routing layer uses the phi-lattice oscillator network (Kuramoto + Spiral8 + softmax top-k) to modulate expert selection in MoE layers. The `--hdgl-alpha 0.2` parameter controls blend strength between standard and phi-modulated routing.

At each token, the critic TD module observes routing quality (top-1 expert confidence) and adjusts the alpha modulator live — low-confidence tokens get stronger HDGL influence, high-confidence tokens fall back to standard top-k.

## Key constants

| Constant | Value |
|---|---|
| PHI | 1.6180339887498948 |
| DN_EMPIRICAL_BETA | 0.360942 |
| Qwen3 RoPE base | 1,000,000 (read from GGUF) |
| Default context | 8192 tokens |
| HTTP port | 11434 (Ollama-compatible) |
| Attention tile (flash) | 32 positions |
| Nucleus candidate cap | 2048 tokens |

