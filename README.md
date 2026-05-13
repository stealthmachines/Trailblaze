# TRAILBLAZE v0.5 — Qwen3 Inference Runtime

CPU-native inference runtime for Qwen3.x GGUF models.  Runs the full transformer
forward pass in portable C11 — no Python, no Docker, no CUDA required.  Exposes an
Ollama-compatible HTTP API so existing tools work unchanged.

## Status

| Capability | Status |
|---|---|
| GGUF mmap loader (v2/v3, all Qwen3 architectures) | ✅ complete |
| Dequantisation: Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K / Q4_0 / Q8_0 / BF16 | ✅ complete |
| Transformer forward pass (RMS-norm, RoPE, GQA, SwiGLU, MoE) | ✅ complete |
| Qwen3 per-head QK-norm | ✅ complete |
| Flash-style tiled attention (O(tile) memory, no heap alloc per token) | ✅ complete |
| Persistent per-session KV cache (up to 16 sessions) | ✅ complete (v0.4) |
| Pre-allocated decode scratch buffers (no per-token malloc) | ✅ complete (v0.4) |
| Ollama HTTP API — /api/generate, /api/chat, /api/tags | ✅ complete |
| Streaming responses (`stream:true` → Transfer-Encoding: chunked) | ✅ complete (v0.4) |
| Request body up to 256 KB, prompts up to 8 KB | ✅ complete (v0.4) |
| CPU SIMD dispatch (AVX-512 / AVX2 / scalar, oscillator-driven) | ✅ complete (v0.5) |
| HDGL phi-lattice MoE expert routing | ✅ implemented, unvalidated |
| CUDA / GPU acceleration | ⚠️ scaffold only (`-DTB_CUDA`) |
| Multi-session concurrent inference | ⚠️ serialised (generate_lock) |

## Benchmark

> These numbers use the synthetic self-test model (4 layers, 512 hidden, 8 experts).
> A real Qwen3-4B (36 layers, 2560 hidden) will show lower throughput.

```bash
bash scripts/build.sh
./bin/tb_infer --benchmark                             # synthetic model only
./bin/tb_infer --model path/to/model.gguf --benchmark  # real throughput
```

## What changed in v0.5

- **Analog dispatch layer** (`src/tb_analog_dispatch.{h,c}`) — bridges the 8D Kuramoto
  oscillator into a CPU SIMD scheduler.  Oscillator phase coefficient of variation (CV)
  selects kernel path:
  - PLUCK (CV > 0.50) → AVX-512 VNNI (highest throughput, scatter-friendly)
  - SUSTAIN / FINETUNE (CV 0.10–0.50) → AVX2 FMA (balanced)
  - LOCK (CV < 0.10) → scalar (phases settled, working set fits in L1)
- **S(U) secondary discriminant** — resonance measure further modulates path selection
  at runtime between SUSTAIN and FINETUNE sub-paths.
- **`TB_InferCtx`** gains `TBOscSnapshot osc_snap` and `TBCpuCaps cpu_caps`.
  `tb_dispatch_detect_caps()` runs once at init (CPUID); SIMD path elected per token.
- `Makefile` / `build.sh` updated to compile `src/tb_analog_dispatch.c`.

## What changed in v0.4

- **KV cache is now persistent** — allocated once per session in `ctx->session_kvs[]`,
  reused across decode calls.  Previously O(N²) in sequence length.
- **Scratch buffers** — `scratch_x / scratch_y / scratch_norm / scratch_logits` pre-allocated.
- **HTTP body reading** — reads until `Content-Length`, up to 256 KB.
- **Streaming** — `stream:true` delivers chunked Transfer-Encoding, one JSON line per token.
- **UAF fix** — `cc` fields copied before `free(cc)` in `tb_handle_conn`.
- **Unified /api/generate + /api/chat handler** — merged, de-duplicated.

## What changed in v0.3

- Qwen3 GGUF metadata keys: `embedding_length`, `block_count`, `head_count`/`head_count_kv`,
  `feed_forward_length`, `expert_count`/`expert_used_count`, `rms_norm_eps`,
  `moe_intermediate_size`, `max_position_embeddings`
- Qwen3 per-head QK-norm (`blk.L.attn_q_norm.weight`, `blk.L.attn_k_norm.weight`)
- All weight dequantisation routed through `tb_gguf_dequant_matvec()`
- MoE packed tensors (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`)
- RoPE base auto-read from GGUF (Qwen3 uses 1,000,000)
- Flash-attention style tiling; O(N+k log k) nucleus sampler

## Available models

All in `~/.ollama/models/blobs/`:

| Model | Quant | Size | Notes |
|---|---|---|---|
| `ermiaazarkhalili/.../qwen3.5-4b-sft-claude-opus-reasoning-unsloth.q3_k_m.gguf` | Q3_K_M | 2.1 GB | **Start here** |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q3_K_XL.gguf` | Q3_K_XL | 4.7 GB | |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q6_K_XL.gguf` | Q6_K_XL | 8.2 GB | |
| `unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf` | Q2_K_XL | 11 GB | |
| `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | Q2_K_XL | 11.5 GB | MoE 128 experts |

## Build

```bash
# Linux / WSL:
bash scripts/build.sh

# Or via make:
make infer

# Debug (ASan + UBSan):
bash scripts/build.sh --debug
```

**Requirements:** `gcc`, `libm`, `libpthread` (standard on Ubuntu/Debian)

## Run

```bash
bash scripts/run.sh                        # auto-select smallest model
bash scripts/run.sh --4b                   # Qwen3.5-4B  (2.1 GB)
bash scripts/run.sh --9b                   # Qwen3.5-9B  Q3
bash scripts/run.sh --27b                  # Qwen3.6-27B Q2
bash scripts/run.sh --35b                  # Qwen3.6-35B MoE Q2
bash scripts/run.sh --hdgl                 # Enable phi-lattice routing
bash scripts/run.sh --model /path/to/model.gguf --port 11434
```

## Test

```bash
# Non-streaming:
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen3","prompt":"What is consciousness?","stream":false}'

# Streaming (token-by-token):
curl -N http://localhost:11434/api/generate \
  -d '{"model":"qwen3","prompt":"Explain RoPE embeddings","stream":true}'

# Chat:
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"hello"}],"stream":false}'

# Model list:
curl http://localhost:11434/api/tags
```

## Architecture

```
Layer 0: tb_phi_lattice.{h,c}    — 8D Kuramoto RK4 oscillator, Weyl lattice, PhiFold AEAD
Layer 1: tb_tensor.{h,c}         — Epoch-aware tensors, flash-tiled KV attention,
                                    O(N+k log k) nucleus sampler
Layer 2: tb_graph.{h,c}          — HDGL graph engine, ERL ledger, CognitionTree
Layer 3: tb_orchestration.{h,c}  — Agent fabric, tool registry (19 built-ins),
                                    unfold engine, HTTP extension routes
Layer 4: tb_infer.c              — Transformer forward pass, HTTP server (Ollama-compat),
                                    persistent KV cache (TB_MAX_SESSIONS=16),
                                    chunked streaming, 256 KB request body
         tb_gguf.c               — GGUF mmap loader (all Qwen3 architectures)
         tb_tokenizer.c          — BPE tokenizer (from GGUF vocab)
Layer 5: tb_semantic_os.{h,c}    — Capability tokens, WuWei codec (5 strategies),
                                    SemanticContext, SemanticCompressor

src/:    tb_analog_dispatch.{h,c} — Oscillator-driven CPU SIMD path selector (NEW in v0.5)
                                    (AVX-512 / AVX2 / scalar)
```

## HDGL routing

When `--hdgl` is passed, expert selection in MoE layers is modulated by the phi-lattice
oscillator (Kuramoto + Spiral8 + softmax top-k blend).  The critic TD module observes
routing quality and adjusts the alpha modulator per token.  Use `--no-hdgl` to compare.

## Key constants

| Constant | Value |
|---|---|
| PHI | 1.6180339887498948 |
| DN_EMPIRICAL_BETA | 0.360942 |
| Qwen3 RoPE base | 1,000,000 (read from GGUF) |
| Default context | 8192 tokens |
| HTTP port | 11434 (Ollama-compatible) |
| Max request body | 256 KB |
| Attention tile | 32 positions |
| Nucleus candidate cap | 2048 tokens |
| Max persistent sessions | 16 |

## License

All files remain property of ZCHG.org pursuant to
https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
Licensing inquiries: charg.chg.wecharg@gmail.com
| Max persistent sessions | 16 |
