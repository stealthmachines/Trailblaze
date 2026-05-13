# TRAILBLAZE v0.7 — Qwen3 Inference Runtime + HDGL Phi-Lattice Routing

CPU-native (with optional CUDA) inference runtime for Qwen3.x GGUF models.
Runs the full transformer forward pass in portable C11.  No Python, no Docker.
Exposes an Ollama-compatible HTTP API so existing tools (Open WebUI, LangChain,
Continue, etc.) work unchanged.

**v0.7 highlights:** HDGL phi-lattice expert routing validated on RTX 2060 (3.0 tok/s),
WuWei compression codec (5 strategies), analog dispatch with CUDA 4-stream async,
multi-session KV persistence, full Qwen3.5 & Qwen3.6 support.

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
| CUDA dequant pipeline (4-stream async, all 8 quant types) | ✅ complete (v0.6) |
| HDGL phi-lattice MoE expert routing | ✅ complete & validated (v0.7) |
| WuWei codec (DELTA_FOLD, PHI_COMPRESS, SPIRAL_PACK, RESONANCE, RAW) | ✅ complete (v0.7) |
| Multi-session concurrent inference | ⚠️ serialised (generate_lock) |

## Quick start

**Windows (native — no WSL required):**
```bat
git clone https://github.com/stealthmachines/Trailblaze.git -b v0.7
cd Trailblaze
:: Open "x64 Native Tools Command Prompt for VS 2022", then:
nmake -f Makefile.win          :: produces bin\tb_infer.exe  (CUDA auto-linked if present)
bin\tb_infer.exe --model "%USERPROFILE%\.ollama\models\blobs\...\model.gguf" --port 11434
```

**Linux / macOS:**
```bash
git clone https://github.com/stealthmachines/Trailblaze.git -b v0.7
cd Trailblaze
bash scripts/build.sh          # produces bin/tb_infer
bash scripts/run.sh --4b       # auto-locates Qwen3.5-4B GGUF
```

**Chat (any platform):**
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen3","prompt":"Hello","stream":false}'
```

**Requirements:**
- Windows: MSVC (VS 2022 Build Tools) — `cl.exe` + `ws2_32.lib`; no POSIX runtime needed (`src/tb_win32.h` shim covers mmap/pthreads/clock_gettime)
- Linux/macOS: `gcc ≥ 11`, `libm`, `libpthread`
- Optional (both): CUDA Toolkit ≥ 11.4 for GPU acceleration

## Build options

**Windows (x64 Native Tools Command Prompt):**
```bat
nmake -f Makefile.win           :: CUDA auto-linked if nvcc on PATH
nmake -f Makefile.win benchmark :: synthetic self-test
nmake -f Makefile.win clean
```

**Linux / macOS:**
```bash
bash scripts/build.sh           # auto-detect CUDA; CPU-only if nvcc absent
bash scripts/build.sh --debug   # ASan + UBSan
bash scripts/build.sh --cuda    # force CUDA build (requires nvcc)
bash scripts/build.sh --no-cuda # force CPU-only
make infer                      # equivalent to build.sh (no CUDA auto-detect)
```

## Run

**Windows:**
```bat
bin\tb_infer.exe --model "%USERPROFILE%\.ollama\models\blobs\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-UD-Q3_K_XL.gguf" --port 11434
bin\tb_infer.exe --benchmark
nmake -f Makefile.win run-4b    :: shorthand for smallest model
nmake -f Makefile.win run-9b    :: Q6_K_XL
```

**Linux / macOS:**
```bash
bash scripts/run.sh                              # auto-select smallest model
bash scripts/run.sh --4b                         # Qwen3.5-4B  Q3_K_M (2.1 GB)
bash scripts/run.sh --9b                         # Qwen3.5-9B  Q3_K_XL (4.7 GB)
bash scripts/run.sh --27b                        # Qwen3.6-27B Q2_K_XL (11 GB)
bash scripts/run.sh --35b                        # Qwen3.6-35B MoE Q2 (11.5 GB)
bash scripts/run.sh --hdgl                       # enable phi-lattice routing
bash scripts/run.sh --model /path/to/model.gguf --port 11434
./bin/tb_infer --benchmark
```

## HTTP API

The server starts on port `11434` by default (Ollama-compatible).

### Generate (non-streaming)

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "prompt": "Explain attention mechanisms",
    "stream": false,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

Response:
```json
{
  "model": "qwen3",
  "done": true,
  "response": "Attention mechanisms allow...",
  "eval_count": 42,
  "eval_duration": 1234000000
}
```

### Generate (streaming, token-by-token)

```bash
curl -N http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "prompt": "Write a haiku about inference",
    "stream": true
  }'
```

Each chunk:
```json
{"model":"qwen3","done":false,"response":"Weights"}
{"model":"qwen3","done":false,"response":" multiply"}
...
{"model":"qwen3","done":true,"eval_count":17,"eval_duration":480000000}
```

### Chat

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "What is phi?"}],
    "stream": false
  }'
```

### Model list

```bash
curl http://localhost:11434/api/tags
```

### Health check

```bash
curl http://localhost:11434/health
# {"status":"ok","runtime":"trailblaze"}
```

## Using with Open WebUI

```bat
:: Windows — start the server:
bin\tb_infer.exe --model "%USERPROFILE%\.ollama\models\blobs\...\model.gguf" --port 11434
```
```bash
# Linux/macOS — start the server:
bash scripts/run.sh --4b
```
Then point Open WebUI at it (Ollama mode, no changes needed):
**Settings → Connections → Ollama → `http://localhost:11434`**

## Available models

Tested with Qwen3/Qwen3.5/Qwen3.6 GGUFs from Hugging Face or Ollama blob cache:

| Model | Quant | Size | Notes |
|---|---|---|---|
| `ermiaazarkhalili/.../qwen3.5-4b-sft-claude-opus-reasoning-unsloth.q3_k_m.gguf` | Q3_K_M | 2.1 GB | **Start here** |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q3_K_XL.gguf` | Q3_K_XL | 4.7 GB | |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q6_K_XL.gguf` | Q6_K_XL | 8.2 GB | |
| `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q8_K_XL.gguf` | Q8_K_XL | 12 GB | |
| `unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf` | Q2_K_XL | 11 GB | |
| `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | Q2_K_XL | 11.5 GB | MoE 128 experts |

Any GGUF in a supported architecture will work — these are just validated.

## Benchmark

> These numbers use the synthetic self-test model (4 layers, 512 hidden, 8 experts).
> A real Qwen3-4B (36 layers, 2560 hidden) will show lower throughput on CPU.
> With CUDA enabled on a modern GPU, expect 3–8× improvement over CPU.

```bash
./bin/tb_infer --benchmark                              # synthetic
./bin/tb_infer --model path/to/model.gguf --benchmark   # real
```

## What changed in v0.6

- **CUDA dequant pipeline** — `tb_gguf_dequant.cu` rebuilt from scaffold to full engine:
  all 8 quant types (Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K / Q4_0 / Q8_0 / BF16 / F32)
  now GPU-accelerated.
- **4-stream async pipeline** — eliminates `cudaDeviceSynchronize()` per matvec call.
  Hidden-state vector `x` uploaded once per token (`tb_cuda_upload_x`); kernels fire on
  rotating streams; async D→H copies overlap with stream N+1 kernel execution.
- **Analog-driven block/grid dispatch** — oscillator phase drives CUDA block size:
  PLUCK → 256 threads/block, SUSTAIN/FINETUNE → 128, LOCK → 64.  S(U) resonance
  discriminant scales grid size ±25%.
- **Q3_K + Q8_K + BF16 + F32 CUDA kernels** added (v0.5 had Q4_K / Q6_K / Q4_0 / Q8_0 only).

## What changed in v0.5

- **Analog dispatch layer** (`src/tb_analog_dispatch.{h,c}`) — bridges the 8D Kuramoto
  oscillator into a CPU SIMD scheduler.  Oscillator phase (CV) selects kernel path:
  - PLUCK (CV > 0.50) → AVX-512 VNNI (highest throughput, scatter-friendly)
  - SUSTAIN/FINETUNE (CV 0.10–0.50) → AVX2 FMA (balanced)
  - LOCK (CV < 0.10) → scalar (phases settled, working set fits in L1)
- **`TB_InferCtx`** gains `TBOscSnapshot osc_snap` and `TBCpuCaps cpu_caps`.
  `tb_dispatch_detect_caps()` runs once at init; SIMD path selected per-token.
- `Makefile` / `build.sh` updated to compile `tb_analog_dispatch.c`.

## What changed in v0.4

- **KV cache persistent** — allocated once per session in `ctx->session_kvs[16]`,
  reused across decode calls.  Previously O(N²) in sequence length.
- **Scratch buffers** — `scratch_x/y/norm/logits` pre-allocated; eliminates ~4
  malloc/free per token.
- **HTTP body reader** — `Content-Length`-aware, reads up to 256 KB (was 8 KB).
- **Streaming** — `stream:true` delivers chunked Transfer-Encoding, one JSON line
  per token.
- **UAF fix** — `cc` fields copied before `free(cc)` in `tb_handle_conn`.
- **Unified /api/generate + /api/chat** — merged handler, de-duplicated logic.

## What changed in v0.3

- Qwen3 GGUF metadata keys: `embedding_length`, `block_count`, `head_count`/`head_count_kv`,
  `feed_forward_length`, `expert_count`/`expert_used_count`, `rms_norm_eps`,
  `moe_intermediate_size`, `max_position_embeddings`
- Qwen3 per-head QK-norm (`blk.L.attn_q_norm.weight`, `blk.L.attn_k_norm.weight`)
- All weight dequantisation through `tb_gguf_dequant_matvec()`
- MoE packed tensors (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`)
- RoPE base auto-read from GGUF (Qwen3 uses 1,000,000)
- Flash-attention style tiling; O(N+k log k) nucleus sampler

## Architecture

```
Layer 0: tb_phi_lattice.{h,c}    — 8D Kuramoto RK4 oscillator, Weyl lattice, PhiFold AEAD
Layer 1: tb_tensor.{h,c}         — Epoch-aware tensors, flash-tiled KV attention,
                                    O(N+k log k) nucleus sampler
Layer 2: tb_graph.{h,c}          — HDGL graph engine, ERL ledger, CognitionTree
         tb_hopfield_exp.{h,c}   — Hopfield associative memory
Layer 3: tb_orchestration.{h,c}  — Agent fabric, tool registry (19 built-ins),
                                    unfold engine, HTTP extension routes
Layer 4: tb_infer.c              — Transformer forward pass, HTTP server (Ollama-compat),
                                    persistent KV cache (TB_MAX_SESSIONS=16),
                                    chunked streaming, 256 KB request body
         tb_gguf.c               — GGUF mmap loader (all Qwen3 architectures)
         tb_gguf_dequant.cu      — CUDA dequant engine (4-stream async, all quant types)
         tb_tokenizer.c          — BPE tokenizer (from GGUF vocab)
Layer 5: tb_semantic_os.{h,c}    — Capability tokens, WuWei codec (5 strategies),
                                    SemanticContext, SemanticCompressor

src/:    tb_analog_dispatch.{h,c} — Oscillator-driven CPU SIMD path selector
                                    (AVX-512 / AVX2 / scalar)
         hdgl_router.{h,c}        — Spiral8 double-strand phi-tau expert routing
         hdgl_critic.{h,c}        — TD(0) critic, 5-feature → value network,
                                    online SGD; adjusts routing alpha per token
         zchg_{http,store,...}.c  — ZCHG transport / key-value store stack
         analog_engine.{h,c}      — Analog compute substrate (ll_analog)
```

## HDGL routing

When `--hdgl` is passed, MoE expert selection is modulated by the phi-lattice oscillator
(Kuramoto + Spiral8 + softmax top-k blend).  The critic TD module observes routing quality
and adjusts the alpha modulator per token.  `--no-hdgl` disables it for comparison.

## CUDA acceleration

Build with CUDA support:

```bat
:: Windows — CUDA linked automatically by Makefile.win if nvcc is on PATH:
nmake -f Makefile.win
```
```bash
# Linux/macOS:
bash scripts/build.sh --cuda       # requires nvcc + CUDA toolkit ≥ 11.4
```

At model load, quantised weight tensors are uploaded to VRAM once via
`tb_cuda_upload_tensor()`.  Per token, `tb_cuda_upload_x()` pushes the hidden-state
vector once; all matvec calls in that token step reuse the device copy.
Four async streams pipeline kernel dispatch and D→H copies.

Oscillator phase drives block size (PLUCK → 256, SUSTAIN/FINETUNE → 128, LOCK → 64).

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
