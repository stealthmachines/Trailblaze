# Per-Layer Batch Test Checklist
# Qwen3.5-9B / TRAILBLAZE v0.8

Each layer runs a **batch of B=8 hidden states** independently.
Inputs: 1 real BOS embedding + 7 seeded random unit vectors.
A layer PASSES only if every sub-check below passes.

---

## Global Pre-Flight (run once before any layer test)

| # | Check | How | Pass Condition |
|---|-------|-----|----------------|
| G1 | Model loaded | tb_gguf_load | arch=qwen35 layers=32 hidden=4096 |
| G2 | GPU upload | tb_gguf_cuda_upload_all | ≥400 tensors uploaded, 0 errors |
| G3 | Embedding row | dequant BOS row (tok 248045) | L2 ∈ [1, 500], no NaN |
| G4 | RMS norm sanity | norm([1,1,...,1]) | output ≈ [w[0], w[1], ...] within 1% |
| G5 | Q6_K spot check | dequant blk.0.attn_q.weight row 0 | L2 > 0.1, no zero-collapse |

---

## Per-Layer Checklist  (repeat for L = 0 … 31)

### Stage 1 — Weight Inventory
| # | Tensor | Expected shape | qtype | Pass Condition |
|---|--------|---------------|-------|----------------|
| 1a | blk.L.attn_norm.weight | [4096] | BF16 or F32 | exists, nelems=4096 |
| 1b | blk.L.attn_q.weight  (or attn_qkv) | [4096×4096] | Q6_K | exists, qtype=14 |
| 1c | blk.L.attn_k.weight | [2048×4096] | Q6_K | exists |
| 1d | blk.L.attn_v.weight | [2048×4096] | Q6_K | exists |
| 1e | blk.L.attn_output.weight | [4096×4096] | Q6_K | exists |
| 1f | blk.L.ffn_norm.weight | [4096] | BF16 or F32 | exists |
| 1g | blk.L.ffn_gate.weight | [14336×4096] | Q6_K | exists |
| 1h | blk.L.ffn_up.weight | [14336×4096] | Q6_K | exists |
| 1i | blk.L.ffn_down.weight | [4096×14336] | Q6_K | exists |
| 1j | blk.L.attn_q_norm.weight (global layers only: L%4==3) | [256] | F32 | exists on L=3,7,11,15,19,23,27,31 |

### Stage 2 — Attention Pre-Norm
| # | Check | Input | Pass Condition |
|---|-------|-------|----------------|
| 2a | RMS norm runs | batch x[0..7], blk.L.attn_norm | L2 ∈ [10, 500] per item |
| 2b | No NaN/Inf | all 8 outputs | bad_count = 0 |
| 2c | Diversity preserved | L2(norm_i - norm_j) for all pairs | min pairwise dist > 0.5 |

### Stage 3 — QKV Projection
| # | Check | Input | Pass Condition |
|---|-------|-------|----------------|
| 3a | Q projection | normed x → W_q matvec | L2 ∈ [1, 5000] |
| 3b | K projection | normed x → W_k matvec | L2 ∈ [1, 5000] |
| 3c | V projection | normed x → W_v matvec | L2 ∈ [1, 5000] |
| 3d | GPU/CPU agreement | run same row CPU vs GPU | max abs diff < 0.05 |
| 3e | No NaN/Inf | Q, K, V for all 8 | bad_count = 0 |

### Stage 4 — QK Norm (global layers only)
| # | Check | Pass Condition |
|---|-------|----------------|
| 4a | Per-head Q norm applied | L2 of each Q head ≈ sqrt(256) ± 20% |
| 4b | Per-head K norm applied | same |
| 4c | Local layers: QK norm tensors absent | check 1j; skip if missing |

### Stage 5 — RoPE
| # | Check | Pass Condition |
|---|-------|----------------|
| 5a | Q after RoPE ≠ Q before | L2 diff > 0.01 |
| 5b | K after RoPE ≠ K before | same |
| 5c | RoPE is position-dependent | rope(q, pos=0) ≠ rope(q, pos=4) |
| 5d | No NaN | bad_count = 0 |

### Stage 6 — Attention
| # | Check | Pass Condition |
|---|-------|----------------|
| 6a | Output L2 reasonable | ∈ [1, 50000] |
| 6b | No NaN/Inf | bad_count = 0 |
| 6c | Different Q/K/V → different attn_out | min pairwise dist > 0.1 |
| 6d | Single-token attn == Q (trivial case) | attn(Q, K=[Q], V=[Q], pos=0) ≈ Q |

### Stage 7 — Output Projection + First Residual
| # | Check | Pass Condition |
|---|-------|----------------|
| 7a | W_o matvec L2 | ∈ [0.01, 500000] |
| 7b | Residual x2 = x + attn_proj | x2 L2 ≈ x L2 ± 50x |
| 7c | No NaN | bad_count = 0 |

### Stage 8 — FFN Pre-Norm
| # | Check | Pass Condition |
|---|-------|----------------|
| 8a | RMS norm of x2 | L2 ∈ [10, 500] |
| 8b | Diversity | min pairwise dist > 0.5 |

### Stage 9 — FFN (SwiGLU)
| # | Check | Pass Condition |
|---|-------|----------------|
| 9a | gate matvec L2 | ∈ [0.1, 500000] |
| 9b | up matvec L2 | same |
| 9c | SwiGLU activations | no NaN, L2 > 0.01 |
| 9d | down matvec L2 | ∈ [0.01, 500000] |
| 9e | CPU/GPU agreement on gate row 0 | max abs diff < 0.05 |

### Stage 10 — Final Residual (Layer Output)
| # | Check | Pass Condition |
|---|-------|----------------|
| 10a | out = x2 + ffn_out | out L2 ∈ [0.1, 1000000] |
| 10b | No NaN/Inf | bad_count = 0 |
| 10c | Batch diversity | min pairwise L2 dist > 1.0 |
| 10d | Output ≠ input | L2(out - in) > 0.5 per item |

---

## Cross-Layer Checks (after all 32 layers pass individually)

| # | Check | Pass Condition |
|---|-------|----------------|
| C1 | Layer 0 output feeds Layer 1 cleanly | no NaN propagation |
| C2 | Hidden state norm stable across layers | L2 ∈ [50, 5000] at every layer |
| C3 | Full 32-layer forward on BOS token | logits over vocab, argmax ≠ 33604 ('lec') |
| C4 | "Paris is the capital of" → token in [France/français/Paris/...] | semantic sanity |

---

## Batching Protocol

```
Batch item | Input source              | Purpose
-----------|---------------------------|----------------------------------
  b=0      | BOS embedding (tok 248045)| Real model input, most diagnostic
  b=1      | Unit vector, seed 0x0001  | Random diversity
  b=2      | Unit vector, seed 0x0002  | Random diversity
  b=3      | Unit vector, seed 0x0003  | Random diversity
  b=4      | All-zeros                 | Linearity / zero-collapse check
  b=5      | All-ones (scaled)         | Uniform input
  b=6      | BOS + noise (5%)          | Robustness / sensitivity
  b=7      | -BOS embedding            | Sign invariance check
```

Each batch item is run through the layer **independently** (fresh KV cache per item).  
This means 8 forward passes per layer, 256 total across 32 layers.

---

## Implementation Plan

1. `build_layer_test.bat`  — compile `tb_layer_test.exe` linking against tb_infer_notest.obj
2. `tb_layer_test.c`       — harness that runs above checklist, prints PASS/FAIL table
3. Run: `tb_layer_test.exe model.gguf --layer 0 --verbose` to debug layer 0 first
4. Run: `tb_layer_test.exe model.gguf --all --stop-first` to find first failing layer

### Key diagnostic: Stage 3d (CPU/GPU agreement)
This is the most likely source of the current bug.
Run matvec for each weight in layer L on the same input via both paths,
compare elementwise. A mismatch > 0.05 identifies the broken quantization path.
```
