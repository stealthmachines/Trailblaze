/*
 * tb_gguf.h — TRAILBLAZE Complete GGUF Loader
 * Q4_0, Q4_K, Q8_0, BF16, F16, F32 + vocab extraction
 */
#pragma once
#ifndef TB_GGUF_H
#define TB_GGUF_H

#include <stdint.h>
#include <stddef.h>
#ifndef TB_TOKENIZER_H
#include "tb_tokenizer.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Tensor info (from GGUF tensor table) ─────────────────────────────────── */
typedef struct {
    char     name[128];
    int      n_dims;
    int64_t  shape[4];
    int      qtype;      /* ggml_type: 0=F32,1=F16,2=Q4_0,8=Q8_0,12=Q4_K,14=Q6_K,30=BF16 */
    uint64_t offset;     /* byte offset in mmapped weights (adjusted for data section) */
    void    *d_data;     /* device (VRAM) pointer — set by tb_cuda_upload_tensor(),
                          * NULL when TB_CUDA not enabled or upload failed.           */
    size_t   d_bytes;    /* byte count of d_data allocation                           */
} TB_GGUFTensorInfo;

/* ── Expert blob layout (from extract_weights.py flat format) ─────────────── */
typedef struct {
    size_t gate_w_off, gate_w_sz;
    size_t gate_s_off, gate_s_sz;
    size_t gate_b_off, gate_b_sz;
    size_t up_w_off,   up_w_sz;
    size_t up_s_off,   up_s_sz;
    size_t up_b_off,   up_b_sz;
    size_t down_w_off, down_w_sz;
    size_t down_s_off, down_s_sz;
    size_t down_b_off, down_b_sz;
    size_t expert_total;
    int    hidden_dim, ffn_dim, group_size;
} TB_ExpertBlobLayout;

/* ── Loaded model ─────────────────────────────────────────────────────────── */
typedef struct {
    /* Architecture */
    char  arch[64];
    int   vocab_size, n_layers, n_heads, n_kv_heads, head_dim;
    int   hidden_dim, ffn_dim, moe_intermediate_size;
    int   n_experts, n_experts_per_tok, group_size;
    float rope_base, norm_eps;
    int   max_seq_len, full_attn_interval;
    int   bos_token_id, eos_token_id;

    /* Tensors */
    TB_GGUFTensorInfo *tensors;
    int                n_tensors;

    /* Fibonacci hash index — O(1) name → tensor lookup (built at load time).
     * Replaces the O(N) linear scan in tb_gguf_find_tensor.
     * Capacity is next power-of-2 above 2×n_tensors (load factor ≤ 50%).
     * Hash: FNV-1a × φ-multiplier, open-addressing with linear probe. */
    TB_GGUFTensorInfo **tensor_index;   /* pointer table, heap allocated */
    uint32_t            tensor_index_cap; /* capacity (power of 2)       */

    /* Tokenizer (extracted from GGUF KV) */
    TB_Tokenizer *tokenizer;  /* NULL if vocab not in GGUF */

    /* mmap */
    void  *weights_data;
    size_t file_size;
    int    weights_fd;
} TB_GGUFLoaded;

/* ── API ──────────────────────────────────────────────────────────────────── */

/* Load model: parses header, extracts vocab, mmaps weights */
TB_GGUFLoaded* tb_gguf_load(const char *path);
void           tb_gguf_free(TB_GGUFLoaded *g);

/* Tensor lookup and data access */
const TB_GGUFTensorInfo* tb_gguf_find_tensor(const TB_GGUFLoaded *g, const char *name);
const void*              tb_gguf_tensor_data(const TB_GGUFLoaded *g,
                                              const TB_GGUFTensorInfo *t);
int64_t                  tb_gguf_tensor_nelems(const TB_GGUFTensorInfo *t);

/* Dequantise one row (n_weights elements) → float32 */
void tb_gguf_dequant_row(const void *data, int qtype, int n_weights, float *out);

/* Dequantise-matvec: out(M) = W(M×K) @ x(K)  — all qtypes.
 * When TB_CUDA is defined and W has a valid d_data device pointer,
 * dispatches to the GPU kernel; otherwise runs the scalar CPU path. */
void tb_gguf_dequant_matvec(const void *W, int qtype, int M, int K,
                              const float *x, float *out);

/* Convenience wrapper that accepts a tensor pointer and dispatches to GPU
 * automatically when the tensor's d_data is populated.
 * When the analog dispatch context is set (tb_dispatch_context_set), uses
 * the Kuramoto oscillator state to select the optimal CPU kernel. */
void tb_gguf_tensor_matvec(const TB_GGUFLoaded *g, const TB_GGUFTensorInfo *t,
                             int M, int K, const float *x, float *out);

/* Set global analog dispatch context — called once from tb_infer_create(). */
#include "../src/tb_analog_dispatch.h"
void tb_dispatch_context_set(const TBOscSnapshot *snap, const TBCpuCaps *caps);

#ifdef TB_CUDA
/* Upload all tensors whose qtype is GPU-accelerated to device VRAM.
 * Call once after tb_gguf_load().  Prints progress to stderr.
 * Returns number of tensors successfully uploaded. */
int tb_gguf_cuda_upload_all(TB_GGUFLoaded *g);
#endif

/* Expert blob layout from model dimensions (compatible with extract_weights.py output) */
int tb_gguf_expert_layout(const TB_GGUFLoaded *g, TB_ExpertBlobLayout *layout);

#ifdef __cplusplus
}
#endif
#endif /* TB_GGUF_H */
