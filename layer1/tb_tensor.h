/*
 * tb_tensor.h — TRAILBLAZE Layer 1: Tensor Runtime (pure C)
 *
 * Epoch-aware tensors, branch-scoped KV cache, BLAS-style matmul,
 * flash-style attention, RMSNorm, Q8_0 quantisation, sampler.
 *
 * NEW from uploaded primitives:
 *   Hopfield update   — tb_hopfield_step()      (semantic convergence)
 *   Hopfield energy   — tb_hopfield_energy()     (stability scoring)
 *   Iterative relax   — tb_hopfield_relax()      (nonlinear recall)
 *   Hebbian train     — tb_hebbian_train()       (associative memory)
 *   Phase coherence   — tb_phase_coherence()     (confidence metric)
 *   Relaxation merge  — tb_relax_merge()         (branch reconciliation)
 *   Random projection — tb_random_projection()   (topology plugin)
 *
 * Build:
 *   gcc -O3 -march=native -std=c11 -DTB_L1_TEST \
 *       layer1/tb_tensor.c layer0/tb_phi_lattice.c -lm -o tb_l1_test
 */

#pragma once
#ifndef TB_TENSOR_H
#define TB_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include "../layer0/tb_phi_lattice.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * DType
 * ============================================================================ */

typedef enum {
    TB_DTYPE_F32 = 0,
    TB_DTYPE_F16 = 1,
    TB_DTYPE_Q8_0= 3,
    TB_DTYPE_I32 = 6,
    TB_DTYPE_I8  = 7,
} TB_DType;

/* ============================================================================
 * Tensor — epoch-aware, branch-scoped
 * ============================================================================ */

#define TB_TENSOR_MAX_DIMS 4
#define TB_Q8_BLOCK_SIZE   32

typedef struct {
    float    *data;          /* float32 data (or NULL for quantised) */
    int8_t   *qdata;         /* Q8_0 quantised data */
    float    *scales;        /* Q8_0 per-block scales */
    size_t    shape[TB_TENSOR_MAX_DIMS];
    int       ndim;
    TB_DType  dtype;
    int32_t   epoch;         /* lattice epoch at creation — checked before use */
    int       branch_id;
    char      name[64];
} TB_Tensor;

/* ============================================================================
 * KV Cache — branch-scoped, COW-forkable, epoch-invalidated
 * ============================================================================ */

typedef struct {
    float  **keys;       /* keys[layer] shape: (n_heads, seq_len, head_dim) */
    float  **vals;
    int      n_layers;
    int      n_heads;
    int      head_dim;
    int      max_seq;
    int      seq_len;
    int      branch_id;
    int32_t  epoch;
} TB_KVCache;

/* ============================================================================
 * Transformer config
 * ============================================================================ */

typedef struct {
    int    vocab_size;
    int    n_layers;
    int    n_heads;
    int    n_kv_heads;
    int    head_dim;
    int    hidden_dim;
    int    ffn_dim;
    float  norm_eps;
    int    use_silu;    /* 1=SwiGLU, 0=GELU */
} TB_TransformerConfig;

/* ============================================================================
 * Hopfield memory (from uploaded document primitives)
 * ============================================================================ */

typedef struct {
    float   *weights;    /* n×n symmetric weight matrix */
    int      n;          /* state dimension */
    int      n_patterns; /* stored patterns so far */
} TB_HopfieldMemory;

/* ============================================================================
 * API: Tensor lifecycle
 * ============================================================================ */

TB_Tensor* tb_tensor_alloc(const size_t *shape, int ndim,
                            TB_DType dtype, int32_t epoch, int branch_id);
void       tb_tensor_free(TB_Tensor *t);
int        tb_tensor_is_valid(const TB_Tensor *t, int32_t current_epoch);
TB_Tensor* tb_tensor_clone(const TB_Tensor *src);
/* Quantise F32 → Q8_0 */
TB_Tensor* tb_tensor_quantize_q8(const TB_Tensor *src, int32_t epoch);
/* Dequantise Q8_0 → F32 */
TB_Tensor* tb_tensor_dequantize_q8(const TB_Tensor *src);

/* ============================================================================
 * API: Core operators
 * ============================================================================ */

/* C = A @ B  (2D only for now) */
void tb_matmul(const float *A, const float *B, float *C,
               int M, int K, int N);

/* C = A @ Wᵀ + b  (linear layer; b may be NULL) */
void tb_linear(const float *x, const float *W, const float *b,
               float *out, int batch_seq, int in_dim, int out_dim);

/* RMSNorm: out = x/rms(x) * w */
void tb_rms_norm(const float *x, const float *w, float *out,
                 int n, float eps);

/* SiLU: out[i] = x[i] * σ(x[i]) */
void tb_silu(const float *x, float *out, int n);

/* GELU approximation */
void tb_gelu(const float *x, float *out, int n);

/* Softmax over last dim of shape (M, N) */
void tb_softmax(float *x, int M, int N);

/* ============================================================================
 * API: Scaled dot-product attention with KV cache
 * ============================================================================ */

/*
 * q: (n_heads, 1, head_dim)
 * k: (n_kv_heads, 1, head_dim)  — new token
 * v: (n_kv_heads, 1, head_dim)
 * out: (n_heads, head_dim)
 */
void tb_attention(const float *q, const float *k, const float *v,
                  TB_KVCache *cache, int layer_idx,
                  int n_heads, int n_kv_heads, int head_dim,
                  float *out);

/* ============================================================================
 * API: KV cache
 * ============================================================================ */

TB_KVCache* tb_kvcache_alloc(int n_layers, int n_heads, int head_dim,
                              int max_seq, int branch_id, int32_t epoch);
void        tb_kvcache_free(TB_KVCache *c);
TB_KVCache* tb_kvcache_fork(const TB_KVCache *src, int new_branch_id);
void        tb_kvcache_invalidate(TB_KVCache *c, int32_t new_epoch);
TB_KVCache* tb_kvcache_reconcile(const TB_KVCache *a, const TB_KVCache *b,
                                  int32_t new_epoch);

/* ============================================================================
 * API: Sampler
 * ============================================================================ */

int tb_sample_greedy(const float *logits, int vocab_size);
int tb_sample_top_p (const float *logits, int vocab_size,
                      float p, float temperature);
int tb_sample_top_k (const float *logits, int vocab_size,
                      int k, float temperature);

/* ============================================================================
 * API: Hopfield associative memory
 * (from uploaded document — KEEP primitives)
 * ============================================================================ */

/* Allocate Hopfield memory for n-dimensional state */
TB_HopfieldMemory* tb_hopfield_alloc(int n);
void               tb_hopfield_free(TB_HopfieldMemory *hm);

/* Store pattern via Hebbian outer product: W += x⊗xᵀ, diag=0 */
void tb_hebbian_train(TB_HopfieldMemory *hm, const float *pattern);

/* Single synchronous update step: state[i] = tanh(β·Σⱼ W[i,j]·state[j]) */
void tb_hopfield_step(float *state, const float *weights, int n, float beta);

/* Energy: E = −½ xᵀWx */
float tb_hopfield_energy(const float *state, const float *weights, int n);

/* Iterative relaxation until convergence (|ΔE| < epsilon or max_iters) */
void tb_hopfield_relax(float *state, const float *weights, int n,
                       float beta, int max_iters, float epsilon);

/* Phase coherence metric: R = |mean(e^{iθ})| ∈ [0,1]
 * R=1 → perfect lock, R=0 → maximally scattered
 * USE: confidence heuristic for scheduler / merge decisions */
float tb_phase_coherence(const float *phase, int n);

/* Weighted interpolation merge: dst = α·a + (1−α)·b */
void tb_relax_merge(float *dst, const float *a, const float *b,
                    int n, float alpha);

/* Iterative merge via Hopfield steps (branch reconciliation) */
void tb_iterative_merge(float *state, const float *weights, int n, int steps);

/* Random projection: output = proj @ input
 * proj: (out_dim × in_dim) — caller provides random Gaussian matrix */
void tb_random_projection(const float *input, float *output,
                           const float *proj, int in_dim, int out_dim);

#ifdef __cplusplus
}
#endif

#endif /* TB_TENSOR_H */
