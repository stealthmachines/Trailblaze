/*
 * tb_infer.h — TRAILBLAZE Inference Runtime
 * Ollama/CUDA replacement: GGUF → HDGL routing → HTTP serving.
 */
#pragma once
#ifndef TB_INFER_H
#define TB_INFER_H

#include <stdint.h>
#include <stddef.h>
#include "../src/tb_analog_dispatch.h"
#include <string.h>   /* memcpy */
#ifdef _WIN32
#  include "../src/tb_win32.h"
#else
#  include <pthread.h>
#endif
#include "../layer0/tb_phi_lattice.h"
#include "../layer1/tb_tensor.h"
#include "../layer2/tb_graph.h"
#include "tb_tokenizer.h"
#include "tb_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TB_GGUFModel is an alias for TB_GGUFLoaded (the complete loaded model) */
typedef TB_GGUFLoaded TB_GGUFModel;

/* ── Expert selection (HDGL-routed MoE) ─────────────────────────────────── */
typedef struct {
    int   expert_indices[8];
    float expert_weights[8];
    int   k;
    int   hdgl_expert;
    float hdgl_alpha;
    int   semantic_octave;
} TB_ExpertSelection;

/* ── Session KV cache ────────────────────────────────────────────────────── */
#define TB_MAX_SESSIONS 16

typedef struct {
    TB_KVCache *kv;
    void       *store;           /* zchg_store_t* */
    char        session_key[64];
    int32_t     epoch;
    int         branch_id;
    int         session_id;
    int         active;
    /* Qwen3.5 Gated DeltaNet recurrent state:
     *   ssm_h[layer][head][d_k][d_v] — row-major, float32
     *   shape: [n_layers × n_heads × head_d_k × head_d_v]
     * Only allocated when the model has SSM layers (full_attn_interval > 0). */
    float      *ssm_h;           /* DeltaNet S matrices */
    int         ssm_n_layers;    /* = m->n_layers */
    int         ssm_n_heads;     /* n_ssm_heads (= ssm_a.shape[0]) */
    int         ssm_head_d_k;    /* key-space dim per head */
    int         ssm_head_d_v;    /* value-space dim per head (= ssm_state_size) */
    /* Conv1d sliding-window state:
     *   conv_h[layer][(kernel-1)][channel]
     *   shape: [n_layers × (conv_kernel-1) × ssm_inner_size] */
    float      *conv_h;          /* causal conv history buffer */
    int         conv_hist;       /* = conv_kernel - 1 (= 3) */
    int         conv_inner_size; /* = ssm_inner_size (= 8192) */
} TB_SessionKV;

/* ── Inference context ───────────────────────────────────────────────────── */
typedef struct {
    TB_GGUFModel     *model;
    TB_PhiLattice    *lattice;
    TB_BackendRegistry registry;
    /* Persistent per-session KV caches — allocated once, reused across tokens */
    TB_SessionKV      session_kvs[TB_MAX_SESSIONS];
    int               n_sessions;
    int               use_hdgl;
    int               use_hdgl_semantic;
    float             hdgl_alpha;
    int               max_new_tokens;
    float             temperature;
    float             top_p;
    int               top_k;
    TB_HopfieldMemory *semantic_mem;
    int                sem_dim;
    double  prefill_ms, decode_ms;
    int     tokens_generated, prompt_tokens;
    /* HDGL_History (28 bytes) — per-context Spiral8 phase state  */
    uint8_t hdgl_routing_state[32];
    /* Layer 3: Cognition tree (wired in tb_infer_create/free) */
    TB_CognitionTree *tree;
    /* Serialise generate calls: prevents data races on lattice/ctx state */
    pthread_mutex_t  generate_lock;
    /* Current-session DeltaNet state pointers — set by tb_infer_decode before layer loop */
    float  *cur_ssm_h;        /* points into session_kvs[slot].ssm_h (NULL if no SSM) */
    float  *cur_conv_h;       /* points into session_kvs[slot].conv_h (NULL if no SSM) */
    int     cur_ssm_n_heads;  /* n_ssm_heads */
    int     cur_ssm_head_d_k; /* head_d_k */
    int     cur_ssm_head_d_v; /* head_d_v */
    int     cur_conv_hist;    /* conv_kernel - 1 */
    int     cur_conv_inner;   /* ssm_inner_size */
    /* Pre-allocated decode scratch — avoids per-token malloc */
    float *scratch_x;      /* [hidden_dim] hidden state */
    float *scratch_y;      /* [hidden_dim] layer output swap buffer */
    float *scratch_norm;   /* [hidden_dim] post-norm output */
    float *scratch_ones;   /* [hidden_dim] unit weight fallback */
    float *scratch_logits; /* [vocab_size] lm-head output */
    /* Analog dispatch: oscillator snapshot + CPU capability cache */
    TBOscSnapshot    osc_snap;   /* updated every harmonic sync */
    TBCpuCaps        cpu_caps;   /* set once at tb_infer_create() */
} TB_InferCtx;

/* ── HTTP server config ──────────────────────────────────────────────────── */
typedef struct {
    TB_InferCtx *ctx;
    char         model_name[128];
    int          port;
    char         host[64];
    int          n_parallel;
    int          verbose;
    /* Layer 3 extension handler: handles /unfold /tools /state /ledger /sse */
    int  (*ext_handler)(int fd, const char *path, const char *method,
                        const char *body, void *ext_ctx);
    void  *ext_ctx;
} TB_ServeConfig;

/* ── API ─────────────────────────────────────────────────────────────────── */
TB_GGUFModel* tb_model_load  (const char *model_path);
void          tb_model_free  (TB_GGUFModel *model);

TB_InferCtx*  tb_infer_create(TB_GGUFModel *model,
                               int use_hdgl, float hdgl_alpha,
                               uint32_t lattice_slots, uint64_t seed);
void          tb_infer_free  (TB_InferCtx *ctx);

int  tb_infer_decode  (TB_InferCtx *ctx, int last_token, int pos, int session_id);
int  tb_infer_generate(TB_InferCtx *ctx, const int *prompt_ids, int n_prompt,
                        int *out_ids, int max_out, int session_id,
                        void (*token_cb)(int,void*), void *cb_ud);
int  tb_serve         (TB_ServeConfig *cfg);

/* Expert routing (HDGL phi-lattice blend) */
TB_ExpertSelection tb_route_experts(TB_InferCtx *ctx, int token_id,
                                     int layer_idx, const float *gate_logits,
                                     int n_experts);

/* RoPE */
void tb_rope_apply(float *q, float *k, int head_dim, int pos, float rope_base, int rotary_dim);

/* BF16 helper (used in tb_infer.c and tb_gguf.c) */
static inline float tb_bf16_to_f32_infer(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float f; memcpy(&f,&b,4); return f;
}

/* Diff utility */
void tb_diff_sources(const char *dir_a, const char *dir_b, const char *out_patch);

/* Single-layer forward pass — exposed for per-layer batch testing.
 * x/out: (hidden_dim,)  kv may be NULL (skips KV write, uses copy-attn fallback) */
int tb_layer_forward(TB_InferCtx *ctx, int layer_idx,
                     const float *x, float *out,
                     TB_KVCache *kv, int pos, int token_id);

#ifdef __cplusplus
}
#endif
#endif /* TB_INFER_H */
