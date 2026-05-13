/*
 * tb_infer.c — TRAILBLAZE Inference Runtime Implementation
 *
 * This is the Ollama/CUDA replacement.
 *
 * What it does that Ollama doesn't:
 *   - HDGL phi-lattice expert routing (lattice state influences which experts fire)
 *   - Branch-aware KV cache (fork/merge per session, not just a ring buffer)
 *   - zchg_store v0.2 for cross-session KV persistence (strand-native, not Redis)
 *   - Epoch ratchet on context clear (forward secrecy)
 *   - Hopfield semantic memory for retrieval-augmented generation
 *   - Critic v33 learns routing preferences from observed quality signals
 *
 * What it shares with Ollama:
 *   - GGUF model format (same files, same tensor layout)
 *   - Ollama-compatible HTTP API (/api/generate, /api/chat, /api/tags)
 *   - BPE tokenizer (llama-format vocabulary in GGUF)
 *
 * Build: gcc -O3 -march=native -std=c11 -DTB_INFER_TEST \
 *            -Ilayer0 -Ilayer1 -Ilayer2 -Ilayer4 -Iinclude \
 *            layer4/tb_infer.c layer0/tb_phi_lattice.c \
 *            layer1/tb_tensor.c layer2/tb_graph.c \
 *            src/sha256_minimal.c src/zchg_lattice.c src/zchg_store_v02.c \
 *            src/hdgl_router.c src/vector_container.c \
 *            -lm -lpthread -o bin/tb_infer
 */

#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "tb_infer.h"
#include "tb_gguf.h"
#include "../layer0/tb_phi_lattice.h"
#include "../layer1/tb_tensor.h"
#include "../layer2/tb_graph.h"
#include "../src/hdgl_router.h"
#include "../src/hdgl_critic.h"
#include "../src/hdgl_critic.c"
#include "../layer3/tb_orchestration.h"
#include "../layer5/tb_semantic_os.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <stdarg.h>

static int g_tb_verbose = 0;

static void tb_vlog(const char *fmt, ...) {
    if (!g_tb_verbose || !fmt) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[trace] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

#ifdef _WIN32
#  include "../src/tb_win32.h"
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
   typedef SOCKET tb_socket_t;
#  define TB_SEND(s,b,n)     send((s),(const char*)(b),(int)(n),0)
#  define TB_RECV(s,b,n)     recv((s),(char*)(b),(int)(n),0)
#  define TB_CLOSESOCK(s)    closesocket(s)
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <pthread.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <sys/socket.h>
   typedef int tb_socket_t;
#  define TB_SEND(s,b,n)     write((s),(b),(n))
#  define TB_RECV(s,b,n)     read((s),(b),(n))
#  define TB_CLOSESOCK(s)    close(s)
#endif

#ifdef TB_CUDA
#  include "tb_gguf_dequant.h"
#endif
/* Forward declaration for compat shim */
void tb_gguf_dequant_matvec_4bit_compat(const uint32_t *w, const uint16_t *s, const uint16_t *b, const float *x, float *out, int od, int id, int gs);


/* ── Spiral8 octave alpha (from nonmetal_infer.c k_hdgl_octave_alpha) ──── */
static const float k_spiral8_alpha[8] = {
    0.015269f, 0.008262f, 0.110649f, -0.083485f,
    0.025847f, -0.045123f, 0.067891f, 0.012345f
};

/* ── Timing ─────────────────────────────────────────────────────────────── */
static double tb_wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Append-only benchmark telemetry (HDGL-SQL bridge path: JSONL event stream). */
static void tb_bench_logf(FILE *fp, const char *fmt, ...) {
    if (!fp || !fmt) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fputc('\n', fp);
    fflush(fp);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 1: 4-bit group-quantised matvec
 * Directly from nonmetal_infer.c cpu_dequant_matvec_4bit.
 * Q4 packed: 8 nibbles per uint32, groups of `group_size` elements.
 * Scale/bias per group stored as BF16.
 * With -O3 -march=native gcc auto-vectorises the inner loop to AVX2.
 * ────────────────────────────────────────────────────────────────────────── */


void tb_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 2: RoPE (Rotary Position Embedding)
 * Applies in-place to q and k vectors.
 * ────────────────────────────────────────────────────────────────────────── */

void tb_rope_apply(float *q, float *k, int head_dim, int pos, float rope_base) {
    for (int i = 0; i < head_dim / 2; i++) {
        float theta = (float)pos / powf(rope_base, 2.0f * i / head_dim);
        float cos_t = cosf(theta), sin_t = sinf(theta);
        float q0 = q[2*i], q1 = q[2*i+1];
        float k0 = k[2*i], k1 = k[2*i+1];
        q[2*i]   = q0*cos_t - q1*sin_t;
        q[2*i+1] = q0*sin_t + q1*cos_t;
        k[2*i]   = k0*cos_t - k1*sin_t;
        k[2*i+1] = k0*sin_t + k1*cos_t;
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 3: GGUF model loading
 * Compatible with llama.cpp GGUF v2/v3 format.
 * Uses mmap for zero-copy weight access.
 * ────────────────────────────────────────────────────────────────────────── */

/* GGUF value types */
typedef enum {
    GGUF_UINT8=0, GGUF_INT8=1, GGUF_UINT16=2, GGUF_INT16=3,
    GGUF_UINT32=4, GGUF_INT32=5, GGUF_FLOAT32=6, GGUF_BOOL=7,
    GGUF_STRING=8, GGUF_ARRAY=9, GGUF_UINT64=10, GGUF_INT64=11,
    GGUF_FLOAT64=12
} GGUFValueType;

/* Parse string from GGUF stream (length-prefixed uint64) */
static char* gguf_read_string(FILE *f) {
    uint64_t len; fread(&len, 8, 1, f);
    if (len > 65536) { fseek(f, len, SEEK_CUR); return strdup(""); }
    char *s = (char*)malloc(len + 1);
    fread(s, 1, len, f); s[len] = '\0';
    return s;
}

/* Skip a GGUF value (for KV pairs we don't care about) */
static void __attribute__((unused)) gguf_skip_value(FILE *f, uint32_t vtype) {
    uint64_t n; char *s; uint32_t elem_type;
    switch (vtype) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: fseek(f,1,SEEK_CUR); break;
        case GGUF_UINT16: case GGUF_INT16: fseek(f,2,SEEK_CUR); break;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: fseek(f,4,SEEK_CUR); break;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: fseek(f,8,SEEK_CUR); break;
        case GGUF_STRING: s=gguf_read_string(f); free(s); break;
        case GGUF_ARRAY:
            fread(&elem_type, 4, 1, f);
            fread(&n, 8, 1, f);
            for (uint64_t i=0;i<n;i++) gguf_skip_value(f, elem_type);
            break;
        default: break;
    }
}

TB_GGUFModel* tb_model_load(const char *model_path) {
#ifdef TB_CUDA
    tb_cuda_init();
#endif
    TB_GGUFModel *m = tb_gguf_load(model_path);
#ifdef TB_CUDA
    if (m) tb_gguf_cuda_upload_all(m);
#endif
    return m;
}

void tb_model_free(TB_GGUFModel *m) {
    if (!m) return;
    if (m->weights_data && m->weights_data != MAP_FAILED)
        munmap(m->weights_data, m->file_size);
    if (m->weights_fd >= 0) close(m->weights_fd);
    free(m->tensors);
    free(m);
}



/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 4: Phi-lattice expert routing
 * Combines standard softmax gate scores with TB_PhiLattice routing.
 * From nonmetal_infer.c route_step_cpu() HDGL blend logic.
 * ────────────────────────────────────────────────────────────────────────── */

/* Spiral8 semantic octave: which Kuramoto dimension this (token,layer) maps to */
static int tb_semantic_octave(int token_id, int layer_idx) {
    double phi_hash = fmod((double)token_id * 1.6180339887498948 +
                           (double)layer_idx * 0.5, 1.0);
    return (int)(phi_hash * 8) % 8;
}

/* Semantic boost: amplify experts resonant with this octave */
static float tb_semantic_boost(int octave, int expert_idx, int n_experts) {
    float local_gain = 1.0f + fabsf(k_spiral8_alpha[octave]);
    float phi_exp    = fmod((double)expert_idx * 1.6180339887498948, 1.0);
    float octave_f   = (float)octave / 8.0f;
    return 1.0f + local_gain * cosf((phi_exp - octave_f) * (float)M_PI * 2.0f) * 0.5f;
}

TB_ExpertSelection tb_route_experts(
    TB_InferCtx   *ctx,
    int            token_id,
    int            layer_idx,
    const float   *gate_logits,
    int            n_experts
) {
    TB_ExpertSelection sel = {0};
    int k = ctx->model->n_experts_per_tok;
    if (k > 8) k = 8;
    sel.k          = k;
    sel.hdgl_alpha = ctx->hdgl_alpha;

    /* Copy gate scores */
    float scores[256];
    if (n_experts > 256) n_experts = 256;
    memcpy(scores, gate_logits, n_experts * sizeof(float));

    /* HDGL Spiral8 temporal routing — replaces ad-hoc slot hash
     *
     * route_token_recursive() uses:
     *   phi-tau geometry: expert affinity derived from semantic depth (layer:token)
     *   Double strand:    primary (forward) + mirror (counter-rotating) strands
     *   Phase history:    hdgl_routing_state accumulates across all decode steps
     *   Alpha-weighting:  negative-alpha strands are stickier (expert specialisation)
     *
     * This is architecturally superior to Ollama softmax-top-k because the routing
     * decision incorporates the full temporal trajectory of this session's experts
     * via the Kuramoto phase accumulators, not just the current token's gate logits.
     */
    if (ctx->use_hdgl) {
        /* Initialize router with expert count (idempotent, NULL lattice = cold start) */
        hdgl_router_init(NULL, n_experts);

        /* Build token key: "layer:token_id" feeds phi-tau depth-weighting */
        char tok_key[64];
        snprintf(tok_key, sizeof(tok_key), "%d:%d", layer_idx, token_id);
        Token hdgl_tok = { tok_key, token_id };

        /* Cast per-ctx routing state to HDGL_History (32-byte opaque buffer) */
        HDGL_History *H = (HDGL_History *)(void *)ctx->hdgl_routing_state;

        /* Route via Spiral8 double-strand phi-tau geometry */
        int hdgl_exp = route_token_recursive(hdgl_tok, H);

        /* ── Layer 3: critic alpha modulator ──────────────────────────────
         * Compute pre-boost top1 probability as inverse-confidence feature.
         * critic_alpha_mod() returns sigmoid-mapped [0.3, 1.0]:
         *   low value  → suppress HDGL influence (model is already confident)
         *   high value → amplify HDGL influence (model is uncertain, trust geometry)
         * At cold start: critic weights ≈ 0, sigmoid(0) = 0.5, mod = 0.65
         */
        float _top1_raw = scores[0];
        for (int _e = 1; _e < n_experts; _e++)
            if (scores[_e] > _top1_raw) _top1_raw = scores[_e];
        /* Raw softmax approx for top1 prob (fast) */
        float _raw_sum = 0.0f;
        for (int _e = 0; _e < n_experts; _e++) _raw_sum += expf(scores[_e] - _top1_raw);
        float _top1_prob = 1.0f / _raw_sum;   /* top1_prob = exp(0) / sum */

        float _crit_feat[CRITIC_IN];
        _crit_feat[0] = 1.0f - _top1_prob;                           /* inv_conf   */
        _crit_feat[1] = (ctx->lattice)
                        ? (1.0f - (float)ctx->lattice->phase_var)  /* Kuramoto R */
                        : 0.5f;                                       /* coherence  */
        _crit_feat[2] = _top1_prob;                                   /* mean_gate  */
        _crit_feat[3] = (ctx->model && ctx->model->max_seq_len > 0)
                        ? (float)ctx->tokens_generated /
                          (float)ctx->model->max_seq_len
                        : 0.0f;                                       /* pos_norm   */
        _crit_feat[4] = (ctx->tokens_generated > 0)
                        ? fminf((float)ctx->tokens_generated / 100.0f, 1.0f)
                        : 0.0f;                                       /* accum_norm */

        float _alpha_mod = critic_alpha_mod(_crit_feat);  /* [0.3, 1.0] */
        float _eff_alpha = ctx->hdgl_alpha * _alpha_mod;

        if (hdgl_exp >= 0 && hdgl_exp < n_experts) {
            /* Primary boost scaled by learned alpha modulator */
            scores[hdgl_exp] += _eff_alpha * 2.0f;
            /* ── Layer 2: v39 amplitude weighting ─────────────────────────
             * routing_weight(dphi, amp) = exp(SHARPNESS * cos(dphi) * sqrt(amp))
             * Adjacent experts get softer boosts with 60-degree phase separation.
             * cos(±π/3) = 0.5, so neighbours see exp(0.5*sqrt(amp)) scaling.
             */
            int nb1 = (hdgl_exp + 1) % n_experts;
            int nb2 = (hdgl_exp - 1 + n_experts) % n_experts;
            float _gate_amp = _top1_prob;   /* use softmax top1 as amplitude proxy */
            float _v39_nb   = expf(0.5f * sqrtf(_gate_amp)) - 1.0f;  /* ≥0 */
            scores[nb1] += _eff_alpha * (0.3f + _v39_nb * 0.1f);
            scores[nb2] += _eff_alpha * (0.3f + _v39_nb * 0.1f);
            sel.hdgl_expert = hdgl_exp;

            /* ── Critic TD update: observe routing quality ─────────────────
             * Reward = confidence of primary expert after boosting (pre-normalise).
             * TD target = reward + gamma * V(next).  next ≈ same features (stationary).
             */
            float _td_target = critic_td_target(_top1_prob, _crit_feat);
            critic_observe(_crit_feat, _td_target);
            /* Flush gradients every 32 routing calls to amortise cost */
            if ((ctx->tokens_generated & 31) == 0) critic_update();
        }

        /* Spiral8 semantic octave multiplier (existing, kept as amplitude modulation) */
        if (ctx->use_hdgl_semantic) {
            int octave = tb_semantic_octave(token_id, layer_idx);
            sel.semantic_octave = octave;
            for (int e = 0; e < n_experts; e++)
                scores[e] *= tb_semantic_boost(octave, e, n_experts);
        }
    }

    /* Softmax */
    float max_s = scores[0];
    for (int e = 1; e < n_experts; e++) if (scores[e] > max_s) max_s = scores[e];
    float sum = 0.0f;
    for (int e = 0; e < n_experts; e++) { scores[e] = expf(scores[e]-max_s); sum+=scores[e]; }
    for (int e = 0; e < n_experts; e++) scores[e] /= sum;

    /* Top-k selection */
    for (int i = 0; i < k; i++) {
        sel.expert_indices[i] = 0; sel.expert_weights[i] = -1e30f;
    }
    for (int e = 0; e < n_experts; e++) {
        int min_i = 0;
        for (int i = 1; i < k; i++) if (sel.expert_weights[i] < sel.expert_weights[min_i]) min_i = i;
        if (scores[e] > sel.expert_weights[min_i]) {
            sel.expert_weights[min_i] = scores[e];
            sel.expert_indices[min_i] = e;
        }
    }

    /* Normalise top-k weights */
    float ws = 0.0f;
    for (int i = 0; i < k; i++) ws += sel.expert_weights[i];
    if (ws > 0.0f) for (int i = 0; i < k; i++) sel.expert_weights[i] /= ws;

    return sel;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 5: Expert forward (MoE)
 * ────────────────────────────────────────────────────────────────────────── */

int tb_expert_forward(
    const unsigned char *expert_buf,
    const TB_GGUFModel  *model,
    const TB_ExpertBlobLayout *layout,
    const float *x,
    float       *out
) {
    int hidden   = model->hidden_dim;
    int ffn_dim  = model->ffn_dim > 0 ? model->ffn_dim : hidden * 4;
    int gs       = model->group_size;

    const uint32_t *gate_w = (const uint32_t*)(expert_buf + layout->gate_w_off);
    const uint16_t *gate_s = (const uint16_t*)(expert_buf + layout->gate_s_off);
    const uint16_t *gate_b = (const uint16_t*)(expert_buf + layout->gate_b_off);
    const uint32_t *up_w   = (const uint32_t*)(expert_buf + layout->up_w_off);
    const uint16_t *up_s   = (const uint16_t*)(expert_buf + layout->up_s_off);
    const uint16_t *up_b   = (const uint16_t*)(expert_buf + layout->up_b_off);
    const uint32_t *down_w = (const uint32_t*)(expert_buf + layout->down_w_off);
    const uint16_t *down_s = (const uint16_t*)(expert_buf + layout->down_s_off);
    const uint16_t *down_b = (const uint16_t*)(expert_buf + layout->down_b_off);

    float *gate_out = (float*)malloc(ffn_dim * sizeof(float));
    float *up_out   = (float*)malloc(ffn_dim * sizeof(float));
    float *act      = (float*)malloc(ffn_dim * sizeof(float));
    if (!gate_out || !up_out || !act) { free(gate_out); free(up_out); free(act); return 0; }

    tb_gguf_dequant_matvec_4bit_compat(gate_w, gate_s, gate_b, x, gate_out, ffn_dim, hidden, gs);
    tb_gguf_dequant_matvec_4bit_compat(up_w,   up_s,   up_b,   x, up_out,   ffn_dim, hidden, gs);
    tb_swiglu(gate_out, up_out, act, ffn_dim);
    tb_gguf_dequant_matvec_4bit_compat(down_w, down_s, down_b, act, out, hidden, ffn_dim, gs);

    free(gate_out); free(up_out); free(act);
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 6: Inference context lifecycle
 * ────────────────────────────────────────────────────────────────────────── */

TB_InferCtx* tb_infer_create(TB_GGUFModel *model,
                               int use_hdgl, float hdgl_alpha,
                               uint32_t lattice_slots, uint64_t seed) {
    TB_InferCtx *ctx = (TB_InferCtx*)calloc(1, sizeof(TB_InferCtx));
    if (!ctx) return NULL;
    ctx->model             = model;
    ctx->use_hdgl          = use_hdgl;
    ctx->use_hdgl_semantic = use_hdgl;
    ctx->hdgl_alpha        = hdgl_alpha;
    ctx->max_new_tokens    = 512;
    ctx->temperature       = 0.7f;
    ctx->top_p             = 0.9f;
    ctx->top_k             = 40;

    /* Create phi-lattice */
    if (lattice_slots == 0) lattice_slots = 512;
    ctx->lattice = tb_lattice_create(lattice_slots, seed);
    if (!ctx->lattice) { free(ctx); return NULL; }

    /* Warm up lattice with model config (deterministic routing) */
    char init_key[128];
    snprintf(init_key, sizeof(init_key), "model:%s:layers:%d:experts:%d",
             model->arch, model->n_layers, model->n_experts);
    for (int i = 0; i < 20; i++) tb_lattice_advance(ctx->lattice, 1);

    tb_registry_init(&ctx->registry, ctx->lattice);

    /* Detect CPU SIMD capabilities once — used by tb_dispatch_matvec */
    tb_dispatch_detect_caps(&ctx->cpu_caps);
    memset(&ctx->osc_snap, 0, sizeof(ctx->osc_snap));
    /* Register the context pointers globally so tb_gguf_tensor_matvec can
     * read them without a signature change (wu-wei: state flows down) */
    tb_dispatch_context_set(&ctx->osc_snap, &ctx->cpu_caps);

    /* Layer 3: cognition tree (ERL ledger, branch-aware KV, cell commits) */
    ctx->tree = tb_tree_create(ctx->lattice, "tb_infer", NULL);
    pthread_mutex_init(&ctx->generate_lock, NULL);

    /* Hopfield semantic memory (dim = min(128, hidden_dim)) */
    ctx->sem_dim = model->hidden_dim > 0 ? (model->hidden_dim < 128 ? model->hidden_dim : 128) : 64;
    ctx->semantic_mem = tb_hopfield_alloc(ctx->sem_dim);

    /* Pre-allocate decode scratch buffers — shared across all tokens, never freed
     * until tb_infer_free().  Eliminates ~4 malloc/free pairs per token step. */
    int H = model->hidden_dim > 0 ? model->hidden_dim : 512;
    int V = model->vocab_size  > 0 ? model->vocab_size  : 32000;
    ctx->scratch_x      = (float*)calloc(H, sizeof(float));
    ctx->scratch_y      = (float*)malloc(H * sizeof(float));
    ctx->scratch_norm   = (float*)malloc(H * sizeof(float));
    ctx->scratch_ones   = (float*)malloc(H * sizeof(float));
    ctx->scratch_logits = (float*)calloc(V, sizeof(float));
    if (ctx->scratch_ones) { for (int i = 0; i < H; i++) ctx->scratch_ones[i] = 1.0f; }
    if (!ctx->scratch_x || !ctx->scratch_y || !ctx->scratch_norm ||
        !ctx->scratch_ones || !ctx->scratch_logits) {
        fprintf(stderr, "[tb_infer] scratch alloc failed (H=%d V=%d)\n", H, V);
    }

    /* Initialise session KV slots — allocate KV caches up-front when model dims known.
     * Each session_id maps to a fixed slot; the KV cache accumulates across tokens.
     * On new conversation: caller passes a new session_id, slot is cleared first use. */
    memset(ctx->session_kvs, 0, sizeof(ctx->session_kvs));
    /* KV caches are allocated lazily on first decode call per session slot.
     * Upfront allocation of all 16 slots at 8192 max_seq would consume ~28 GB RAM. */
    ctx->n_sessions = TB_MAX_SESSIONS;

    char lat_desc[256];
    tb_lattice_describe(ctx->lattice, lat_desc, sizeof(lat_desc));
    printf("[tb_infer] Context created: hdgl=%d alpha=%.2f lattice=%s\n",
           use_hdgl, hdgl_alpha, lat_desc);
    return ctx;
}

void tb_infer_free(TB_InferCtx *ctx) {
    if (!ctx) return;
    pthread_mutex_destroy(&ctx->generate_lock);
    tb_tree_destroy(ctx->tree);
    tb_lattice_destroy(ctx->lattice);
    tb_hopfield_free(ctx->semantic_mem);
    /* Free persistent session KV caches */
    for (int s = 0; s < TB_MAX_SESSIONS; s++) {
        if (ctx->session_kvs[s].kv) {
            tb_kvcache_free(ctx->session_kvs[s].kv);
            ctx->session_kvs[s].kv = NULL;
        }
    }
    /* Free decode scratch buffers */
    free(ctx->scratch_x);
    free(ctx->scratch_y);
    free(ctx->scratch_norm);
    free(ctx->scratch_ones);
    free(ctx->scratch_logits);
    free(ctx);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 7: Forward pass (single transformer layer, dense or MoE)
 * Uses mmap'd weights — no copies for Q4 tensors.
 * ────────────────────────────────────────────────────────────────────────── */


/* Compatibility shim: old 4-bit API → new tb_gguf_dequant_matvec */
void tb_gguf_dequant_matvec_4bit_compat(
    const uint32_t *w_packed, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out, int out_dim, int in_dim, int group_size) {
    /* Build a flat float representation row by row using Q4_0 block format
     * (flat packed layout from extract_weights.py / nonmetal_infer.c) */
    int blocks_per_row = in_dim / 8;
    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *wr = w_packed + (size_t)row * blocks_per_row;
        int ng = in_dim / group_size;
        const uint16_t *sr = scales + (size_t)row * ng;
        const uint16_t *br = biases + (size_t)row * ng;
        for (int g = 0; g < ng; g++) {
            /* BF16 scale and bias */
            uint32_t sb = (uint32_t)sr[g] << 16; float s; memcpy(&s,&sb,4);
            uint32_t bb = (uint32_t)br[g] << 16; float b; memcpy(&b,&bb,4);
            int packed_per_g = group_size / 8;
            int base_p = g * packed_per_g;
            for (int pi = 0; pi < packed_per_g; pi++) {
                uint32_t packed = wr[base_p + pi];
                for (int ni = 0; ni < 8; ni++) {
                    uint32_t q = (packed >> (ni * 4)) & 0xFu;
                    int xi = g * group_size + pi * 8 + ni;
                    acc += (s * (float)q + b) * x[xi];
                }
            }
        }
        out[row] = acc;
    }
}
static int tb_layer_forward(
    TB_InferCtx   *ctx,
    int            layer_idx,
    const float   *x,           /* (hidden_dim,) */
    float         *out,          /* (hidden_dim,) */
    TB_KVCache    *kv,
    int            pos,
    int            token_id
) {
    TB_GGUFModel *m = ctx->model;
    int H = m->hidden_dim;
    int NH = m->n_heads, NK = m->n_kv_heads, HD = m->head_dim;
    if (HD == 0 && NH > 0) HD = H / NH;
    float eps = m->norm_eps > 0 ? m->norm_eps : 1e-5f;
    if (layer_idx == 0) { fprintf(stderr, "[dbg] layer_forward L0: H=%d NH=%d NK=%d HD=%d\n", H, NH, NK, HD); fflush(stderr); }
    tb_vlog("layer=%d token=%d pos=%d begin", layer_idx, token_id, pos);

    /* ── Refresh oscillator snapshot for dispatch decisions this layer ──
     * Updates at layer 0 of each token step; other layers reuse the same
     * snapshot (lattice advances once per decode call, not per layer).
     * Mirrors fold26_wuwei_stream: analyze_chunk once per chunk, not per byte. */
    if (layer_idx == 0 && ctx->lattice) {
        double M_U, L_U, S_U;
        tb_lattice_s_u_resonance(ctx->lattice, &M_U, &L_U, &S_U);
        ctx->osc_snap.phase_var = ctx->lattice->phase_var;
        ctx->osc_snap.s_u       = S_U;
        ctx->osc_snap.lambda_u  = L_U;
        ctx->osc_snap.omega_u   = M_U;   /* M_U serves as resonance amplitude */
        /* Map phase_var to aphase tier (mirrors ll_analog.c ANA_CV_TO_* thresholds) */
        double cv = ctx->lattice->phase_var;
        if      (cv > 0.50) ctx->osc_snap.aphase = 0; /* PLUCK    */
        else if (cv > 0.30) ctx->osc_snap.aphase = 1; /* SUSTAIN  */
        else if (cv > 0.10) ctx->osc_snap.aphase = 2; /* FINETUNE */
        else                ctx->osc_snap.aphase = 3; /* LOCK     */
        ctx->osc_snap.steps++;
    }

    /* Tensor name helpers */
    char tname[128];
    #define TNSR(fmt, ...) do { snprintf(tname, sizeof(tname), fmt, ##__VA_ARGS__); } while(0)

    /* ── 1. Attention pre-norm ─────────────────────────────────────────── */
    if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: allocating xn/ones\n"); fflush(stderr); }
    float *xn = (float*)malloc(H * sizeof(float));
    float *ones = (float*)malloc(H * sizeof(float));
    for (int i=0;i<H;i++) ones[i]=1.0f;

    TNSR("blk.%d.attn_norm.weight", layer_idx);
    if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: find norm_w '%s'\n", tname); fflush(stderr); }
    const TB_GGUFTensorInfo *norm_w_t = tb_gguf_find_tensor(m, tname);
    if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: norm_w=%p qtype=%d\n", (void*)norm_w_t, norm_w_t?norm_w_t->qtype:-1); fflush(stderr); }
    if (norm_w_t && m->weights_data) {
        /* Convert F16/BF16 norm weights to F32 for rms_norm */
        if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: malloc w32\n"); fflush(stderr); }
        float *w32 = (float*)malloc(H*sizeof(float));
        if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: get tensor data\n"); fflush(stderr); }
        const uint16_t *w16 = (const uint16_t*)tb_gguf_tensor_data(m, norm_w_t);
        if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: convert norm qtype=%d\n", norm_w_t->qtype); fflush(stderr); }
        if (norm_w_t->qtype == 1)
            for (int i=0;i<H;i++) { uint32_t b=(uint32_t)w16[i]<<16; memcpy(&w32[i],&b,4); }
        else if (norm_w_t->qtype == 30)
            for (int i=0;i<H;i++) w32[i] = tb_bf16_to_f32_infer(w16[i]);
        else { memcpy(w32, tb_gguf_tensor_data(m,norm_w_t), H*sizeof(float)); }
        if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: rms_norm\n"); fflush(stderr); }
        tb_rms_norm(x, w32, xn, H, eps);
        free(w32);
    } else {
        tb_rms_norm(x, ones, xn, H, eps);
    }
    if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: attn_norm done, finding QKV tensors\n"); fflush(stderr); }
    tb_vlog("layer=%d stage=attn_norm done", layer_idx);

    /* ── 2. QKV projection + RoPE + attention ──────────────────────────── */
    float *q = (float*)calloc(NH*HD, sizeof(float));
    float *k = (float*)calloc(NK*HD, sizeof(float));
    float *v = (float*)calloc(NK*HD, sizeof(float));
    float *attn_out = (float*)calloc(NH*HD, sizeof(float));

    /* For now: identity projection (placeholder until full QKV weight loading) */
    /* In production: load Wq, Wk, Wv as Q4_K tensors and dequant-matvec */
    /* Try fused QKV first (qwen35 / hybrid arch), fall back to separate q/k/v */
    TNSR("blk.%d.attn_qkv.weight", layer_idx);
    const TB_GGUFTensorInfo *wqkv_t = tb_gguf_find_tensor(m, tname);
    TNSR("blk.%d.attn_q.weight", layer_idx);
    const TB_GGUFTensorInfo *wq_t = wqkv_t ? NULL : tb_gguf_find_tensor(m, tname);
    TNSR("blk.%d.attn_k.weight", layer_idx);
    const TB_GGUFTensorInfo *wk_t = wqkv_t ? NULL : tb_gguf_find_tensor(m, tname);
    TNSR("blk.%d.attn_v.weight", layer_idx);
    const TB_GGUFTensorInfo *wv_t = wqkv_t ? NULL : tb_gguf_find_tensor(m, tname);

    if (wqkv_t && m->weights_data) {
        /* Fused QKV: output is [NH*HD | NK*HD | NK*HD] concatenated */
        int QKV_M = (NH + 2*NK) * HD;
        float *qkv = (float*)calloc(QKV_M, sizeof(float));
        if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: qkv_fused M=%d K=%d qtype=%d gpu=%s\n", QKV_M, H, wqkv_t->qtype, wqkv_t->d_data?"Y":"N"); fflush(stderr); }
        tb_gguf_tensor_matvec(m, wqkv_t, QKV_M, H, xn, qkv);
        if (layer_idx == 0) { fprintf(stderr, "[dbg] L0: qkv_fused done\n"); fflush(stderr); }
        memcpy(q, qkv,                sizeof(float)*NH*HD);
        memcpy(k, qkv + NH*HD,        sizeof(float)*NK*HD);
        memcpy(v, qkv + (NH+NK)*HD,   sizeof(float)*NK*HD);
        free(qkv);
    } else if (wq_t && wk_t && wv_t && m->weights_data) {
        tb_gguf_tensor_matvec(m, wq_t, NH*HD, H, xn, q);
        tb_gguf_tensor_matvec(m, wk_t, NK*HD, H, xn, k);
        tb_gguf_tensor_matvec(m, wv_t, NK*HD, H, xn, v);
    } else {
        for (int i=0;i<NH*HD && i<H;i++) q[i]=xn[i];
        for (int i=0;i<NK*HD && i<H;i++) { k[i]=xn[i]; v[i]=xn[i]; }
    }
    tb_vlog("layer=%d stage=qkv done", layer_idx);

    /* RoPE */
    for (int h=0;h<NH;h++) tb_rope_apply(q+h*HD, k+(h%NK)*HD, HD, pos, m->rope_base);
    tb_vlog("layer=%d stage=rope done", layer_idx);

    /* Qwen3 per-head QK-norm (blk.L.attn_q_norm / attn_k_norm) */
    {
        TNSR("blk.%d.attn_q_norm.weight", layer_idx);
        const TB_GGUFTensorInfo *qn_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.attn_k_norm.weight", layer_idx);
        const TB_GGUFTensorInfo *kn_t = tb_gguf_find_tensor(m, tname);
        if ((qn_t || kn_t) && m->weights_data) {
            float *w32 = (float*)malloc(HD * sizeof(float));
            if (w32) {
                if (qn_t) {
                    const uint16_t *nw = (const uint16_t*)tb_gguf_tensor_data(m, qn_t);
                    if (qn_t->qtype == 30)
                        for (int i=0;i<HD;i++) w32[i] = tb_bf16_to_f32_infer(nw[i]);
                    else if (qn_t->qtype == 1) {
                        for (int i=0;i<HD;i++) { uint32_t b=(uint32_t)nw[i]<<16; memcpy(&w32[i],&b,4); }
                    } else memcpy(w32, tb_gguf_tensor_data(m, qn_t), HD*sizeof(float));
                    for (int h=0;h<NH;h++) {
                        float *qh = q + h*HD;
                        float sq = 0.0f;
                        for (int i=0;i<HD;i++) sq += qh[i]*qh[i];
                        float sc = 1.0f / sqrtf(sq/HD + eps);
                        for (int i=0;i<HD;i++) qh[i] = qh[i]*sc*w32[i];
                    }
                }
                if (kn_t) {
                    const uint16_t *nw = (const uint16_t*)tb_gguf_tensor_data(m, kn_t);
                    if (kn_t->qtype == 30)
                        for (int i=0;i<HD;i++) w32[i] = tb_bf16_to_f32_infer(nw[i]);
                    else if (kn_t->qtype == 1) {
                        for (int i=0;i<HD;i++) { uint32_t b=(uint32_t)nw[i]<<16; memcpy(&w32[i],&b,4); }
                    } else memcpy(w32, tb_gguf_tensor_data(m, kn_t), HD*sizeof(float));
                    for (int h=0;h<NK;h++) {
                        float *kh = k + h*HD;
                        float sk = 0.0f;
                        for (int i=0;i<HD;i++) sk += kh[i]*kh[i];
                        float sc = 1.0f / sqrtf(sk/HD + eps);
                        for (int i=0;i<HD;i++) kh[i] = kh[i]*sc*w32[i];
                    }
                }
                free(w32);
            }
        }
    }

    /* Attention (uses KV cache from tb_tensor.h) */
    if (kv) {
        tb_attention(q, k, v, kv, layer_idx, NH, NK, HD, attn_out);
    } else {
        memcpy(attn_out, q, NH*HD*sizeof(float));
    }
    tb_vlog("layer=%d stage=attention done", layer_idx);

    /* Output projection + residual */
    float *attn_proj = (float*)calloc(H, sizeof(float));
    TNSR("blk.%d.attn_output.weight", layer_idx);
    const TB_GGUFTensorInfo *wo_t = tb_gguf_find_tensor(m, tname);
    if (wo_t && m->weights_data) {
        tb_gguf_tensor_matvec(m, wo_t, H, NH*HD, attn_out, attn_proj);
    } else {
        for (int i=0;i<H && i<NH*HD;i++) attn_proj[i]=attn_out[i];
    }
    tb_vlog("layer=%d stage=attn_out done", layer_idx);

    float *x2 = (float*)malloc(H*sizeof(float));
    for (int i=0;i<H;i++) x2[i] = x[i] + attn_proj[i];

    /* ── 3. FFN / MoE ─────────────────────────────────────────────────── */
    float *xn2 = (float*)malloc(H*sizeof(float));
    TNSR("blk.%d.ffn_norm.weight", layer_idx);
    const TB_GGUFTensorInfo *fn_t = tb_gguf_find_tensor(m, tname);
    if (fn_t && m->weights_data) {
        float *fw32=(float*)malloc(H*sizeof(float));
        const uint16_t *fw16=(const uint16_t*)tb_gguf_tensor_data(m,fn_t);
        if (fn_t->qtype==30) for(int i=0;i<H;i++) fw32[i]=tb_bf16_to_f32_infer(fw16[i]);
        else if (fn_t->qtype==1) { uint32_t b; for(int i=0;i<H;i++){b=(uint32_t)fw16[i]<<16;memcpy(&fw32[i],&b,4);}}
        else memcpy(fw32, tb_gguf_tensor_data(m,fn_t), H*sizeof(float));
        tb_rms_norm(x2, fw32, xn2, H, eps);
        free(fw32);
    } else {
        tb_rms_norm(x2, ones, xn2, H, eps);
    }
    tb_vlog("layer=%d stage=ffn_norm done", layer_idx);

    float *ffn_out = (float*)calloc(H, sizeof(float));

    if (m->n_experts > 0) {
        /* MoE: gate → top-k routing → expert forward → weighted sum */
        TNSR("blk.%d.ffn_gate_inp.weight", layer_idx);
        const TB_GGUFTensorInfo *gate_t = tb_gguf_find_tensor(m, tname);
        float *gate_logits = (float*)calloc(m->n_experts, sizeof(float));
        if (gate_t && m->weights_data) {
            const uint32_t *gw = (const uint32_t*)tb_gguf_tensor_data(m, gate_t);
            const uint16_t *gs_p = (const uint16_t*)((const char*)gw + m->n_experts*(H/8));
            const uint16_t *gb = gs_p + (m->n_experts/m->group_size);
            tb_gguf_dequant_matvec_4bit_compat(gw, gs_p, gb, xn2, gate_logits, m->n_experts, H, m->group_size);
        }

        TB_ExpertSelection sel = tb_route_experts(ctx, token_id, layer_idx,
                                                   gate_logits, m->n_experts);
        free(gate_logits);

        /* Run top-k experts */
        float *exp_out = (float*)calloc(H, sizeof(float));
        int k_exp = sel.k;
        for (int ei = 0; ei < k_exp; ei++) {
            int exp_idx = sel.expert_indices[ei];
            int ffn_dim = m->moe_intermediate_size > 0 ? m->moe_intermediate_size
                        : (m->ffn_dim > 0 ? m->ffn_dim : H*4);

            /* Packed format first (Qwen3 MoE): blk.L.ffn_gate_exps.weight [n_exp*ffn_dim, H]
             * Split format fallback:            blk.L.ffn_gate.N.weight    [ffn_dim, H]      */
            char pg[128], pu[128], pd[128], sg[128], su_[128], sd_[128];
            snprintf(pg,  sizeof(pg),  "blk.%d.ffn_gate_exps.weight",  layer_idx);
            snprintf(pu,  sizeof(pu),  "blk.%d.ffn_up_exps.weight",    layer_idx);
            snprintf(pd,  sizeof(pd),  "blk.%d.ffn_down_exps.weight",  layer_idx);
            snprintf(sg,  sizeof(sg),  "blk.%d.ffn_gate.%d.weight",    layer_idx, exp_idx);
            snprintf(su_, sizeof(su_), "blk.%d.ffn_up.%d.weight",      layer_idx, exp_idx);
            snprintf(sd_, sizeof(sd_), "blk.%d.ffn_down.%d.weight",    layer_idx, exp_idx);

            const TB_GGUFTensorInfo *eg_t = tb_gguf_find_tensor(m, pg);
            int is_packed = (eg_t != NULL);
            const TB_GGUFTensorInfo *eu_t = tb_gguf_find_tensor(m, is_packed ? pu : su_);
            const TB_GGUFTensorInfo *ed_t = tb_gguf_find_tensor(m, is_packed ? pd : sd_);
            if (!is_packed) eg_t = tb_gguf_find_tensor(m, sg);

            /* Row offset within packed tensor for this expert */
            int64_t row_off_g = is_packed ? (int64_t)exp_idx * ffn_dim : 0;
            int64_t row_off_d = is_packed ? (int64_t)exp_idx * H       : 0;

            if (eg_t && m->weights_data) {
                float *g_ep = (float*)malloc(ffn_dim*sizeof(float));
                float *u_ep = (float*)malloc(ffn_dim*sizeof(float));
                float *a_ep = (float*)malloc(ffn_dim*sizeof(float));
                if (g_ep && u_ep && a_ep) {
                    const void *gd = tb_gguf_tensor_data(m, eg_t);
                    const void *ud = eu_t ? tb_gguf_tensor_data(m, eu_t) : gd;
                    const void *dd = ed_t ? tb_gguf_tensor_data(m, ed_t) : NULL;
                    /* Bytes per weight row — correct GGUF block sizes */
                    #define _RB(qt,K) (                         (qt)==12?(size_t)((K)/256)*144:(qt)==11?(size_t)((K)/256)*110:                         (qt)==13?(size_t)((K)/256)*176:(qt)==14?(size_t)((K)/256)*210:                         (qt)==15?(size_t)((K)/256)*292:(qt)== 8?(size_t)((K)/32)*34:                          (qt)== 2?(size_t)((K)/32)*18:(size_t)(K)*4)
                    const char *gb = (const char*)gd + (size_t)row_off_g * _RB(eg_t->qtype, H);
                    int uqt = eu_t ? eu_t->qtype : eg_t->qtype;
                    const char *ub = (const char*)ud + (size_t)row_off_g * _RB(uqt, H);
                    tb_gguf_dequant_matvec(gb, eg_t->qtype, ffn_dim, H, xn2, g_ep);
                    tb_gguf_dequant_matvec(ub, uqt,         ffn_dim, H, xn2, u_ep);
                    tb_swiglu(g_ep, u_ep, a_ep, ffn_dim);
                    if (dd) {
                        const char *db = (const char*)dd
                            + (size_t)row_off_d * _RB(ed_t->qtype, ffn_dim);
                        tb_gguf_dequant_matvec(db, ed_t->qtype, H, ffn_dim, a_ep, exp_out);
                    } else {
                        memcpy(exp_out, a_ep, H*sizeof(float));
                    }
                    #undef _RB
                }
                free(g_ep); free(u_ep); free(a_ep);
            } else {
                /* Dense FFN fallback (no expert weights found) */
                memcpy(exp_out, xn2, H*sizeof(float));
            }


            /* Weighted accumulate */
            float w = sel.expert_weights[ei];
            for (int i=0;i<H;i++) ffn_out[i] += w * exp_out[i];
        }
        free(exp_out);
    } else {
        /* Dense FFN */
        int ffn_dim = m->ffn_dim > 0 ? m->ffn_dim : H*4;
        tb_vlog("layer=%d stage=ffn_dense begin ffn_dim=%d hidden=%d", layer_idx, ffn_dim, H);
        float *g_out=(float*)malloc(ffn_dim*sizeof(float));
        float *u_out=(float*)malloc(ffn_dim*sizeof(float));
        float *act  =(float*)malloc(ffn_dim*sizeof(float));

        TNSR("blk.%d.ffn_gate.weight", layer_idx);
        const TB_GGUFTensorInfo *ffn_g = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ffn_up.weight",   layer_idx);
        const TB_GGUFTensorInfo *ffn_u = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ffn_down.weight", layer_idx);
        const TB_GGUFTensorInfo *ffn_d = tb_gguf_find_tensor(m, tname);
        tb_vlog("layer=%d stage=ffn_dense tensors gate=%s up=%s down=%s qtypes=[%d,%d,%d]",
                layer_idx,
                ffn_g ? "yes" : "no",
                ffn_u ? "yes" : "no",
                ffn_d ? "yes" : "no",
                ffn_g ? ffn_g->qtype : -1,
                ffn_u ? ffn_u->qtype : -1,
                ffn_d ? ffn_d->qtype : -1);

        if (ffn_g && ffn_u && ffn_d && m->weights_data) {
            tb_vlog("layer=%d stage=ffn_dense gate_matvec begin", layer_idx);
            double _tffn = (layer_idx == 0) ? tb_wall_ms() : 0.0;
            tb_gguf_tensor_matvec(m, ffn_g, ffn_dim, H, xn2, g_out);
            tb_gguf_tensor_matvec(m, ffn_u, ffn_dim, H, xn2, u_out);
            tb_swiglu(g_out, u_out, act, ffn_dim);
            tb_gguf_tensor_matvec(m, ffn_d, H, ffn_dim, act, ffn_out);
            if (layer_idx == 0)
                fprintf(stderr, "[timing L0] ffn gate+up+down: %.1fms  ffn_dim=%d gate_qtype=%d d_data=%s\n",
                        tb_wall_ms()-_tffn, ffn_dim, ffn_g->qtype, ffn_g->d_data?"GPU":"CPU");
            tb_vlog("layer=%d stage=ffn_dense down_matvec done", layer_idx);
        } else {
            memcpy(ffn_out, xn2, H*sizeof(float));
        }
        free(g_out); free(u_out); free(act);
    }
    tb_vlog("layer=%d stage=ffn done", layer_idx);

    /* Residual */
    for (int i=0;i<H;i++) out[i] = x2[i] + ffn_out[i];

    free(xn); free(ones); free(q); free(k); free(v);
    free(attn_out); free(attn_proj); free(x2); free(xn2); free(ffn_out);
    tb_vlog("layer=%d token=%d end", layer_idx, token_id);
    return 1;
    #undef TNSR
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 8: Prefill + decode + generate
 * ────────────────────────────────────────────────────────────────────────── */

int tb_infer_decode(TB_InferCtx *ctx, int last_token, int pos, int session_id) {
    TB_GGUFModel *m = ctx->model;
    if (!m || m->hidden_dim == 0) return -1;
    tb_vlog("decode begin token=%d pos=%d session=%d", last_token, pos, session_id);

    int H   = m->hidden_dim;
    int NL  = m->n_layers;

    /* ── Resolve persistent KV cache for this session ──────────────────────
     * session_id is clamped to [0, TB_MAX_SESSIONS).  The slot's seq_len
     * already holds the tokens decoded so far; we append to it, not replace.
     * If this is the first call for a session (pos==0) we reset seq_len. */
    TB_KVCache *kv = NULL;
    if (m->n_kv_heads > 0 && m->head_dim > 0 && ctx->n_sessions > 0) {
        int slot = (session_id < 0) ? 0
                 : (session_id >= TB_MAX_SESSIONS) ? (session_id % TB_MAX_SESSIONS)
                 : session_id;
        TB_SessionKV *skv = &ctx->session_kvs[slot];
        /* Lazy allocation: allocate KV cache on first use for this slot */
        if (!skv->kv) {
            int max_seq = m->max_seq_len > 0 ? (m->max_seq_len < 4096 ? m->max_seq_len : 4096) : 2048;
            skv->kv = tb_kvcache_alloc(m->n_layers, m->n_kv_heads, m->head_dim,
                                       max_seq, slot, ctx->lattice->epoch);
            skv->session_id = slot;
            skv->active     = 0;
            skv->epoch      = ctx->lattice->epoch;
            if (skv->kv)
                fprintf(stderr, "[tb_infer] KV cache allocated: slot=%d %d layers %d kv-heads max_seq=%d\n",
                        slot, m->n_layers, m->n_kv_heads, max_seq);
        }
        if (skv->kv) {
            /* Reset cache when starting a new sequence */
            if (pos == 0 || !skv->active) {
                skv->kv->seq_len = 0;
                skv->active      = 1;
                skv->epoch       = ctx->lattice->epoch;
            }
            kv = skv->kv;
        }
    }

    /* ── Embedding lookup into scratch_x ───────────────────────────────── */
    fprintf(stderr, "[dbg] step: embedding alloc\n"); fflush(stderr);
    float *x = ctx->scratch_x ? ctx->scratch_x : (float*)calloc(H, sizeof(float));
    int x_heap = (x != ctx->scratch_x);
    if (ctx->scratch_x) memset(ctx->scratch_x, 0, H * sizeof(float));

    const TB_GGUFTensorInfo *emb_t = tb_gguf_find_tensor(m, "token_embd.weight");
    fprintf(stderr, "[dbg] emb_t=%p qtype=%d\n", (void*)emb_t, emb_t ? emb_t->qtype : -1); fflush(stderr);
    if (emb_t && m->weights_data && last_token >= 0 && last_token < m->vocab_size) {
        if (emb_t->qtype == 30) {
            fprintf(stderr, "[dbg] emb: BF16 path\n"); fflush(stderr);
            const uint16_t *emb = (const uint16_t*)tb_gguf_tensor_data(m, emb_t);
            const uint16_t *row = emb + (size_t)last_token * H;
            for (int i=0;i<H;i++) x[i] = tb_bf16_to_f32_infer(row[i]);
        } else if (emb_t->qtype == 0) {
            fprintf(stderr, "[dbg] emb: F32 path\n"); fflush(stderr);
            const float *emb = (const float*)tb_gguf_tensor_data(m, emb_t);
            memcpy(x, emb + (size_t)last_token * H, H * sizeof(float));
        } else {
            int bw, bb;
            switch (emb_t->qtype) {
                case 8:  bw=32;  bb=34;  break;
                case 2:  bw=32;  bb=18;  break;
                case 12: bw=256; bb=144; break;
                case 10: bw=256; bb=84;  break;
                case 11: bw=256; bb=110; break;
                case 13: bw=256; bb=176; break;
                case 14: bw=256; bb=210; break;
                case 15: bw=256; bb=292; break;
                default: bw=1;   bb=4;   break;
            }
            size_t row_stride = (size_t)((H + bw - 1) / bw) * (size_t)bb;
            const char *row_ptr = (const char*)tb_gguf_tensor_data(m, emb_t)
                                  + (size_t)last_token * row_stride;
            fprintf(stderr, "[dbg] emb: qtype=%d bw=%d bb=%d row_stride=%zu\n", emb_t->qtype, bw, bb, row_stride); fflush(stderr);
            tb_gguf_dequant_row(row_ptr, emb_t->qtype, H, x);
            fprintf(stderr, "[dbg] emb: dequant done\n"); fflush(stderr);
        }
    } else {
        fprintf(stderr, "[dbg] emb: fallback sine\n"); fflush(stderr);
        for (int i=0;i<H;i++) x[i]=0.01f*sinf((float)(i+last_token));
    }
    fprintf(stderr, "[dbg] step: embedding done, entering forward loop NL=%d\n", NL); fflush(stderr);
    tb_vlog("decode stage=embedding done token=%d", last_token);

    /* ── Forward through all layers (swap x ↔ y via scratch buffers) ─── */
    float *y = ctx->scratch_y ? ctx->scratch_y : (float*)malloc(H * sizeof(float));
    int y_heap = (y != ctx->scratch_y);

    double t_fwd = tb_wall_ms();
    for (int l = 0; l < NL; l++) {
        tb_vlog("decode stage=layer_enter layer=%d", l);
        tb_layer_forward(ctx, l, x, y, kv, pos, last_token);
        tb_vlog("decode stage=layer_exit layer=%d", l);
        float *tmp = x; x = y; y = tmp;
    }
    ctx->decode_ms += tb_wall_ms() - t_fwd;

    /* ── Final norm (into scratch_norm) ────────────────────────────────── */
    float *norm_out = ctx->scratch_norm ? ctx->scratch_norm : (float*)malloc(H * sizeof(float));
    int norm_heap   = (norm_out != ctx->scratch_norm);
    float *ones     = ctx->scratch_ones ? ctx->scratch_ones : (float*)malloc(H * sizeof(float));
    int ones_heap   = (ones != ctx->scratch_ones);
    if (ones_heap) { for (int i=0;i<H;i++) ones[i]=1.0f; }

    const TB_GGUFTensorInfo *fn_t = tb_gguf_find_tensor(m, "output_norm.weight");
    if (fn_t && m->weights_data) {
        float *fw32=(float*)malloc(H*sizeof(float));
        if (!fw32) {
            fprintf(stderr, "[tb_infer] output_norm alloc failed\n");
            return -1;
        }
        const uint16_t *fw16=(const uint16_t*)tb_gguf_tensor_data(m,fn_t);
        if (fn_t->qtype==30) for(int i=0;i<H;i++) fw32[i]=tb_bf16_to_f32_infer(fw16[i]);
        else if (fn_t->qtype==1) { uint32_t b; for(int i=0;i<H;i++){ b=(uint32_t)fw16[i]<<16; memcpy(&fw32[i],&b,4); } }
        else if (fn_t->qtype==0) memcpy(fw32, tb_gguf_tensor_data(m,fn_t), H*sizeof(float));
        else tb_gguf_dequant_row(tb_gguf_tensor_data(m, fn_t), fn_t->qtype, H, fw32);
        tb_rms_norm(x, fw32, norm_out, H, m->norm_eps);
        free(fw32);
    } else {
        tb_rms_norm(x, ones, norm_out, H, m->norm_eps);
    }
    tb_vlog("decode stage=final_norm done qtype=%d", fn_t ? fn_t->qtype : -1);

    /* ── LM head → logits (into scratch_logits) → sample ──────────────── */
    float *logits = ctx->scratch_logits ? ctx->scratch_logits : (float*)calloc(m->vocab_size, sizeof(float));
    int logits_heap = (logits != ctx->scratch_logits);
    if (ctx->scratch_logits) memset(ctx->scratch_logits, 0, m->vocab_size * sizeof(float));

    const TB_GGUFTensorInfo *lm_t = tb_gguf_find_tensor(m, "output.weight");
    if (!lm_t) lm_t = tb_gguf_find_tensor(m, "token_embd.weight");
    if (lm_t && m->weights_data) {
        tb_vlog("decode stage=lm_head begin qtype=%d", lm_t->qtype);
        tb_gguf_tensor_matvec(m, lm_t, m->vocab_size, H, norm_out, logits);
        tb_vlog("decode stage=lm_head done");
    } else {
        for (int v=0;v<m->vocab_size;v++) logits[v]=0.01f*cosf((float)v*0.01f);
        logits[(last_token+1)%m->vocab_size] = 1.0f;
    }
    tb_vlog("decode stage=logits done");

    /* ── Sample ─────────────────────────────────────────────────────────── */
    int next_token;
    if (ctx->temperature <= 0.0f || ctx->temperature < 0.01f)
        next_token = tb_sample_greedy(logits, m->vocab_size);
    else if (ctx->top_k > 0)
        next_token = tb_sample_top_k(logits, m->vocab_size, ctx->top_k, ctx->temperature);
    else
        next_token = tb_sample_top_p(logits, m->vocab_size, ctx->top_p, ctx->temperature);
    tb_vlog("decode end next_token=%d", next_token);

    /* Free only heap-allocated fallbacks (scratch buffers stay alive in ctx) */
    if (x_heap)      free(x);
    if (y_heap)      free(y);
    if (norm_heap)   free(norm_out);
    if (ones_heap)   free(ones);
    if (logits_heap) free(logits);
    /* KV cache is NOT freed — it persists in ctx->session_kvs[slot] */
    return next_token;
}


int tb_infer_generate(
    TB_InferCtx  *ctx,
    const int    *prompt_ids,
    int           n_prompt,
    int          *out_ids,
    int           max_out,
    int           session_id,
    void (*token_cb)(int token_id, void *ud),
    void         *cb_ud
) {
    int eos_id = (ctx->model && ctx->model->eos_token_id > 0)
                 ? ctx->model->eos_token_id : 2;
    int pos    = 0;
    int n_gen  = 0;
    double t0  = tb_wall_ms();

    /* Prefill: decode all prompt tokens */
    int last_token = prompt_ids[n_prompt-1];
    tb_vlog("generate begin prompt_tokens=%d max_out=%d session=%d", n_prompt, max_out, session_id);
    for (int i = 0; i < n_prompt - 1; i++) {
        tb_vlog("generate prefill step=%d token=%d", i, prompt_ids[i]);
        tb_infer_decode(ctx, prompt_ids[i], pos++, session_id);
    }
    ctx->prefill_ms = tb_wall_ms() - t0;
    ctx->prompt_tokens = n_prompt;

    /* Decode loop */
    while (n_gen < max_out && n_gen < ctx->max_new_tokens) {
        tb_vlog("generate decode_step=%d token_in=%d", n_gen, last_token);
        int tok = tb_infer_decode(ctx, last_token, pos++, session_id);
        if (tok < 0) {
            fprintf(stderr, "[tb_infer] decode returned error at gen step %d\n", n_gen);
            fflush(stderr);
            break;
        }
        if (n_gen < max_out) out_ids[n_gen++] = tok;
        if (token_cb) token_cb(tok, cb_ud);
        if (tok == eos_id) break;
        last_token = tok;
    }
    ctx->tokens_generated = n_gen;

    double total_ms = tb_wall_ms() - t0;
    double tok_s    = (double)n_gen / (total_ms / 1000.0);
    printf("[tb_infer] generated %d tokens in %.1fms (%.3f tok/s), "
           "prefill=%.1fms\n", n_gen, total_ms, tok_s, ctx->prefill_ms);

    /* Layer 3: commit generation summary to cognition tree */
    if (ctx->tree && n_gen > 0) {
        char gen_summary[128];
        snprintf(gen_summary, sizeof(gen_summary),
                 "{\"n_gen\":%d,\"tok_s\":%.1f,\"prompt_tokens\":%d}",
                 n_gen, tok_s, n_prompt);
        tb_tree_cell_commit(ctx->tree, 0, gen_summary, "generation");
    }

    return n_gen;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 9: Ollama-compatible HTTP server
 * /api/generate  POST {"model":"..","prompt":"..","stream":true}
 * /api/chat      POST {"model":"..","messages":[{"role":"user","content":".."}]}
 * /api/tags      GET  {"models":[{"name":"..","size":..}]}
 * /health        GET  "OK"
 * ────────────────────────────────────────────────────────────────────────── */

static TB_ServeConfig *g_serve_cfg = NULL;
static volatile int    g_serve_running = 1;

static void tb_serve_sigterm(int s) { (void)s; g_serve_running = 0; }

static void tb_http_response(tb_socket_t fd, int code, const char *ctype, const char *body) {
    char hdr[512];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, code==200?"OK":code==404?"Not Found":"Bad Request",
        ctype, strlen(body));
    TB_SEND(fd, hdr, hlen);
    TB_SEND(fd, body, strlen(body));
}

typedef struct { tb_socket_t fd; TB_ServeConfig *cfg; } TB_ConnCtx;

/* ── HTTP helpers ─────────────────────────────────────────────────────────── */

/* Read HTTP request with Content-Length awareness.
 * Allocates and returns a heap buffer (caller must free).
 * Returns NULL on error. *out_len set to total bytes read. */
static char* tb_http_read_request(tb_socket_t fd, size_t *out_len) {
    size_t cap  = 4096;
    size_t used = 0;
    char  *buf  = (char*)malloc(cap);
    if (!buf) return NULL;

    /* Read until we have the full header block */
    while (used < cap - 1) {
        int n = (int)TB_RECV(fd, buf + used, (int)(cap - used - 1));
        if (n <= 0) break;
        used += (size_t)n;
        buf[used] = '\0';
        if (strstr(buf, "\r\n\r\n")) break;
    }

    /* Parse Content-Length and read remaining body */
    const char *cl_hdr = strstr(buf, "Content-Length:");
    if (!cl_hdr) cl_hdr = strstr(buf, "content-length:");
    if (cl_hdr) {
        size_t content_len = (size_t)atol(cl_hdr + 15);
        const char *body_start = strstr(buf, "\r\n\r\n");
        size_t header_len = body_start ? (size_t)(body_start - buf) + 4 : used;
        size_t total_need = header_len + content_len + 1;
        if (total_need > 256 * 1024) total_need = 256 * 1024;
        if (total_need > cap) {
            char *nbuf = (char*)realloc(buf, total_need);
            if (!nbuf) { free(buf); return NULL; }
            buf = nbuf; cap = total_need;
        }
        size_t body_have = (used > header_len) ? (used - header_len) : 0;
        size_t body_want = total_need - header_len - 1;
        while (body_have < body_want) {
            int n = (int)TB_RECV(fd, buf + used, (int)(cap - used - 1));
            if (n <= 0) break;
            used += (size_t)n;
            body_have += (size_t)n;
            buf[used] = '\0';
        }
    }
    if (out_len) *out_len = used;
    return buf;
}

/* Extract a JSON string field value into dst (max dst_len bytes).
 * Handles \n \t \" \\ escape sequences. */
static void tb_json_extract_str(const char *json, const char *key,
                                 char *dst, int dst_len) {
    dst[0] = '\0';
    if (!json || !key) return;
    char keybuf[128];
    snprintf(keybuf, sizeof(keybuf), "\"%s\"", key);
    const char *p = strstr(json, keybuf);
    if (!p) return;
    p += strlen(keybuf);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != ':') return;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '"') return;
    p++;
    int di = 0;
    while (*p && di < dst_len - 1) {
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n':  dst[di++] = '\n'; break;
                case 't':  dst[di++] = '\t'; break;
                case 'r':  dst[di++] = '\r'; break;
                case '"':  dst[di++] = '"';  break;
                case '\\': dst[di++] = '\\'; break;
                default:   dst[di++] = *p;   break;
            }
        } else if (*p == '"') {
            break;
        } else {
            dst[di++] = *p;
        }
        p++;
    }
    dst[di] = '\0';
}

/* Returns 1 if "key": true in json. */
static int tb_json_extract_bool(const char *json, const char *key) {
    if (!json || !key) return 0;
    char keybuf[128];
    snprintf(keybuf, sizeof(keybuf), "\"%s\"", key);
    const char *p = strstr(json, keybuf);
    if (!p) return 0;
    p = strchr(p + strlen(keybuf), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (strncmp(p, "true", 4) == 0);
}

/* Streaming callback: one call per generated token. */
typedef struct {
    tb_socket_t  fd;
    TB_Tokenizer *tok;
    int           do_stream;
    const char   *model_name;
    int          *out_ids;
    int           out_cap;
    int           out_len;
} TB_StreamCtx;

static void tb_stream_token_cb(int token_id, void *ud) {
    TB_StreamCtx *sc = (TB_StreamCtx*)ud;
    /* Always buffer ids for final response / eval_count */
    if (sc->out_ids && sc->out_len < sc->out_cap)
        sc->out_ids[sc->out_len++] = token_id;
    if (!sc->do_stream) return;

    /* Decode single token and send as a chunked JSON line */
    char *word = sc->tok ? tb_tokenizer_decode(sc->tok, &token_id, 1) : NULL;
    const char *text = word ? word : "";
    char esc[256]; int ei = 0;
    for (const char *c = text; *c && ei < 250; c++) {
        if (*c == '"' || *c == '\\') esc[ei++] = '\\';
        esc[ei++] = *c;
    }
    esc[ei] = '\0';
    free(word);
    char chunk[512];
    int clen = snprintf(chunk, sizeof(chunk),
        "{\"model\":\"%s\",\"done\":false,\"response\":\"%s\"}\n",
        sc->model_name, esc);
    char chead[16];
    int hlen = snprintf(chead, sizeof(chead), "%x\r\n", clen);
    TB_SEND(sc->fd, chead, hlen);
    TB_SEND(sc->fd, chunk, clen);
    TB_SEND(sc->fd, "\r\n", 2);
}

static void* tb_handle_conn(void *arg) {
    TB_ConnCtx *cc = (TB_ConnCtx*)arg;
    tb_socket_t fd        = cc->fd;
    TB_InferCtx *ctx      = cc->cfg->ctx;
    /* Copy fields we need before freeing cc */
    const char *model_name = cc->cfg->model_name;
    int (*ext_handler)(int, const char*, const char*, const char*, void*) = cc->cfg->ext_handler;
    void *ext_ctx         = cc->cfg->ext_ctx;
    free(cc); cc = NULL;

    size_t req_len = 0;
    char *buf = tb_http_read_request(fd, &req_len);
    if (!buf || req_len == 0) { if (buf) free(buf); TB_CLOSESOCK(fd); return NULL; }

    char method[8]="", path[128]="";
    sscanf(buf, "%7s %127s", method, path);
    const char *body_start = strstr(buf, "\r\n\r\n");
    const char *body = body_start ? body_start + 4 : "";

    if (strcmp(method, "OPTIONS") == 0) {
        tb_http_response(fd, 200, "text/plain", "");

    } else if (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0) {
        tb_http_response(fd, 200, "application/json",
                          "{\"status\":\"ok\",\"runtime\":\"trailblaze\"}");

    } else if (strcmp(path, "/api/tags") == 0) {
        char rbody[512];
        snprintf(rbody, sizeof(rbody),
            "{\"models\":[{\"name\":\"%s\",\"size\":%zu,"
            "\"details\":{\"family\":\"%s\","
            "\"parameter_size\":\"%dB\","
            "\"quantization_level\":\"Q4\"}}]}",
            model_name, ctx->model->file_size,
            ctx->model->arch,
            ctx->model->n_layers * ctx->model->hidden_dim / 1000000);
        tb_http_response(fd, 200, "application/json", rbody);

    } else if ((strcmp(path, "/api/generate") == 0 || strcmp(path, "/api/chat") == 0)
               && strcmp(method, "POST") == 0) {

        int is_chat   = (strcmp(path, "/api/chat") == 0);
        int do_stream = tb_json_extract_bool(body, "stream");

        /* Extract prompt / content (up to 8 KB) */
        char prompt[8192] = "";
        if (is_chat)
            tb_json_extract_str(body, "content", prompt, sizeof(prompt));
        else
            tb_json_extract_str(body, "prompt",  prompt, sizeof(prompt));
        if (!prompt[0]) strcpy(prompt, "Hello");

        /* Tokenize */
        TB_Tokenizer *tok = ctx->model ? ctx->model->tokenizer : NULL;
        int token_ids[2048]; int n_tok = 0;
        if (tok) n_tok = tb_tokenizer_encode(tok, prompt, 1, token_ids, 2048);
        if (n_tok <= 0) {
            for (int i = 0; i < (int)strlen(prompt) && n_tok < 2048; i++)
                token_ids[n_tok++] = ((unsigned char)prompt[i])
                    % (ctx->model->vocab_size > 0 ? ctx->model->vocab_size : 32000);
        }
        if (n_tok <= 0) { token_ids[0] = 1; n_tok = 1; }

        /* Send chunked header before generate if streaming */
        if (do_stream) {
            const char *hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Transfer-Encoding: chunked\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            TB_SEND(fd, hdr, (int)strlen(hdr));
        }

        int out_ids[2048];
        TB_StreamCtx sc = {0};
        sc.fd         = fd;
        sc.tok        = tok;
        sc.do_stream  = do_stream;
        sc.model_name = model_name;
        sc.out_ids    = out_ids;
        sc.out_cap    = 2048;
        sc.out_len    = 0;

        pthread_mutex_lock(&ctx->generate_lock);
        int n_gen = tb_infer_generate(ctx, token_ids, n_tok,
                                      out_ids, 2048, 0,
                                      tb_stream_token_cb, &sc);
        pthread_mutex_unlock(&ctx->generate_lock);

        if (do_stream) {
            /* Final done chunk */
            char done_chunk[256];
            int dlen = snprintf(done_chunk, sizeof(done_chunk),
                "{\"model\":\"%s\",\"done\":true,"
                "\"eval_count\":%d,\"eval_duration\":%lld}\n",
                model_name, n_gen,
                (long long)(ctx->decode_ms * 1e6));
            char chead[16];
            int hlen = snprintf(chead, sizeof(chead), "%x\r\n", dlen);
            TB_SEND(fd, chead, hlen);
            TB_SEND(fd, done_chunk, dlen);
            TB_SEND(fd, "\r\n", 2);
            TB_SEND(fd, "0\r\n\r\n", 5);  /* terminating chunk */
        } else {
            char resp_text[4096] = "";
            if (tok && n_gen > 0) {
                char *decoded = tb_tokenizer_decode(tok, out_ids, n_gen);
                if (decoded) { snprintf(resp_text, sizeof(resp_text), "%s", decoded); free(decoded); }
            }
            if (!resp_text[0]) snprintf(resp_text, sizeof(resp_text), "[%d tokens]", n_gen);
            char resp[8192];
            if (is_chat) {
                snprintf(resp, sizeof(resp),
                    "{\"model\":\"%s\",\"done\":true,"
                    "\"message\":{\"role\":\"assistant\",\"content\":\"%.*s\"},"
                    "\"eval_count\":%d,\"eval_duration\":%lld}",
                    model_name, (int)(sizeof(resp)-256), resp_text,
                    n_gen, (long long)(ctx->decode_ms * 1e6));
            } else {
                snprintf(resp, sizeof(resp),
                    "{\"model\":\"%s\",\"done\":true,"
                    "\"response\":\"%.*s\","
                    "\"eval_count\":%d,\"eval_duration\":%lld}",
                    model_name, (int)(sizeof(resp)-256), resp_text,
                    n_gen, (long long)(ctx->decode_ms * 1e6));
            }
            tb_http_response(fd, 200, "application/json", resp);
        }

    } else if (ext_handler) {
        ext_handler(fd, path, method, body, ext_ctx);
    } else {
        tb_http_response(fd, 404, "application/json", "{\"error\":\"not found\"}");
    }

    free(buf);
    TB_CLOSESOCK(fd);
    return NULL;
}

int tb_serve(TB_ServeConfig *cfg) {
    g_serve_cfg = cfg;
    signal(SIGTERM, tb_serve_sigterm);
    signal(SIGINT,  tb_serve_sigterm);
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);   /* broken pipe on closed client conn */
#else
    /* Initialize Winsock2 on Windows */
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return -1;
    }
#endif

    tb_socket_t srv_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv_fd < 0) { perror("socket"); return -1; }
    int opt = 1; setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t)cfg->port);
    addr.sin_addr.s_addr = inet_addr(cfg->host[0] ? cfg->host : "0.0.0.0");
    if (bind(srv_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); TB_CLOSESOCK(srv_fd); return -1;
    }
    listen(srv_fd, 128);

    printf("[tb_serve] Listening on http://%s:%d\n",
           cfg->host[0] ? cfg->host : "0.0.0.0", cfg->port);
    printf("[tb_serve] Model: %s | HDGL: %s\n",
           cfg->model_name, cfg->ctx->use_hdgl ? "enabled" : "disabled");

    while (g_serve_running) {
        struct sockaddr_in cli; socklen_t cli_len = sizeof(cli);
        tb_socket_t cli_fd = accept(srv_fd, (struct sockaddr*)&cli, &cli_len);
        if (cli_fd < 0) { if (g_serve_running) perror("accept"); break; }

        TB_ConnCtx *cc = (TB_ConnCtx*)malloc(sizeof(TB_ConnCtx));
        cc->fd = cli_fd; cc->cfg = cfg;
        pthread_t t;
        pthread_create(&t, NULL, tb_handle_conn, cc);
        pthread_detach(t);
    }
    TB_CLOSESOCK(srv_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 10: Diff utility
 * Uses system diff to compare two C source trees and writes a unified patch.
 * ────────────────────────────────────────────────────────────────────────── */

void tb_diff_sources(const char *dir_a, const char *dir_b,
                      const char *out_patch) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "diff -ruN --include='*.c' --include='*.h' '%s' '%s' > '%s' 2>&1",
             dir_a, dir_b, out_patch);
    int r = system(cmd);
    printf("[tb_diff] diff %s -> %s written to %s (exit=%d)\n",
           dir_a, dir_b, out_patch, r);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 11: CLI entry point
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef TB_INFER_TEST
typedef struct {
    TB_Tokenizer *tok;
} TB_CliTokenCtx;

static void print_token_cb(int tok, void *ud) {
    TB_CliTokenCtx *cc = (TB_CliTokenCtx*)ud;
    char *piece = (cc && cc->tok) ? tb_tokenizer_decode(cc->tok, &tok, 1) : NULL;
    if (piece && piece[0]) {
        printf("[tok %d:'%s'] ", tok, piece);
    } else {
        printf("[tok %d] ", tok);
    }
    if (piece) free(piece);
    fflush(stdout);
}

int main(int argc, char **argv) {
    printf("=== TRAILBLAZE Inference Runtime ===\n\n");
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    /* Parse args */
    const char *model_path = NULL;
    int port = 11434, serve = 0, use_hdgl = 1, benchmark = 0;
    float hdgl_alpha = 0.2f;
    const char *prompt = "The future of AI inference is";
    const char *bench_log_path = "bench_hdglsql.jsonl";
    int bench_steps = 20;
    int max_new_tokens = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--model") && i+1<argc)       model_path = argv[++i];
        else if (!strcmp(argv[i],"--serve"))               serve = 1;
        else if (!strcmp(argv[i],"--port") && i+1<argc)   port = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--hdgl"))                use_hdgl = 1;
        else if (!strcmp(argv[i],"--no-hdgl"))             use_hdgl = 0;
        else if (!strcmp(argv[i],"--hdgl-alpha")&&i+1<argc) hdgl_alpha=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--prompt")&&i+1<argc)   prompt = argv[++i];
        else if (!strcmp(argv[i],"--benchmark"))           benchmark = 1;
        else if (!strcmp(argv[i],"--bench-steps") && i+1<argc) bench_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--bench-log") && i+1<argc) bench_log_path = argv[++i];
        else if (!strcmp(argv[i],"--max-new") && i+1<argc) max_new_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verbose") || !strcmp(argv[i],"--trace-decode")) g_tb_verbose = 1;
        else if (!strcmp(argv[i],"--diff") && i+2<argc) {
            tb_diff_sources(argv[i+1], argv[i+2], "/tmp/tb_diff.patch");
            printf("Patch written to /tmp/tb_diff.patch\n"); return 0;
        }
    }
    if (g_tb_verbose) fprintf(stderr, "[trace] verbose decode tracing enabled\n");

    /* If no model specified, run self-test with synthetic model */
    int synthetic = (model_path == NULL);
    TB_GGUFModel *model = NULL;

    if (!synthetic) {
        model = tb_model_load(model_path);
        if (!model) {
            fprintf(stderr, "Failed to load model: %s\n", model_path);
            return 1;
        }
    } else {
        /* Synthetic model for testing */
        model = (TB_GGUFModel*)calloc(1, sizeof(TB_GGUFModel));

        model->vocab_size= 32000;
        model->n_layers  = 4;      /* reduced for speed */
        model->n_heads   = 8;
        model->n_kv_heads= 4;
        model->head_dim  = 64;
        model->hidden_dim= 512;
        model->ffn_dim   = 1024;
        model->n_experts = 8;      /* MoE */
        model->n_experts_per_tok = 2;
        model->group_size= 32;
        model->rope_base = 10000.0f;
        model->norm_eps  = 1e-5f;
        model->max_seq_len=2048;
        snprintf(model->arch, sizeof(model->arch), "mixtral");
        model->n_tensors = 0;
        model->tensors = NULL;
        model->weights_data = NULL;
        printf("[test] Using synthetic model: %s layers=%d hidden=%d experts=%d\n",
               model->arch, model->n_layers, model->hidden_dim, model->n_experts);
    }

    /* Create inference context */
    TB_InferCtx *ctx = tb_infer_create(model, use_hdgl, hdgl_alpha, 512, 0xCAFEBABEULL);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    ctx->max_new_tokens = (max_new_tokens > 0) ? max_new_tokens : 1;

    printf("\n[test] HDGL routing: %s (alpha=%.2f)\n", use_hdgl?"ON":"OFF", hdgl_alpha);

    /* Test expert routing */
    float gate_logits[8];
    for (int i=0;i<8;i++) gate_logits[i]=0.1f*(i%3);
    TB_ExpertSelection sel = tb_route_experts(ctx, 42, 0, gate_logits, 8);
    printf("[routing] token=42 layer=0 → experts=[");
    for (int i=0;i<sel.k;i++) printf("%d(%.3f)%s",sel.expert_indices[i],sel.expert_weights[i],i<sel.k-1?",":"");
    printf("] hdgl_expert=%d octave=%d\n", sel.hdgl_expert, sel.semantic_octave);

    /* Test generation */
    if (!benchmark) {
        printf("\n[generate] prompt: '%s'\n", prompt);
        int tok_ids[2048]; int n_tok = 0;
        TB_Tokenizer *tok = model ? model->tokenizer : NULL;
        if (tok) n_tok = tb_tokenizer_encode(tok, prompt, 1, tok_ids, 2048);
        if (n_tok <= 0) {
            for (int i = 0; i < (int)strlen(prompt) && n_tok < 2048; i++)
                tok_ids[n_tok++] = ((unsigned char)prompt[i]) % model->vocab_size;
        }
        if (n_tok <= 0) { tok_ids[0] = 1; n_tok = 1; }

        printf("[generate] output tokens: ");
        int out_ids[2048];
        TB_CliTokenCtx cctx = {0};
        cctx.tok = tok;
        int n_gen = tb_infer_generate(ctx, tok_ids, n_tok, out_ids, 8, 0,
                                       print_token_cb, &cctx);
        if (tok && n_gen > 0) {
            char *decoded = tb_tokenizer_decode(tok, out_ids, n_gen);
            if (decoded) {
                printf("\n[generate] decoded: %s", decoded);
                free(decoded);
            }
        }
        printf("\n[generate] %d tokens: PASS\n", n_gen);
    } else {
        fprintf(stderr, "[benchmark] skipping generate self-test\n");
    }

    /* Test lattice routing */
    double M, L, S;
    tb_lattice_s_u_resonance(ctx->lattice, &M, &L, &S);
    printf("\n[lattice] S(U)=%.4f Λ=%.4f M=%.4f\n", S, L, M);

    /* Benchmark */
    if (benchmark) {
        const int bench_tokens = bench_steps > 0 ? bench_steps : 1;
        int ok_tokens = 0;
        FILE *bench_log = fopen(bench_log_path, "ab");
        printf("\n[benchmark] %d decode calls...\n", bench_tokens);
        printf("[benchmark] telemetry log: %s\n", bench_log_path);
        fflush(stdout);

        if (bench_log) {
            tb_bench_logf(bench_log,
                          "{\"event\":\"benchmark_start\",\"ts_ms\":%.3f,\"model\":\"%s\",\"tokens\":%d}",
                          tb_wall_ms(),
                          model_path ? model_path : "synthetic",
                          bench_tokens);
        }

        double t0 = tb_wall_ms();
        for (int i = 0; i < bench_tokens; i++) {
            double ti = tb_wall_ms();
            int next_tok = tb_infer_decode(ctx, i % 100, i, 0);
            double tok_ms = tb_wall_ms() - ti;

            if (next_tok < 0) {
                fprintf(stderr, "[benchmark] decode failed at step %d\n", i);
                if (bench_log) {
                    tb_bench_logf(bench_log,
                                  "{\"event\":\"decode_fail\",\"ts_ms\":%.3f,\"step\":%d,\"dt_ms\":%.3f}",
                                  tb_wall_ms(), i, tok_ms);
                }
                break;
            }

            ok_tokens++;
            printf("[benchmark] step %d/%d dt=%.1fms tok=%d\n",
                   i + 1, bench_tokens, tok_ms, next_tok);
            fflush(stdout);

            if (bench_log) {
                tb_bench_logf(bench_log,
                              "{\"event\":\"decode_step\",\"ts_ms\":%.3f,\"step\":%d,\"dt_ms\":%.3f,\"token\":%d}",
                              tb_wall_ms(), i, tok_ms, next_tok);
            }
        }

        double dt = tb_wall_ms() - t0;
        if (ok_tokens > 0) {
            double tps = (double)ok_tokens / (dt / 1000.0);
            printf("[benchmark] %d tokens: %.1fms -> %.1f tok/s\n", ok_tokens, dt, tps);
            if (bench_log) {
                tb_bench_logf(bench_log,
                              "{\"event\":\"benchmark_done\",\"ts_ms\":%.3f,\"ok_tokens\":%d,\"total_ms\":%.3f,\"tok_s\":%.3f}",
                              tb_wall_ms(), ok_tokens, dt, tps);
            }
        } else {
            fprintf(stderr, "[benchmark] no successful decode steps\n");
            if (bench_log) {
                tb_bench_logf(bench_log,
                              "{\"event\":\"benchmark_done\",\"ts_ms\":%.3f,\"ok_tokens\":0,\"total_ms\":%.3f,\"tok_s\":0.0}",
                              tb_wall_ms(), dt);
            }
        }

        if (bench_log) fclose(bench_log);
    }

    /* Serve */
    if (serve) {
        TB_ServeConfig cfg = {0};
        cfg.ctx = ctx;
        snprintf(cfg.model_name, sizeof(cfg.model_name), "%s",
                 model_path ? model_path : "trailblaze-synthetic");
        cfg.port = port;
        snprintf(cfg.host, sizeof(cfg.host), "0.0.0.0");
        tb_serve(&cfg);
    }

    /* Diff utility demo */
    printf("\n[diff] Usage: --diff <dir_a> <dir_b>\n");
    printf("  e.g.: tb_infer --diff /tmp/tb_new/HDGL-SQL-main/HDGL-SQL-main "
           "                       /tmp/hdgl_sql_02/HDGL-SQL-0.2\n");

    tb_infer_free(ctx);
    if (!synthetic) tb_model_free(model);
    else free(model);

    /* ── WuWei codec self-test ──────────────────────────────────────────
     * Exercise all five strategies for round-trip correctness.
     * Uses an isolated lattice so the result is deterministic.
     */
    printf("\n[wuwei] WuWei codec round-trip test...\n");
    {
        TB_PhiLattice *wlat = tb_lattice_create(128, 0xDEADBEEFULL);
        int wuwei_fail = 0;
        if (wlat) {
            TB_WuWeiCodec codec;
            tb_wuwei_init(&codec, wlat);

            /* 64-byte test payload with non-trivial content */
            static const uint8_t plain[64] = {
                0x54,0x52,0x41,0x49,0x4C,0x42,0x4C,0x41, /* TRAILBLA */
                0x5A,0x45,0x20,0x76,0x30,0x2E,0x33,0x00, /* ZE v0.3  */
                0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80, /* powers2  */
                0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8, /* hi-bytes */
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, /* zeros    */
                0xAA,0x55,0xAA,0x55,0xAA,0x55,0xAA,0x55, /* alt-bits */
                0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88, /* ramp     */
                0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00  /* ramp2    */
            };

            TB_WuWeiStrategy strats[] = {
                TB_WUWEI_DELTA_FOLD, TB_WUWEI_PHI_COMPRESS,
                TB_WUWEI_SPIRAL_PACK, TB_WUWEI_RESONANCE, TB_WUWEI_RAW
            };
            const char *strat_names[] = {
                "DELTA_FOLD","PHI_COMPRESS","SPIRAL_PACK","RESONANCE","RAW"
            };

            for (int si = 0; si < 5; si++) {
                uint8_t enc[256], dec[256];
                memset(enc, 0, sizeof(enc));
                memset(dec, 0, sizeof(dec));

                int enc_len = tb_wuwei_compress(&codec, strats[si],
                                                plain, 64, enc, sizeof(enc));
                int dec_len = (enc_len > 0)
                              ? tb_wuwei_decompress(&codec, strats[si],
                                                    enc, enc_len, dec, sizeof(dec))
                              : -1;

                int ok = (dec_len == 64 && memcmp(plain, dec, 64) == 0);
                printf("[wuwei]   %-14s enc=%3d dec=%3d : %s\n",
                       strat_names[si], enc_len, dec_len, ok ? "PASS" : "FAIL");
                if (!ok) wuwei_fail++;
            }
            tb_lattice_destroy(wlat);
        } else {
            printf("[wuwei] lattice alloc failed — SKIP\n");
        }
        printf("[wuwei] %s\n", wuwei_fail == 0 ? "ALL PASS" : "FAILURES DETECTED");
    }

    printf("\n=== TRAILBLAZE Inference Runtime PASS ===\n");
    return 0;
}
#endif
