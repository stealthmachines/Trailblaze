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

static void tb_vec_stats(const float *v, int n, float *l2_out, float *maxabs_out, int *bad_out) {
    double l2 = 0.0;
    float maxabs = 0.0f;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float x = v[i];
        if (!isfinite(x)) {
            bad++;
            continue;
        }
        l2 += (double)x * (double)x;
        float ax = fabsf(x);
        if (ax > maxabs) maxabs = ax;
    }
    if (l2_out) *l2_out = (float)sqrt(l2);
    if (maxabs_out) *maxabs_out = maxabs;
    if (bad_out) *bad_out = bad;
}

static void tb_nanhunt_stage(int pos, int layer_idx, const char *stage, const float *v, int n) {
    if (pos >= 8 || !v || n <= 0 || !stage) return;
    float l2 = 0.0f, maxabs = 0.0f;
    int bad = 0;
    tb_vec_stats(v, n, &l2, &maxabs, &bad);
    fprintf(stderr, "[nanhunt2 pos=%d L%d %s] bad=%d L2=%.4f maxabs=%.4f\n",
            pos, layer_idx, stage, bad, l2, maxabs);
    fflush(stderr);
}

/* Smoke gate — fires at every pipeline → transition.
 * Catches zero-collapse (missing tensor / unimplemented quant) and
 * norm-explosion before they corrupt downstream stages silently.
 * lo/hi are loose L2 bounds; pass 0/0 to skip range check. */
static void tb_smoke(int pos, int layer_idx, const char *stage,
                     const float *v, int n,
                     float lo, float hi) {
    if (!v || n <= 0) return;
    float l2 = 0.0f, maxabs = 0.0f; int bad = 0;
    tb_vec_stats(v, n, &l2, &maxabs, &bad);
    const char *verdict = "OK";
    if (bad > 0)             verdict = "FAIL:NAN";
    else if (l2 < 1e-9f)     verdict = "FAIL:ZERO";
    else if (hi > 0 && l2 > hi) verdict = "FAIL:EXPLODE";
    else if (lo > 0 && l2 < lo) verdict = "FAIL:DEAD";
    if (verdict[0]=='F' || pos < 4)
        fprintf(stderr, "[smoke pos=%d L%02d %-20s] L2=%9.2f maxabs=%8.3f  %s\n",
                pos, layer_idx, stage, l2, maxabs, verdict);
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

void tb_rope_apply(float *q, float *k, int head_dim, int pos,
                   float rope_base, int rotary_dim) {
    /* Partial RoPE: if rotary_dim < head_dim, rotate only the first
     * rotary_dim elements of each head and leave dims [rotary_dim..head_dim)
     * unchanged.  rotary_dim==0 means full rotation (all head_dim dims).
     * Required for Qwen3.5 which sets rope.dimension_count=64 on 256-dim heads.
     */
    int n_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    for (int i = 0; i < n_rot / 2; i++) {
        float theta = (float)pos / powf(rope_base, 2.0f * i / n_rot);
        float cos_t = cosf(theta), sin_t = sinf(theta);
        if (q) {
            float q0 = q[2*i], q1 = q[2*i+1];
            q[2*i]   = q0*cos_t - q1*sin_t;
            q[2*i+1] = q0*sin_t + q1*cos_t;
        }
        if (k) {
            float k0 = k[2*i], k1 = k[2*i+1];
            k[2*i]   = k0*cos_t - k1*sin_t;
            k[2*i+1] = k0*sin_t + k1*cos_t;
        }
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
            /* ── Margin-capped boost ───────────────────────────────────────
             * Root cause of top-1 flip: an unconditional additive boost of
             * _eff_alpha*2.0 (~0.26) in post-softmax space overrides the
             * gate whenever the gate margin is smaller than that value,
             * which is the common case for most real MoE layers.
             *
             * Fix: operate in pre-softmax (logit) space and cap the boost
             * so it cannot push hdgl_exp above the current gate top-1.
             * The HDGL signal can still promote a non-top expert within the
             * top-k set — it just can't override a clear gate preference.
             *
             * margin = logit_top1 - logit_hdgl_exp  (≥ 0 by definition)
             * max_boost = margin * _eff_alpha        (scales with uncertainty)
             * boost = min(_eff_alpha * 2.0, max_boost)
             *
             * When hdgl_exp IS the gate top-1 (margin=0): boost=0 (no-op).
             * When gate is fully uncertain (all logits equal, margin=0): boost=0.
             * When gate has slight preference (margin=0.5): boost ≤ 0.5*alpha.
             * HDGL influence is now strictly proportional to gate uncertainty.
             */
            float _margin = _top1_raw - scores[hdgl_exp];  /* ≥ 0 */
            float _max_boost = _margin * _eff_alpha;
            float _boost = _eff_alpha * 2.0f;
            if (_boost > _max_boost) _boost = _max_boost;
            scores[hdgl_exp] += _boost;

            /* Neighbour boosts: same margin cap, reduced weight */
            int nb1 = (hdgl_exp + 1) % n_experts;
            int nb2 = (hdgl_exp - 1 + n_experts) % n_experts;
            float _gate_amp = _top1_prob;   /* use softmax top1 as amplitude proxy */
            float _v39_nb   = expf(0.5f * sqrtf(_gate_amp)) - 1.0f;  /* ≥0 */
            float _nb_boost_raw = _eff_alpha * (0.3f + _v39_nb * 0.1f);
            float _nb_margin1 = _top1_raw - scores[nb1];
            float _nb_margin2 = _top1_raw - scores[nb2];
            float _nb_max1 = _nb_margin1 * _eff_alpha;
            float _nb_max2 = _nb_margin2 * _eff_alpha;
            scores[nb1] += (_nb_boost_raw < _nb_max1) ? _nb_boost_raw : _nb_max1;
            scores[nb2] += (_nb_boost_raw < _nb_max2) ? _nb_boost_raw : _nb_max2;
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

    /* ── Gate top-1 preservation clamp ─────────────────────────────────────
     * After all HDGL modifications (additive boost + semantic multiplier),
     * enforce that the gate's top-1 expert is not overridden unless the gate
     * was genuinely uncertain (top-1 prob ≤ 1/n_experts * 1.5, i.e. less than
     * 50% above uniform).  When the gate had a clear preference, swap the
     * HDGL-boosted top-1 back to the gate top-1 if they differ.
     *
     * This is the authoritative single-point enforcement of the invariant:
     *   HDGL can influence the TOP-K SET and WEIGHTS, but cannot override
     *   a confident gate's primary expert choice.
     */
    {
        /* Find the gate top-1 (recorded before HDGL mods as _top1_raw index) */
        int _gate_top1 = 0;
        for (int _e = 1; _e < n_experts; _e++)
            if (gate_logits[_e] > gate_logits[_gate_top1]) _gate_top1 = _e;

        /* Find the post-HDGL top-1 */
        int _hdgl_top1 = 0;
        for (int _e = 1; _e < n_experts; _e++)
            if (scores[_e] > scores[_hdgl_top1]) _hdgl_top1 = _e;

        /* Uniform probability = 1/n_experts; threshold = 1.5x uniform */
        float _uniform = 1.0f / (float)n_experts;
        float _threshold = _uniform * 1.5f;

        /* Compute gate top-1 probability from original logits for confidence check */
        float _gt1_max = gate_logits[_gate_top1];
        float _gt1_sum = 0.0f;
        for (int _e = 0; _e < n_experts; _e++) _gt1_sum += expf(gate_logits[_e] - _gt1_max);
        float _gate_top1_prob = 1.0f / _gt1_sum;

        if (_hdgl_top1 != _gate_top1 && _gate_top1_prob > _threshold) {
            /* Gate was confident; HDGL flipped the top-1 — swap scores back */
            float _tmp = scores[_gate_top1];
            scores[_gate_top1] = scores[_hdgl_top1] + 1e-6f; /* ensure strict top-1 */
            scores[_hdgl_top1] = _tmp;
        }
    }

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
    ctx->use_hdgl_semantic = 0;  /* Semantic octave multiplier disabled by default.
                                  * It applies a per-token multiplicative boost to
                                  * post-softmax scores that can flip rankings even
                                  * when the HDGL additive boost is margin-capped.
                                  * Enable explicitly with a dedicated flag if needed. */
    ctx->hdgl_alpha        = hdgl_alpha;
    ctx->max_new_tokens    = 512;
    ctx->temperature       = 0.0f;  /* greedy for diagnostics */
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
        free(ctx->session_kvs[s].ssm_h);
        ctx->session_kvs[s].ssm_h = NULL;
        free(ctx->session_kvs[s].conv_h);
        ctx->session_kvs[s].conv_h = NULL;
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

/* GQA attention with separate Q/K and V head dims (Qwen3.5: HD_k=256, HD_v=128).
 * q[n_heads*HD_k], k[NK_k*HD_k], v[NK_v*HD_v], out[n_heads*HD_v]. */
static void tb_attention_split_kv(
    const float *q, const float *k, const float *v,
    TB_KVCache *cache, int layer_idx,
    int n_heads, int NK_k, int HD_k,
    int NK_v, int HD_v,
    int pos, float *out)
{
#define ATTN_TILE 32
    float scale = 1.0f / sqrtf((float)HD_k);
    int   total = pos + 1;

    for (int h = 0; h < NK_k; h++) {
        float *kslot = cache->keys[layer_idx]
                       + (size_t)h * cache->max_seq * HD_k
                       + (size_t)pos * HD_k;
        memcpy(kslot, k + h * HD_k, HD_k * sizeof(float));
    }
    for (int h = 0; h < NK_v; h++) {
        float *vslot = cache->vals[layer_idx]
                       + (size_t)h * cache->max_seq * HD_v
                       + (size_t)pos * HD_v;
        memcpy(vslot, v + h * HD_v, HD_v * sizeof(float));
    }
    if (layer_idx == 0) cache->seq_len++;

    float tile_scores[ATTN_TILE];
    for (int h = 0; h < n_heads; h++) {
        int kv_k = (NK_k > 0) ? (h * NK_k / n_heads) : 0;
        int kv_v = (NK_v > 0) ? (h * NK_v / n_heads) : 0;
        const float *qh  = q + (size_t)h * HD_k;
        float       *outh = out + (size_t)h * HD_v;
        memset(outh, 0, HD_v * sizeof(float));

        float m_i = -1e38f, l_i = 0.0f;
        for (int t0 = 0; t0 < total; t0 += ATTN_TILE) {
            int tend = t0 + ATTN_TILE; if (tend > total) tend = total;
            int tn = tend - t0;
            for (int ti = 0; ti < tn; ti++) {
                const float *kt = cache->keys[layer_idx]
                    + (size_t)kv_k * cache->max_seq * HD_k
                    + (size_t)(t0 + ti) * HD_k;
                float dot = 0.0f;
                for (int d = 0; d < HD_k; d++) dot += qh[d] * kt[d];
                tile_scores[ti] = dot * scale;
            }
            float m_new = m_i;
            for (int ti = 0; ti < tn; ti++) if (tile_scores[ti] > m_new) m_new = tile_scores[ti];
            float rsc = expf(m_i - m_new), l_new = rsc * l_i;
            for (int ti = 0; ti < tn; ti++) { tile_scores[ti] = expf(tile_scores[ti] - m_new); l_new += tile_scores[ti]; }
            for (int d = 0; d < HD_v; d++) outh[d] *= rsc;
            for (int ti = 0; ti < tn; ti++) {
                const float *vt = cache->vals[layer_idx]
                    + (size_t)kv_v * cache->max_seq * HD_v
                    + (size_t)(t0 + ti) * HD_v;
                float a = tile_scores[ti];
                for (int d = 0; d < HD_v; d++) outh[d] += a * vt[d];
            }
            m_i = m_new; l_i = l_new;
        }
        if (l_i > 0.0f) { float inv_l = 1.0f/l_i; for (int d=0;d<HD_v;d++) outh[d]*=inv_l; }
    }
#undef ATTN_TILE
}

int tb_layer_forward(
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
    float *xn = (float*)malloc(H * sizeof(float));
    float *ones = (float*)malloc(H * sizeof(float));
    for (int i=0;i<H;i++) ones[i]=1.0f;

    TNSR("blk.%d.attn_norm.weight", layer_idx);
    const TB_GGUFTensorInfo *norm_w_t = tb_gguf_find_tensor(m, tname);
    if (norm_w_t && m->weights_data) {
        float *w32 = (float*)malloc(H*sizeof(float));
        const uint16_t *w16 = (const uint16_t*)tb_gguf_tensor_data(m, norm_w_t);
        if (norm_w_t->qtype == 1)
            for (int i=0;i<H;i++) { uint32_t b=(uint32_t)w16[i]<<16; memcpy(&w32[i],&b,4); }
        else if (norm_w_t->qtype == 30)
            for (int i=0;i<H;i++) w32[i] = tb_bf16_to_f32_infer(w16[i]);
        else { memcpy(w32, tb_gguf_tensor_data(m,norm_w_t), H*sizeof(float)); }
        tb_rms_norm(x, w32, xn, H, eps);
        free(w32);
    } else {
        tb_rms_norm(x, ones, xn, H, eps);
    }
    tb_vlog("layer=%d stage=attn_norm done", layer_idx);
    tb_smoke(pos, layer_idx, "1:attn_norm→qkv",    xn, H,       10.0f, 500.0f);

    /* ── 2. QKV + Attention or DeltaNet (local vs. global layers) ─────── */
    /* Global layers: L % full_attn_interval == full_attn_interval-1
     * Local layers:  everything else — use Gated DeltaNet, not softmax attention */
    int is_local = (m->full_attn_interval > 0 &&
                    (layer_idx % m->full_attn_interval != m->full_attn_interval - 1));

    float *q        = NULL;
    float *k        = NULL;
    float *v        = NULL;
    float *attn_out = NULL;
    float *attn_proj = (float*)calloc(H, sizeof(float));

    if (!is_local) {
        /* ── GLOBAL LAYER: standard GQA softmax attention ──────────────── */
        /* Try fused QKV first, fall back to separate q/k/v */
        TNSR("blk.%d.attn_qkv.weight", layer_idx);
        const TB_GGUFTensorInfo *wqkv_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.attn_q.weight", layer_idx);
        const TB_GGUFTensorInfo *wq_t = wqkv_t ? NULL : tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.attn_k.weight", layer_idx);
        const TB_GGUFTensorInfo *wk_t = wqkv_t ? NULL : tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.attn_v.weight", layer_idx);
        const TB_GGUFTensorInfo *wv_t = wqkv_t ? NULL : tb_gguf_find_tensor(m, tname);

        /* Note: Q weight has M=8192 but NH*HD=16*256=4096. We use only the first
         * NH*HD rows of W_q (the actual attention Q). The extra rows in W_q are
         * architecture-specific and not needed for basic inference. NK is always
         * derived from the K weight shape to handle GQA correctly. */
        if (wk_t && wk_t->n_dims >= 2 && HD > 0) {
            int nk_actual = (int)(tb_gguf_tensor_nelems(wk_t) / ((int64_t)H * HD));
            if (nk_actual > 0 && nk_actual != NK) NK = nk_actual;
        }
        q        = (float*)calloc(NH*HD, sizeof(float));
        k        = (float*)calloc(NK*HD, sizeof(float));
        v        = (float*)calloc(NK*HD, sizeof(float));
        attn_out = (float*)calloc(NH*HD, sizeof(float));

        if (wqkv_t && m->weights_data) {
            int QKV_M = (NH + 2*NK) * HD;
            float *qkv_buf = (float*)calloc(QKV_M, sizeof(float));
            tb_gguf_tensor_matvec(m, wqkv_t, QKV_M, H, xn, qkv_buf);
            memcpy(q, qkv_buf,              sizeof(float)*NH*HD);
            memcpy(k, qkv_buf + NH*HD,      sizeof(float)*NK*HD);
            memcpy(v, qkv_buf + (NH+NK)*HD, sizeof(float)*NK*HD);
            free(qkv_buf);
        } else if (wq_t && wk_t && wv_t && m->weights_data) {
            if (pos < 2)
                fprintf(stderr, "[sep_qkv L%d] q=%d k=%d v=%d NH=%d NK=%d HD=%d\n",
                        layer_idx, wq_t->qtype, wk_t->qtype, wv_t->qtype, NH, NK, HD);
            tb_gguf_tensor_matvec(m, wq_t, NH*HD, H, xn, q);
            tb_gguf_tensor_matvec(m, wk_t, NK*HD, H, xn, k);
            tb_gguf_tensor_matvec(m, wv_t, NK*HD, H, xn, v);
        } else {
            for (int i=0;i<NH*HD && i<H;i++) q[i]=xn[i];
            for (int i=0;i<NK*HD && i<H;i++) { k[i]=xn[i]; v[i]=xn[i]; }
        }
        tb_vlog("layer=%d stage=qkv done", layer_idx);

        /* Qwen3 per-head QK-norm */
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
                            float sq = eps;
                            for (int i=0;i<HD;i++) sq += qh[i]*qh[i];
                            float sc = 1.0f / sqrtf(sq/HD);
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
                            float sk = eps;
                            for (int i=0;i<HD;i++) sk += kh[i]*kh[i];
                            float sc = 1.0f / sqrtf(sk/HD);
                            for (int i=0;i<HD;i++) kh[i] = kh[i]*sc*w32[i];
                        }
                    }
                    free(w32);
                }
            }
        }

        /* RoPE */
        for (int h=0;h<NH;h++) tb_rope_apply(q+h*HD, NULL,   HD, pos, m->rope_base, m->rope_rotary_dim);
        for (int h=0;h<NK;h++) tb_rope_apply(NULL,   k+h*HD, HD, pos, m->rope_base, m->rope_rotary_dim);
        tb_vlog("layer=%d stage=rope done", layer_idx);

        /* Softmax attention */
        if (kv) {
            tb_attention(q, k, v, kv, layer_idx, NH, NK, HD, pos, attn_out);
        } else {
            memcpy(attn_out, q, NH*HD*sizeof(float));
        }
        tb_vlog("layer=%d stage=attention done", layer_idx);

        /* Output projection: attn_output.weight (global layers only) */
        TNSR("blk.%d.attn_output.weight", layer_idx);
        const TB_GGUFTensorInfo *wo_t = tb_gguf_find_tensor(m, tname);
        if (!wo_t) { TNSR("blk.%d.attn_out.weight", layer_idx); wo_t = tb_gguf_find_tensor(m, tname); }
        if (wo_t && m->weights_data) {
            /* Use wo_t's actual K dim: handles GQA where W_o only maps partial heads */
            int wo_K = (H > 0) ? (int)(tb_gguf_tensor_nelems(wo_t) / (int64_t)H) : NH*HD;
            if (wo_K <= 0) wo_K = NH*HD;
            tb_gguf_tensor_matvec(m, wo_t, H, wo_K, attn_out, attn_proj);
        } else {
            int copy_sz = NH*HD < H ? NH*HD : H;
            for (int i=0;i<copy_sz;i++) attn_proj[i] = attn_out[i];
        }
        tb_vlog("layer=%d stage=attn_out done", layer_idx);

    } else {
        /* ── LOCAL LAYER: Gated DeltaNet ────────────────────────────────── */
        /*
         * Architecture (Qwen3.5):
         *   xn → wqkv [H→inner]     → QKV (Q:key_dim, K:key_dim, V:value_dim)
         *   xn → wgate [H→H]        → Z (output gate)
         *   xn → ssm_alpha [H→nh]   + dt_bias → softplus → * ssm_a → g_t (per-head)
         *   xn → ssm_beta  [H→nh]   → sigmoid → beta_t (per-head)
         *   conv1d(QKV) → silu      → Q_c, K_c, V_c
         *   L2-norm Q_c, K_c per head
         *   DeltaNet recurrence per head:
         *     S = g*S + k ⊗ (β*(v − g*S^Tk))
         *     out = S^T q
         *   gated_rms_norm(out, ssm_norm) * silu(Z) → ssm_out.weight → attn_proj
         */
        TNSR("blk.%d.attn_qkv.weight", layer_idx);
        const TB_GGUFTensorInfo *wqkv_t  = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.attn_gate.weight", layer_idx);
        const TB_GGUFTensorInfo *wgate_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ssm_a", layer_idx);
        const TB_GGUFTensorInfo *ssm_a_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ssm_alpha.weight", layer_idx);
        const TB_GGUFTensorInfo *alpha_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ssm_beta.weight", layer_idx);
        const TB_GGUFTensorInfo *beta_wt = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ssm_conv1d.weight", layer_idx);
        const TB_GGUFTensorInfo *conv_t  = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ssm_norm.weight", layer_idx);
        const TB_GGUFTensorInfo *snorm_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.ssm_out.weight", layer_idx);
        const TB_GGUFTensorInfo *sout_t  = tb_gguf_find_tensor(m, tname);

        /* Dimensions — must match allocation in tb_infer_decode:
         *   n_h   = ssm_a.shape[0] = 32 heads
         *   hdk   = 64  (key head dim)
         *   hdv   = 128 (value head dim = ssm_state_size = ssm_inner_size/n_h)
         *   vdim  = n_h * hdv = 4096 (value dim, = ssm_inner_size)
         *   kdim  = n_h * hdk = 2048 (key/query dim)
         *   qkv_total = 2*kdim + vdim = 8192 (wqkv output size, conv channels) */
        int n_h   = ctx->cur_ssm_n_heads  > 0 ? ctx->cur_ssm_n_heads  : 32;
        int hdk   = ctx->cur_ssm_head_d_k > 0 ? ctx->cur_ssm_head_d_k : 64;
        int hdv   = ctx->cur_ssm_head_d_v > 0 ? ctx->cur_ssm_head_d_v : 128;
        int vdim  = n_h * hdv;         /* 4096 = ssm_inner_size */
        int kdim  = n_h * hdk;         /* 2048 */
        int qkv_total = 2*kdim + vdim; /* 8192 = wqkv output size */
        int inner = m->ssm_inner_size > 0 ? m->ssm_inner_size : vdim; /* 4096 */
        int chist = ctx->cur_conv_hist  > 0 ? ctx->cur_conv_hist  : 3;
        int cinn  = ctx->cur_conv_inner > 0 ? ctx->cur_conv_inner : qkv_total;

        /* --- Step 1: QKV projection [H → qkv_total] --- */
        float *qkv_buf = (float*)calloc(qkv_total, sizeof(float));
        if (wqkv_t && m->weights_data)
            tb_gguf_tensor_matvec(m, wqkv_t, qkv_total, H, xn, qkv_buf);

        /* --- Step 2: Gate Z [H → H] --- */
        float *Z_gate = (float*)calloc(H, sizeof(float));
        if (wgate_t && m->weights_data)
            tb_gguf_tensor_matvec(m, wgate_t, H, H, xn, Z_gate);

        /* --- Step 3: Per-head alpha/beta → g_t, beta_t --- */
        float *g_t    = (float*)calloc(n_h, sizeof(float));
        float *beta_t = (float*)calloc(n_h, sizeof(float));
        for (int h=0; h<n_h; h++) { g_t[h]=0.9f; beta_t[h]=1.0f; } /* defaults */
        if (alpha_t && beta_wt && ssm_a_t && m->weights_data) {
            float *alpha_proj = (float*)calloc(n_h, sizeof(float));
            float *beta_proj  = (float*)calloc(n_h, sizeof(float));
            tb_gguf_tensor_matvec(m, alpha_t, n_h, H, xn, alpha_proj);
            tb_gguf_tensor_matvec(m, beta_wt,  n_h, H, xn, beta_proj);

            const float *a_log = (const float*)tb_gguf_tensor_data(m, ssm_a_t);
            float dt_buf[128] = {0};
            TNSR("blk.%d.ssm_dt.bias", layer_idx);
            const TB_GGUFTensorInfo *dt_t = tb_gguf_find_tensor(m, tname);
            if (dt_t && m->weights_data) {
                int nd = n_h < 128 ? n_h : 128;
                const float *dtr = (const float*)tb_gguf_tensor_data(m, dt_t);
                for (int i=0;i<nd;i++) dt_buf[i] = dtr[i];
            }
            for (int h=0; h<n_h; h++) {
                float a_in = alpha_proj[h] + dt_buf[h];
                /* softplus: numerically stable form */
                float sp = (a_in >= 0.0f) ? a_in + logf(1.0f + expf(-a_in))
                                          : logf(1.0f + expf(a_in));
                /* g = exp(-a_log * sp): a_log > 0, sp > 0 → g ∈ (0,1) */
                g_t[h] = expf(-(a_log[h] > 0.0f ? a_log[h] : -a_log[h]) * sp);
                beta_t[h] = 1.0f / (1.0f + expf(-beta_proj[h]));
            }
            free(alpha_proj); free(beta_proj);
        }

        /* --- Step 4: Causal depthwise Conv1d over all QKV channels --- */
        /* conv_t GGUF shape: [K=4, M=8192] — data stored as M rows × K cols (GGUF convention)
         * Element [channel c, kern k] = cw[c*ksz + k], ksz = chist+1 = 4
         * History buffer: ch_ptr[hist_slot * cinn + channel] */
        float *conv_out = (float*)calloc(qkv_total, sizeof(float));
        float *ch_ptr   = ctx->cur_conv_h
                        ? ctx->cur_conv_h + (size_t)layer_idx * chist * cinn
                        : NULL;
        if (conv_t && m->weights_data && ch_ptr && cinn == qkv_total) {
            const float *cw = (const float*)tb_gguf_tensor_data(m, conv_t);
            int ksz = chist + 1;
            for (int c=0; c<qkv_total; c++) {
                float vc = 0.0f;
                for (int k=0; k<chist; k++)
                    vc += ch_ptr[k*cinn + c] * cw[c*ksz + k];
                vc += qkv_buf[c] * cw[c*ksz + chist];
                conv_out[c] = vc;
            }
            if (chist >= 2)
                memmove(ch_ptr, ch_ptr + cinn, (chist-1)*cinn*sizeof(float));
            memcpy(ch_ptr + (chist-1)*cinn, qkv_buf, cinn*sizeof(float));
        } else {
            memcpy(conv_out, qkv_buf, qkv_total * sizeof(float));
            if (ch_ptr) {
                if (chist >= 2)
                    memmove(ch_ptr, ch_ptr + cinn, (chist-1)*cinn*sizeof(float));
                memcpy(ch_ptr + (chist-1)*cinn, qkv_buf,
                       (cinn < qkv_total ? cinn : qkv_total)*sizeof(float));
            }
        }
        /* SiLU on conv output */
        for (int c=0; c<qkv_total; c++)
            conv_out[c] *= 1.0f / (1.0f + expf(-conv_out[c]));

        free(qkv_buf);

        /* Split conv_out[qkv_total] → Q[kdim], K[kdim], V[vdim] */
        float *Q_buf = conv_out;          /* [0 : kdim)   = [n_h*hdk] */
        float *K_buf = conv_out + kdim;   /* [kdim : 2*kdim) */
        float *V_buf = conv_out + 2*kdim; /* [2*kdim : qkv_total) = [vdim] */

        /* --- Step 5: L2-normalize Q, K per head --- */
        for (int h=0; h<n_h; h++) {
            float *qh = Q_buf + h*hdk, *kh = K_buf + h*hdk;
            float sq=1e-8f, sk=1e-8f;
            for (int i=0;i<hdk;i++) { sq += qh[i]*qh[i]; sk += kh[i]*kh[i]; }
            float iscq = 1.0f/sqrtf(sq), isck = 1.0f/sqrtf(sk);
            for (int i=0;i<hdk;i++) { qh[i]*=iscq; kh[i]*=isck; }
        }

        /* --- Step 6: DeltaNet recurrence per head --- */
        /* S[h]: [hdk × hdv] row-major — updated IN PLACE in session state */
        float *dn_out = (float*)calloc(vdim, sizeof(float));
        if (ctx->cur_ssm_h) {
            for (int h=0; h<n_h; h++) {
                float *qh = Q_buf + h*hdk;
                float *kh = K_buf + h*hdk;
                float *vh = V_buf + h*hdv;
                float *Sh = ctx->cur_ssm_h
                          + (size_t)layer_idx * n_h * hdk * hdv
                          + (size_t)h * hdk * hdv;
                float gh  = g_t[h];
                float bh  = beta_t[h];

                /* prediction = g * S^T k ∈ [hdv] */
                float *pred = (float*)calloc(hdv, sizeof(float));
                for (int j=0; j<hdk; j++) {
                    float gkj = gh * kh[j];
                    for (int i=0; i<hdv; i++)
                        pred[i] += Sh[j*hdv + i] * gkj;
                }
                /* delta_v = beta * (v - pred) ∈ [hdv] */
                for (int i=0; i<hdv; i++) pred[i] = bh * (vh[i] - pred[i]);

                /* S = g*S + k ⊗ delta_v  (in-place) */
                for (int j=0; j<hdk; j++) {
                    float kj = kh[j];
                    for (int i=0; i<hdv; i++)
                        Sh[j*hdv + i] = gh * Sh[j*hdv + i] + kj * pred[i];
                }

                /* out_h = S^T q_h ∈ [hdv] */
                float *oh = dn_out + h*hdv;
                for (int j=0; j<hdk; j++) {
                    float qj = qh[j];
                    for (int i=0; i<hdv; i++)
                        oh[i] += Sh[j*hdv + i] * qj;
                }
                free(pred);
            }
        } else {
            /* No state: approximate with V */
            if (vdim <= H) memcpy(dn_out, V_buf, vdim*sizeof(float));
        }

        free(conv_out);  /* frees Q_buf/K_buf/V_buf too (same allocation) */

        /* --- Step 7: Gated RMSNorm per head + silu(Z) gate --- */
        /* ssm_norm.weight [hdv=128]: shared per-element norm weights */
        float *snorm_w = NULL;
        if (snorm_t && m->weights_data) {
            snorm_w = (float*)malloc(hdv * sizeof(float));
            if (snorm_t->qtype == 0)
                memcpy(snorm_w, tb_gguf_tensor_data(m, snorm_t), hdv*sizeof(float));
            else {
                const uint16_t *sw = (const uint16_t*)tb_gguf_tensor_data(m, snorm_t);
                if (snorm_t->qtype == 30)
                    for (int i=0;i<hdv;i++) snorm_w[i] = tb_bf16_to_f32_infer(sw[i]);
                else  /* F32 fallback */
                    memcpy(snorm_w, tb_gguf_tensor_data(m, snorm_t), hdv*sizeof(float));
            }
        }
        for (int h=0; h<n_h; h++) {
            float *oh = dn_out + h*hdv;
            float rms_sq = eps;
            for (int i=0;i<hdv;i++) rms_sq += oh[i]*oh[i];
            float sc = 1.0f / sqrtf(rms_sq / (float)hdv);
            float *zh = Z_gate + h*hdv;
            for (int i=0;i<hdv;i++) {
                float ni = oh[i] * sc;
                if (snorm_w) ni *= snorm_w[i];
                float zi = zh[i];
                oh[i] = ni * (zi / (1.0f + expf(-zi)));  /* silu gate */
            }
        }
        if (snorm_w) free(snorm_w);

        /* --- Step 8: Output projection ssm_out.weight [H×H] --- */
        if (sout_t && m->weights_data)
            tb_gguf_tensor_matvec(m, sout_t, H, vdim, dn_out, attn_proj);
        else if (vdim == H)
            memcpy(attn_proj, dn_out, H*sizeof(float));

        if (pos < 2 && layer_idx < 2)
            fprintf(stderr, "[dn_L%d pos=%d] n_h=%d hdk=%d hdv=%d qkv=%d g0=%.4f b0=%.4f\n",
                    layer_idx, pos, n_h, hdk, hdv, qkv_total, g_t[0], beta_t[0]);

        free(dn_out); free(Z_gate); free(g_t); free(beta_t);
        tb_vlog("layer=%d stage=deltanet done", layer_idx);
    }

    float *x2 = (float*)malloc(H*sizeof(float));
    for (int i=0;i<H;i++) x2[i] = x[i] + attn_proj[i];
    tb_smoke(pos, layer_idx, "7:resid_attn→ffn",   x2, H,           0.1f, 500000.0f);

    /* ── 3. FFN / MoE ─────────────────────────────────────────────────── */
    float *xn2 = (float*)malloc(H*sizeof(float));
    TNSR("blk.%d.ffn_norm.weight", layer_idx);
    const TB_GGUFTensorInfo *fn_t = tb_gguf_find_tensor(m, tname);
    /* Qwen3.5 names this "post_attention_norm.weight" */
    if (!fn_t) { TNSR("blk.%d.post_attention_norm.weight", layer_idx); fn_t = tb_gguf_find_tensor(m, tname); }
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
    tb_smoke(pos, layer_idx, "8:ffn_norm→ffn",     xn2, H,          10.0f, 500.0f);

    float *ffn_out = (float*)calloc(H, sizeof(float));

    if (m->n_experts > 0) {
        /* MoE: gate → top-k routing → expert forward → weighted sum */
        TNSR("blk.%d.ffn_gate_inp.weight", layer_idx);
        const TB_GGUFTensorInfo *gate_t = tb_gguf_find_tensor(m, tname);
        float *gate_logits = (float*)calloc(m->n_experts, sizeof(float));
        if (gate_t && m->weights_data) {
            /* Fix 1: use the standard GGUF dequant path (handles Q4_K, Q6_K, Q8_0, …).
             * The previous code assumed a custom extract_weights.py packed Q4 layout
             * that does not match real GGUF files, producing garbage gate logits and
             * therefore wrong expert selection at every MoE layer. */
            tb_gguf_tensor_matvec(m, gate_t, m->n_experts, H, xn2, gate_logits);
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
                    /* Bytes per weight row in a packed expert tensor.
                     * Each entry is: (K / block_elements) * bytes_per_block
                     * Values verified against ggml-quants.h sizeof(block_*) :
                     *   F32 (0)    : K*4
                     *   F16 (1)    : K*2          was: K*4 (2x wrong) — FIXED
                     *   Q4_0 (2)   : (K/32)*18
                     *   Q8_0 (8)   : (K/32)*34
                     *   Q2_K (10)  : (K/256)*84
                     *   Q3_K (11)  : (K/256)*110
                     *   Q4_K (12)  : (K/256)*144
                     *   Q5_K (13)  : (K/256)*176
                     *   Q6_K (14)  : (K/256)*210
                     *   Q8_K (15)  : (K/256)*292
                     *   IQ4_NL(20) : (K/32)*18    was: K*4 (~7x wrong) — FIXED
                     *   IQ2_XS(21) : (K/256)*74   was: (K/256)*110 — FIXED
                     *   BF16 (30)  : K*2          was: K*4 (2x wrong) — FIXED
                     */
                    #define _RB(qt,K) (\
                        (qt)== 0 ? (size_t)(K)*4          : \
                        (qt)== 1 ? (size_t)(K)*2          : \
                        (qt)== 2 ? (size_t)((K)/32)*18    : \
                        (qt)== 8 ? (size_t)((K)/32)*34    : \
                        (qt)==10 ? (size_t)((K)/256)*84   : \
                        (qt)==11 ? (size_t)((K)/256)*110  : \
                        (qt)==12 ? (size_t)((K)/256)*144  : \
                        (qt)==13 ? (size_t)((K)/256)*176  : \
                        (qt)==14 ? (size_t)((K)/256)*210  : \
                        (qt)==15 ? (size_t)((K)/256)*292  : \
                        (qt)==20 ? (size_t)((K)/32)*18    : \
                        (qt)==21 ? (size_t)((K)/256)*110  : \
                        (qt)==23 ? (size_t)((K)/256)*136  : \
                        (qt)==30 ? (size_t)(K)*2          : \
                                   (size_t)(K)*4)
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
            double _tffn = (layer_idx <= 5) ? tb_wall_ms() : 0.0;
            tb_gguf_tensor_matvec(m, ffn_g, ffn_dim, H, xn2, g_out);
            tb_gguf_tensor_matvec(m, ffn_u, ffn_dim, H, xn2, u_out);
            tb_swiglu(g_out, u_out, act, ffn_dim);
            tb_gguf_tensor_matvec(m, ffn_d, H, ffn_dim, act, ffn_out);
            if (layer_idx <= 5)
                fprintf(stderr, "[timing L%d] ffn gate+up+down: %.1fms  ffn_dim=%d gate_qtype=%d up_qtype=%d down_qtype=%d gpu=%s\n",
                        layer_idx, tb_wall_ms()-_tffn, ffn_dim,
                        ffn_g->qtype, ffn_u->qtype, ffn_d->qtype,
                        ffn_g->d_data?"Y":"N");
            tb_vlog("layer=%d stage=ffn_dense down_matvec done", layer_idx);
        } else {
            memcpy(ffn_out, xn2, H*sizeof(float));
        }
        free(g_out); free(u_out); free(act);
    }
    tb_vlog("layer=%d stage=ffn done", layer_idx);
    tb_smoke(pos, layer_idx, "9:ffn→resid",        ffn_out, H,      0.1f, 500000.0f);

    /* Residual */
    for (int i=0;i<H;i++) out[i] = x2[i] + ffn_out[i];
    tb_smoke(pos, layer_idx, "10:resid_out→next",  out, H,          0.1f, 1000000.0f);

    /* Per-layer diagnostics for the last prompt position (pos==5) */
    if (pos == 5) {
        float ss = 0.0f;
        for (int i=0;i<H;i++) ss += out[i]*out[i];
        float norm_out = sqrtf(ss / H);
        /* Also measure attn contribution */
        float sa = 0.0f;
        for (int i=0;i<H;i++) sa += attn_proj[i]*attn_proj[i];
        fprintf(stderr, "[diag p5 L%d] out_rms=%.4f attn_rms=%.4f is_local=%d\n",
                layer_idx, norm_out, sqrtf(sa/H), is_local);
    }

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
            /* DeltaNet recurrent state + conv1d history — allocated for hybrid models.
             * Actual Qwen3.5-9B tensor shapes (verified from GGUF):
             *   ssm_a.shape        = [32]      → n_heads = 32
             *   ssm_alpha/beta     = [H→32]    → per-head alpha/beta output dim = n_heads
             *   ssm_norm.weight    = [128]      → head_d_v = 128
             *   ssm_inner_size     = 4096       = n_heads * head_d_v (value dim)
             *   attn_qkv.shape     = [H, 8192]  → QKV total = 8192
             *   head_d_k           = (8192 - n_heads*head_d_v) / (2*n_heads) = 64
             *   conv1d.shape       = [4, 8192]  → conv channels = 8192 */
            if (m->full_attn_interval > 0 && m->ssm_inner_size > 0) {
                /* n_heads from ssm_a tensor (32 for Qwen3.5-9B) */
                int n_heads = 32;
                const TB_GGUFTensorInfo *ssm_a_t = tb_gguf_find_tensor(m, "blk.0.ssm_a");
                if (ssm_a_t && ssm_a_t->shape[0] > 0)
                    n_heads = (int)ssm_a_t->shape[0];

                /* head_d_v = ssm_inner_size / n_heads = 4096/32 = 128 */
                int head_d_v = (n_heads > 0) ? (m->ssm_inner_size / n_heads) : 128;
                if (head_d_v <= 0) head_d_v = 128;

                /* QKV total from wqkv weight shape, fallback to 2*ssm_inner_size */
                int qkv_total = 2 * m->ssm_inner_size;
                const TB_GGUFTensorInfo *wqkv_t = tb_gguf_find_tensor(m, "blk.0.attn_qkv.weight");
                if (wqkv_t && wqkv_t->n_dims >= 2 && wqkv_t->shape[1] > 0)
                    qkv_total = (int)wqkv_t->shape[1];

                /* head_d_k = (qkv_total - n_heads*head_d_v) / (2*n_heads) = 64 */
                int head_d_k = (qkv_total - n_heads * head_d_v) / (2 * n_heads);
                if (head_d_k <= 0) head_d_k = 64;

                /* conv channels = qkv_total = 8192 */
                int conv_inner = qkv_total;
                const TB_GGUFTensorInfo *conv_t = tb_gguf_find_tensor(m, "blk.0.ssm_conv1d");
                if (conv_t && conv_t->n_dims >= 2 && conv_t->shape[1] > 0)
                    conv_inner = (int)conv_t->shape[1];

                skv->ssm_n_layers  = m->n_layers;
                skv->ssm_n_heads   = n_heads;
                skv->ssm_head_d_k  = head_d_k;
                skv->ssm_head_d_v  = head_d_v;
                size_t ssm_bytes   = (size_t)m->n_layers * n_heads * head_d_k * head_d_v * sizeof(float);
                skv->ssm_h = (float*)calloc(1, ssm_bytes);

                int conv_hist  = (m->ssm_conv_kernel > 1) ? (m->ssm_conv_kernel - 1) : 3;
                skv->conv_hist       = conv_hist;
                skv->conv_inner_size = conv_inner;
                size_t conv_bytes    = (size_t)m->n_layers * conv_hist * conv_inner * sizeof(float);
                skv->conv_h = (float*)calloc(1, conv_bytes);

                fprintf(stderr, "[tb_infer] DeltaNet state: %d layers x %d heads x [d_k=%d x d_v=%d] = %.1f MB\n",
                        m->n_layers, n_heads, head_d_k, head_d_v, ssm_bytes/(1024.0*1024.0));
                fprintf(stderr, "[tb_infer] Conv1d state:   %d layers x %d hist x %d ch = %.1f MB\n",
                        m->n_layers, conv_hist, conv_inner, conv_bytes/(1024.0*1024.0));
            }
        }
        if (skv->kv) {
            /* Reset cache when starting a new sequence */
            if (pos == 0 || !skv->active) {
                skv->kv->seq_len = 0;
                skv->active      = 1;
                skv->epoch       = ctx->lattice->epoch;
                /* Zero DeltaNet state and conv state at start of new sequence */
                if (skv->ssm_h)
                    memset(skv->ssm_h, 0,
                           (size_t)skv->ssm_n_layers * skv->ssm_n_heads
                           * skv->ssm_head_d_k * skv->ssm_head_d_v * sizeof(float));
                if (skv->conv_h)
                    memset(skv->conv_h, 0,
                           (size_t)skv->ssm_n_layers * skv->conv_hist
                           * skv->conv_inner_size * sizeof(float));
            }
            kv = skv->kv;
        }
        /* Expose current-session DeltaNet state via ctx for tb_layer_forward */
        ctx->cur_ssm_h       = skv->ssm_h;
        ctx->cur_conv_h      = skv->conv_h;
        ctx->cur_ssm_n_heads = skv->ssm_n_heads > 0 ? skv->ssm_n_heads : 32;
        ctx->cur_ssm_head_d_k= skv->ssm_head_d_k > 0 ? skv->ssm_head_d_k : 64;
        ctx->cur_ssm_head_d_v= skv->ssm_head_d_v > 0 ? skv->ssm_head_d_v : 128;
        ctx->cur_conv_hist   = skv->conv_hist > 0 ? skv->conv_hist : 3;
        ctx->cur_conv_inner  = skv->conv_inner_size > 0 ? skv->conv_inner_size : 8192;
    }

    /* ── Embedding lookup into scratch_x ───────────────────────────────── */
    float *x = ctx->scratch_x ? ctx->scratch_x : (float*)calloc(H, sizeof(float));
    int x_heap = (x != ctx->scratch_x);
    if (ctx->scratch_x) memset(ctx->scratch_x, 0, H * sizeof(float));

    const TB_GGUFTensorInfo *emb_t = tb_gguf_find_tensor(m, "token_embd.weight");
    if (emb_t && m->weights_data && last_token >= 0 && last_token < m->vocab_size) {
        if (emb_t->qtype == 30) {
            const uint16_t *emb = (const uint16_t*)tb_gguf_tensor_data(m, emb_t);
            const uint16_t *row = emb + (size_t)last_token * H;
            for (int i=0;i<H;i++) x[i] = tb_bf16_to_f32_infer(row[i]);
        } else if (emb_t->qtype == 0) {
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
            tb_gguf_dequant_row(row_ptr, emb_t->qtype, H, x);
        }
    } else {
        for (int i=0;i<H;i++) x[i]=0.01f*sinf((float)(i+last_token));
    }
    tb_vlog("decode stage=embedding done token=%d", last_token);

    /* ── Forward through all layers (swap x ↔ y via scratch buffers) ─── */
    float *y = ctx->scratch_y ? ctx->scratch_y : (float*)malloc(H * sizeof(float));
    int y_heap = (y != ctx->scratch_y);

    /* NaN hunt: check embedding and first few layer outputs — fires for pos 0..7 */
    int _do_hunt = (pos < 8);
    if (_do_hunt) {
        float emb_l2=0.0f; int emb_nan=0;
        for(int _i=0;_i<H;_i++){
            if(isnan(x[_i])||isinf(x[_i])) emb_nan++;
            else emb_l2+=x[_i]*x[_i];
        }
        fprintf(stderr,"[nanhunt pos=%d] emb tok=%d qtype=%d: nan/inf=%d L2=%.4f\n",
                pos, last_token, emb_t?emb_t->qtype:-1, emb_nan, sqrtf(emb_l2));
        fflush(stderr);
    }

    double t_fwd = tb_wall_ms();
    for (int l = 0; l < NL; l++) {
        tb_vlog("decode stage=layer_enter layer=%d", l);
        tb_layer_forward(ctx, l, x, y, kv, pos, last_token);
        tb_vlog("decode stage=layer_exit layer=%d", l);
        /* NaN hunt: print every 4th layer + final, only for early positions */
        if (_do_hunt && (l % 4 == 3 || l == NL-1)) {
            float l2=0.0f; int nans=0;
            for(int _i=0;_i<H;_i++){
                if(isnan(y[_i])||isinf(y[_i])) nans++;
                else l2+=y[_i]*y[_i];
            }
            fprintf(stderr,"[nanhunt pos=%d] after layer %d: nan/inf=%d L2=%.4f\n",pos,l,nans,sqrtf(l2));
            fflush(stderr);
        }
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
    if (pos == 0) {
        if (lm_t) fprintf(stderr,"[lmdiag] output.weight qtype=%d shape=[%lld,%lld] gpu=%s vocab_size=%d H=%d\n",
                lm_t->qtype, (long long)lm_t->shape[0], (long long)lm_t->shape[1],
                lm_t->d_data?"Y":"N", m->vocab_size, H);
        if (fn_t) fprintf(stderr,"[lmdiag] output_norm.weight qtype=%d shape=[%lld] gpu=%s\n",
                fn_t->qtype, (long long)fn_t->shape[0], fn_t->d_data?"Y":"N");
        fflush(stderr);
    }
    if (pos < 8) {
        /* norm_out L2 */
        float n2 = 0.0f; for (int _i=0;_i<H;_i++) n2+=norm_out[_i]*norm_out[_i];
        fprintf(stderr,"[lmdiag pos=%d] lm_t=%s gpu=%s  norm_out_L2=%.4f\n",
                pos, lm_t?lm_t->name:"NULL", lm_t&&lm_t->d_data?"Y":"N", sqrtf(n2));
        fflush(stderr);
    }
    if (lm_t && m->weights_data) {
        tb_gguf_tensor_matvec(m, lm_t, m->vocab_size, H, norm_out, logits);
    } else {
        for (int v=0;v<m->vocab_size;v++) logits[v]=0.01f*cosf((float)v*0.01f);
        logits[(last_token+1)%m->vocab_size] = 1.0f;
    }

    /* top-5 logits diagnostic — fires for pos 0..7 */
    if (pos < 8) {
        /* find top-5 */
        int top5[5]={0,0,0,0,0}; float tv[5]={-1e38f,-1e38f,-1e38f,-1e38f,-1e38f};
        for (int _v=0;_v<m->vocab_size;_v++) {
            if (logits[_v] > tv[4]) {
                tv[4]=logits[_v]; top5[4]=_v;
                for (int _j=3;_j>=0&&tv[_j+1]>tv[_j];_j--) {
                    float _ft=tv[_j]; tv[_j]=tv[_j+1]; tv[_j+1]=_ft;
                    int _fi=top5[_j]; top5[_j]=top5[_j+1]; top5[_j+1]=_fi;
                }
            }
        }
        fprintf(stderr,"[lmdiag pos=%d] top5 logits: ",pos);
        for (int _j=0;_j<5;_j++) fprintf(stderr,"[%d]=%.3f ",top5[_j],tv[_j]);
        fprintf(stderr,"\n"); fflush(stderr);
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
    fprintf(stderr, "[generate] prompt ids (%d): ", n_prompt);
    for (int i = 0; i < n_prompt; i++) fprintf(stderr, "%d ", prompt_ids[i]);
    fprintf(stderr, "\n"); fflush(stderr);
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

/* Forward declaration — tb_json_extract_str is defined after these helpers. */
static void tb_json_extract_str(const char *json, const char *key, char *dst, int dst_len);

/* ── Chat-template helpers ────────────────────────────────────────────────── */

/* Extract ALL messages from the Ollama-format messages array.
 * Walks the JSON linearly: finds each {"role":"...","content":"..."} object.
 * Writes up to max_msgs role/content pairs into the parallel arrays.
 * Returns the number of messages found. */
static int tb_json_extract_messages(const char *json,
                                     char roles[][32], char contents[][4096],
                                     int max_msgs) {
    if (!json) return 0;
    int count = 0;
    const char *p = json;
    /* Locate the "messages" array */
    const char *arr = strstr(p, "\"messages\"");
    if (!arr) return 0;
    arr = strchr(arr + 10, '[');
    if (!arr) return 0;
    p = arr + 1;
    /* Walk objects within the array */
    while (*p && count < max_msgs) {
        /* Skip whitespace / commas between objects */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ',')) p++;
        if (*p != '{') break;
        /* Find "role" */
        const char *role_key = strstr(p, "\"role\"");
        const char *cont_key = strstr(p, "\"content\"");
        /* Find the closing brace of this object (naively — fine for flat objects) */
        const char *obj_end = strchr(p + 1, '}');
        if (!obj_end) break;
        if (role_key && role_key < obj_end && cont_key && cont_key < obj_end) {
            /* Extract role */
            const char *rp = strchr(role_key + 6, ':');
            if (rp) {
                while (*rp == ':' || *rp == ' ' || *rp == '\t') rp++;
                if (*rp == '"') {
                    rp++;
                    int ri = 0;
                    while (*rp && *rp != '"' && ri < 31) roles[count][ri++] = *rp++;
                    roles[count][ri] = '\0';
                }
            }
            /* Extract content using existing helper (searches from obj start) */
            char tmp_obj[8192];
            int obj_len = (int)(obj_end - p + 1);
            if (obj_len >= (int)sizeof(tmp_obj)) obj_len = (int)sizeof(tmp_obj) - 1;
            memcpy(tmp_obj, p, (size_t)obj_len);
            tmp_obj[obj_len] = '\0';
            tb_json_extract_str(tmp_obj, "content", contents[count], 4096);
            count++;
        }
        p = obj_end + 1;
    }
    return count;
}

/* Apply a simplified chat template to produce the full prompt string.
 *
 * Strategy (in priority order):
 *
 * 1. If the model stores a tokenizer.chat_template Jinja2 string we check
 *    for known marker substrings to identify the template family and format
 *    accordingly.  This covers the most common open-weight families:
 *      - ChatML  (<|im_start|> / <|im_end|>)         — Qwen, Mistral-Nemo, etc.
 *      - Llama-3 (<|start_header_id|> / <|eot_id|>)  — Meta Llama-3.x
 *      - Phi-3   (<|user|> / <|end|>)                — Microsoft Phi-3 / 3.5
 *      - Gemma   (<start_of_turn> / <end_of_turn>)   — Google Gemma-2
 *      - Mistral [INST] / [/INST]                    — Mistral-7B v0.x
 *
 * 2. If no template string is present, fall back to plain ChatML which is
 *    the de-facto default for most recent instruction-tuned GGUF models.
 *
 * The output is written into `out` (capacity `out_cap`).
 * Returns 0 on success, -1 if the buffer was truncated.
 */
static int tb_apply_chat_template(const char *tmpl,
                                   char roles[][32], char contents[][4096],
                                   int n_msgs,
                                   char *out, int out_cap) {
    out[0] = '\0';
    int pos = 0;

#define TCAT(s) do { \
    int _l = (int)strlen(s); \
    if (pos + _l < out_cap - 1) { memcpy(out+pos, s, (size_t)_l); pos += _l; out[pos]='\0'; } \
    else { out[out_cap-1]='\0'; return -1; } \
} while(0)

    /* Detect template family from the stored Jinja string */
    int use_llama3  = tmpl && strstr(tmpl, "<|start_header_id|>") != NULL;
    int use_phi3    = tmpl && strstr(tmpl, "<|user|>")            != NULL;
    int use_gemma   = tmpl && strstr(tmpl, "<start_of_turn>")     != NULL;
    int use_mistral = tmpl && strstr(tmpl, "[INST]")              != NULL
                           && !strstr(tmpl, "<|im_start|>");
    /* ChatML is the default (also explicit if template contains im_start) */

    if (use_llama3) {
        /* Meta Llama-3 format */
        TCAT("<|begin_of_text|>");
        for (int i = 0; i < n_msgs; i++) {
            TCAT("<|start_header_id|>"); TCAT(roles[i]); TCAT("<|end_header_id|>\n\n");
            TCAT(contents[i]);
            TCAT("<|eot_id|>");
        }
        TCAT("<|start_header_id|>assistant<|end_header_id|>\n\n");

    } else if (use_phi3) {
        /* Microsoft Phi-3 / 3.5 format */
        for (int i = 0; i < n_msgs; i++) {
            TCAT("<|"); TCAT(roles[i]); TCAT("|>\n");
            TCAT(contents[i]);
            TCAT("<|end|>\n");
        }
        TCAT("<|assistant|>\n");

    } else if (use_gemma) {
        /* Google Gemma-2 format */
        for (int i = 0; i < n_msgs; i++) {
            const char *turn = strcmp(roles[i], "assistant") == 0 ? "model" : "user";
            TCAT("<start_of_turn>"); TCAT(turn); TCAT("\n");
            TCAT(contents[i]);
            TCAT("<end_of_turn>\n");
        }
        TCAT("<start_of_turn>model\n");

    } else if (use_mistral) {
        /* Mistral v0.x [INST] format — system prompt prepended to first user turn */
        int first_user = 1;
        TCAT("<s>");
        for (int i = 0; i < n_msgs; i++) {
            if (strcmp(roles[i], "system") == 0) {
                /* Mistral has no dedicated system tag; inject before first [INST] */
                if (first_user) {
                    /* Will be prepended when we hit the next user message */
                    /* Store it temporarily in contents[-1] style by embedding inline */
                    char sys_buf[4096];
                    snprintf(sys_buf, sizeof(sys_buf), "%s\n\n", contents[i]);
                    TCAT("[INST] "); TCAT(sys_buf);
                    first_user = 0;
                }
            } else if (strcmp(roles[i], "user") == 0) {
                if (first_user) { TCAT("[INST] "); first_user = 0; }
                TCAT(contents[i]); TCAT(" [/INST]");
            } else {
                TCAT(contents[i]); TCAT("</s><s>");
            }
        }

    } else {
        /* ChatML — default for Qwen, Mistral-Nemo, OpenHermes, etc. */
        for (int i = 0; i < n_msgs; i++) {
            TCAT("<|im_start|>"); TCAT(roles[i]); TCAT("\n");
            TCAT(contents[i]);
            TCAT("<|im_end|>\n");
        }
        TCAT("<|im_start|>assistant\n");
    }
#undef TCAT
    return 0;
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

    /* Bug 4 fix: decoding a single token with tb_tokenizer_decode always starts
     * with pos=0, so the pos>0 guard that strips the leading space from a
     * word-marker-prefixed token never fires.  Every word-starting token therefore
     * gets a spurious leading space in the streamed output.
     *
     * Fix: decode the current token in context of the previous one (two-token
     * window), then subtract the already-emitted prefix to isolate the new piece.
     * For the very first generated token, strip any leading space directly. */
    char *word = NULL;
    if (sc->tok) {
        if (sc->out_len >= 2) {
            int pair[2] = { sc->out_ids[sc->out_len - 2], token_id };
            char *pair_str = tb_tokenizer_decode(sc->tok, pair, 2);
            char *prev_str = tb_tokenizer_decode(sc->tok, &sc->out_ids[sc->out_len - 2], 1);
            if (pair_str && prev_str) {
                int skip = (int)strlen(prev_str);
                word = strdup(pair_str + skip);
            } else {
                word = pair_str ? strdup(pair_str) : NULL;
            }
            free(pair_str);
            free(prev_str);
        } else {
            word = tb_tokenizer_decode(sc->tok, &token_id, 1);
            /* First token: strip any leading space (pos=0 suppression did not fire). */
            if (word && word[0] == ' ')
                memmove(word, word + 1, strlen(word)); /* includes NUL */
        }
    }
    const char *text = word ? word : "";
    /* Escape for JSON — heap buffer sized to worst-case (2x input + NUL).
     * Also handle \n/\r/\t so the JSON stays valid when tokens contain newlines. */
    int tlen = (int)strlen(text);
    char *esc = (char*)malloc((size_t)(tlen * 2 + 4));
    int ei = 0;
    if (esc) {
        for (const char *c = text; *c; c++) {
            if      (*c == '"'  || *c == '\\') { esc[ei++] = '\\'; esc[ei++] = *c; }
            else if (*c == '\n')               { esc[ei++] = '\\'; esc[ei++] = 'n'; }
            else if (*c == '\r')               { esc[ei++] = '\\'; esc[ei++] = 'r'; }
            else if (*c == '\t')               { esc[ei++] = '\\'; esc[ei++] = 't'; }
            else                              { esc[ei++] = *c; }
        }
        esc[ei] = '\0';
    }
    free(word);
    char chunk[512];
    int clen = snprintf(chunk, sizeof(chunk),
        "{\"model\":\"%s\",\"done\":false,\"response\":\"%s\"}\n",
        sc->model_name, esc ? esc : "");
    free(esc);
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

        /* Build the prompt string ─────────────────────────────────────────── */
        char prompt[16384] = "";

        if (is_chat) {
            /* /api/chat: body = {"messages":[{"role":"user","content":"..."},...]} */
            char msg_roles[32][32];
            char msg_contents[32][4096];
            int n_msgs = tb_json_extract_messages(body, msg_roles, msg_contents, 32);

            if (n_msgs > 0) {
                const char *tmpl = (ctx->model) ? ctx->model->chat_template : NULL;
                int rc = tb_apply_chat_template(tmpl,
                                                msg_roles, msg_contents, n_msgs,
                                                prompt, (int)sizeof(prompt));
                if (rc < 0)
                    fprintf(stderr, "[chat] warning: prompt truncated to %zu bytes\n",
                            sizeof(prompt));
            } else {
                /* Fallback: try legacy flat "content" key */
                tb_json_extract_str(body, "content", prompt, sizeof(prompt));
            }
        } else {
            /* /api/generate: body = {"prompt":"..."} */
            tb_json_extract_str(body, "prompt", prompt, sizeof(prompt));
        }
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

#ifndef TB_NO_MAIN
int main(int argc, char **argv) {
    printf("=== TRAILBLAZE Inference Runtime ===\n\n");
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    /* Parse args */
    const char *model_path = NULL;
    int port = 11434, serve = 0, use_hdgl = 0, benchmark = 0;  /* HDGL off by default: use --hdgl to enable */
    float hdgl_alpha = 0.2f;
    float override_temp = -1.0f;   /* -1 means "use default" */
    int   add_bos = 1;             /* prepend BOS token */
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
        else if ((!strcmp(argv[i],"--temp")||!strcmp(argv[i],"--temperature")) && i+1<argc) {
            override_temp = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i],"--no-bos"))                add_bos = 0;
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
    if (override_temp >= 0.0f) ctx->temperature = override_temp;

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
        if (tok) n_tok = tb_tokenizer_encode(tok, prompt, add_bos, tok_ids, 2048);
        if (n_tok <= 0) {
            for (int i = 0; i < (int)strlen(prompt) && n_tok < 2048; i++)
                tok_ids[n_tok++] = ((unsigned char)prompt[i]) % model->vocab_size;
        }
        if (n_tok <= 0) { tok_ids[0] = 1; n_tok = 1; }

        /* Per-token decode diagnostic */
        fprintf(stderr, "[dbg] tok=%p n_tok=%d\n", (void*)tok, n_tok); fflush(stderr);
        if (tok) {
            fprintf(stderr, "[generate] prompt token decode (%d): ", n_tok);
            for (int i = 0; i < n_tok; i++) {
                char *piece = tb_tokenizer_decode(tok, &tok_ids[i], 1);
                fprintf(stderr, "[%d='%s'] ", tok_ids[i], piece ? piece : "?");
                if (piece) free(piece);
            }
            fprintf(stderr, "\n"); fflush(stderr);
        }

        printf("[generate] output tokens: ");
        int out_ids[2048];
        TB_CliTokenCtx cctx = {0};
        cctx.tok = tok;
        int n_gen = tb_infer_generate(ctx, tok_ids, n_tok, out_ids, max_new_tokens, 0,
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
#endif /* TB_NO_MAIN */
#endif /* TB_INFER_TEST */
