/*
 * tb_layer_test.c — TRAILBLAZE per-layer batch test harness
 *
 * Runs a configurable batch of hidden-state inputs through each transformer
 * layer individually, printing a pass/fail checklist for every stage.
 *
 * Build: build_layer_test.bat
 * Usage: tb_layer_test.exe <model.gguf> [options]
 *   --layer N     test only layer N (default: all 0..n_layers-1)
 *   --batch B     batch size: number of different inputs per layer (default 8)
 *   --stop-first  stop after first FAIL
 *   --verbose     also print OK stages
 *   --no-gpu      skip GPU matvec (CPU path only)
 */

#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "layer4/tb_infer.h"
#include "layer4/tb_gguf.h"
#include "layer1/tb_tensor.h"

/* ── Tolerances ───────────────────────────────────────────────────────────── */
#define NORM_LO          0.05f   /* output L2 must be above this (zero-collapse) */
#define NORM_HI          1e7f    /* output L2 must be below this (explosion)     */
#define DIVERSITY_MIN    0.01f   /* min L2 distance between any two batch outputs */

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static void vec_stats(const float *v, int n,
                      float *l2_out, float *maxabs_out, int *bad_out) {
    double l2 = 0.0;
    float maxabs = 0.0f;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(v[i])) { bad++; continue; }
        l2 += (double)v[i] * (double)v[i];
        float a = fabsf(v[i]);
        if (a > maxabs) maxabs = a;
    }
    if (l2_out)    *l2_out    = (float)sqrt(l2);
    if (maxabs_out)*maxabs_out = maxabs;
    if (bad_out)   *bad_out   = bad;
}

/* L2 distance between two float vectors */
static float vec_dist(const float *a, const float *b, int n) {
    double d = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        d += diff * diff;
    }
    return (float)sqrt(d);
}

/* Seeded LCG for reproducible random hidden states */
static uint32_t lcg_state;
static float lcg_randf(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

/* Fill v[n] with unit-normal-ish values scaled to typical embedding magnitude */
static void fill_hidden(float *v, int n, uint32_t seed) {
    lcg_state = seed;
    double ss = 0.0;
    for (int i = 0; i < n; i++) { v[i] = lcg_randf(); ss += (double)v[i]*(double)v[i]; }
    /* Normalise to L2 ~ sqrt(n) (same order as real embeddings after rms-norm) */
    float scale = (float)(sqrt((double)n) / sqrt(ss + 1e-12));
    for (int i = 0; i < n; i++) v[i] *= scale;
}

/* ── Weight inventory check ───────────────────────────────────────────────── */

typedef struct {
    const char *name;
    int optional;          /* 1 = warn not fail */
} TensorSpec;

static int check_tensor(const TB_GGUFLoaded *m, const char *name, int optional) {
    const TB_GGUFTensorInfo *t = tb_gguf_find_tensor(m, name);
    if (!t) {
        printf("    [%s] %-40s  MISSING%s\n",
               optional ? "?" : "F", name, optional ? " (optional)" : "");
        return optional ? 1 : 0;
    }
    int64_t ne = tb_gguf_tensor_nelems(t);
    printf("    [✓] %-40s  qtype=%2d  nelems=%lld  gpu=%s\n",
           name, t->qtype, (long long)ne, t->d_data ? "Y" : "N");
    return 1;
}

static int check_layer_weights(const TB_GGUFLoaded *m, int L, int verbose) {
    char n[128];
    int ok = 1;
    if (verbose) printf("  [weights L%02d]\n", L);

#define CK(fmt, ...) do { \
    snprintf(n, sizeof(n), fmt, ##__VA_ARGS__); \
    if (!check_tensor(m, n, 0)) ok = 0; \
} while(0)
#define CKOPT(fmt, ...) do { \
    snprintf(n, sizeof(n), fmt, ##__VA_ARGS__); \
    check_tensor(m, n, 1); \
} while(0)

    CK("blk.%d.attn_norm.weight", L);
    CK("blk.%d.ffn_norm.weight",  L);

    /* QKV: either fused or separate */
    const TB_GGUFTensorInfo *fused = NULL;
    snprintf(n, sizeof(n), "blk.%d.attn_qkv.weight", L);
    fused = tb_gguf_find_tensor(m, n);
    if (fused) {
        if (verbose) check_tensor(m, n, 0);
    } else {
        CK("blk.%d.attn_q.weight", L);
        CK("blk.%d.attn_k.weight", L);
        CK("blk.%d.attn_v.weight", L);
        CKOPT("blk.%d.attn_q_norm.weight", L);
        CKOPT("blk.%d.attn_k_norm.weight", L);
    }
    CK("blk.%d.attn_output.weight", L);

    /* FFN: packed experts, split experts, or dense */
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_exps.weight", L);
    const TB_GGUFTensorInfo *packed = tb_gguf_find_tensor(m, n);
    if (packed) {
        if (verbose) {
            check_tensor(m, n, 0);
            snprintf(n, sizeof(n), "blk.%d.ffn_up_exps.weight",   L); check_tensor(m, n, 0);
            snprintf(n, sizeof(n), "blk.%d.ffn_down_exps.weight", L); check_tensor(m, n, 0);
        }
    } else {
        CK("blk.%d.ffn_gate.weight", L);
        CK("blk.%d.ffn_up.weight",   L);
        CK("blk.%d.ffn_down.weight", L);
    }
#undef CK
#undef CKOPT
    return ok;
}

/* ── Stage result ─────────────────────────────────────────────────────────── */

#define MAX_STAGES 16
typedef struct {
    char  name[32];
    int   pass;        /* 1=pass 0=fail */
    float l2;
    float maxabs;
    int   bad;         /* NaN/Inf count */
} Stage;

/* ── Per-layer batch test ─────────────────────────────────────────────────── */

typedef struct {
    int   layer;
    int   batch_size;
    int   all_pass;
    int   n_stages;
    Stage stages[MAX_STAGES];
    /* batch diversity */
    float min_pairwise_dist;
    float output_vs_input_dist;  /* avg L2(out_i - in_i) */
} LayerResult;

static void record_stage(LayerResult *r, const char *name,
                         const float *v, int n) {
    if (r->n_stages >= MAX_STAGES) return;
    Stage *s = &r->stages[r->n_stages++];
    strncpy(s->name, name, 31); s->name[31] = '\0';
    vec_stats(v, n, &s->l2, &s->maxabs, &s->bad);
    s->pass = (s->bad == 0) && (s->l2 > NORM_LO) && (s->l2 < NORM_HI);
    if (!s->pass) r->all_pass = 0;
}

static LayerResult run_layer_batch(TB_InferCtx *ctx, int L, int B) {
    LayerResult r;
    memset(&r, 0, sizeof(r));
    r.layer = L;
    r.batch_size = B;
    r.all_pass = 1;

    TB_GGUFModel *m = ctx->model;
    int H = m->hidden_dim;
    if (H <= 0) { r.all_pass = 0; return r; }

    /* Allocate batch arrays */
    float **in  = (float**)malloc(B * sizeof(float*));
    float **out = (float**)malloc(B * sizeof(float*));
    for (int b = 0; b < B; b++) {
        in[b]  = (float*)malloc(H * sizeof(float));
        out[b] = (float*)calloc(H, sizeof(float));
    }

    /* Generate diverse inputs:
     *   b=0: BOS embedding from model (most realistic)
     *   b=1..B-1: random unit vectors with different seeds */
    {
        const TB_GGUFTensorInfo *emb = tb_gguf_find_tensor(m, "token_embd.weight");
        if (emb && m->weights_data) {
            /* Compute byte stride per row from qtype */
            size_t row_bytes;
            switch (emb->qtype) {
                case 8:  row_bytes = (size_t)(H/32)*34;   break;
                case 14: row_bytes = (size_t)(H/256)*210; break;
                case 12: row_bytes = (size_t)(H/256)*144; break;
                case 30: /* fall-through */
                case  1: row_bytes = (size_t)H*2;         break;
                case  0: row_bytes = (size_t)H*4;         break;
                default: row_bytes = (size_t)H*4;         break;
            }
            const void *bos_row = (const char*)tb_gguf_tensor_data(m, emb)
                                + (size_t)m->bos_token_id * row_bytes;
            tb_gguf_dequant_row(bos_row, emb->qtype, H, in[0]);
        } else {
            fill_hidden(in[0], H, 0xBEEF0000u);
        }
    }
    for (int b = 1; b < B; b++)
        fill_hidden(in[b], H, 0xBEEF0000u + (uint32_t)b * 0x9E3779B9u);

    /* Run each batch item through layer L independently (pos=b, token=b) */
    /* Allocate a fresh KV cache so attention has a clean state per item */
    int max_seq = B + 4;
    TB_KVCache *kv = tb_kvcache_alloc(m->n_layers, m->n_kv_heads, m->head_dim,
                                      max_seq, /*slot*/99, 0);

    int n_fail = 0;
    for (int b = 0; b < B; b++) {
        if (kv) kv->seq_len = 0;  /* reset cache per item — independent tests */
        int ok = tb_layer_forward(ctx, L, in[b], out[b], kv, /*pos*/b, /*token*/b);
        if (!ok) { n_fail++; r.all_pass = 0; }
    }
    if (kv) tb_kvcache_free(kv);

    /* Aggregate output stats across batch */
    {
        /* Flatten all outputs into one big vector for combined stats */
        float total_l2 = 0.0f, total_max = 0.0f;
        int total_bad = 0;
        for (int b = 0; b < B; b++) {
            float l2, mx; int bad;
            vec_stats(out[b], H, &l2, &mx, &bad);
            total_l2 += l2;
            if (mx > total_max) total_max = mx;
            total_bad += bad;
        }
        Stage *s = &r.stages[r.n_stages++];
        snprintf(s->name, 32, "output_stats");
        s->l2     = total_l2 / B;
        s->maxabs = total_max;
        s->bad    = total_bad;
        s->pass   = (total_bad == 0)
                 && (total_l2 / B > NORM_LO)
                 && (total_l2 / B < NORM_HI)
                 && (n_fail == 0);
        if (!s->pass) r.all_pass = 0;
    }

    /* Diversity: min pairwise L2 distance between outputs */
    r.min_pairwise_dist = 1e30f;
    for (int i = 0; i < B; i++)
        for (int j = i+1; j < B; j++) {
            float d = vec_dist(out[i], out[j], H);
            if (d < r.min_pairwise_dist) r.min_pairwise_dist = d;
        }
    if (r.min_pairwise_dist < DIVERSITY_MIN) r.all_pass = 0;

    /* Input→output distance: verify layer actually does something */
    double io_sum = 0.0;
    for (int b = 0; b < B; b++)
        io_sum += vec_dist(in[b], out[b], H);
    r.output_vs_input_dist = (float)(io_sum / B);

    for (int b = 0; b < B; b++) { free(in[b]); free(out[b]); }
    free(in); free(out);
    return r;
}

/* ── Print result ─────────────────────────────────────────────────────────── */

static void print_result(const LayerResult *r, int verbose) {
    const char *verdict = r->all_pass ? "PASS" : "FAIL";
    printf("  Layer %02d: %-4s  "
           "out_L2=%.2f  diversity=%.3f  io_dist=%.2f\n",
           r->layer, verdict,
           r->stages[r->n_stages-1].l2,
           r->min_pairwise_dist,
           r->output_vs_input_dist);

    if (!r->all_pass || verbose) {
        for (int s = 0; s < r->n_stages; s++) {
            const Stage *st = &r->stages[s];
            printf("    [%s] %-20s  L2=%10.3f  maxabs=%9.4f  bad=%d\n",
                   st->pass ? "✓" : "F",
                   st->name, st->l2, st->maxabs, st->bad);
        }
        if (r->min_pairwise_dist < DIVERSITY_MIN)
            printf("    [F] diversity=%.4f < threshold %.4f  (mode collapse?)\n",
                   r->min_pairwise_dist, DIVERSITY_MIN);
    }
}

/* ── Tensor inventory dump ────────────────────────────────────────────────── */

static void dump_layer_inventory(const TB_GGUFLoaded *m, int L) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "blk.%d.", L);
    printf("  Tensors for layer %d:\n", L);
    for (int i = 0; i < m->n_tensors; i++) {
        const TB_GGUFTensorInfo *t = &m->tensors[i];
        if (strncmp(t->name, prefix, strlen(prefix)) == 0) {
            printf("    qtype=%2d  nelems=%8lld  gpu=%-1s  %s\n",
                   t->qtype, (long long)tb_gguf_tensor_nelems(t),
                   t->d_data ? "Y" : "N", t->name);
        }
    }
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--layer N] [--batch B]"
                " [--stop-first] [--verbose] [--inventory] [--no-gpu]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];

    /* Parse options */
    int opt_layer     = -1;    /* -1 = all */
    int opt_batch     = 8;
    int opt_stop      = 0;
    int opt_verbose   = 0;
    int opt_inventory = 0;
    int opt_no_gpu    = 0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--layer")    && i+1 < argc) { opt_layer   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--batch")  && i+1 < argc) { opt_batch   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--stop-first"))            { opt_stop     = 1; }
        else if (!strcmp(argv[i], "--verbose"))               { opt_verbose  = 1; }
        else if (!strcmp(argv[i], "--inventory"))             { opt_inventory = 1; }
        else if (!strcmp(argv[i], "--no-gpu"))                { opt_no_gpu   = 1; }
    }
    if (opt_batch < 1) opt_batch = 1;
    if (opt_batch > 64) opt_batch = 64;

    printf("=== TRAILBLAZE Layer Test ===\n");
    printf("model: %s\n", model_path);
    printf("batch: %d  stop-first: %d\n\n", opt_batch, opt_stop);

    /* Load model */
    printf("[load] loading model...\n");
    fflush(stdout);
    TB_GGUFModel *m = tb_model_load(model_path);
    if (!m) { fprintf(stderr, "FAIL: tb_model_load returned NULL\n"); return 1; }
    printf("[load] arch=%s layers=%d hidden=%d heads=%d/%d ffn=%d vocab=%d\n",
           m->arch, m->n_layers, m->hidden_dim,
           m->n_heads, m->n_kv_heads, m->ffn_dim, m->vocab_size);

#ifdef TB_CUDA
    if (!opt_no_gpu) {
        printf("[cuda] uploading tensors to GPU...\n"); fflush(stdout);
        int n_up = tb_gguf_cuda_upload_all(m);
        printf("[cuda] %d tensors uploaded\n", n_up);
    }
#else
    (void)opt_no_gpu;
    printf("[info] TB_CUDA not defined — CPU path only\n");
#endif

    /* Create inference context */
    printf("[ctx]  creating inference context...\n"); fflush(stdout);
    TB_InferCtx *ctx = tb_infer_create(m, /*use_hdgl*/1, /*alpha*/0.20f,
                                        /*lattice_slots*/512, /*seed*/0xCAFEBABEull);
    if (!ctx) { fprintf(stderr, "FAIL: tb_infer_create returned NULL\n"); tb_model_free(m); return 1; }
    printf("[ctx]  OK  hidden=%d\n\n", m->hidden_dim);

    int n_layers = m->n_layers;
    int L_start  = (opt_layer >= 0) ? opt_layer : 0;
    int L_end    = (opt_layer >= 0) ? opt_layer : n_layers - 1;

    int n_pass = 0, n_fail = 0;

    for (int L = L_start; L <= L_end; L++) {
        printf("Layer %02d/%02d\n", L, n_layers - 1);

        /* Tensor inventory */
        if (opt_inventory)
            dump_layer_inventory(m, L);

        /* Weight check */
        int wt_ok = check_layer_weights(m, L, opt_verbose);
        if (!wt_ok) {
            printf("  Layer %02d: FAIL  (missing required tensors)\n", L);
            n_fail++;
            if (opt_stop) break;
            continue;
        }

        /* Batch forward */
        LayerResult r = run_layer_batch(ctx, L, opt_batch);
        print_result(&r, opt_verbose);

        if (r.all_pass) n_pass++;
        else            n_fail++;

        if (!r.all_pass && opt_stop) {
            printf("\n[stop-first] stopping after first FAIL\n");
            break;
        }
        fflush(stdout);
    }

    printf("\n=== Summary: %d/%d layers PASS ===\n", n_pass, n_pass + n_fail);

    tb_infer_free(ctx);
    tb_model_free(m);
    return (n_fail == 0) ? 0 : 1;
}
