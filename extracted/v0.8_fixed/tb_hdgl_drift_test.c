/*
 * tb_hdgl_drift_test.c — HDGL routing drift test
 *
 * Contrived empirical test: no real model needed.
 * We feed tb_route_experts() a fixed, repeating gate-logit vector over N tokens
 * and measure how much the HDGL routing state causes expert selection to diverge
 * from the gate-only baseline over time.
 *
 * Build:
 *   gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -DTB_INFER_TEST -DTB_NO_MAIN \
 *       -Ilayer0 -Ilayer1 -Ilayer2 -Ilayer3 -Ilayer4 -Ilayer5 -Iinclude -Isrc \
 *       tb_hdgl_drift_test.c \
 *       layer4/tb_infer.c layer4/tb_gguf.c layer4/tb_tokenizer.c \
 *       layer3/tb_orchestration.c layer5/tb_semantic_os.c \
 *       layer0/tb_phi_lattice.c layer1/tb_tensor.c layer2/tb_graph.c \
 *       src/sha256_minimal.c src/hdgl_bootloaderz.c src/hdgl_router.c \
 *       src/vector_container.c src/analog_engine.c src/tb_analog_dispatch.c \
 *       -lm -lpthread -o bin/tb_hdgl_drift_test
 *
 * Usage: ./bin/tb_hdgl_drift_test [--tokens N] [--experts E] [--layers L] [--alpha F]
 */

#define TB_INFER_TEST
#define TB_NO_MAIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "layer4/tb_infer.h"

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* Softmax top-k: returns the k experts the gate logits alone would choose,
 * with their normalised weights. This is the ground-truth baseline. */
static void gate_only_topk(const float *logits, int n_experts, int k,
                            int *out_idx, float *out_w) {
    float scores[256];
    float max_s = logits[0];
    for (int e = 1; e < n_experts; e++) if (logits[e] > max_s) max_s = logits[e];
    float sum = 0.0f;
    for (int e = 0; e < n_experts; e++) { scores[e] = expf(logits[e] - max_s); sum += scores[e]; }
    for (int e = 0; e < n_experts; e++) scores[e] /= sum;

    /* Simple top-k selection */
    int   idx[256]; for (int e = 0; e < n_experts; e++) idx[e] = e;
    /* Partial selection sort for top-k */
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i+1; j < n_experts; j++)
            if (scores[j] > scores[best]) best = j;
        float tf = scores[i]; scores[i] = scores[best]; scores[best] = tf;
        int   ti = idx[i];   idx[i]   = idx[best];   idx[best]   = ti;
    }
    float wsum = 0.0f;
    for (int i = 0; i < k; i++) { out_idx[i] = idx[i]; wsum += scores[i]; }
    for (int i = 0; i < k; i++) out_w[i] = (wsum > 0) ? scores[i] / wsum : 1.0f / k;
}

/* Jaccard similarity of two top-k expert sets (order-independent). */
static float expert_jaccard(const int *a, const int *b, int k) {
    int intersect = 0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            if (a[i] == b[j]) { intersect++; break; }
    int union_size = 2*k - intersect;
    return union_size > 0 ? (float)intersect / (float)union_size : 1.0f;
}

/* L1 distance between two normalised weight vectors (same index order). */
static float weight_l1(const int *idx_a, const float *w_a,
                       const int *idx_b, const float *w_b, int k) {
    /* Build full weight arrays indexed by expert */
    float wa[256] = {0}, wb[256] = {0};
    for (int i = 0; i < k; i++) wa[idx_a[i]] = w_a[i];
    for (int i = 0; i < k; i++) wb[idx_b[i]] = w_b[i];
    float l1 = 0.0f;
    /* only need to iterate over experts in either set */
    for (int i = 0; i < k; i++) { l1 += fabsf(wa[idx_a[i]] - wb[idx_a[i]]); }
    for (int i = 0; i < k; i++) {
        int found = 0;
        for (int j = 0; j < k; j++) if (idx_b[i] == idx_a[j]) { found=1; break; }
        if (!found) l1 += wb[idx_b[i]];
    }
    return l1;
}

/* ── Gate logit scenarios ─────────────────────────────────────────────────── */
typedef struct { const char *name; float logits[16]; } Scenario;

/* Various gate logit shapes that represent real model behaviour:
 *  confident  — one expert dominates (top1 >> rest)
 *  balanced   — all experts nearly equal (high entropy)
 *  two_way    — two experts nearly tied, rest low
 *  noisy      — random-ish with no clear winner
 *  repeating  — cycles through dominant experts token by token
 */
static void fill_logits(int scenario_idx, int token, int n_experts, float *logits) {
    switch (scenario_idx % 5) {
    case 0: /* confident: expert 0 strongly preferred */
        for (int e = 0; e < n_experts; e++) logits[e] = -2.0f;
        logits[0] = 4.0f;
        break;
    case 1: /* balanced: uniform */
        for (int e = 0; e < n_experts; e++) logits[e] = 0.0f;
        break;
    case 2: /* two_way: experts 1 and 3 tied */
        for (int e = 0; e < n_experts; e++) logits[e] = -3.0f;
        logits[1] = 2.0f; logits[3] = 2.0f;
        break;
    case 3: /* noisy: deterministic pseudo-random */
        for (int e = 0; e < n_experts; e++)
            logits[e] = sinf((float)(token * 7 + e * 13)) * 2.0f;
        break;
    case 4: /* cycling: dominant expert rotates every 8 tokens */
        for (int e = 0; e < n_experts; e++) logits[e] = -1.0f;
        logits[(token / 8) % n_experts] = 3.5f;
        break;
    }
}

static const char *scenario_name(int s) {
    switch (s % 5) {
    case 0: return "confident ";
    case 1: return "balanced  ";
    case 2: return "two_way   ";
    case 3: return "noisy     ";
    case 4: return "cycling   ";
    default: return "unknown   ";
    }
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    int   n_tokens  = 512;
    int   n_experts = 8;
    int   k         = 2;          /* experts per token (n_experts_per_tok) */
    int   n_layers  = 32;
    float alpha     = 0.2f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens")  && i+1<argc) n_tokens  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--experts") && i+1<argc) n_experts = atoi(argv[++i]);
        if (!strcmp(argv[i], "--layers")  && i+1<argc) n_layers  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--alpha")   && i+1<argc) alpha     = (float)atof(argv[++i]);
    }

    printf("=== HDGL Routing Drift Test ===\n");
    printf("tokens=%d  experts=%d  top-k=%d  layers=%d  alpha=%.3f\n\n",
           n_tokens, n_experts, k, n_layers, alpha);

    /* Synthetic minimal model — only fields tb_route_experts reads */
    TB_GGUFModel model;
    memset(&model, 0, sizeof(model));
    model.vocab_size        = 32000;
    model.n_layers          = n_layers;
    model.n_experts         = n_experts;
    model.n_experts_per_tok = k;
    model.hidden_dim        = 4096;
    snprintf(model.arch, sizeof(model.arch), "mixtral");

    /* Two contexts: one HDGL-off (baseline), one HDGL-on */
    TB_InferCtx *ctx_off = tb_infer_create(&model, 0, 0.0f, 512, 0xDEADBEEFULL);
    TB_InferCtx *ctx_on  = tb_infer_create(&model, 1, alpha, 512, 0xDEADBEEFULL);
    if (!ctx_off || !ctx_on) {
        fprintf(stderr, "Failed to create inference contexts\n");
        return 1;
    }

    /* ── Per-scenario results ────────────────────────────────────────────── */
    printf("%-12s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "Scenario", "Window", "Jaccard↓", "WeightL1↑", "DriftFrac", "Verdict");
    printf("%-12s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "------------", "--------", "----------", "----------", "----------", "----------");

    int overall_fail = 0;

    for (int sc = 0; sc < 5; sc++) {
        /* Reset HDGL state for each scenario */
        memset(ctx_on->hdgl_routing_state, 0, sizeof(ctx_on->hdgl_routing_state));

        /* Accumulate metrics over three windows: early (0-63), mid (64-255), late (256-511) */
        struct { int start, end; const char *label; } windows[] = {
            {0,      63,  "early   "},
            {64,    255,  "mid     "},
            {256, n_tokens-1, "late    "},
        };
        int n_windows = (n_tokens >= 256) ? 3 : (n_tokens >= 64) ? 2 : 1;

        int first_row = 1;
        for (int w = 0; w < n_windows; w++) {
            double sum_jaccard = 0.0, sum_l1 = 0.0;
            int    n_mismatch  = 0;
            int    n_samples   = 0;

            for (int tok = windows[w].start; tok <= windows[w].end && tok < n_tokens; tok++) {
                /* Same gate logits seen by both paths */
                float gate[256];
                fill_logits(sc, tok, n_experts, gate);

                /* Baseline: gate-only top-k (no HDGL state) */
                int   base_idx[8]; float base_w[8];
                gate_only_topk(gate, n_experts, k, base_idx, base_w);

                /* Run HDGL-on routing at layer 0 (state accumulates) */
                /* We also run all other layers to accumulate state realistically */
                for (int layer = 0; layer < n_layers; layer++) {
                    TB_ExpertSelection sel = tb_route_experts(
                        ctx_on, tok * n_layers + layer, layer, gate, n_experts);

                    /* Only compare layer 0 to avoid redundant stats */
                    if (layer == 0) {
                        float hdgl_w[8] = {0};
                        /* Reconstruct per-expert weight from sel */
                        for (int i = 0; i < sel.k; i++)
                            hdgl_w[sel.expert_indices[i]] = sel.expert_weights[i];

                        float jac = expert_jaccard(base_idx, sel.expert_indices, k);
                        float l1  = weight_l1(base_idx, base_w,
                                              sel.expert_indices, sel.expert_weights, k);
                        sum_jaccard += jac;
                        sum_l1      += l1;
                        if (jac < 1.0f) n_mismatch++;
                        n_samples++;
                    }
                }
            }

            double avg_jac = sum_jaccard / n_samples;
            double avg_l1  = sum_l1      / n_samples;
            double drift_frac = (double)n_mismatch / n_samples;

            /* Verdict: FAIL if >20% of tokens get wrong experts OR mean L1 > 0.3 */
            int fail = (drift_frac > 0.20 || avg_l1 > 0.30);
            if (fail) overall_fail = 1;

            if (first_row) {
                printf("%-12s  %-8s  %10.4f  %10.4f  %10.4f  %s\n",
                       scenario_name(sc), windows[w].label,
                       avg_jac, avg_l1, drift_frac,
                       fail ? "\033[31mFAIL\033[0m" : "\033[32mOK\033[0m");
                first_row = 0;
            } else {
                printf("%-12s  %-8s  %10.4f  %10.4f  %10.4f  %s\n",
                       "", windows[w].label,
                       avg_jac, avg_l1, drift_frac,
                       fail ? "\033[31mFAIL\033[0m" : "\033[32mOK\033[0m");
            }
        }
        printf("\n");
    }

    /* ── Phase drift over time: track hdgl_routing_state evolution ─────── */
    printf("=== Phase drift: HDGL primary_phase over %d tokens (confident gate) ===\n",
           n_tokens);
    {
        memset(ctx_on->hdgl_routing_state, 0, sizeof(ctx_on->hdgl_routing_state));
        double prev_phase = 0.0;
        double total_phase_delta = 0.0;
        int    n_phase_samples = 0;
        int    checkpoints[] = {0, 1, 7, 15, 31, 63, 127, 255, 511, -1};

        printf("  tok   primary_phase      delta_per_tok  cumulative_delta\n");
        int cp = 0;
        float gate[256];
        fill_logits(0, 0, n_experts, gate);  /* confident gate, fixed */

        for (int tok = 0; tok < n_tokens && tok <= 511; tok++) {
            for (int layer = 0; layer < n_layers; layer++)
                tb_route_experts(ctx_on, tok, layer, gate, n_experts);

            /* Read primary_phase from the 32-byte opaque history buffer.
             * HDGL_History layout: last_feedback(u32) last_expert_id(i32)
             *                      primary_phase(f64) mirror_phase(f64) strand_idx(i32)
             * primary_phase is at byte offset 8. */
            double ph;
            memcpy(&ph, ctx_on->hdgl_routing_state + 8, sizeof(double));

            double delta = fabs(ph - prev_phase);
            total_phase_delta += delta;
            n_phase_samples++;
            prev_phase = ph;

            if (tok == checkpoints[cp]) {
                printf("  %4d  %+18.6f  %14.6f  %16.6f\n",
                       tok, ph, delta,
                       total_phase_delta);
                cp++;
                if (checkpoints[cp] < 0) break;
            }
        }

        double avg_delta = (n_phase_samples > 0)
                           ? total_phase_delta / n_phase_samples : 0.0;
        printf("\n  avg phase delta per token: %.6f\n", avg_delta);
        if (avg_delta > 0.05)
            printf("  \033[31mWARN: phase accumulates significantly — "
                   "routing will drift over long sequences.\033[0m\n");
        else
            printf("  \033[32mOK: phase stable.\033[0m\n");
    }

    printf("\n=== Summary ===\n");
    if (overall_fail)
        printf("\033[31mFAIL: HDGL routing diverges from gate baseline beyond "
               "acceptable thresholds.\n"
               "      Expert selection is being overridden by phase state, not gate logits.\n"
               "      Use --no-hdgl for coherent output.\033[0m\n");
    else
        printf("\033[32mOK: HDGL drift within acceptable bounds "
               "(Jaccard≥0.8, L1≤0.3, drift_frac≤0.20).\033[0m\n");

    tb_infer_free(ctx_off);
    tb_infer_free(ctx_on);
    printf("\n");
    return overall_fail ? 1 : 0;
}
