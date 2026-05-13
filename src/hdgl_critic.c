/*
 * src/hdgl_critic.c — TRAILBLAZE v0.3 Routing Critic
 *
 * TD(0)-learning mini-network that modulates hdgl_alpha based on routing quality.
 * Adapted from experimental/hdgl_critic_v33.c — same architecture, fixed header.
 *
 * Architecture: CRITIC_IN=5 → H1=8 → H2=4 → OUT=1
 * Activations:  ELU (hidden), linear (output)
 * Learning:     SGD + L2, accumulated gradients, flush via critic_update()
 *
 * Build: included via tb_infer.c (no separate compilation needed for now)
 *        For standalone test: gcc -O2 -std=c11 -DCRITIC_TEST src/hdgl_critic.c -lm
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "hdgl_critic.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ── Network dimensions ──────────────────────────────────────────────────── */
#define H1   8
#define H2   4
/* Use CRITIC_OUT to avoid clash with Windows SAL annotation #define OUT */
#define CRITIC_OUT 1

#define W1_N (CRITIC_IN * H1)
#define B1_N H1
#define W2_N (H1 * H2)
#define B2_N H2
#define W3_N (H2 * CRITIC_OUT)
#define B3_N CRITIC_OUT
#define TOTAL_WEIGHTS (W1_N + B1_N + W2_N + B2_N + W3_N + B3_N)

/* ── Learning hyperparameters ─────────────────────────────────────────────── */
#define CRITIC_LR     0.001f
#define CRITIC_GAMMA  0.95f
#define CRITIC_LAMBDA 0.01f

/* ── Global network state ─────────────────────────────────────────────────── */
static float W1[W1_N], b1[B1_N];
static float W2[W2_N], b2[B2_N];
static float W3[W3_N], b3[B3_N];
static float g_W1[W1_N], g_b1[B1_N];
static float g_W2[W2_N], g_b2[B2_N];
static float g_W3[W3_N], g_b3[B3_N];
static float h1_buf[H1], h2_buf[H2];
static float reward_accum = 0.0f;
static int   n_updates    = 0;
static int   initialised  = 0;

/* ── Xavier init ─────────────────────────────────────────────────────────── */
static float xavier(int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    return (2.0f * (float)rand() / (float)RAND_MAX - 1.0f) * limit;
}

void critic_init(void) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < W1_N; i++) W1[i] = xavier(CRITIC_IN, H1);
    for (int i = 0; i < B1_N; i++) b1[i] = 0.0f;
    for (int i = 0; i < W2_N; i++) W2[i] = xavier(H1, H2);
    for (int i = 0; i < B2_N; i++) b2[i] = 0.0f;
    for (int i = 0; i < W3_N; i++) W3[i] = xavier(H2, CRITIC_OUT);
    for (int i = 0; i < B3_N; i++) b3[i] = 0.0f;
    memset(g_W1, 0, sizeof(g_W1)); memset(g_b1, 0, sizeof(g_b1));
    memset(g_W2, 0, sizeof(g_W2)); memset(g_b2, 0, sizeof(g_b2));
    memset(g_W3, 0, sizeof(g_W3)); memset(g_b3, 0, sizeof(g_b3));
    reward_accum = 0.0f;
    n_updates    = 0;
    initialised  = 1;
}

/* ── ELU ─────────────────────────────────────────────────────────────────── */
static float elu(float x)   { return (x >= 0.0f) ? x : (expf(x) - 1.0f); }
static float elu_d(float y) { return (y >= 0.0f) ? 1.0f : (y + 1.0f); }

/* ── Sigmoid helper for alpha modulation output ──────────────────────────── */
static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ── Forward pass ─────────────────────────────────────────────────────────── */
float critic_forward(const float s[CRITIC_IN]) {
    if (!initialised) critic_init();

    for (int i = 0; i < H1; i++) {
        float z = b1[i];
        for (int j = 0; j < CRITIC_IN; j++) z += W1[i*CRITIC_IN + j] * s[j];
        h1_buf[i] = elu(z);
    }
    for (int i = 0; i < H2; i++) {
        float z = b2[i];
        for (int j = 0; j < H1; j++) z += W2[i*H1 + j] * h1_buf[j];
        h2_buf[i] = elu(z);
    }
    float out = b3[0];
    for (int j = 0; j < H2; j++) out += W3[j] * h2_buf[j];
    return out;
}

/*
 * critic_alpha_mod — map critic value to a [0.3, 1.0] alpha multiplier.
 * At cold start (weights ≈ 0) → sigmoid(0) = 0.5 → alpha_mod = 0.65.
 * After learning, high-confidence routing → critic value rises → alpha_mod → 1.0.
 * Low-quality routing → alpha_mod → 0.3 (suppresses HDGL influence).
 */
float critic_alpha_mod(const float s[CRITIC_IN]) {
    float v = critic_forward(s);
    return 0.3f + 0.7f * sigmoidf(v);
}

/* ── Observe + accumulate gradients (TD error) ───────────────────────────── */
void critic_observe(const float s[CRITIC_IN], float target) {
    if (!initialised) critic_init();

    float pred  = critic_forward(s);
    float delta = target - pred;

    float d_out = -2.0f * delta;
    for (int j = 0; j < H2; j++) g_W3[j] += d_out * h2_buf[j];
    g_b3[0] += d_out;

    float d_h2[H2];
    for (int j = 0; j < H2; j++) d_h2[j] = d_out * W3[j] * elu_d(h2_buf[j]);
    for (int i = 0; i < H2; i++) {
        g_b2[i] += d_h2[i];
        for (int j = 0; j < H1; j++) g_W2[i*H1 + j] += d_h2[i] * h1_buf[j];
    }

    float d_h1[H1] = {0};
    for (int j = 0; j < H1; j++) {
        for (int i = 0; i < H2; i++) d_h1[j] += d_h2[i] * W2[i*H1 + j];
        d_h1[j] *= elu_d(h1_buf[j]);
    }
    for (int i = 0; i < H1; i++) {
        g_b1[i] += d_h1[i];
        for (int j = 0; j < CRITIC_IN; j++) g_W1[i*CRITIC_IN + j] += d_h1[i] * s[j];
    }

    reward_accum += fabsf(delta);
    n_updates++;
}

/* ── Apply gradients (SGD + L2) ───────────────────────────────────────────── */
void critic_update(void) {
    if (n_updates == 0) return;
    float lr = CRITIC_LR / (float)n_updates;

    for (int i = 0; i < W1_N; i++) { W1[i] -= lr*(g_W1[i] + CRITIC_LAMBDA*W1[i]); g_W1[i]=0; }
    for (int i = 0; i < B1_N; i++) { b1[i] -= lr*g_b1[i]; g_b1[i]=0; }
    for (int i = 0; i < W2_N; i++) { W2[i] -= lr*(g_W2[i] + CRITIC_LAMBDA*W2[i]); g_W2[i]=0; }
    for (int i = 0; i < B2_N; i++) { b2[i] -= lr*g_b2[i]; g_b2[i]=0; }
    for (int i = 0; i < W3_N; i++) { W3[i] -= lr*(g_W3[i] + CRITIC_LAMBDA*W3[i]); g_W3[i]=0; }
    for (int i = 0; i < B3_N; i++) { b3[i] -= lr*g_b3[i]; g_b3[i]=0; }

    n_updates    = 0;
    reward_accum = 0.0f;
}

/* ── TD target ────────────────────────────────────────────────────────────── */
float critic_td_target(float observed_reward, const float s_next[CRITIC_IN]) {
    return observed_reward + CRITIC_GAMMA * critic_forward(s_next);
}

/* ── Save / load ──────────────────────────────────────────────────────────── */
int critic_save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    float buf[TOTAL_WEIGHTS];
    float *p = buf;
    memcpy(p, W1, W1_N*sizeof(float)); p+=W1_N;
    memcpy(p, b1, B1_N*sizeof(float)); p+=B1_N;
    memcpy(p, W2, W2_N*sizeof(float)); p+=W2_N;
    memcpy(p, b2, B2_N*sizeof(float)); p+=B2_N;
    memcpy(p, W3, W3_N*sizeof(float)); p+=W3_N;
    memcpy(p, b3, B3_N*sizeof(float));
    fwrite(buf, sizeof(float), TOTAL_WEIGHTS, f);
    fclose(f);
    return 0;
}

int critic_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    float buf[TOTAL_WEIGHTS];
    if ((int)fread(buf, sizeof(float), TOTAL_WEIGHTS, f) != TOTAL_WEIGHTS) {
        fclose(f); return -1;
    }
    fclose(f);
    float *p = buf;
    memcpy(W1, p, W1_N*sizeof(float)); p+=W1_N;
    memcpy(b1, p, B1_N*sizeof(float)); p+=B1_N;
    memcpy(W2, p, W2_N*sizeof(float)); p+=W2_N;
    memcpy(b2, p, B2_N*sizeof(float)); p+=B2_N;
    memcpy(W3, p, W3_N*sizeof(float)); p+=W3_N;
    memcpy(b3, p, B3_N*sizeof(float));
    initialised = 1;
    return 0;
}
