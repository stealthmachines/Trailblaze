/*
 * src/hdgl_critic_v33.c — HDGL Critic v33 Implementation
 *
 * TD(0)-learning mini-network for RL-driven backend scheduling.
 * Header: include/hdgl_critic_v33.h (from analog-prime-main)
 *
 * Architecture: 5-input → 8-hidden → 1-output (state value estimator)
 * Feature vector: [residue, coherence, amplitude, r_h_norm, acc_norm]
 *
 * Used in TRAILBLAZE Layer 4 to learn routing preferences from observed
 * task latencies and quality signals (replaces hardcoded Dₙ cost model).
 *
 * Build: gcc -O2 -std=c11 -DCRITIC_TEST src/hdgl_critic_v33.c -lm
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "hdgl_critic_v33.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ── Network dimensions ───────────────────────────────────────────────────── */
#define H1    8      /* hidden layer 1 */
#define H2    4      /* hidden layer 2 */
#define OUT   1

#define W1_N  (CRITIC_IN * H1)
#define B1_N  H1
#define W2_N  (H1 * H2)
#define B2_N  H2
#define W3_N  (H2 * OUT)
#define B3_N  OUT

#define TOTAL_WEIGHTS (W1_N + B1_N + W2_N + B2_N + W3_N + B3_N)

/* ── Learning hyperparameters ─────────────────────────────────────────────── */
#define CRITIC_LR      0.001f
#define CRITIC_GAMMA   0.95f    /* discount factor */
#define CRITIC_LAMBDA  0.01f    /* L2 regularisation */

/* ── Global network state ─────────────────────────────────────────────────── */
static float W1[W1_N], b1[B1_N];
static float W2[W2_N], b2[B2_N];
static float W3[W3_N], b3[B3_N];
static float g_W1[W1_N], g_b1[B1_N];
static float g_W2[W2_N], g_b2[B2_N];
static float g_W3[W3_N], g_b3[B3_N];
static float h1_buf[H1], h2_buf[H2];   /* activations from last forward */
static float reward_accum = 0.0f;
static int   n_updates    = 0;
static int   initialised  = 0;

/* ── Xavier initialisation ────────────────────────────────────────────────── */
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
    for (int i = 0; i < W3_N; i++) W3[i] = xavier(H2, OUT);
    for (int i = 0; i < B3_N; i++) b3[i] = 0.0f;
    memset(g_W1,0,sizeof(g_W1)); memset(g_b1,0,sizeof(g_b1));
    memset(g_W2,0,sizeof(g_W2)); memset(g_b2,0,sizeof(g_b2));
    memset(g_W3,0,sizeof(g_W3)); memset(g_b3,0,sizeof(g_b3));
    reward_accum = 0.0f;
    n_updates    = 0;
    initialised  = 1;
}

/* ── ELU activation ───────────────────────────────────────────────────────── */
static float elu(float x) {
    return (x >= 0.0f) ? x : (expf(x) - 1.0f);
}
static float elu_d(float y) {
    return (y >= 0.0f) ? 1.0f : (y + 1.0f);  /* dy/dx where y=elu(x) */
}

/* ── Forward pass ─────────────────────────────────────────────────────────── */
float critic_forward(const float s[CRITIC_IN]) {
    if (!initialised) critic_init();

    /* Layer 1: h1 = ELU(W1 @ s + b1) */
    for (int i = 0; i < H1; i++) {
        float z = b1[i];
        for (int j = 0; j < CRITIC_IN; j++) z += W1[i*CRITIC_IN + j] * s[j];
        h1_buf[i] = elu(z);
    }

    /* Layer 2: h2 = ELU(W2 @ h1 + b2) */
    for (int i = 0; i < H2; i++) {
        float z = b2[i];
        for (int j = 0; j < H1; j++) z += W2[i*H1 + j] * h1_buf[j];
        h2_buf[i] = elu(z);
    }

    /* Layer 3: out = W3 @ h2 + b3  (linear, no activation) */
    float out = b3[0];
    for (int j = 0; j < H2; j++) out += W3[j] * h2_buf[j];
    return out;
}

/* ── Observe + accumulate gradients ───────────────────────────────────────── */
void critic_observe(const float s[CRITIC_IN], float target) {
    if (!initialised) critic_init();

    float pred = critic_forward(s);
    float delta = target - pred;   /* TD error */

    /* Backprop — layer 3 */
    float d_out = -2.0f * delta;   /* MSE gradient */
    for (int j = 0; j < H2; j++) g_W3[j] += d_out * h2_buf[j];
    g_b3[0] += d_out;

    /* Layer 2 */
    float d_h2[H2];
    for (int j = 0; j < H2; j++) d_h2[j] = d_out * W3[j] * elu_d(h2_buf[j]);
    for (int i = 0; i < H2; i++) {
        g_b2[i] += d_h2[i];
        for (int j = 0; j < H1; j++) g_W2[i*H1 + j] += d_h2[i] * h1_buf[j];
    }

    /* Layer 1 */
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

/* ── Apply accumulated gradients (SGD + L2) ───────────────────────────────── */
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

/* ── Utilities ────────────────────────────────────────────────────────────── */
int critic_weight_count(void) { return TOTAL_WEIGHTS; }

void critic_pack_weights(float *out) {
    memcpy(out, W1, W1_N*sizeof(float)); out += W1_N;
    memcpy(out, b1, B1_N*sizeof(float)); out += B1_N;
    memcpy(out, W2, W2_N*sizeof(float)); out += W2_N;
    memcpy(out, b2, B2_N*sizeof(float)); out += B2_N;
    memcpy(out, W3, W3_N*sizeof(float)); out += W3_N;
    memcpy(out, b3, B3_N*sizeof(float));
}

void critic_print_stats(void) {
    printf("[critic_v33] weights=%d updates=%d reward_accum=%.4f\n",
           TOTAL_WEIGHTS, n_updates, reward_accum);
    float V_nominal[CRITIC_IN] = {0.5f, 0.5f, 1.0f, 0.5f, 0.5f};
    printf("[critic_v33] V(nominal)=%.4f\n", critic_forward(V_nominal));
}

int critic_save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    float buf[TOTAL_WEIGHTS];
    critic_pack_weights(buf);
    fwrite(buf, sizeof(float), TOTAL_WEIGHTS, f);
    fclose(f);
    return 0;
}

int critic_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    float buf[TOTAL_WEIGHTS];
    size_t n = fread(buf, sizeof(float), TOTAL_WEIGHTS, f);
    fclose(f);
    if ((int)n != TOTAL_WEIGHTS) return -1;
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

/* ────────────────────────────────────────────────────────────────────────────
 * Self-test
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef CRITIC_TEST
#include <assert.h>

int main(void) {
    printf("=== HDGL Critic v33 Test ===\n\n");
    critic_init();
    printf("[init] %d weights\n", critic_weight_count());

    /* Nominal state */
    float s_nom[CRITIC_IN] = {0.5f, 0.5f, 1.0f, 0.5f, 0.5f};
    float v0 = critic_forward(s_nom);
    printf("[forward] V(nominal)=%.4f\n", v0);

    /* Train toward target=1.0 for 200 iterations */
    float target = 1.0f;
    float loss_before = fabsf(critic_forward(s_nom) - target);
    for (int i = 0; i < 200; i++) {
        critic_observe(s_nom, target);
        if (i % 20 == 19) critic_update();
    }
    critic_update();
    float loss_after = fabsf(critic_forward(s_nom) - target);
    printf("[train] loss %.4f → %.4f (200 steps)\n", loss_before, loss_after);
    assert(loss_after < loss_before);
    printf("[train] PASS (loss decreased)\n\n");

    /* TD target */
    float s_next[CRITIC_IN] = {0.6f, 0.4f, 1.2f, 0.6f, 0.6f};
    float td = critic_td_target(0.5f, s_next);
    printf("[td_target] r=0.5 + γ·V(s')=%.4f = %.4f\n",
           critic_forward(s_next), td);

    /* Different states should give different values */
    float s_good[CRITIC_IN] = {0.1f, 0.9f, 1.5f, 0.8f, 0.7f};
    float s_bad[CRITIC_IN]  = {0.9f, 0.1f, 0.2f, 0.1f, 0.1f};
    float v_good = critic_forward(s_good);
    float v_bad  = critic_forward(s_bad);
    printf("[forward] V(high-coherence)=%.4f  V(low-coherence)=%.4f\n",
           v_good, v_bad);

    /* Save/load */
    critic_save("/tmp/critic_v33_test.bin");
    float w_before = critic_forward(s_nom);
    critic_init();  /* reset */
    critic_load("/tmp/critic_v33_test.bin");
    float w_after = critic_forward(s_nom);
    assert(fabsf(w_before - w_after) < 1e-6f);
    printf("[save/load] PASS (V=%.4f restored)\n", w_after);

    critic_print_stats();
    printf("\n=== Critic v33 PASS ===\n");
    return 0;
}
#endif
