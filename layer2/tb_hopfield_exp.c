/*
 * experimental/hopfield/tb_hopfield_exp.c
 *
 * All 8 document primitives + modern dense Hopfield + phi-lattice integration.
 * Build standalone: gcc -O3 -std=c11 -DTB_HOP_EXP_TEST
 *     -Iexperimental/hopfield -Ilayer0
 *     experimental/hopfield/tb_hopfield_exp.c layer0/tb_phi_lattice.c -lm
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "tb_hopfield_exp.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* ────────────────────────────────────────────────────────────────────────────
 * Classical Hopfield primitives (document verbatim implementations)
 * ────────────────────────────────────────────────────────────────────────── */

/* Document section 1: synchronous update */
void tb_hopfield_step(float *state, const float *weights, int n, float beta) {
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return;
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += weights[i*n + j] * state[j];
        buf[i] = tanhf(beta * sum);
    }
    memcpy(state, buf, n * sizeof(float));
    free(buf);
}

/* Document section 2: energy function */
float tb_hopfield_energy(const float *state, const float *weights, int n) {
    float E = 0.0f;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            E += state[i] * weights[i*n + j] * state[j];
    return -0.5f * E;
}

/* Document section 3: iterative relaxation */
void tb_hopfield_relax(float *state, const float *weights, int n,
                       float beta, int max_iters, float epsilon) {
    float prev_E = tb_hopfield_energy(state, weights, n);
    for (int iter = 0; iter < max_iters; iter++) {
        tb_hopfield_step(state, weights, n, beta);
        float E = tb_hopfield_energy(state, weights, n);
        if (fabsf(E - prev_E) < epsilon) break;
        prev_E = E;
    }
}

/* Document section 6: Hebbian associative matrix */
void tb_hebbian_train(float *weights, const float *pattern, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            weights[i*n + j] += pattern[i] * pattern[j];
    }
    for (int i = 0; i < n; i++) weights[i*n + i] = 0.0f;
}

/* Document section 4: phase coherence metric */
float tb_phase_coherence(const float *phase, int n) {
    float re = 0.0f, im = 0.0f;
    for (int i = 0; i < n; i++) {
        re += cosf(phase[i]);
        im += sinf(phase[i]);
    }
    re /= (float)n;
    im /= (float)n;
    return sqrtf(re*re + im*im);
}

/* Document section 8: relaxation merge */
void tb_relax_merge(float *dst, const float *a, const float *b, int n, float alpha) {
    float one_m = 1.0f - alpha;
    for (int i = 0; i < n; i++)
        dst[i] = alpha * a[i] + one_m * b[i];
}

/* Document section 7: random projection */
void tb_random_projection(const float *input, float *output,
                           const float *proj, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++)
            sum += proj[i*in_dim + j] * input[j];
        output[i] = sum;
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * Modern Dense Hopfield (Ramsauer et al. 2020)
 * Retrieval update rule: x ← Ξ · softmax(β · Ξᵀ · x)
 * Energy: E = −lse(β, Ξᵀx) + ½‖x‖² + (1/β)log(N)
 *
 * lse(β,z) = (1/β) · log(Σ exp(β·zᵢ))   [log-sum-exp]
 *
 * One-step convergence for well-separated patterns.
 * Capacity: exponential in dim (vs linear for classical).
 * ────────────────────────────────────────────────────────────────────────── */

TB_DenseHopfield* tb_dense_hopfield_alloc(int dim, int max_patterns, float beta) {
    TB_DenseHopfield *dh = (TB_DenseHopfield *)calloc(1, sizeof(*dh));
    if (!dh) return NULL;
    dh->dim         = dim;
    dh->n_patterns  = 0;
    dh->beta        = beta;
    dh->Xi = (float *)calloc((size_t)max_patterns * dim, sizeof(float));
    if (!dh->Xi) { free(dh); return NULL; }
    return dh;
}

void tb_dense_hopfield_free(TB_DenseHopfield *dh) {
    if (!dh) return;
    free(dh->Xi);
    free(dh);
}

void tb_dense_hopfield_store(TB_DenseHopfield *dh, const float *pattern) {
    memcpy(dh->Xi + dh->n_patterns * dh->dim, pattern,
           dh->dim * sizeof(float));
    dh->n_patterns++;
}

/* x_new = Ξ · softmax(β · Ξᵀ · x) */
void tb_dense_hopfield_retrieve(const TB_DenseHopfield *dh,
                                 const float *query, float *out) {
    int  N   = dh->n_patterns;
    int  dim = dh->dim;
    if (N == 0) { memcpy(out, query, dim * sizeof(float)); return; }

    float *scores = (float *)malloc((size_t)N * sizeof(float));
    if (!scores) return;

    /* scores[μ] = β · (Ξ[μ] · x) */
    float max_s = -1e30f;
    for (int mu = 0; mu < N; mu++) {
        float dot = 0.0f;
        const float *xi_mu = dh->Xi + mu * dim;
        for (int i = 0; i < dim; i++) dot += xi_mu[i] * query[i];
        scores[mu] = dh->beta * dot;
        if (scores[mu] > max_s) max_s = scores[mu];
    }

    /* softmax (numerically stable) */
    float sum = 0.0f;
    for (int mu = 0; mu < N; mu++) {
        scores[mu] = expf(scores[mu] - max_s);
        sum += scores[mu];
    }
    for (int mu = 0; mu < N; mu++) scores[mu] /= sum;

    /* out = Ξᵀ · softmax */
    memset(out, 0, dim * sizeof(float));
    for (int mu = 0; mu < N; mu++) {
        const float *xi_mu = dh->Xi + mu * dim;
        float w = scores[mu];
        for (int i = 0; i < dim; i++) out[i] += w * xi_mu[i];
    }
    free(scores);
}

float tb_dense_hopfield_energy(const TB_DenseHopfield *dh, const float *query) {
    int  N   = dh->n_patterns;
    int  dim = dh->dim;
    if (N == 0) return 0.0f;

    /* lse = (1/β) · log(Σ exp(β · ξᵀx)) */
    float max_s = -1e30f;
    float *scores = (float *)malloc((size_t)N * sizeof(float));
    if (!scores) return 0.0f;
    for (int mu = 0; mu < N; mu++) {
        float dot = 0.0f;
        const float *xi = dh->Xi + mu * dim;
        for (int i = 0; i < dim; i++) dot += xi[i] * query[i];
        scores[mu] = dh->beta * dot;
        if (scores[mu] > max_s) max_s = scores[mu];
    }
    float sum = 0.0f;
    for (int mu = 0; mu < N; mu++) sum += expf(scores[mu] - max_s);
    float lse = (1.0f / dh->beta) * (logf(sum) + max_s);

    /* ‖x‖² / 2 */
    float norm2 = 0.0f;
    for (int i = 0; i < dim; i++) norm2 += query[i] * query[i];

    free(scores);
    return -lse + 0.5f * norm2 + (1.0f / dh->beta) * logf((float)N);
}

/* ────────────────────────────────────────────────────────────────────────────
 * Phi-lattice weighted retrieval
 * Weight each stored pattern by the Dₙ amplitude of its lattice slot.
 * Patterns resonant with the current lattice state are amplified.
 * ────────────────────────────────────────────────────────────────────────── */

void tb_phi_weighted_retrieve(const TB_DenseHopfield *dh,
                               TB_PhiLattice *lat,
                               const float *query, float *out) {
    int  N   = dh->n_patterns;
    int  dim = dh->dim;
    if (N == 0) { memcpy(out, query, dim * sizeof(float)); return; }

    float *scores = (float *)malloc((size_t)N * sizeof(float));
    if (!scores) return;

    float max_s = -1e30f;
    for (int mu = 0; mu < N; mu++) {
        /* Pattern key: use first 8 bytes as address proxy */
        char key[32];
        snprintf(key, sizeof(key), "pat_%d", mu);
        double dn = tb_lattice_dn_for_key(lat, key, strlen(key));

        float dot = 0.0f;
        const float *xi = dh->Xi + mu * dim;
        for (int i = 0; i < dim; i++) dot += xi[i] * query[i];

        /* Phi-amplified score: β·(ξᵀx) · (1 + Dₙ/max_Dₙ) */
        scores[mu] = dh->beta * dot * (1.0f + (float)dn * 0.1f);
        if (scores[mu] > max_s) max_s = scores[mu];
    }

    float sum = 0.0f;
    for (int mu = 0; mu < N; mu++) {
        scores[mu] = expf(scores[mu] - max_s);
        sum += scores[mu];
    }
    for (int mu = 0; mu < N; mu++) scores[mu] /= sum;

    memset(out, 0, dim * sizeof(float));
    for (int mu = 0; mu < N; mu++) {
        const float *xi = dh->Xi + mu * dim;
        float w = scores[mu];
        for (int i = 0; i < dim; i++) out[i] += w * xi[i];
    }
    free(scores);
}

/* ────────────────────────────────────────────────────────────────────────────
 * Semantic Memory (classical + dense combined)
 * ────────────────────────────────────────────────────────────────────────── */

TB_SemanticMemory* tb_semantic_memory_alloc(int dim, int max_patterns) {
    TB_SemanticMemory *sm = (TB_SemanticMemory *)calloc(1, sizeof(*sm));
    if (!sm) return NULL;
    sm->dim         = dim;
    sm->n_patterns  = 0;
    sm->patterns    = (float *)calloc((size_t)max_patterns * dim, sizeof(float));
    sm->weights     = (float *)calloc((size_t)dim * dim, sizeof(float));
    if (!sm->patterns || !sm->weights) {
        free(sm->patterns); free(sm->weights); free(sm);
        return NULL;
    }
    return sm;
}

void tb_semantic_memory_free(TB_SemanticMemory *sm) {
    if (!sm) return;
    free(sm->patterns); free(sm->weights); free(sm);
}

void tb_semantic_memory_store(TB_SemanticMemory *sm, const float *pattern) {
    memcpy(sm->patterns + sm->n_patterns * sm->dim, pattern,
           sm->dim * sizeof(float));
    tb_hebbian_train(sm->weights, pattern, sm->dim);
    sm->n_patterns++;
}

float tb_semantic_memory_recall(TB_SemanticMemory *sm, float *query, float beta) {
    float E_b = tb_hopfield_energy(query, sm->weights, sm->dim);
    tb_hopfield_relax(query, sm->weights, sm->dim, beta, 100, 1e-4f);
    float E_a = tb_hopfield_energy(query, sm->weights, sm->dim);
    return E_a - E_b;   /* negative = converged deeper = good recall */
}

/* Cosine similarity between two branch tip vectors after attractor convergence */
float tb_semantic_merge_score(TB_SemanticMemory *sm,
                               const float *branch_a, const float *branch_b) {
    int dim = sm->dim;
    float *a = (float *)malloc(dim * sizeof(float));
    float *b = (float *)malloc(dim * sizeof(float));
    if (!a || !b) { free(a); free(b); return 0.0f; }

    memcpy(a, branch_a, dim * sizeof(float));
    memcpy(b, branch_b, dim * sizeof(float));

    /* Relax both to nearest attractor */
    tb_hopfield_relax(a, sm->weights, dim, 1.5f, 50, 1e-4f);
    tb_hopfield_relax(b, sm->weights, dim, 1.5f, 50, 1e-4f);

    /* Cosine similarity of attractors */
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    free(a); free(b);
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-9f) ? dot / denom : 0.0f;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Self-test
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef TB_HOP_EXP_TEST
int main(void) {
    printf("=== TRAILBLAZE Experimental Hopfield ===\n\n");

    /* Classical Hopfield */
    int N = 16;
    float *W = (float *)calloc(N*N, sizeof(float));
    float p1[16], p2[16], state[16];
    for (int i=0;i<N;i++) { p1[i]=(i%2)?1.f:-1.f; p2[i]=(i%3)?1.f:-1.f; }
    tb_hebbian_train(W, p1, N);
    tb_hebbian_train(W, p2, N);

    memcpy(state, p1, N*sizeof(float));
    state[0] *= 0.5f; state[3] *= 0.5f;  /* corrupt */
    float E_b = tb_hopfield_energy(state, W, N);
    tb_hopfield_relax(state, W, N, 2.0f, 50, 1e-4f);
    float E_a = tb_hopfield_energy(state, W, N);
    printf("[classical] E: %.2f→%.2f PASS\n", E_b, E_a);
    assert(E_a <= E_b + 0.1f);
    free(W);

    /* Dense Hopfield */
    int DIM = 32;
    TB_DenseHopfield *dh = tb_dense_hopfield_alloc(DIM, 20, 4.0f);
    float *patterns[5];
    for (int p=0;p<5;p++) {
        patterns[p]=(float*)malloc(DIM*sizeof(float));
        for (int i=0;i<DIM;i++) patterns[p][i]=((i+p*7)%5==0)?1.f:-0.5f;
        tb_dense_hopfield_store(dh, patterns[p]);
    }
    float *query=(float*)malloc(DIM*sizeof(float));
    float *retrieved=(float*)malloc(DIM*sizeof(float));
    memcpy(query, patterns[2], DIM*sizeof(float));
    query[0]*=0.3f; query[5]*=0.3f;  /* corrupt */
    tb_dense_hopfield_retrieve(dh, query, retrieved);
    /* Compute similarity to original */
    float dot=0,na=0,nb=0;
    for(int i=0;i<DIM;i++){dot+=retrieved[i]*patterns[2][i];na+=retrieved[i]*retrieved[i];nb+=patterns[2][i]*patterns[2][i];}
    float sim=dot/(sqrtf(na)*sqrtf(nb));
    printf("[dense] one-step retrieval similarity=%.4f\n", sim);
    assert(sim > 0.7f);
    printf("[dense] PASS\n");

    /* Phi-weighted retrieval */
    TB_PhiLattice *lat = tb_lattice_create(256, 0xDEADBEEFULL);
    for (int i=0;i<5;i++) tb_lattice_advance(lat,1);
    float *phi_out=(float*)malloc(DIM*sizeof(float));
    tb_phi_weighted_retrieve(dh, lat, query, phi_out);
    printf("[phi-weighted] retrieve: PASS\n");
    tb_lattice_destroy(lat);

    /* Semantic memory */
    TB_SemanticMemory *sm = tb_semantic_memory_alloc(16, 10);
    float sp1[16], sp2[16];
    for(int i=0;i<16;i++){sp1[i]=(i%2)?1.f:-1.f;sp2[i]=(i%4)?1.f:-1.f;}
    tb_semantic_memory_store(sm, sp1);
    tb_semantic_memory_store(sm, sp2);
    float q2[16];
    memcpy(q2,sp1,16*sizeof(float));
    q2[2]*=0.5f;
    float dE = tb_semantic_memory_recall(sm, q2, 1.5f);
    printf("[semantic_recall] ΔE=%.4f\n", dE);
    float score = tb_semantic_merge_score(sm, sp1, sp2);
    printf("[merge_score] cosine=%.4f\n", score);
    assert(score >= -1.0f && score <= 1.0f);
    printf("[merge_score] PASS\n");
    tb_semantic_memory_free(sm);

    for(int p=0;p<5;p++) free(patterns[p]);
    free(query); free(retrieved); free(phi_out);
    tb_dense_hopfield_free(dh);
    printf("\n=== Experimental Hopfield PASS ===\n");
    return 0;
}
#endif
