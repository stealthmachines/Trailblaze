/*
 * experimental/hopfield/tb_hopfield_exp.h
 *
 * Standalone Hopfield primitives for use as a semantic overlay in Layer 5.
 * Per the uploaded document: these belong in experimental/ NOT in Layer 0/1 core.
 *
 * Extended beyond the 8 document primitives to include:
 *   - Modern dense Hopfield (Ramsauer et al. 2020): exponential energy
 *   - Sparse pattern retrieval (top-k activation)
 *   - Branch-aware memory consolidation
 *   - Phi-lattice weighted attention query
 */

#pragma once
#ifndef TB_HOPFIELD_EXP_H
#define TB_HOPFIELD_EXP_H

#include <stdint.h>
#include <stddef.h>
#include "../../layer0/tb_phi_lattice.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Classical Hopfield (from uploaded document — verbatim)
 * ============================================================================ */

/* state[i] = tanh(β · Σⱼ W[i,j] · state[j])  */
void tb_hopfield_step(float *state, const float *weights, int n, float beta);

/* E = −½ xᵀWx */
float tb_hopfield_energy(const float *state, const float *weights, int n);

/* Iterative relaxation until |ΔE| < epsilon */
void tb_hopfield_relax(float *state, const float *weights, int n,
                       float beta, int max_iters, float epsilon);

/* W += x⊗xᵀ, diag = 0  */
void tb_hebbian_train(float *weights, const float *pattern, int n);

/* R = |mean(e^{iθ})| — phase coherence */
float tb_phase_coherence(const float *phase, int n);

/* dst = α·a + (1−α)·b */
void tb_relax_merge(float *dst, const float *a, const float *b, int n, float alpha);

/* output = proj @ input */
void tb_random_projection(const float *input, float *output,
                           const float *proj, int in_dim, int out_dim);

/* ============================================================================
 * Modern Dense Hopfield (Ramsauer et al. 2020)
 * Energy: E = −lse(β, Ξᵀx) + ½xᵀx + (1/β)·log(N) + (M/2)·log(2π)
 * Retrieval: x_new = Ξ · softmax(β · Ξᵀ · x)
 *
 * Capacity: O(2^(n/2)) vs classical O(n/log(n))
 * Retrieval: one step vs many
 * Use: high-capacity semantic pattern library for branch merge scoring
 * ============================================================================ */

typedef struct {
    float  *Xi;        /* stored patterns matrix: (n_patterns × dim) */
    int     n_patterns;
    int     dim;
    float   beta;      /* inverse temperature; higher = sharper retrieval */
} TB_DenseHopfield;

TB_DenseHopfield* tb_dense_hopfield_alloc(int dim, int max_patterns, float beta);
void              tb_dense_hopfield_free(TB_DenseHopfield *dh);
void              tb_dense_hopfield_store(TB_DenseHopfield *dh, const float *pattern);

/* One-shot retrieval: x_new = Ξ · softmax(β · Ξᵀ · x) */
void tb_dense_hopfield_retrieve(const TB_DenseHopfield *dh,
                                 const float *query, float *out);

/* Energy of query state under stored patterns */
float tb_dense_hopfield_energy(const TB_DenseHopfield *dh, const float *query);

/* ============================================================================
 * Phi-lattice weighted retrieval
 * Use Dₙ amplitudes as weights on stored patterns during retrieval.
 * Patterns near high-Dₙ slots are preferred (resonant recall).
 * ============================================================================ */

void tb_phi_weighted_retrieve(const TB_DenseHopfield *dh,
                               TB_PhiLattice *lat,
                               const float *query, float *out);

/* ============================================================================
 * Semantic consolidation: compress branch history to Hopfield attractor
 * ============================================================================ */

typedef struct {
    float *patterns;     /* (n_patterns × dim) flattened */
    int    n_patterns;
    int    dim;
    float *weights;      /* n×n Hopfield weight matrix */
} TB_SemanticMemory;

TB_SemanticMemory* tb_semantic_memory_alloc(int dim, int max_patterns);
void               tb_semantic_memory_free(TB_SemanticMemory *sm);
void               tb_semantic_memory_store(TB_SemanticMemory *sm,
                                             const float *pattern);
/* Retrieve closest attractor to query, return convergence energy */
float              tb_semantic_memory_recall(TB_SemanticMemory *sm,
                                              float *query, float beta);
/* Branch merge scoring: similarity of two branch tip attractors */
float              tb_semantic_merge_score(TB_SemanticMemory *sm,
                                            const float *branch_a,
                                            const float *branch_b);

#ifdef __cplusplus
}
#endif

#endif /* TB_HOPFIELD_EXP_H */
