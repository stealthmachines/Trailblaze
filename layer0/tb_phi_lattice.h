/*
 * tb_phi_lattice.h — TRAILBLAZE Layer 0
 * Phi-irrational resonance lattice, 8D Kuramoto oscillator,
 * phi_fold hash, PhiStream AEAD, backend registry.
 *
 * Pure C11, no external deps except libm.
 * Ported from: conscious-128-bit-floor/hdgl_bootloaderz.h,
 *              prime_ui.c, ll_analog.c (AnaOsc8D / APhase model).
 *
 * Build: gcc -O3 -march=native -std=c11 -lm tb_phi_lattice.c -o tb_l0_test
 */

#pragma once
#ifndef TB_PHI_LATTICE_H
#define TB_PHI_LATTICE_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants (mirrors ll_analog.c ANA_* + hdgl_analog_v31.c)
 * ============================================================================ */

#define TB_PHI              1.6180339887498948
#define TB_LN_PHI           0.4812118250596035
#define TB_LN2              0.6931471805599453
#define TB_PI               3.14159265358979323846
#define TB_ANA_DIMS         8
#define TB_GAMMA            0.02
#define TB_K_COUPLING       1.0
#define TB_CONSENSUS_EPS    1e-6
#define TB_CONSENSUS_N      100
#define TB_ANA_DT           0.01
#define TB_ANA_SHA_INTERVAL 8
#define TB_ANA_HARM_ALPHA   0.8
#define TB_ANA_HARM_PASSES  4
#define TB_MAX_SLOTS        524288   /* 2^19 — practical lattice max */
#define TB_DEFAULT_SLOTS    4096

/* APA flags (from hdgl_bootloaderz.h) */
#define TB_FLAG_SIGN_NEG    (1u << 0)
#define TB_FLAG_IS_NAN      (1u << 1)
#define TB_FLAG_GOI         (1u << 2)
#define TB_FLAG_GUZ         (1u << 3)
#define TB_FLAG_CONSENSUS   (1u << 4)

/* ============================================================================
 * Spiral8: 8-polytope geometry table
 * ============================================================================ */

typedef struct {
    int     dim;        /* 1..8 */
    int     wave_mode;  /* +1 propagating / 0 standing / -1 absorbing */
    double  alpha;      /* phase offset fraction */
    const char *name;
} TB_Spiral8Entry;

static const TB_Spiral8Entry TB_SPIRAL8[8] = {
    {1,  1, 0.000, "segment"},
    {2,  0, 0.250, "square"},
    {3, -1, 0.500, "cube"},
    {4,  1, 0.750, "tesseract"},
    {5,  0, 0.125, "5-cube"},
    {6, -1, 0.375, "6-cube"},
    {7,  1, 0.625, "7-cube"},
    {8,  0, 0.875, "8-cube"},
};

/* ============================================================================
 * Slot4096 — the fundamental lattice cell
 * (from hdgl_bootloaderz.h Slot4096 struct)
 * ============================================================================ */

typedef struct {
    double   value;          /* Weyl fractional part ∈ [0,1) */
    double   phase;          /* Kuramoto oscillator phase */
    double   freq;           /* natural frequency ωᵢ */
    double   amp_re;         /* complex amplitude real */
    double   amp_im;         /* complex amplitude imag */
    double   dn_amplitude;   /* Dₙ(r) resonance value */
    int      dimension;      /* Spiral8 dim 1..8 */
    int      wave_mode;      /* +1 / 0 / -1 */
    uint32_t flags;          /* TB_FLAG_* */
} TB_Slot;

/* ============================================================================
 * PhiFold — lattice-keyed hash (no SHA, no XOR)
 * S-box rebuilt from live lattice state on every epoch advance.
 * From conscious-128-bit-floor/prime_ui.c phi_fold_hash32/64.
 * ============================================================================ */

typedef struct {
    uint8_t  sbox_a[256];
    uint8_t  sbox_b[256];
    int      epoch_cached;   /* invalidated by tb_lattice_advance() */
} TB_PhiFold;

/* ============================================================================
 * PhiStream AEAD — additive Z/256Z stream cipher
 * Envelope: ctr[8] | tag[32] | ciphertext[n]
 * From conscious-128-bit-floor/prime_ui.c phi_stream_seal/open.
 * ============================================================================ */

typedef struct {
    TB_PhiFold *pf;
    uint64_t    ctr;
} TB_PhiStream;

/* ============================================================================
 * PhiLattice — the central state object
 * ============================================================================ */

typedef struct {
    TB_Slot    *slots;
    uint32_t    n_slots;
    int32_t     epoch;
    double      time;
    double      omega;       /* global driving frequency */
    double      phase_var;   /* Kuramoto order parameter: 1−R ∈ [0,1] */
    int         consensus_steps;
    uint64_t    seed_value;

    TB_PhiFold   phi_fold;
    TB_PhiStream phi_stream;
    uint8_t      prk[32];    /* current epoch PRK */
} TB_PhiLattice;

/* ============================================================================
 * Backend abstraction
 * ============================================================================ */

typedef enum {
    TB_BACKEND_CPU_SCALAR = 0,
    TB_BACKEND_CPU_AVX2   = 1,
    TB_BACKEND_ANALOG     = 2,   /* 8D Kuramoto, CUDA-free oracle */
    TB_BACKEND_METAL      = 3,
    TB_BACKEND_CUDA       = 4,
    TB_BACKEND_SPIRV      = 5,
    TB_BACKEND_LLVM_JIT   = 6,
    TB_BACKEND_COUNT      = 7,
} TB_Backend;

typedef enum {
    TB_OP_MATMUL        = 0,
    TB_OP_ATTENTION     = 1,
    TB_OP_ELEMENTWISE   = 2,
    TB_OP_REDUCTION     = 3,
    TB_OP_NTT           = 4,
    TB_OP_LATTICE_UPD   = 5,
    TB_OP_KURAMOTO      = 6,
    TB_OP_PHI_FOLD      = 7,
    TB_OP_COUNT         = 8,
} TB_OpClass;

typedef struct {
    TB_Backend  type;
    int         available;
    int         compute_units;
    /* cost model: T(n) ≈ a·nᵏ + b */
    double      cost_a[TB_OP_COUNT];
    double      cost_k[TB_OP_COUNT];
    double      cost_b[TB_OP_COUNT];
} TB_BackendDesc;

typedef struct {
    TB_BackendDesc descs[TB_BACKEND_COUNT];
    int            n_backends;
    TB_PhiLattice *lattice;      /* for Dₙ-modulated selection */
} TB_BackendRegistry;

/* ============================================================================
 * API: Lattice lifecycle
 * ============================================================================ */

/* Allocate + seed a lattice. seed=0 → entropy from OS + perf counters. */
TB_PhiLattice* tb_lattice_create(uint32_t n_slots, uint64_t seed);
void           tb_lattice_destroy(TB_PhiLattice *lat);

/* Run Kuramoto steps then ratchet epoch (lk_advance equivalent).
 * Rebuilds S-box, harvests 4-source entropy, invalidates sealed state. */
void tb_lattice_advance(TB_PhiLattice *lat, int steps);

/* Phi-lattice u128 address: phi_fold_hash64(parts...) → low 16 bytes as u128 */
void tb_lattice_phi_addr(TB_PhiLattice *lat,
                         const uint8_t **parts, const size_t *lens, int n_parts,
                         uint8_t out_addr[16]);

/* phi-spiral slot assignment: FNV1a→φ²-multiply→slot index */
uint32_t tb_lattice_slot_for_key(TB_PhiLattice *lat,
                                 const char *key, size_t key_len);

/* Dₙ(r) amplitude at slot for key */
double tb_lattice_dn_for_key(TB_PhiLattice *lat,
                              const char *key, size_t key_len);

/* U-field resonance S(U) — prime invariant discriminant */
void tb_lattice_s_u_resonance(TB_PhiLattice *lat,
                               double *out_M, double *out_L, double *out_S);

/* Human-readable describe into buf (JSON) */
int  tb_lattice_describe(TB_PhiLattice *lat, char *buf, size_t buf_len);

/* ============================================================================
 * API: PhiFold hash
 * ============================================================================ */

void tb_phi_fold_refresh(TB_PhiLattice *lat);
void tb_phi_fold_hash32(TB_PhiLattice *lat, const uint8_t *data, size_t n, uint8_t out[32]);
void tb_phi_fold_hash64(TB_PhiLattice *lat, const uint8_t *data, size_t n, uint8_t out[64]);
void tb_phi_fold_prk   (TB_PhiLattice *lat, const char *ctx, uint8_t out[32]);

/* ============================================================================
 * API: PhiStream AEAD
 * ============================================================================ */

/* Returns bytes written to out_buf (= 40 + plaintext_len).
 * Caller must provide out_buf of at least plaintext_len + 40 bytes. */
size_t tb_phi_stream_seal  (TB_PhiLattice *lat,
                             const uint8_t *pt,  size_t pt_len,
                             const char    *ctx,
                             uint8_t       *out, size_t out_cap);

/* Returns plaintext length on success, 0 on tamper/auth failure.
 * out must be >= envelope_len - 40 bytes. */
size_t tb_phi_stream_unseal(TB_PhiLattice *lat,
                             const uint8_t *env, size_t env_len,
                             const char    *ctx,
                             uint8_t       *out, size_t out_cap);

/* ============================================================================
 * API: Backend registry
 * ============================================================================ */

void tb_registry_init   (TB_BackendRegistry *reg, TB_PhiLattice *lat);
void tb_registry_add    (TB_BackendRegistry *reg, TB_BackendDesc desc);
TB_Backend tb_registry_select(TB_BackendRegistry *reg, TB_OpClass op, size_t n);

/* ============================================================================
 * Internal: Kuramoto step (exported for Layer 1 analog backend)
 * ============================================================================ */

void tb_kuramoto_step(TB_PhiLattice *lat, double dt);
void tb_update_dn_amplitudes(TB_PhiLattice *lat);

#ifdef __cplusplus
}
#endif

#endif /* TB_PHI_LATTICE_H */
