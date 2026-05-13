/* tb_analog_dispatch.h — Analog-driven compute dispatcher
 *
 * Bridges ll_analog.c's 8D Kuramoto oscillator (AnaOsc8D) into the
 * TRAILBLAZE inference engine as a hardware-path scheduler.
 *
 * The fold26 wu-wei insight applied to matmul:
 *   analyze_chunk(data) → DataCharacteristics → CompressionStrategy
 *   ≡
 *   oscillator_state(osc) → TBDispatchClass → matvec kernel selection
 *
 * Phase-to-path mapping (mirrors fold26_wuwei_stream select_chunk_strategy):
 *
 *   APHASE_PLUCK    (CV > 0.50) — high variance, phases scattered:
 *     → DISPATCH_AVX512: full 512-bit VNNI path, highest throughput,
 *       scatter-friendly (cache misses expected anyway)
 *
 *   APHASE_SUSTAIN  (CV 0.30–0.50) — absorbing structure:
 *     → DISPATCH_AVX2: 256-bit FMA path, balanced throughput/latency
 *
 *   APHASE_FINETUNE (CV 0.10–0.30) — refinement:
 *     → DISPATCH_AVX2: same path, tighter prefetch hints
 *
 *   APHASE_LOCK     (CV < 0.10) — settled consensus:
 *     → DISPATCH_SCALAR: cache-warm scalar path; at lock, working set
 *       fits in L1 and scalar avoids AVX transition penalty
 *
 * S(U) resonance discriminant (from ll_analog harmonic sync):
 *   S(U) = |Ω·e^(iπΛ)+1|  — encodes prime/composite resonance
 *   Used as a secondary signal: S(U) near 0 → phases nearly locked →
 *   bias toward scalar even if aphase hasn't transitioned yet.
 *
 * Build: compiled as part of tb_infer via build.sh (no CUDA required).
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Dispatch class: maps oscillator state → matvec kernel ────────────────── */
typedef enum {
    TB_DISPATCH_SCALAR  = 0,  /* scalar reference path (cache-warm, lock phase) */
    TB_DISPATCH_AVX2    = 1,  /* 256-bit AVX2 + FMA (sustain / finetune)        */
    TB_DISPATCH_AVX512  = 2,  /* 512-bit AVX-512 VNNI (pluck / high variance)   */
} TBDispatchClass;

/* ── Oscillator state snapshot (extracted from AnaOsc8D, no ll_analog.h dep) ─ */
typedef struct {
    double  phase_var;   /* CV = 1 − R ∈ [0,1]: Kuramoto coherence measure     */
    double  s_u;         /* S(U) = |Ω·e^(iπΛ)+1|: resonance discriminant       */
    double  lambda_u;    /* Λ_φ^(U): phi-log depth of field state               */
    double  omega_u;     /* Ω^(U): resonance envelope (persists between syncs)  */
    int     aphase;      /* 0=Pluck 1=Sustain 2=FineTune 3=Lock                 */
    int     steps;       /* total RK4 steps (used for warmup detection)         */
} TBOscSnapshot;

/* ── Runtime capability flags (set once at startup) ──────────────────────── */
typedef struct {
    int have_avx2;
    int have_avx512f;
    int have_avx512vnni;
    int have_fma;
} TBCpuCaps;

/* ── API ──────────────────────────────────────────────────────────────────── */

/* Detect CPU capabilities (call once at startup). */
void tb_dispatch_detect_caps(TBCpuCaps *caps);

/* Choose dispatch class given oscillator snapshot and detected caps.
 * Implements the fold26 wu-wei strategy selector for compute paths. */
TBDispatchClass tb_dispatch_classify(const TBOscSnapshot *snap,
                                     const TBCpuCaps *caps);

/* ── Fused dequant+dot kernels ────────────────────────────────────────────── *
 * Each operates on one row of the weight matrix (K weights) and returns
 * the dot product with x[0..K-1].  No intermediate float buffer needed.     */

/* Scalar: reference implementation, always correct. */
float tb_dot_q4k_scalar(const uint8_t *row, int K, const float *x);
float tb_dot_q3k_scalar(const uint8_t *row, int K, const float *x);
float tb_dot_q8_scalar (const uint8_t *row, int K, const float *x);
float tb_dot_bf16_scalar(const uint16_t *row, int K, const float *x);

/* AVX2: 256-bit FMA fused dequant+dot. */
float tb_dot_q4k_avx2(const uint8_t *row, int K, const float *x);
float tb_dot_q8_avx2 (const uint8_t *row, int K, const float *x);
float tb_dot_bf16_avx2(const uint16_t *row, int K, const float *x);

/* AVX-512: 512-bit VNNI/FMA fused dequant+dot. */
float tb_dot_q4k_avx512(const uint8_t *row, int K, const float *x);
float tb_dot_q8_avx512 (const uint8_t *row, int K, const float *x);

/* ── Dispatched matvec (replaces tb_gguf_dequant_matvec inner loop) ───────── *
 * W: packed weight matrix (M rows × K cols in native quant format)
 * qtype: GGUF quant type (0=F32, 1=F16, 30=BF16, 2=Q4_0, 8=Q8_0, 12=Q4_K…)
 * snap: current oscillator state (may be NULL → falls back to scalar)       */
void tb_dispatch_matvec(const void *W, int qtype, int M, int K,
                        const float *x, float *out,
                        const TBOscSnapshot *snap,
                        const TBCpuCaps *caps);

#ifdef __cplusplus
}
#endif
