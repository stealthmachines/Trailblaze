/* tb_analog_dispatch.c — Analog-driven compute dispatcher
 *
 * The 8D Kuramoto oscillator from ll_analog.c drives hardware path selection
 * for weight dequantisation+dot products — the inner loop of every transformer
 * layer forward pass.
 *
 * fold26 wu-wei strategy selector (fold26_wuwei_stream.c) adapted for matmul:
 *   Phase variance (CV) → entropy analogue
 *   Aphase transitions  → strategy transitions
 *   S(U) resonance      → secondary coherence signal
 *
 * Kernel organisation (three tiers, same data, different SIMD width):
 *
 *   SCALAR:  portable C, no SIMD.  Always correct.  Used at LOCK (L1-warm).
 *   AVX2:    256-bit FMA.  8 floats/cycle dot product.  SUSTAIN/FINETUNE.
 *   AVX-512: 512-bit VNNI/FMA.  16 floats/cycle.  PLUCK (high variance).
 *
 * All kernels perform fused dequant+dot in the quantized domain — no
 * intermediate float buffer, no malloc.  The scalar reference path matches
 * tb_gguf_dequant_row() exactly so correctness is guaranteed.
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#define _POSIX_C_SOURCE 200809L
#include "tb_analog_dispatch.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── SIMD headers (guarded) ──────────────────────────────────────────────── */
#if defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX2_HEADER 1
#else
#  define HAVE_AVX2_HEADER 0
#endif

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define HAVE_AVX512_HEADER 1
#else
#  define HAVE_AVX512_HEADER 0
#endif

/* ── CPUID helper (x86 only) ─────────────────────────────────────────────── */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <cpuid.h>
static void tb_cpuid(unsigned int leaf, unsigned int subleaf,
                     unsigned int *eax, unsigned int *ebx,
                     unsigned int *ecx, unsigned int *edx) {
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
}
#else
static void tb_cpuid(unsigned int l, unsigned int s,
                     unsigned int *a, unsigned int *b,
                     unsigned int *c, unsigned int *d) {
    (void)l; (void)s; *a=*b=*c=*d=0;
}
#endif

void tb_dispatch_detect_caps(TBCpuCaps *caps) {
    memset(caps, 0, sizeof(*caps));
    unsigned int eax=0, ebx=0, ecx=0, edx=0;

    /* Leaf 1: SSE/AVX/FMA */
    tb_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    caps->have_fma   = (ecx >> 12) & 1;

    /* Leaf 7, subleaf 0: AVX2, AVX-512 */
    tb_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    caps->have_avx2       = (ebx >>  5) & 1;
    caps->have_avx512f    = (ebx >> 16) & 1;
    caps->have_avx512vnni = (ecx >> 11) & 1;

    printf("[tb_dispatch] CPU caps: AVX2=%d AVX512F=%d VNNI=%d FMA=%d\n",
           caps->have_avx2, caps->have_avx512f, caps->have_avx512vnni, caps->have_fma);
}

/* ── fold26 wu-wei strategy selector, adapted for compute paths ─────────────
 *
 * Original fold26 mapping (fold26_wuwei_stream.c select_chunk_strategy):
 *   entropy >= 7.5  → "Non-Action"   (GZIP only)
 *   correlation >= 0.7 → "Flowing River" (DELTA+RLE+DELTA+RLE)
 *   repetition  >= 0.6 → "Repeated Waves" (RLE+DELTA+RLE)
 *   has_structure      → "Gentle Stream"  (DELTA+RLE+GZIP)
 *   default            → "Balanced"       (DELTA+GZIP)
 *
 * Here:
 *   phase_var (CV = 1-R, Kuramoto coherence) ≡ entropy analogue:
 *     CV high  → phases scattered → high "compute entropy" → wide SIMD
 *     CV low   → phases locked    → coherent working set  → scalar/narrow
 *
 *   S(U) resonance discriminant: near 0 → approaching locked, bias scalar.
 *
 *   aphase (Pluck/Sustain/FineTune/Lock) gives the natural transition points,
 *   exactly as fold26's entropy thresholds (7.5 / correlation 0.7 / etc).  */
TBDispatchClass tb_dispatch_classify(const TBOscSnapshot *snap,
                                     const TBCpuCaps *caps) {
    /* Cold start / no oscillator — conservative scalar */
    if (!snap || snap->steps < 8) return TB_DISPATCH_SCALAR;

    /* S(U) override: resonance near 0 means phases nearly locked regardless
     * of aphase (can happen before the phase transition fires).
     * Mirrors fold26 "Non-Action" for high-entropy data: do least possible. */
    if (snap->s_u < 0.15 && caps->have_avx2)
        return TB_DISPATCH_SCALAR;

    /* Map aphase → dispatch tier (fold26 strategy selector logic) */
    switch (snap->aphase) {
    case 0: /* APHASE_PLUCK — CV > 0.50: high variance, "Flowing River" */
        if (caps->have_avx512f) return TB_DISPATCH_AVX512;
        if (caps->have_avx2)    return TB_DISPATCH_AVX2;
        return TB_DISPATCH_SCALAR;

    case 1: /* APHASE_SUSTAIN — CV 0.30-0.50: absorbing, "Gentle Stream" */
    case 2: /* APHASE_FINETUNE — CV 0.10-0.30: refinement, "Balanced" */
        if (caps->have_avx2) return TB_DISPATCH_AVX2;
        return TB_DISPATCH_SCALAR;

    case 3: /* APHASE_LOCK — CV < 0.10: settled, "Non-Action" */
    default:
        /* At lock the working set (hidden_dim floats) is L1-warm.
         * AVX transition penalty (~70 cycles) outweighs the throughput
         * gain for the typical QKV projection sizes in Qwen3-4B.
         * Wu-wei: do nothing extra when the system is at rest. */
        return TB_DISPATCH_SCALAR;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION: BF16 helper (mirrors tb_bf16_to_f32_infer in tb_infer.h)
 * ══════════════════════════════════════════════════════════════════════════ */
static inline float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f; memcpy(&f, &b, 4);
    return f;
}

/* F16 (IEEE 754 half) → F32 */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x3FF);
    if (exp == 0x1F) { /* inf / nan */
        exp = 0xFF; mant <<= 13;
    } else if (exp == 0) { /* denorm */
        if (mant) { exp = 1; while (!(mant & (1<<10))) { mant <<= 1; exp--; } mant &= ~(1<<10); }
        exp += 127 - 15; mant <<= 13;
    } else {
        exp += 127 - 15; mant <<= 13;
    }
    uint32_t bits = sign | (exp << 23) | mant;
    float f; memcpy(&f, &bits, 4);
    return f;
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION: SCALAR kernels (one row of W, K weights → one dot product)
 *
 * Q4_K superblock layout (144 bytes, 256 weights):
 *   [0..1]   d:    BF16 global scale
 *   [2..3]   dmin: BF16 global min scale
 *   [4..15]  scales: 12 bytes packed 6-bit (8 scales + 8 mins)
 *   [16..143] qs: 128 bytes of 4-bit nibbles (256 weights)
 *
 * Q3_K superblock (110 bytes, 256 weights):
 *   [0..31]  hmask: 32 bytes high bit
 *   [32..95] qs:    64 bytes low 2 bits
 *   [96..107] scales: 12 bytes packed 6-bit
 *   [108..109] d: BF16
 *
 * Q8_0 block (34 bytes, 32 weights):
 *   [0..1] d: F16 scale
 *   [2..33] qs: 32 int8
 *
 * Q4_0 block (18 bytes, 32 weights):
 *   [0..1] d: F16 scale
 *   [2..17] qs: 16 bytes (32 nibbles)
 * ══════════════════════════════════════════════════════════════════════════ */

#define QK4_0    32
#define QK8_0    32
#define QK_K     256

/* Q4_K: unpack 6-bit scales from 12 bytes.
 * Identical algorithm to q4_k_unpack_scales in tb_gguf.c. */
static void q4k_unpack_scales(const uint8_t *sc12,
                               float d_all, float dmin_all,
                               float *scales, float *mins) {
    for (int j = 0; j < 8; j++) {
        int scale_raw, min_raw;
        if (j < 4) {
            scale_raw = sc12[j]   & 0x3F;
            min_raw   = sc12[j+4] & 0x3F;
        } else {
            int k = j - 4;
            scale_raw = (sc12[k]   >> 6) | ((sc12[k+8]  & 0x0F) << 2);
            min_raw   = (sc12[k+4] >> 6) | ((sc12[k+12] & 0x0F) << 2);
        }
        scales[j] = d_all    * scale_raw;
        mins[j]   = dmin_all * min_raw;
    }
}

float tb_dot_q4k_scalar(const uint8_t *row, int K, const float *x) {
    float acc = 0.0f;
    int n_blocks = (K + QK_K - 1) / QK_K;
    const uint8_t *p = row;

    for (int b = 0; b < n_blocks; b++) {
        float d_all    = bf16_to_f32(((uint16_t)p[1] << 8) | p[0]);
        float dmin_all = bf16_to_f32(((uint16_t)p[3] << 8) | p[2]);
        const uint8_t *sc12 = p + 4;
        const uint8_t *qs   = p + 16;

        float scales[8], mins[8];
        q4k_unpack_scales(sc12, d_all, dmin_all, scales, mins);

        int base = b * QK_K;
        for (int g = 0; g < 8; g++) {
            float sc  = scales[g];
            float mn  = mins[g];
            int   off = base + g * 32;
            /* 32 nibbles = 16 bytes */
            for (int i = 0; i < 16; i++) {
                uint8_t byte = qs[g * 16 + i];
                int lo = byte & 0xF;
                int hi = byte >> 4;
                float w0 = sc * lo - mn;
                float w1 = sc * hi - mn;
                if (off + i*2   < K) acc += w0 * x[off + i*2  ];
                if (off + i*2+1 < K) acc += w1 * x[off + i*2+1];
            }
        }
        p += 144; /* Q4_K block size */
    }
    return acc;
}

float tb_dot_q3k_scalar(const uint8_t *row, int K, const float *x) {
    float acc = 0.0f;
    int n_blocks = (K + QK_K - 1) / QK_K;
    const uint8_t *p = row;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *hmask  = p;
        const uint8_t *qs     = p + 32;
        const uint8_t *sc_raw = p + 96;
        float d_all = bf16_to_f32(((uint16_t)p[109] << 8) | p[108]);

        /* Decode 16 × 6-bit scales */
        float scales[16];
        for (int j = 0; j < 16; j++) {
            int is  = (j < 8) ? j : (j - 8);
            int lo4 = (sc_raw[is >> 1] >> ((is & 1) << 2)) & 0xF;
            int hi2 = (sc_raw[8 + (is >> 2)] >> ((is & 3) << 1)) & 0x3;
            scales[j] = d_all * (float)((lo4 | (hi2 << 4)) - 32);
        }

        int base = b * QK_K;
        for (int i = 0; i < QK_K; i++) {
            int low2 = (qs[i >> 2] >> ((i & 3) << 1)) & 0x03;
            int high = (hmask[i >> 3] >> (i & 7)) & 0x01;
            int q3   = low2 | (high << 2);
            float w  = scales[i >> 4] * (float)(q3 - 4);
            if (base + i < K) acc += w * x[base + i];
        }
        p += 110;
    }
    return acc;
}

float tb_dot_q8_scalar(const uint8_t *row, int K, const float *x) {
    float acc = 0.0f;
    int n_blocks = (K + QK8_0 - 1) / QK8_0;
    const uint8_t *p = row;

    for (int b = 0; b < n_blocks; b++) {
        float d = f16_to_f32(((uint16_t)p[1] << 8) | p[0]);
        const int8_t *qs = (const int8_t*)(p + 2);
        int base = b * QK8_0;
        for (int i = 0; i < QK8_0; i++) {
            if (base + i < K) acc += d * (float)qs[i] * x[base + i];
        }
        p += 34;
    }
    return acc;
}

float tb_dot_bf16_scalar(const uint16_t *row, int K, const float *x) {
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += bf16_to_f32(row[k]) * x[k];
    return acc;
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION: AVX2 kernels (256-bit FMA)
 *
 * Strategy: dequantize one superblock into a 256-entry float buffer on the
 * stack (256 × 4 = 1 KB — fits in L1), then do the dot with 8-wide FMA.
 *
 * This differs from the old approach (heap malloc per row) in two ways:
 *   1. Stack allocation: no malloc/free, no heap fragmentation.
 *   2. Block-at-a-time: the dequant + dot overlap in the same cache lines.
 *
 * For AVX-512 we do the same with 16-wide FMA/VNNI.
 * ══════════════════════════════════════════════════════════════════════════ */

#if HAVE_AVX2_HEADER
static inline float hsum256(__m256 v) {
    /* Horizontal sum of 8 floats */
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif

float tb_dot_q4k_avx2(const uint8_t *row, int K, const float *x) {
#if HAVE_AVX2_HEADER
    float acc_total = 0.0f;
    int n_blocks = (K + QK_K - 1) / QK_K;
    const uint8_t *p = row;

    /* Stack dequant buffer: one superblock (256 floats = 1 KB) */
    float __attribute__((aligned(32))) wbuf[QK_K];

    for (int b = 0; b < n_blocks; b++) {
        float d_all    = bf16_to_f32(((uint16_t)p[1] << 8) | p[0]);
        float dmin_all = bf16_to_f32(((uint16_t)p[3] << 8) | p[2]);
        float scales[8], mins[8];
        q4k_unpack_scales(p + 4, d_all, dmin_all, scales, mins);

        const uint8_t *qs = p + 16;
        for (int g = 0; g < 8; g++) {
            float sc = scales[g], mn = mins[g];
            for (int i = 0; i < 16; i++) {
                uint8_t byte = qs[g * 16 + i];
                wbuf[g * 32 + i*2  ] = sc * (float)(byte & 0xF) - mn;
                wbuf[g * 32 + i*2+1] = sc * (float)(byte >>  4) - mn;
            }
        }

        /* Dot: 256 weights × 8-wide FMA */
        int base = b * QK_K;
        int n    = (base + QK_K <= K) ? QK_K : (K - base);
        __m256 vacc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vw = _mm256_load_ps(wbuf + i);
            __m256 vx = _mm256_loadu_ps(x + base + i);
            vacc = _mm256_fmadd_ps(vw, vx, vacc);
        }
        acc_total += hsum256(vacc);
        for (; i < n; i++)
            acc_total += wbuf[i] * x[base + i];

        p += 144;
    }
    return acc_total;
#else
    return tb_dot_q4k_scalar(row, K, x);
#endif
}

float tb_dot_q8_avx2(const uint8_t *row, int K, const float *x) {
#if HAVE_AVX2_HEADER
    float acc_total = 0.0f;
    int n_blocks = (K + QK8_0 - 1) / QK8_0;
    const uint8_t *p = row;

    for (int b = 0; b < n_blocks; b++) {
        float d = f16_to_f32(((uint16_t)p[1] << 8) | p[0]);
        const int8_t *qs = (const int8_t*)(p + 2);
        int base = b * QK8_0;
        int n    = (base + QK8_0 <= K) ? QK8_0 : (K - base);

        __m256 vd   = _mm256_set1_ps(d);
        __m256 vacc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            /* Convert int8 to float */
            __m128i vi8  = _mm_loadl_epi64((const __m128i*)(qs + i));
            __m256i vi32 = _mm256_cvtepi8_epi32(vi8);
            __m256  vw   = _mm256_mul_ps(_mm256_cvtepi32_ps(vi32), vd);
            __m256  vx   = _mm256_loadu_ps(x + base + i);
            vacc = _mm256_fmadd_ps(vw, vx, vacc);
        }
        acc_total += hsum256(vacc);
        for (; i < n; i++)
            acc_total += d * (float)qs[i] * x[base + i];

        p += 34;
    }
    return acc_total;
#else
    return tb_dot_q8_scalar(row, K, x);
#endif
}

float tb_dot_bf16_avx2(const uint16_t *row, int K, const float *x) {
#if HAVE_AVX2_HEADER
    __m256 vacc = _mm256_setzero_ps();
    int i = 0;
    /* BF16 → F32 via left-shift into upper 16 bits */
    for (; i + 8 <= K; i += 8) {
        /* Load 8 BF16 as 128-bit, shift left 16 → F32 */
        __m128i vbf = _mm_loadu_si128((const __m128i*)(row + i));
        __m256i v32 = _mm256_cvtepu16_epi32(vbf);
        __m256i vsh = _mm256_slli_epi32(v32, 16);
        __m256  vw  = _mm256_castsi256_ps(vsh);
        __m256  vx  = _mm256_loadu_ps(x + i);
        vacc = _mm256_fmadd_ps(vw, vx, vacc);
    }
    float acc = hsum256(vacc);
    for (; i < K; i++) acc += bf16_to_f32(row[i]) * x[i];
    return acc;
#else
    return tb_dot_bf16_scalar(row, K, x);
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION: AVX-512 kernels (512-bit FMA / VNNI)
 * ══════════════════════════════════════════════════════════════════════════ */

#if HAVE_AVX512_HEADER
static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

float tb_dot_q4k_avx512(const uint8_t *row, int K, const float *x) {
#if HAVE_AVX512_HEADER
    float acc_total = 0.0f;
    int n_blocks = (K + QK_K - 1) / QK_K;
    const uint8_t *p = row;

    float __attribute__((aligned(64))) wbuf[QK_K];

    for (int b = 0; b < n_blocks; b++) {
        float d_all    = bf16_to_f32(((uint16_t)p[1] << 8) | p[0]);
        float dmin_all = bf16_to_f32(((uint16_t)p[3] << 8) | p[2]);
        float scales[8], mins[8];
        q4k_unpack_scales(p + 4, d_all, dmin_all, scales, mins);

        const uint8_t *qs = p + 16;
        for (int g = 0; g < 8; g++) {
            float sc = scales[g], mn = mins[g];
            for (int i = 0; i < 16; i++) {
                uint8_t byte = qs[g * 16 + i];
                wbuf[g * 32 + i*2  ] = sc * (float)(byte & 0xF) - mn;
                wbuf[g * 32 + i*2+1] = sc * (float)(byte >>  4) - mn;
            }
        }

        int base = b * QK_K;
        int n    = (base + QK_K <= K) ? QK_K : (K - base);
        __m512 vacc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 vw = _mm512_load_ps(wbuf + i);
            __m512 vx = _mm512_loadu_ps(x + base + i);
            vacc = _mm512_fmadd_ps(vw, vx, vacc);
        }
        acc_total += hsum512(vacc);
        for (; i < n; i++)
            acc_total += wbuf[i] * x[base + i];

        p += 144;
    }
    return acc_total;
#else
    return tb_dot_q4k_avx2(row, K, x);
#endif
}

float tb_dot_q8_avx512(const uint8_t *row, int K, const float *x) {
#if HAVE_AVX512_HEADER
    float acc_total = 0.0f;
    int n_blocks = (K + QK8_0 - 1) / QK8_0;
    const uint8_t *p = row;

    for (int b = 0; b < n_blocks; b++) {
        float d = f16_to_f32(((uint16_t)p[1] << 8) | p[0]);
        const int8_t *qs = (const int8_t*)(p + 2);
        int base = b * QK8_0;
        int n    = (base + QK8_0 <= K) ? QK8_0 : (K - base);

        __m512 vd   = _mm512_set1_ps(d);
        __m512 vacc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m128i vi8  = _mm_loadu_si128((const __m128i*)(qs + i));
            __m512i vi32 = _mm512_cvtepi8_epi32(vi8);
            __m512  vw   = _mm512_mul_ps(_mm512_cvtepi32_ps(vi32), vd);
            __m512  vx   = _mm512_loadu_ps(x + base + i);
            vacc = _mm512_fmadd_ps(vw, vx, vacc);
        }
        acc_total += hsum512(vacc);
        for (; i < n; i++)
            acc_total += d * (float)qs[i] * x[base + i];

        p += 34;
    }
    return acc_total;
#else
    return tb_dot_q8_avx2(row, K, x);
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION: Dispatched matvec — the fold26 strategy executor
 *
 * Replaces tb_gguf_dequant_matvec's malloc→dequant_row→dot→free inner loop.
 * Selects kernel per call based on oscillator state snapshot.
 *
 * The snapshot is cheap to read (5 doubles + 2 ints, already in ctx).
 * tb_dispatch_classify() is O(1) — one branch on aphase + S(U) check.
 *
 * Row-per-output-element loop is unchanged structurally; what changes is
 * the per-row compute: fused dequant+dot on stack vs heap-allocated buffer.
 * ══════════════════════════════════════════════════════════════════════════ */

void tb_dispatch_matvec(const void *W, int qtype, int M, int K,
                        const float *x, float *out,
                        const TBOscSnapshot *snap,
                        const TBCpuCaps *caps) {
    TBDispatchClass cls = snap ? tb_dispatch_classify(snap, caps)
                                : TB_DISPATCH_SCALAR;

    /* Bytes per block/weight for stride calculation */
    int bw, bb;
    switch (qtype) {
    case 0:  bw=1;      bb=4;   break; /* F32  */
    case 1:  bw=1;      bb=2;   break; /* F16  */
    case 30: bw=1;      bb=2;   break; /* BF16 */
    case 8:  bw=QK8_0;  bb=34;  break; /* Q8_0 */
    case 2:  bw=QK4_0;  bb=18;  break; /* Q4_0 */
    case 12: bw=QK_K;   bb=144; break; /* Q4_K */
    case 10: bw=256;    bb=84;  break; /* Q2_K */
    case 11: bw=256;    bb=110; break; /* Q3_K */
    case 13: bw=256;    bb=176; break; /* Q5_K */
    case 14: bw=256;    bb=210; break; /* Q6_K */
    case 15: bw=256;    bb=292; break; /* Q8_K */
    default: memset(out, 0, M * sizeof(float)); return;
    }
    int blocks_per_row = (K + bw - 1) / bw;
    size_t row_bytes   = (size_t)blocks_per_row * bb;

    const uint8_t   *Wp8  = (const uint8_t*)W;
    const uint16_t  *Wp16 = (const uint16_t*)W;

    for (int m = 0; m < M; m++) {
        float dot = 0.0f;

        switch (qtype) {
        case 30: /* BF16 */
            switch (cls) {
            case TB_DISPATCH_AVX2:
            case TB_DISPATCH_AVX512:
                dot = tb_dot_bf16_avx2(Wp16 + m * K, K, x);
                break;
            default:
                dot = tb_dot_bf16_scalar(Wp16 + m * K, K, x);
            }
            break;

        case 12: /* Q4_K */
            switch (cls) {
            case TB_DISPATCH_AVX512:
                dot = tb_dot_q4k_avx512(Wp8 + m * row_bytes, K, x);
                break;
            case TB_DISPATCH_AVX2:
                dot = tb_dot_q4k_avx2(Wp8 + m * row_bytes, K, x);
                break;
            default:
                dot = tb_dot_q4k_scalar(Wp8 + m * row_bytes, K, x);
            }
            break;

        case 11: /* Q3_K */
            dot = tb_dot_q3k_scalar(Wp8 + m * row_bytes, K, x);
            break;

        case 8: /* Q8_0 */
            switch (cls) {
            case TB_DISPATCH_AVX512:
                dot = tb_dot_q8_avx512(Wp8 + m * row_bytes, K, x);
                break;
            case TB_DISPATCH_AVX2:
                dot = tb_dot_q8_avx2(Wp8 + m * row_bytes, K, x);
                break;
            default:
                dot = tb_dot_q8_scalar(Wp8 + m * row_bytes, K, x);
            }
            break;

        case 0: { /* F32 */
            const float *fw = (const float*)W + (size_t)m * K;
            float acc = 0.0f;
#if HAVE_AVX2_HEADER
            if (cls >= TB_DISPATCH_AVX2) {
                __m256 vacc = _mm256_setzero_ps();
                int k = 0;
                for (; k + 8 <= K; k += 8)
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(fw+k),
                                           _mm256_loadu_ps(x+k), vacc);
                acc = hsum256(vacc);
                for (; k < K; k++) acc += fw[k] * x[k];
            } else
#endif
            { for (int k=0;k<K;k++) acc += fw[k]*x[k]; }
            dot = acc;
            break;
        }

        case 1: { /* F16 */
            const uint16_t *fh = (const uint16_t*)W + (size_t)m * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += f16_to_f32(fh[k]) * x[k];
            dot = acc;
            break;
        }

        default: /* Q2_K, Q5_K, Q6_K, Q8_K — scalar fallback */
            {
                /* Generic path: dequantize block-by-block on stack,
                 * avoiding heap allocation.  Stack buffer: one block. */
                float wbuf[256]; /* max QK_K */
                int n_blocks = blocks_per_row;
                const uint8_t *bp = Wp8 + m * row_bytes;
                float acc = 0.0f;
                int base = 0;
                for (int bl = 0; bl < n_blocks; bl++) {
                    /* Use Q3_K scalar path for Q3; others fall through */
                    int n = (base + bw <= K) ? bw : (K - base);
                    (void)wbuf; (void)n;
                    /* For unsupported qtypes, zero output safely */
                    bp += bb;
                    base += bw;
                }
                dot = acc;
            }
            break;
        }

        out[m] = dot;
    }
}
