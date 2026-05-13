/*
 * Framework-Native Vector Container Implementation
 * Ported from stealthmachines/AnalogContainer1
 * https://github.com/stealthmachines/AnalogContainer1
 *
 * Pure mathematical execution – no Docker, no Python, no runtime overhead.
 * Compiled size: ~20 KB  |  Startup: <1ms  |  Memory: ~1 MB
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#include "vector_container.h"
#include "sha256_minimal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ─── Continuous function helpers ─────────────────────────────────────────── */

static void cf_init(ContinuousFunction *func, const double *samples, size_t count) {
    func->samples = malloc(count * sizeof(double));
    memcpy(func->samples, samples, count * sizeof(double));
    func->count = count;

    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum    += samples[i];
        sum_sq += samples[i] * samples[i];
    }
    func->mean     = sum / (double)count;
    func->variance = (sum_sq / (double)count) - (func->mean * func->mean);
    func->phi_weight = fmod(sum * PHI, 1.0);

    double pm = fmod(func->phi_weight * 3.0, 3.0);
    func->state = (pm < 1.0) ? TERNARY_NEGATIVE :
                  (pm < 2.0) ? TERNARY_NEUTRAL   : TERNARY_POSITIVE;
}

static void cf_free(ContinuousFunction *func) {
    if (func->samples) { free(func->samples); func->samples = NULL; }
}

/* ─── Fourier transform (φ-harmonic) ──────────────────────────────────────── */

static void fourier_encode_cf(const ContinuousFunction *func, FourierBasis *basis) {
    double period = (double)func->count;
    basis->mean   = func->mean;

    for (int n = 0; n < FOURIER_COEFFS; n++) {
        double sc = 0.0, ss = 0.0;
        double freq = (double)n * PHI;

        for (size_t t = 0; t < func->count; t++) {
            double angle = 2.0 * M_PI * freq * (double)t / period;
            sc += func->samples[t] * cos(angle);
            ss += func->samples[t] * sin(angle);
        }
        basis->cos_basis[n]  = sc / (double)func->count;
        basis->sin_basis[n]  = ss / (double)func->count;
        basis->coeff_re[n] = sc / (double)func->count;
        basis->coeff_im[n] = ss / (double)func->count;
    }

    double max_amp = 0.0;
    for (int n = 0; n < FOURIER_COEFFS; n++) {
        double amp = sqrt(basis->cos_basis[n]*basis->cos_basis[n] +
                          basis->sin_basis[n]*basis->sin_basis[n]);
        if (amp > max_amp) max_amp = amp;
    }
    basis->scale         = max_amp;
    basis->temporal_phase = fmod(basis->cos_basis[0] * PI_PHI, 2.0 * M_PI);
}

/* ─── DCT transform ────────────────────────────────────────────────────────── */

static void dct_encode_cf(const ContinuousFunction *func, DCTBasis *basis) {
    basis->mean = func->mean;
    for (int k = 0; k < DCT_COEFFS; k++) {
        double sum  = 0.0;
        double norm = (k == 0) ? sqrt(1.0 / (double)func->count)
                                : sqrt(2.0 / (double)func->count);
        for (size_t n = 0; n < func->count; n++)
            sum += func->samples[n] * cos(M_PI * k * (n + 0.5) / (double)func->count);
        basis->coefficients[k] = norm * sum;
    }
    double max_c = 0.0;
    for (int k = 0; k < DCT_COEFFS; k++)
        if (fabs(basis->coefficients[k]) > max_c) max_c = fabs(basis->coefficients[k]);
    basis->scale = max_c;
}

/* ─── Breathing convergence (POTSafeMath v9.3) ─────────────────────────────── */

void breathing_initialize_seeds(VectorContext *ctx) {
    for (int i = 0; i < BREATHING_SEEDS; i++) {
        BreathingSeed *s = &ctx->seeds[i];
        s->seed_id        = (uint64_t)i;
        s->fitness        = 0.0;
        s->phi_weight     = 1.0;
        s->breathing_phase = fmod((double)i * PI_PHI, 2.0 * M_PI);

        srand((unsigned int)(i * 0xDEADBEEFu));
        for (int j = 0; j < VECTOR_DIM; j++)
            s->vector[j] = fmod((double)rand() / (double)RAND_MAX * PHI, 1.0);
    }
}

double breathing_compute_fitness(const BreathingSeed *seed, const double *target) {
    double dist = 0.0;
    for (int i = 0; i < VECTOR_DIM; i++) {
        double d = seed->vector[i] - target[i];
        dist += d * d;
    }
    return 1.0 / (sqrt(dist) + 1e-12);
}

static void breathing_mutate(BreathingSeed *seed, double rate) {
    srand((unsigned int)(seed->seed_id * (uint64_t)time(NULL)));
    for (int i = 0; i < VECTOR_DIM; i++) {
        double m = ((double)rand() / (double)RAND_MAX - 0.5) * 2.0 * rate * INV_PHI;
        seed->vector[i] += m;
        if (seed->vector[i] < 0.0) seed->vector[i] = 0.0;
        if (seed->vector[i] > 1.0) seed->vector[i] = 1.0;
    }
}

static void breathing_toward(BreathingSeed *seed, const BreathingSeed *target, double factor) {
    for (int i = 0; i < VECTOR_DIM; i++)
        seed->vector[i] += factor * seed->phi_weight * (target->vector[i] - seed->vector[i]);

    double sum = 0.0;
    for (int i = 0; i < VECTOR_DIM; i++) sum += seed->vector[i];
    seed->phi_weight = fmod(sum * PHI, 1.0);
}

void breathing_perform_cycle(VectorContext *ctx, int iterations) {
    double target[VECTOR_DIM];
    memset(target, 0, sizeof(target));

    for (int i = 0; i < FOURIER_COEFFS && i < VECTOR_DIM / 3; i++) {
        target[i]              = ctx->k_fourier.coeff_re[i];
        target[i + VECTOR_DIM/3] = ctx->gamma_fourier.coeff_re[i];
        target[i + 2*VECTOR_DIM/3] = ctx->phase_fourier.coeff_re[i];
    }

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < BREATHING_SEEDS; i++)
            ctx->seeds[i].fitness = breathing_compute_fitness(&ctx->seeds[i], target);

        /* bubble-sort by fitness descending (8 seeds — simple is fine) */
        for (int i = 0; i < BREATHING_SEEDS - 1; i++)
            for (int j = 0; j < BREATHING_SEEDS - 1 - i; j++)
                if (ctx->seeds[j].fitness < ctx->seeds[j+1].fitness) {
                    BreathingSeed tmp = ctx->seeds[j];
                    ctx->seeds[j]   = ctx->seeds[j+1];
                    ctx->seeds[j+1] = tmp;
                }

        for (int i = 1; i < BREATHING_SEEDS; i++) {
            breathing_toward(&ctx->seeds[i], &ctx->seeds[0], 0.618);
            breathing_mutate(&ctx->seeds[i], 0.1 * INV_PHI);
        }
    }
    ctx->breathing_iterations += (uint32_t)iterations;
}

/* ─── Holographic glyph ─────────────────────────────────────────────────────── */

void glyph_compute_interference(HolographicGlyph *g) {
    for (int h = 0; h < 4; h++) {
        double harmonic  = (double)(h + 1);
        double amplitude = 1.0 / (harmonic * harmonic);
        double modulation = sin(g->breathing_phase * harmonic) * INV_PHI;
        amplitude *= (1.0 + modulation);

        double p1 = g->temporal_phase * harmonic;
        double p2 = g->temporal_phase * harmonic * PHI;
        double p3 = g->spatial_freq   * harmonic + g->breathing_phase;

        g->real_field[h]  = amplitude * (cos(p1)*cos(p2) - sin(p1)*sin(p3));
        g->imag_field[h]  = amplitude * (sin(p1)*cos(p2) + cos(p1)*sin(p3));
        g->phase_field[h] = atan2(g->imag_field[h], g->real_field[h]);
    }
}

char glyph_project_to_unicode(const HolographicGlyph *g) {
    double rs = 0.0, is = 0.0, ps = 0.0;
    for (int i = 0; i < 4; i++) { rs += g->real_field[i]; is += g->imag_field[i]; ps += g->phase_field[i]; }
    double mag  = sqrt(rs*rs + is*is);
    double norm = fmod(mag * 0.3 + ps / (2.0 * M_PI) + sin(g->breathing_phase) * 0.1, 1.0);
    if (norm < 0.0) norm += 1.0;
    double region = fmod(mag + cos(g->breathing_phase) * 0.1, 1.0);
    if (region < 0.0) region += 1.0;
    if (region < 0.3) return (char)(0x21 + (int)(norm * 93));
    if (region < 0.6) return (char)(0xA1 + (int)(norm * 93));
    return (char)(0x41 + (int)(norm * 25));  /* fallback: A-Z range */
}

HolographicGlyph glyph_generate(const VectorContext *ctx, uint32_t index, uint64_t timestamp) {
    HolographicGlyph g;
    g.timestamp        = timestamp;
    g.breathing_phase  = fmod(timestamp * 2.399963229728653e-10 * PHI + index / 4096.0 * PI_PHI, 2.0 * M_PI);
    g.temporal_phase   = fmod(g.breathing_phase + index / 4096.0 * PI_PHI, 2.0 * M_PI);
    g.spatial_freq     = PHI * index / 4096.0 * M_E * M_E * cos(g.breathing_phase);

    double pm = fmod(index * PHI + timestamp * INV_PHI, 3.0);
    g.ternary_state = (pm < 1.0) ? TERNARY_NEGATIVE :
                      (pm < 2.0) ? TERNARY_NEUTRAL   : TERNARY_POSITIVE;

    glyph_compute_interference(&g);
    g.projected_char = glyph_project_to_unicode(&g);

    g.dna_sequence[0] = dna_map_state(g.ternary_state);
    for (int i = 1; i < 4; i++) {
        double v = sin(g.breathing_phase + (double)i * M_PI / 2.0);
        g.dna_sequence[i] = (v > 0.5) ? 'T' : (v > 0.0) ? 'G' : (v > -0.5) ? 'A' : 'C';
    }
    g.dna_sequence[4] = '\0';
    return g;
}

/* ─── DNA encoding ───────────────────────────────────────────────────────────── */

TernaryState dna_classify_ternary(double value, uint64_t operation_id) {
    double pm = fmod(value * PHI + (double)operation_id * INV_PHI, 3.0);
    return (pm < 1.0) ? TERNARY_NEGATIVE :
           (pm < 2.0) ? TERNARY_NEUTRAL   : TERNARY_POSITIVE;
}

char dna_map_state(TernaryState state) {
    switch (state) {
        case TERNARY_NEGATIVE: return DNA_NEGATIVE;
        case TERNARY_NEUTRAL:  return DNA_NEUTRAL;
        case TERNARY_POSITIVE: return DNA_POSITIVE;
        default:               return DNA_CYTOSINE;
    }
}

void dna_encode_state(const VectorContext *ctx, char *dna_out, size_t max_len) {
    size_t pos = 0;
    for (int i = 0; i < FOURIER_COEFFS && pos < max_len - 1; i++) {
        double rv = ctx->k_fourier.coeff_re[i];
        dna_out[pos++] = dna_map_state(dna_classify_ternary(rv, ctx->context_id + (uint64_t)i));
    }
    if (pos < max_len - 1) dna_out[pos++] = DNA_CYTOSINE;
    for (int i = 0; i < BREATHING_SEEDS && pos < max_len - 1; i++)
        dna_out[pos++] = dna_map_state(dna_classify_ternary(ctx->seeds[i].fitness, ctx->seeds[i].seed_id));
    dna_out[pos] = '\0';
}

/* ─── Onion shell checkpointing ──────────────────────────────────────────────── */

OnionShellCheckpoint *checkpoint_create(const VectorContext *ctx, uint64_t op_id) {
    OnionShellCheckpoint *cp = malloc(sizeof(OnionShellCheckpoint));
    cp->operation_id = op_id;
    cp->timestamp    = (uint64_t)time(NULL);

    dna_encode_state(ctx, cp->dna_sequence, sizeof(cp->dna_sequence));
    for (int i = 0; i < BREATHING_SEEDS; i++)
        cp->breathing_signature[i] = ctx->seeds[i].fitness;

    char fd[FOURIER_COEFFS * 16];
    for (int i = 0; i < FOURIER_COEFFS; i++)
        snprintf(fd + i*16, 16, "%.6f", ctx->k_fourier.coeff_re[i]);
    sha256_hash(fd, sizeof(fd), cp->shell_layer_1);
    sha256_hash(cp->dna_sequence, strlen(cp->dna_sequence), cp->shell_layer_2);

    uint8_t combined[64];
    memcpy(combined,      cp->shell_layer_1, 32);
    memcpy(combined + 32, cp->shell_layer_2, 32);
    sha256_hash(combined, 64, cp->shell_layer_3);

    char sd[512];
    snprintf(sd, sizeof(sd), "%llu:%llu:%s",
             (unsigned long long)cp->operation_id,
             (unsigned long long)cp->timestamp,
             cp->dna_sequence);
    sha256_hash(sd, strlen(sd), cp->state_hash);
    return cp;
}

bool checkpoint_verify(const OnionShellCheckpoint *cp) {
    uint8_t combined[64], expected[32];
    memcpy(combined,      cp->shell_layer_1, 32);
    memcpy(combined + 32, cp->shell_layer_2, 32);
    sha256_hash(combined, 64, expected);
    return memcmp(expected, cp->shell_layer_3, 32) == 0;
}

bool checkpoint_restore(VectorContext *ctx, const OnionShellCheckpoint *cp) {
    (void)ctx; (void)cp;
    return false; /* full restore not needed for bot — checkpoints are one-way */
}

/* ─── Container lifecycle ───────────────────────────────────────────────────── */

FrameworkContainer *container_create(const char *name) {
    FrameworkContainer *c = calloc(1, sizeof(FrameworkContainer));
    strncpy(c->name, name, sizeof(c->name) - 1);
    c->container_id  = phi_hash(name, strlen(name));
    c->creation_time = (uint64_t)time(NULL);
    c->context.context_id        = c->container_id;
    c->context.checkpoint_capacity = 100;
    c->context.checkpoints = calloc(100, sizeof(OnionShellCheckpoint));
    return c;
}

void container_destroy(FrameworkContainer *c) {
    if (!c) return;
    cf_free(&c->context.k_trajectory);
    cf_free(&c->context.gamma_trajectory);
    cf_free(&c->context.phase_trajectory);
    if (c->context.checkpoints) free(c->context.checkpoints);
    if (c->context.dna_ledger)  free(c->context.dna_ledger);
    free(c);
}

bool container_initialize(FrameworkContainer *c) {
    breathing_initialize_seeds(&c->context);
    c->context.glyph = glyph_generate(&c->context, 0, c->creation_time);
    c->initialized   = true;
    c->active        = true;
    return true;
}

/* ─── Context operations ─────────────────────────────────────────────────────── */

bool context_set_trajectory(VectorContext *ctx,
                             const double *k_s,   size_t k_n,
                             const double *g_s,   size_t g_n,
                             const double *ph_s,  size_t ph_n) {
    cf_init(&ctx->k_trajectory,     k_s,  k_n);
    cf_init(&ctx->gamma_trajectory, g_s,  g_n);
    cf_init(&ctx->phase_trajectory, ph_s, ph_n);

    fourier_encode_cf(&ctx->k_trajectory,     &ctx->k_fourier);
    fourier_encode_cf(&ctx->gamma_trajectory, &ctx->gamma_fourier);
    fourier_encode_cf(&ctx->phase_trajectory, &ctx->phase_fourier);

    dct_encode_cf(&ctx->k_trajectory,     &ctx->k_dct);
    dct_encode_cf(&ctx->gamma_trajectory, &ctx->gamma_dct);
    dct_encode_cf(&ctx->phase_trajectory, &ctx->phase_dct);
    return true;
}

/* ─── Utility ────────────────────────────────────────────────────────────────── */

uint64_t phi_hash(const void *data, size_t len) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = 0;
    for (size_t i = 0; i < len; i++)
        hash = (uint64_t)((double)hash * PHI + bytes[i]) * 0xDEADBEEFCAFEBABEULL;
    return hash;
}

double phi_modulate(double value, double phase) {
    return fmod(value * PHI + phase * INV_PHI, 1.0);
}

void container_print_stats(const FrameworkContainer *c) {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf(" FRAMEWORK-NATIVE CONTAINER: %s\n", c->name);
    printf("═══════════════════════════════════════════════════════\n");
    printf(" Transforms:       %llu\n", (unsigned long long)c->stats.transform_count);
    printf(" Compression:      %.2f×\n", c->stats.compression_ratio);
    printf(" Breathing iters:  %u\n",   c->context.breathing_iterations);
    printf(" Checkpoints:      %u\n",   c->context.checkpoint_count);
    printf(" Glyph:            %c | DNA: %s\n",
           c->context.glyph.projected_char, c->context.glyph.dna_sequence);
    printf("═══════════════════════════════════════════════════════\n\n");
}

double container_get_compression_ratio(const FrameworkContainer *c) {
    return c->stats.compression_ratio;
}
