/*
 * tb_phi_lattice.c — TRAILBLAZE Layer 0 Implementation
 *
 * Derived from:
 *   ll_analog.c        — 8D Kuramoto, APhase adaptive model, U-field
 *   hdgl_analog_v31.c  — GRA closed-form, spectral kernel, consensus
 *   prime_ui.c         — phi_fold_hash32/64, phi_stream, epoch ratchet
 *   zchg_lattice.c     — phi-tau strand routing, EMA, provisioner pipeline
 *
 * Build: gcc -O3 -march=native -std=c11 -lm tb_phi_lattice.c -o tb_l0_test
 */

#ifndef _WIN32
#  define _POSIX_C_SOURCE 199309L
#endif
#include "tb_phi_lattice.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#  include "../src/tb_win32.h"
#endif

/* ── Fibonacci + Prime tables (from hdgl_analog_v31.c GRA closed-form) ─── */
static const uint64_t FIB_TABLE[9]   = {0,1,1,2,3,5,8,13,21};
static const uint64_t PRIME_TABLE[9] = {2,3,5,7,11,13,17,19,23};

/* ── Entropy source: POSIX clock_gettime ─────────────────────────────────── */
static uint64_t tb_perf_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── Simple xorshift64 for slot seeding ─────────────────────────────────── */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 1: Kuramoto dynamics (from ll_analog.c ana_rk4_step + mean-field)
 * ────────────────────────────────────────────────────────────────────────────
 * Mean-field optimisation: O(n) not O(n²).
 *   re_sum = Σ cos(θᵢ),  im_sum = Σ sin(θᵢ)
 *   coupling_i = K/n · (im_sum·cos(θᵢ) − re_sum·sin(θᵢ))   [O(n) identity]
 * Kuramoto order parameter R = √(re_sum²+im_sum²)/n → CV = 1−R
 * ────────────────────────────────────────────────────────────────────────── */

void tb_kuramoto_step(TB_PhiLattice *lat, double dt) {
    uint32_t n    = lat->n_slots;
    TB_Slot *slots = lat->slots;

    /* Compute mean field O(n) */
    double re_sum = 0.0, im_sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        re_sum += slots[i].amp_re;
        im_sum += slots[i].amp_im;
    }
    double R  = sqrt(re_sum*re_sum + im_sum*im_sum) / n;
    double cv = 1.0 - R;
    lat->phase_var = cv;

    /* Adaptive K/γ (from ll_analog.c APHASE_* / ANA_COUPLING[] / ANA_GAMMA[])
     * Four phases based on cv thresholds: Pluck → Sustain → FineTune → Lock */
    double K, gamma;
    if      (cv > 0.50) { K = 5.0; gamma = 0.005; }   /* Pluck:    1000:1 */
    else if (cv > 0.30) { K = 3.0; gamma = 0.008; }   /* Sustain        */
    else if (cv > 0.10) { K = 2.0; gamma = 0.010; }   /* FineTune       */
    else                { K = 1.8; gamma = 0.012; }   /* Lock           */

    /* VCO floor: ω ≥ 10% of natural (from ANA_VCO_BASE=0.1) */
    double vco_floor = 0.1 + 0.9 * cv;

    for (uint32_t i = 0; i < n; i++) {
        TB_Slot *s = &slots[i];
        double coupling = (K / n) * (im_sum * s->amp_re - re_sum * s->amp_im);
        double eff_freq = s->freq * (1.0 + 0.1 * s->wave_mode) * vco_floor * lat->omega;

        s->phase = fmod(s->phase + (eff_freq + coupling) * dt, 2.0 * TB_PI);
        if (s->phase < 0.0) s->phase += 2.0 * TB_PI;

        /* Amplitude update with damping */
        double exp_gamma = exp(-gamma * dt);
        s->amp_re = cos(s->phase) * exp_gamma;
        s->amp_im = sin(s->phase) * exp_gamma;

        /* Normalise to unit circle */
        double mag = sqrt(s->amp_re*s->amp_re + s->amp_im*s->amp_im);
        if (mag > 1e-9) { s->amp_re /= mag; s->amp_im /= mag; }
    }

    /* Consensus detection */
    if (cv < TB_CONSENSUS_EPS) {
        lat->consensus_steps++;
        if (lat->consensus_steps >= TB_CONSENSUS_N) {
            for (uint32_t i = 0; i < n; i++)
                slots[i].flags |= TB_FLAG_CONSENSUS;
        }
    } else {
        lat->consensus_steps = 0;
    }
    lat->time += dt;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 2: Dₙ(r) amplitude update
 * (from hdgl_analog_v31.c gra_rn_closed + hdgl_bootloaderz.h Dₙ formula)
 * Dₙ(r) = √(φ · Fₙ · 2ⁿ · ∏Pₖ · Ω) · r^((n+1)/8)
 * ────────────────────────────────────────────────────────────────────────── */

void tb_update_dn_amplitudes(TB_PhiLattice *lat) {
    for (uint32_t i = 0; i < lat->n_slots; i++) {
        TB_Slot *s = &lat->slots[i];
        int      n = s->dimension;    /* 1..8 */
        double   r = s->value;

        uint64_t Fn      = FIB_TABLE[n];
        double   two_n   = (double)(1u << n);
        uint64_t prod_p  = 1;
        for (int k = 0; k < n; k++) prod_p *= PRIME_TABLE[k];

        double frac_rn   = r * n - floor(r * n);
        double omega     = 0.5 + 0.5 * sin(TB_PI * frac_rn * TB_PHI);
        double inside    = TB_PHI * Fn * two_n * prod_p * omega;
        double k_exp     = (n + 1) / 8.0;

        s->dn_amplitude  = (inside > 0.0 && r > 1e-12)
                           ? sqrt(inside) * pow(r, k_exp)
                           : 0.0;
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 3: Lattice seeding (Weyl sequence + Kuramoto warm-up)
 * ────────────────────────────────────────────────────────────────────────── */

static void tb_lattice_seed(TB_PhiLattice *lat, uint64_t seed) {
    lat->seed_value = seed;
    uint64_t rng    = seed ^ 0xDEADBEEFCAFEBABEULL;

    for (uint32_t k = 1; k <= lat->n_slots; k++) {
        TB_Slot *s   = &lat->slots[k-1];
        int      geo = (k - 1) % 8;

        /* Weyl: fractional part of k·φ */
        double kphi  = (double)k * TB_PHI;
        s->value     = kphi - floor(kphi);
        s->phase     = 2.0 * TB_PI * s->value;
        s->freq      = pow(TB_PHI, TB_SPIRAL8[geo].dim / 8.0);
        s->amp_re    = cos(s->phase);
        s->amp_im    = sin(s->phase);
        s->dn_amplitude = 0.0;
        s->dimension = TB_SPIRAL8[geo].dim;
        s->wave_mode = TB_SPIRAL8[geo].wave_mode;
        s->flags     = 0;

        /* Inject RNG entropy into value */
        uint64_t rv  = xorshift64(&rng);
        s->value     = s->value + (double)(rv & 0xFF) / 256.0;
        s->value    -= floor(s->value);
    }

    /* 50 Kuramoto warm-up steps (dissolve local structure) */
    for (int i = 0; i < 50; i++)
        tb_kuramoto_step(lat, 0.1);

    tb_update_dn_amplitudes(lat);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 4: PhiFold — lattice-keyed S-box hash
 * (from prime_ui.c phi_fold_hash32/64)
 * No SHA, no XOR — pure modular arithmetic over lattice slots.
 * ────────────────────────────────────────────────────────────────────────── */

void tb_phi_fold_refresh(TB_PhiLattice *lat) {
    if (lat->phi_fold.epoch_cached == lat->epoch) return;

    TB_Slot *slots = lat->slots;
    uint32_t n     = lat->n_slots;

    /* Build sbox_a from offset 1024, sbox_b from offset 2048 (Fisher-Yates) */
    for (int box = 0; box < 2; box++) {
        uint32_t off = box ? 2048u : 1024u;
        uint8_t *sb  = box ? lat->phi_fold.sbox_b : lat->phi_fold.sbox_a;
        for (int i = 0; i < 256; i++) sb[i] = (uint8_t)i;
        for (int i = 255; i > 0; i--) {
            uint32_t slot_idx = (off + (uint32_t)i * 3) % n;
            int j = (int)(slots[slot_idx].value * 255.999) % (i + 1);
            if (j < 0) j = 0;
            uint8_t tmp = sb[i]; sb[i] = sb[j]; sb[j] = tmp;
        }
    }
    lat->phi_fold.epoch_cached = lat->epoch;
}

void tb_phi_fold_hash32(TB_PhiLattice *lat,
                        const uint8_t *data, size_t n,
                        uint8_t out[32]) {
    tb_phi_fold_refresh(lat);
    TB_Slot *slots  = lat->slots;
    uint32_t ns     = lat->n_slots;
    uint8_t  phi_b  = (uint8_t)((int)(TB_PHI * 1000) & 0xFF);
    uint8_t  acc[32];
    memset(acc, 0, 32);

    for (size_t i = 0; i < n; i++) {
        uint8_t sv = (uint8_t)((int)(slots[i % ns].value * 255.999) & 0xFF);
        acc[i % 32] = (uint8_t)((acc[i % 32] + data[i] + sv) & 0xFF);
    }

    const uint8_t *sb = lat->phi_fold.sbox_a;
    for (int r = 0; r < 12; r++) {
        for (int j = 0; j < 32; j++) {
            int src = (j + r * 7) % 32;
            uint8_t sv = (uint8_t)((acc[j] + phi_b + acc[src]) & 0xFF);
            sv = (uint8_t)((sv >> 1) | ((sv & 1) << 7));
            acc[j] = sb[sv];
        }
    }
    memcpy(out, acc, 32);
}

void tb_phi_fold_hash64(TB_PhiLattice *lat,
                        const uint8_t *data, size_t n,
                        uint8_t out[64]) {
    tb_phi_fold_refresh(lat);
    TB_Slot *slots  = lat->slots;
    uint32_t ns     = lat->n_slots;
    uint8_t  phi_b  = (uint8_t)((int)(TB_PHI * 1000) & 0xFF);
    uint8_t  lo[32], hi[32];
    memset(lo, 0, 32); memset(hi, 0, 32);

    for (size_t i = 0; i < n; i++) {
        uint8_t sf = (uint8_t)((int)(slots[i % ns].value * 255.999) & 0xFF);
        uint8_t sr = (uint8_t)((int)(slots[(ns-1 - i%ns)].value * 255.999) & 0xFF);
        lo[i % 32] = (uint8_t)((lo[i%32] + data[i] + sf) & 0xFF);
        hi[i % 32] = (uint8_t)((hi[i%32] + data[i] + sr) & 0xFF);
    }

    const uint8_t *sa = lat->phi_fold.sbox_a;
    const uint8_t *sb = lat->phi_fold.sbox_b;
    for (int r = 0; r < 12; r++) {
        for (int j = 0; j < 32; j++) {
            int src = (j + r * 7) % 32;
            uint8_t sl = (uint8_t)((lo[j] + phi_b + lo[src]) & 0xFF);
            uint8_t sh = (uint8_t)((hi[j] + phi_b + hi[src]) & 0xFF);
            lo[j] = sa[(uint8_t)((sl ^ (hi[j] >> 1)) & 0xFF)];
            hi[j] = sb[(uint8_t)((sh ^ (lo[j] >> 1)) & 0xFF)];
        }
    }
    memcpy(out,    lo, 32);
    memcpy(out+32, hi, 32);
}

void tb_phi_fold_prk(TB_PhiLattice *lat, const char *ctx, uint8_t out[32]) {
    uint8_t cb[256]; size_t cl = strlen(ctx);
    if (cl > 255) cl = 255;
    memcpy(cb, ctx, cl);
    uint8_t h1[32];
    tb_phi_fold_hash32(lat, cb, cl, h1);
    /* two-phase: hash32(hash32(ctx) || ctx) */
    uint8_t combined[32 + 256];
    memcpy(combined, h1, 32);
    memcpy(combined + 32, cb, cl);
    tb_phi_fold_hash32(lat, combined, 32 + cl, out);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 5: PhiStream AEAD
 * Additive Z/256Z: ciphertext[i] = (plaintext[i] + keystream[i]) mod 256
 * Tag = phi_fold_hash32(ciphertext) — verified before decryption.
 * ────────────────────────────────────────────────────────────────────────── */

static void phi_stream_keystream(TB_PhiLattice *lat, const uint8_t prk[32],
                                  uint64_t ctr, size_t length, uint8_t *ks) {
    size_t pos  = 0;
    uint32_t blk = 0;
    while (pos < length) {
        /* block input: prk[32] || ctr[8] || blk[4] */
        uint8_t block_in[44];
        memcpy(block_in, prk, 32);
        memcpy(block_in + 32, &ctr, 8);
        memcpy(block_in + 40, &blk, 4);
        uint8_t h[32];
        tb_phi_fold_hash32(lat, block_in, 44, h);
        size_t take = (length - pos < 32) ? (length - pos) : 32;
        memcpy(ks + pos, h, take);
        pos += take;
        blk++;
    }
}

size_t tb_phi_stream_seal(TB_PhiLattice *lat,
                           const uint8_t *pt, size_t pt_len,
                           const char *ctx,
                           uint8_t *out, size_t out_cap) {
    if (out_cap < pt_len + 40) return 0;

    uint8_t prk[32];
    tb_phi_fold_prk(lat, ctx, prk);

    uint64_t ctr = lat->phi_stream.ctr++;

    /* Keystream */
    uint8_t *ks = (uint8_t *)malloc(pt_len);
    if (!ks) return 0;
    phi_stream_keystream(lat, prk, ctr, pt_len, ks);

    /* Encrypt: additive mod 256 */
    uint8_t *ct = out + 40;
    for (size_t i = 0; i < pt_len; i++)
        ct[i] = (uint8_t)((pt[i] + ks[i]) & 0xFF);
    free(ks);

    /* Tag = phi_fold_hash32(ciphertext) */
    uint8_t tag[32];
    tb_phi_fold_hash32(lat, ct, pt_len, tag);

    /* Envelope: ctr[8] | tag[32] | ciphertext */
    memcpy(out,     &ctr, 8);
    memcpy(out + 8, tag,  32);
    return pt_len + 40;
}

size_t tb_phi_stream_unseal(TB_PhiLattice *lat,
                              const uint8_t *env, size_t env_len,
                              const char *ctx,
                              uint8_t *out, size_t out_cap) {
    if (env_len < 40) return 0;
    size_t ct_len = env_len - 40;
    if (out_cap < ct_len) return 0;

    uint64_t ctr;
    memcpy(&ctr, env, 8);
    const uint8_t *tag_stored = env + 8;
    const uint8_t *ct         = env + 40;

    /* Verify tag (constant-time) */
    uint8_t tag_exp[32];
    tb_phi_fold_hash32(lat, ct, ct_len, tag_exp);
    uint8_t diff = 0;
    for (int i = 0; i < 32; i++) diff |= (tag_stored[i] ^ tag_exp[i]);
    if (diff != 0) return 0;

    /* Decrypt */
    uint8_t prk[32];
    tb_phi_fold_prk(lat, ctx, prk);
    uint8_t *ks = (uint8_t *)malloc(ct_len);
    if (!ks) return 0;
    phi_stream_keystream(lat, prk, ctr, ct_len, ks);

    for (size_t i = 0; i < ct_len; i++)
        out[i] = (uint8_t)((ct[i] - ks[i] + 256) & 0xFF);
    free(ks);
    return ct_len;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 6: Lattice lifecycle
 * ────────────────────────────────────────────────────────────────────────── */

TB_PhiLattice* tb_lattice_create(uint32_t n_slots, uint64_t seed) {
    if (n_slots < 8)         n_slots = 8;
    if (n_slots > TB_MAX_SLOTS) n_slots = TB_MAX_SLOTS;

    TB_PhiLattice *lat = (TB_PhiLattice *)calloc(1, sizeof(TB_PhiLattice));
    if (!lat) return NULL;

    lat->slots = (TB_Slot *)calloc(n_slots, sizeof(TB_Slot));
    if (!lat->slots) { free(lat); return NULL; }

    lat->n_slots   = n_slots;
    lat->epoch     = 0;
    lat->time      = 0.0;
    lat->omega     = 1.0;
    lat->phase_var = 1.0;
    lat->phi_fold.epoch_cached = -1;

    /* 4-source entropy (from lk_advance entropy model in prime_ui.c) */
    if (seed == 0) {
        uint64_t t1  = tb_perf_ns();
        uint64_t t2  = (uint64_t)time(NULL);
        uint64_t adr = (uint64_t)(uintptr_t)lat;
        /* OS random */
        FILE *f = fopen("/dev/urandom", "rb");
        uint64_t rnd = 0;
        if (f) { (void)fread(&rnd, 8, 1, f); fclose(f); }
        seed = t1 ^ t2 ^ adr ^ rnd;
    }

    tb_lattice_seed(lat, seed);

    /* Initial PRK */
    lat->phi_stream.pf  = &lat->phi_fold;
    lat->phi_stream.ctr = seed & 0xFFFFFFFFULL;
    tb_phi_fold_prk(lat, "trailblaze::init", lat->prk);

    return lat;
}

void tb_lattice_destroy(TB_PhiLattice *lat) {
    if (!lat) return;
    free(lat->slots);
    free(lat);
}

void tb_lattice_advance(TB_PhiLattice *lat, int steps) {
    for (int i = 0; i < steps; i++)
        tb_kuramoto_step(lat, TB_ANA_DT);

    /* 4-source entropy harvest */
    uint64_t t1  = tb_perf_ns() & 0xFFFF;
    uint64_t t2  = (uint64_t)time(NULL) & 0xFFFF;
    uint64_t rnd = 0;
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) { (void)fread(&rnd, 2, 1, f); fclose(f); }
    uint64_t ent = (t1 ^ t2 ^ rnd);
    uint8_t  ent_bytes[4];
    memcpy(ent_bytes, &ent, 4);

    /* Fold entropy into first 32 slots */
    uint32_t fold_n = lat->n_slots < 32 ? lat->n_slots : 32;
    for (uint32_t i = 0; i < fold_n; i++) {
        lat->slots[i].value += ent_bytes[i % 4] / 256.0;
        lat->slots[i].value -= floor(lat->slots[i].value);
    }

    tb_update_dn_amplitudes(lat);
    lat->epoch++;
    lat->phi_fold.epoch_cached = -1;   /* invalidate S-box */

    /* Rebuild PRK for new epoch */
    char ctx[64];
    snprintf(ctx, sizeof(ctx), "trailblaze::epoch::%d", lat->epoch);
    tb_phi_fold_prk(lat, ctx, lat->prk);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 7: Phi-lattice addressing and routing
 * ────────────────────────────────────────────────────────────────────────── */

void tb_lattice_phi_addr(TB_PhiLattice *lat,
                          const uint8_t **parts, const size_t *lens, int n_parts,
                          uint8_t out_addr[16]) {
    /* Concatenate parts and hash64 */
    size_t total = 0;
    for (int i = 0; i < n_parts; i++) total += lens[i];
    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) { memset(out_addr, 0, 16); return; }
    size_t off = 0;
    for (int i = 0; i < n_parts; i++) {
        memcpy(buf + off, parts[i], lens[i]);
        off += lens[i];
    }
    uint8_t h64[64];
    tb_phi_fold_hash64(lat, buf, total, h64);
    memcpy(out_addr, h64, 16);
    free(buf);
}

uint32_t tb_lattice_slot_for_key(TB_PhiLattice *lat,
                                   const char *key, size_t key_len) {
    uint8_t h[32];
    tb_phi_fold_hash32(lat, (const uint8_t *)key, key_len, h);
    uint32_t raw;
    memcpy(&raw, h, 4);
    /* phi²-multiply to get fractional: raw * φ² / 2^32 → slot */
    double frac = fmod((double)raw * TB_PHI * TB_PHI / 4294967296.0, 1.0);
    if (frac < 0.0) frac += 1.0;
    return (uint32_t)(frac * lat->n_slots) % lat->n_slots;
}

double tb_lattice_dn_for_key(TB_PhiLattice *lat,
                               const char *key, size_t key_len) {
    uint32_t slot = tb_lattice_slot_for_key(lat, key, key_len);
    return lat->slots[slot].dn_amplitude;
}

/* U-field resonance S(U) — the prime invariant discriminant.
 * From ll_analog.c: ana_u_field() and U-field spectral projection. */
void tb_lattice_s_u_resonance(TB_PhiLattice *lat,
                                double *out_M, double *out_L, double *out_S) {
    double re_sum = 0.0, im_sum = 0.0;
    for (uint32_t i = 0; i < lat->n_slots; i++) {
        re_sum += lat->slots[i].amp_re;
        im_sum += lat->slots[i].amp_im;
    }
    double M_U = sqrt(re_sum*re_sum + im_sum*im_sum) / lat->n_slots;
    if (M_U < 1e-12) { *out_M = 0; *out_L = 0; *out_S = 0; return; }

    double L_U   = log(M_U * lat->n_slots) / TB_LN_PHI - 1.0 / (2.0 * TB_PHI);
    double frac_L = L_U - floor(L_U);
    double Om_U  = 0.5 + 0.5 * sin(TB_PI * frac_L * TB_PHI);
    double s_re  = Om_U * cos(TB_PI * L_U) + 1.0;
    double s_im  = Om_U * sin(TB_PI * L_U);
    double S_U   = sqrt(s_re*s_re + s_im*s_im);

    *out_M = M_U;
    *out_L = L_U;
    *out_S = S_U;
}

int tb_lattice_describe(TB_PhiLattice *lat, char *buf, size_t buf_len) {
    double M, L, S;
    tb_lattice_s_u_resonance(lat, &M, &L, &S);
    return snprintf(buf, buf_len,
        "{\"epoch\":%d,\"n_slots\":%u,\"time\":%.4f,"
        "\"phase_var\":%.6f,\"consensus\":%s,"
        "\"M_U\":%.4f,\"Lambda_U\":%.4f,\"S_U\":%.4f,\"seed\":\"0x%016llx\"}",
        lat->epoch, lat->n_slots, lat->time,
        lat->phase_var,
        (lat->slots[0].flags & TB_FLAG_CONSENSUS) ? "true" : "false",
        M, L, S,
        (unsigned long long)lat->seed_value);
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 8: Backend registry
 * ────────────────────────────────────────────────────────────────────────── */

void tb_registry_init(TB_BackendRegistry *reg, TB_PhiLattice *lat) {
    memset(reg, 0, sizeof(*reg));
    reg->lattice = lat;

    /* CPU_AVX2 (available everywhere via compiler auto-vectorisation) */
    TB_BackendDesc cpu = {0};
    cpu.type           = TB_BACKEND_CPU_AVX2;
    cpu.available      = 1;
    cpu.compute_units  = 4;    /* conservative; updated by bench */
    /* cost T(n) = a·nᵏ + b */
    cpu.cost_a[TB_OP_MATMUL]      = 1e-9; cpu.cost_k[TB_OP_MATMUL]      = 2.0; cpu.cost_b[TB_OP_MATMUL]      = 1e-4;
    cpu.cost_a[TB_OP_ATTENTION]   = 2e-9; cpu.cost_k[TB_OP_ATTENTION]   = 2.0; cpu.cost_b[TB_OP_ATTENTION]   = 1e-4;
    cpu.cost_a[TB_OP_ELEMENTWISE] = 1e-10;cpu.cost_k[TB_OP_ELEMENTWISE] = 1.0; cpu.cost_b[TB_OP_ELEMENTWISE] = 1e-6;
    cpu.cost_a[TB_OP_PHI_FOLD]    = 1e-7; cpu.cost_k[TB_OP_PHI_FOLD]    = 1.0; cpu.cost_b[TB_OP_PHI_FOLD]    = 1e-5;
    tb_registry_add(reg, cpu);

    /* ANALOG (8D Kuramoto — correctness oracle, always available) */
    TB_BackendDesc analog = {0};
    analog.type          = TB_BACKEND_ANALOG;
    analog.available     = 1;
    analog.compute_units = 1;
    analog.cost_a[TB_OP_KURAMOTO]    = 1e-8; analog.cost_k[TB_OP_KURAMOTO]    = 1.0; analog.cost_b[TB_OP_KURAMOTO]    = 1e-5;
    analog.cost_a[TB_OP_LATTICE_UPD] = 1e-8; analog.cost_k[TB_OP_LATTICE_UPD] = 1.0; analog.cost_b[TB_OP_LATTICE_UPD] = 1e-5;
    tb_registry_add(reg, analog);
}

void tb_registry_add(TB_BackendRegistry *reg, TB_BackendDesc desc) {
    if (reg->n_backends >= TB_BACKEND_COUNT) return;
    reg->descs[reg->n_backends++] = desc;
}

TB_Backend tb_registry_select(TB_BackendRegistry *reg, TB_OpClass op, size_t n) {
    TB_Backend best_b = TB_BACKEND_CPU_AVX2;
    double     best_c = 1e30;

    for (int i = 0; i < reg->n_backends; i++) {
        TB_BackendDesc *d = &reg->descs[i];
        if (!d->available) continue;
        double a = d->cost_a[op], k = d->cost_k[op], b = d->cost_b[op];
        if (a == 0.0 && b == 0.0) continue;   /* op not supported */
        double cost = a * pow((double)n, k) + b;

        /* Dₙ-modulated: high Dₙ slot → cost reduction (from Kuramoto scheduler) */
        char key[32];
        snprintf(key, sizeof(key), "%d:%d", d->type, op);
        double dn = tb_lattice_dn_for_key(reg->lattice, key, strlen(key));
        cost *= 1.0 / (1.0 + dn * 0.1);

        if (cost < best_c) { best_c = cost; best_b = d->type; }
    }
    return best_b;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 9: Self-test (compiled as main when TB_L0_TEST defined)
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef TB_L0_TEST
#include <assert.h>

int main(void) {
    printf("=== TRAILBLAZE Layer 0 C Self-Test ===\n\n");

    TB_PhiLattice *lat = tb_lattice_create(512, 0xDEADBEEFCAFEBABEULL);
    assert(lat != NULL);

    /* Advance 100 steps */
    for (int i = 0; i < 10; i++) tb_lattice_advance(lat, 10);
    char desc[512];
    tb_lattice_describe(lat, desc, sizeof(desc));
    printf("[init] %s\n", desc);

    /* phi_fold32 avalanche test */
    uint8_t h1[32], h2[32];
    tb_phi_fold_hash32(lat, (const uint8_t *)"hello world", 11, h1);
    tb_phi_fold_hash32(lat, (const uint8_t *)"hello worle", 11, h2);
    int diff_bits = 0;
    for (int i = 0; i < 32; i++) diff_bits += __builtin_popcount(h1[i] ^ h2[i]);
    double pct = diff_bits / 256.0 * 100.0;
    printf("\n[phi_fold32] avalanche: %.1f%% bits changed (expect ~50%%)\n", pct);
    assert(pct > 30.0 && pct < 70.0);
    printf("[phi_fold32] PASS\n");

    /* PhiStream seal/unseal */
    const char *msg = "TRAILBLAZE persistent cognition substrate";
    size_t msg_len  = strlen(msg);
    uint8_t env[1024];
    size_t env_len = tb_phi_stream_seal(lat,
        (const uint8_t *)msg, msg_len, "test", env, sizeof(env));
    assert(env_len == msg_len + 40);

    uint8_t plain[1024];
    size_t pt_len = tb_phi_stream_unseal(lat, env, env_len, "test", plain, sizeof(plain));
    assert(pt_len == msg_len);
    assert(memcmp(plain, msg, msg_len) == 0);
    printf("\n[phi_stream] seal/unseal roundtrip: PASS\n");

    /* Tamper detection */
    env[45] ^= 0xFF;
    size_t bad = tb_phi_stream_unseal(lat, env, env_len, "test", plain, sizeof(plain));
    assert(bad == 0);
    printf("[phi_stream] tamper detection: PASS\n");
    env[45] ^= 0xFF;  /* restore */

    /* Epoch advance → hash changes */
    uint8_t h3[32];
    tb_phi_fold_hash32(lat, (const uint8_t *)"hello world", 11, h3);
    int old_ep = lat->epoch;
    tb_lattice_advance(lat, 1);
    uint8_t h4[32];
    tb_phi_fold_hash32(lat, (const uint8_t *)"hello world", 11, h4);
    assert(lat->epoch == old_ep + 1);
    assert(memcmp(h3, h4, 32) != 0);
    printf("\n[epoch] advance %d→%d, hash changed: PASS\n", old_ep, lat->epoch);

    /* Forward secrecy: old envelope fails after epoch advance */
    size_t old_bad = tb_phi_stream_unseal(lat, env, env_len, "test", plain, sizeof(plain));
    assert(old_bad == 0);
    printf("[epoch] forward secrecy (old envelope rejected): PASS\n");

    /* Slot routing */
    uint32_t slot = tb_lattice_slot_for_key(lat, "matmul", 6);
    double dn      = lat->slots[slot].dn_amplitude;
    printf("\n[routing] 'matmul' → slot %u, Dₙ=%.4f\n", slot, dn);

    /* Backend registry */
    TB_BackendRegistry reg;
    tb_registry_init(&reg, lat);
    TB_Backend sel = tb_registry_select(&reg, TB_OP_MATMUL, 4096);
    printf("\n[backend] MATMUL(4096) → %s\n", sel == TB_BACKEND_CPU_AVX2 ? "CPU_AVX2" : "other");

    /* U-field resonance */
    double M, L, S;
    tb_lattice_s_u_resonance(lat, &M, &L, &S);
    printf("\n[S(U)] M=%.4f Λ=%.4f S=%.4f\n", M, L, S);

    tb_lattice_destroy(lat);
    printf("\n=== Layer 0 C PASS ===\n");
    return 0;
}
#endif /* TB_L0_TEST */
