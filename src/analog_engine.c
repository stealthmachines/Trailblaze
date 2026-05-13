/*
 * analog_engine.c — 8D Kuramoto RK4 oscillator + HDGL token bridge
 *
 * Extracted RK4 core from stealthmachines/AnalogContainer1 (analog_codec_native.c).
 * Thread-free, synchronous, pure-C, no stdlib threads.
 * https://github.com/stealthmachines/AnalogContainer1
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#include "analog_engine.h"
#include "sha256_minimal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── Adaptive phase parameters — Wu Wei K/γ ratios (WU_WEI_ANALYSIS.md) ──── */
/* CORRECTED from V4.0 analysis: K/γ must be 1000:1 at Pluck for convergence */
const double APHASE_GAMMA[4]      = {0.005, 0.008, 0.010, 0.012};
const double APHASE_COUPLING[4]   = {5.0,   3.0,   2.0,   1.8};

/* ── Base(∞) φ-seeds for init (from hdgl_bridge_v40.c ANALOG_DIMS/seeds) ─── */
static const double BASE_INF_SEEDS[8] = {
    1.6180339887,   /* φ¹  D₁ */
    2.6180339887,   /* φ²  D₂ */
    3.6180339887,   /* φ³  D₃ */
    4.8541019662,   /* φ⁴  D₄ */
    5.6180339887,   /* φ⁵  D₅ */
    6.4721359549,   /* φ⁶  D₆ */
    7.8541019662,   /* φ⁷  D₇ */
    8.3141592654,   /* φ⁸  D₈ */
};

/* ── Deterministic pseudo-random (mirrors det_rand in AnalogContainer1) ──── */
static inline double det_rand(uint64_t seed) {
    uint64_t x = seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return (double)(x * 0x2545F4914F6CDD1DULL) / (double)UINT64_MAX;
}

/* ── Initialise ─────────────────────────────────────────────────────────────── */
void analog8_init(AnalogState8D *s, uint64_t seed, double gamma, double k_coupling) {
    memset(s, 0, sizeof(*s));
    /* Start in Pluck phase — low damping, gentle coupling to excite system */
    s->aphase     = APHASE_PLUCK;
    s->gamma      = APHASE_GAMMA[APHASE_PLUCK];
    s->k_coupling = APHASE_COUPLING[APHASE_PLUCK];
    (void)gamma;      /* caller params included for API compat; Pluck wins at init */
    (void)k_coupling;

    for (int i = 0; i < ANG_DIMS; i++) {
        /* Base(∞) φ-seeds for deterministic initial conditions (V4.0) */
        s->re[i]     = BASE_INF_SEEDS[i];
        s->im[i]     = ANG_INV_PHI * det_rand(seed + (uint64_t)i * 7ULL);
        s->freqs[i]  = 1.0 + 0.5 * det_rand(seed + (uint64_t)i);
        s->phases[i] = 2.0 * ANG_PI * det_rand(seed + 100ULL + (uint64_t)i);
    }
}

/* ── Band tuning — rescale natural frequencies toward target Hz ─────────────
 * Called after analog8_init when --alpha-mode / --gamma-mode is active.
 * ─────────────────────────────────────────────────────────────────────────── */
void analog8_tune_band(AnalogState8D *s, double target_hz) {
    /* target in rad/step: omega = 2π × Hz × dt, where dt = 0.01 */
    double target_rad = target_hz * 2.0 * ANG_PI * 0.01;
    /* Compute current mean */
    double mean = 0.0;
    for (int i = 0; i < ANG_DIMS; i++) mean += s->freqs[i];
    mean /= (double)ANG_DIMS;
    if (mean < 1e-9) mean = 1.0;
    /* Scale all freqs proportionally to new mean */
    double ratio = target_rad / mean;
    for (int i = 0; i < ANG_DIMS; i++)
        s->freqs[i] *= ratio;
    /* Update mean_freq and bw_band immediately so callers can read them */
    s->mean_freq = target_rad;
    if      (target_rad < 0.251) s->bw_band = BWB_DELTA;
    else if (target_rad < 0.503) s->bw_band = BWB_THETA;
    else if (target_rad < 0.817) s->bw_band = BWB_ALPHA;
    else if (target_rad < 1.885) s->bw_band = BWB_BETA;
    else                         s->bw_band = BWB_GAMMA;
}

/* ── RK4 single step ─────────────────────────────────────────────────────────
 * Direct port of rk4_step() from AnalogContainer1/analog_codec_native.c.
 * Uses the current s->phases[] for all four sub-steps (the AnalogContainer1
 * simplification — phase advances atomically at step end).
 * ─────────────────────────────────────────────────────────────────────────── */
void analog8_rk4_step(AnalogState8D *s, double dt) {
    /* Temp storage for the four RK4 stages */
    double k1_re[ANG_DIMS], k1_im[ANG_DIMS];
    double k2_re[ANG_DIMS], k2_im[ANG_DIMS];
    double k3_re[ANG_DIMS], k3_im[ANG_DIMS];
    double k4_re[ANG_DIMS], k4_im[ANG_DIMS];
    double tr[ANG_DIMS],    ti[ANG_DIMS];

/* Evaluate the Kuramoto derivative at a given (re[], im[]) state  ─────────
 * dz_i/dt = ω_i·e^{iθ_i} + K·Σ_{j≠i}|z_j – z_i| – γ·z_i
 *            ─── from AnalogContainer1 rk4_step ───────────────────────── */
#define EVAL(src_re, src_im, out_re, out_im) \
    for (int _i = 0; _i < ANG_DIMS; _i++) { \
        double _coupling = 0.0; \
        for (int _j = 0; _j < ANG_DIMS; _j++) { \
            if (_j != _i) { \
                double _dr = (src_re)[_j] - (src_re)[_i]; \
                double _di = (src_im)[_j] - (src_im)[_i]; \
                _coupling += s->k_coupling * sqrt(_dr*_dr + _di*_di); \
            } \
        } \
        double _cph = cos(s->phases[_i]); \
        double _sph = sin(s->phases[_i]); \
        (out_re)[_i] = s->freqs[_i] * _cph + _coupling - s->gamma * (src_re)[_i]; \
        (out_im)[_i] = s->freqs[_i] * _sph + _coupling - s->gamma * (src_im)[_i]; \
        double _mag  = sqrt((out_re)[_i]*(out_re)[_i] + (out_im)[_i]*(out_im)[_i]); \
        if (_mag > ANG_SAT) { \
            (out_re)[_i] *= ANG_SAT / _mag; \
            (out_im)[_i] *= ANG_SAT / _mag; \
        } \
    }

    /* k1 */
    EVAL(s->re, s->im, k1_re, k1_im);

    /* k2: state + 0.5·dt·k1 */
    for (int i = 0; i < ANG_DIMS; i++) {
        tr[i] = s->re[i] + 0.5*dt*k1_re[i];
        ti[i] = s->im[i] + 0.5*dt*k1_im[i];
    }
    EVAL(tr, ti, k2_re, k2_im);

    /* k3: state + 0.5·dt·k2 */
    for (int i = 0; i < ANG_DIMS; i++) {
        tr[i] = s->re[i] + 0.5*dt*k2_re[i];
        ti[i] = s->im[i] + 0.5*dt*k2_im[i];
    }
    EVAL(tr, ti, k3_re, k3_im);

    /* k4: state + dt·k3 */
    for (int i = 0; i < ANG_DIMS; i++) {
        tr[i] = s->re[i] + dt*k3_re[i];
        ti[i] = s->im[i] + dt*k3_im[i];
    }
    EVAL(tr, ti, k4_re, k4_im);

#undef EVAL

    /* Combine: y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4) * dt/6 */
    for (int i = 0; i < ANG_DIMS; i++) {
        s->re[i] += (k1_re[i] + 2.0*k2_re[i] + 2.0*k3_re[i] + k4_re[i]) * dt / 6.0;
        s->im[i] += (k1_im[i] + 2.0*k2_im[i] + 2.0*k3_im[i] + k4_im[i]) * dt / 6.0;
        /* Advance phases atomically at step end (AnalogContainer1 convention) */
        s->phases[i] = fmod(s->phases[i] + s->freqs[i] * dt, 2.0 * ANG_PI);
    }

    s->step_count++;
    s->prev_phase_var = s->phase_var;    /* save before update for trend */
    s->phase_var = analog8_phase_var(s);

    /* Brainwave band detection: mean_freq in rad/step, bands in rad/step
     * (hz × 2π × dt=0.01)  delta<0.251, theta<0.503, alpha<0.817,
     *  beta<1.885, gamma≤1.885+                                          */
    {
        double mf = 0.0;
        for (int i = 0; i < ANG_DIMS; i++) mf += s->freqs[i];
        mf /= (double)ANG_DIMS;
        s->mean_freq = mf;
        if      (mf < 0.2513) s->bw_band = BWB_DELTA;
        else if (mf < 0.5027) s->bw_band = BWB_THETA;
        else if (mf < 0.8168) s->bw_band = BWB_ALPHA;
        else if (mf < 1.8850) s->bw_band = BWB_BETA;
        else                  s->bw_band = BWB_GAMMA;
    }

    /* Rolling phase_var history for CV-based lock detection */
    s->phase_history[s->hist_idx] = s->phase_var;
    s->hist_idx = (s->hist_idx + 1) % ANG_PHASE_HIST;
    if (s->hist_count < ANG_PHASE_HIST) s->hist_count++;

    /* ── Emergency hard damping — Wu Wei Phase 0 (not stored as aphase) ───────
     * If variance explodes (>10), apply emergency damping this step only.  */
    if (s->phase_var > ANG_EMERGENCY_VAR) {
        s->gamma      = 0.040;  /* Emergency: K/γ=12.5:1 — stop chaos        */
        s->k_coupling = 0.5;
    }

    /* ── Adaptive phase advance (V4.0 Wu Wei) ──────────────────────────────
     * Compute CV of recent phase_var window. When CV drops below threshold,
     * advance to the next phase with higher coupling / lower damping.        */
    if (s->hist_count >= ANG_LOCK_WINDOW && s->aphase < APHASE_LOCK) {
        double sum = 0.0;
        for (int i = 0; i < ANG_LOCK_WINDOW; i++) {
            int idx = (s->hist_idx - 1 - i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
            sum += s->phase_history[idx];
        }
        double mean = sum / (double)ANG_LOCK_WINDOW;
        if (mean > 1e-12) {
            double sq = 0.0;
            for (int i = 0; i < ANG_LOCK_WINDOW; i++) {
                int idx = (s->hist_idx - 1 - i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
                double d = s->phase_history[idx] - mean;
                sq += d * d;
            }
            double cv = sqrt(sq / (double)ANG_LOCK_WINDOW) / mean;
            double trend = s->phase_var - s->prev_phase_var;  /* +rising, -falling */

            static const double thresholds[3] = {
                ANG_CV_TO_SUSTAIN, ANG_CV_TO_FINETUNE, ANG_CV_TO_LOCK
            };
            /* Advance when CV below threshold AND variance is falling        */
            if (cv < thresholds[s->aphase] && trend <= 0.0) {
                s->aphase = (APhase)(s->aphase + 1);
                s->gamma      = APHASE_GAMMA[s->aphase];
                s->k_coupling = APHASE_COUPLING[s->aphase];
            }
            /* Fallback: regression — if variance is rising above threshold  */
            else if (trend > 0.0 && s->phase_var > 0.1 && s->aphase > APHASE_PLUCK) {
                s->aphase = (APhase)(s->aphase - 1);
                s->gamma      = APHASE_GAMMA[s->aphase];
                s->k_coupling = APHASE_COUPLING[s->aphase];
            }
        }
    }

    /* ── SHA-256 hybrid feedback correction (V4.0 "Defeating Shannon") ─────
     * Every ANG_SHA_INTERVAL steps: encode state → hash → contract phases
     * toward circular mean.  Active in all phases including LOCK — the
     * contractions maintain the low pvar attractor found during early phases. */
    if (!s->sha_disabled && s->step_count % ANG_SHA_INTERVAL == 0) {
        analog8_sha_feedback(s);
    }
}

/* ── Phase variance ─────────────────────────────────────────────────────────── */
double analog8_phase_var(const AnalogState8D *s) {
    double sum = 0.0;
    for (int i = 0; i < ANG_DIMS; i++) sum += s->phases[i];
    double mean = sum / (double)ANG_DIMS;

    double var = 0.0;
    for (int i = 0; i < ANG_DIMS; i++) {
        double d = s->phases[i] - mean;
        var += d * d;
    }
    return var / (double)ANG_DIMS;
}

/* ── Consensus lock detection (mirrors AnalogContainer1 is_at_equilibrium) ── */
int analog8_is_locked(AnalogState8D *s) {
    if (s->hist_count < ANG_LOCK_WINDOW) return 0;

    double sum = 0.0;
    for (int i = 0; i < ANG_LOCK_WINDOW; i++) {
        int idx = (s->hist_idx - 1 - i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
        sum += s->phase_history[idx];
    }
    double mean = sum / (double)ANG_LOCK_WINDOW;
    if (mean < 1e-12) return 0; /* trivially zero — not truly locked */

    double sq_sum = 0.0;
    for (int i = 0; i < ANG_LOCK_WINDOW; i++) {
        int idx = (s->hist_idx - 1 - i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
        double d = s->phase_history[idx] - mean;
        sq_sum += d * d;
    }
    double cv = sqrt(sq_sum / (double)ANG_LOCK_WINDOW) / mean;
    return cv < ANG_LOCK_CV;
}

/* ── Apply token perturbation ───────────────────────────────────────────────── */
void analog8_apply_token(AnalogState8D *s, const TokenEntry *e, int token_idx) {
    int dim = ((unsigned)token_idx) % ANG_DIMS;
    /* φ-blended frequency nudge toward e->k (mirrors breathing_toward) */
    s->freqs[dim] += ANG_INV_PHI * (e->k - s->freqs[dim]);
    /* Global damping softly pulled toward token's damping */
    s->gamma += 0.05 * ANG_INV_PHI * (e->gamma - s->gamma);
    /* Phase nudge */
    s->phases[dim] = fmod(s->phases[dim] + e->phase * 0.1, 2.0 * ANG_PI);
}

/* ── Evolve until lock or timeout ───────────────────────────────────────────── */
double analog8_score_until_lock(AnalogState8D *s, int max_steps) {
    for (int i = 0; i < max_steps; i++) {
        analog8_rk4_step(s, ANG_DT);
        if (analog8_is_locked(s)) {
            s->locked    = 1;
            s->lock_step = s->step_count;
            return s->phase_var;
        }
    }
    s->locked = 0;
    return s->phase_var;
}

/* ── Token → entry mapping ───────────────────────────────────────────────────
 * φ-spiral encoding consistent with hdgl_corpus_seeder phi_fourier_encode
 * and hdgl_router phi_tau_encode.  expert_id maps expert rank to γ offset.
 * ──────────────────────────────────────────────────────────────────────── */
TokenEntry token_to_analog_entry(uint32_t token_id, uint32_t expert_id, int num_experts) {
    TokenEntry e;

    /* k: φ-spiral of token_id — range [0.5, 2.5] */
    double t = (double)(token_id & 0xFFFF) / 65536.0;
    e.k = 1.5 + sin(t * 2.0 * ANG_PI * ANG_PHI);

    /* gamma: expert_id linearly maps to [0.010, 0.050] */
    int ne = (num_experts > 0) ? num_experts : 512;
    e.gamma = 0.010 + 0.040 * ((double)(expert_id % (unsigned)ne) / (double)ne);

    /* phase: FNV1a-derived, range [0, 2π] */
    uint64_t h = (uint64_t)token_id * 0xDEADBEEFCAFEBABEULL;
    h ^= h >> 17; h ^= h << 31; h ^= h >> 8;
    e.phase = 2.0 * ANG_PI * ((double)(h & 0xFFFFFFu) / (double)0x1000000u);

    return e;
}

/* ── SHA-256 hybrid feedback correction ────────────────────────────────────
 * Implements S_{n+1} = H( D_n(r) ⊕ R_n ) feedback loop from V4.0:
 *   1. Encode 8×(re,im) pairs → 128 bytes (IEEE 754 doubles, raw)
 *   2. SHA-256 hash
 *   3. Decode 8 angle perturbations from first 64 bytes of hash output
 *      (wrap-around: 32 bytes → 8×4-byte uint32 → map to [-0.1, +0.1] rad)
 *   4. Add perturbations to phases[] — gently drifts system out of local
 *      minima while preserving near-consensus oscillation.
 * Effect: eliminates accumulated double-precision noise every 100 steps.
 * ─────────────────────────────────────────────────────────────────────────── */
void analog8_sha_feedback(AnalogState8D *s) {
    /* Encode state: 8 × (re[i], im[i]) = 8 × 16 bytes = 128 bytes */
    uint8_t raw[128];
    for (int i = 0; i < ANG_DIMS; i++) {
        memcpy(raw + i * 16,      &s->re[i], 8);
        memcpy(raw + i * 16 + 8,  &s->im[i], 8);
    }

    uint8_t hash[32];
    sha256_hash(raw, 128, hash);

    /* Constructive SHA: contract phases toward circular mean.
     * This complements Kuramoto coupling — helps borderline oscillators
     * commit to the locked state, actively reducing phase variance.
     *
     * Circular mean (correct handling of phase wraparound):
     *   mean = atan2(mean(sin(φᵢ)), mean(cos(φᵢ)))
     *
     * Each oscillator is pulled toward the circular mean with strength
     * proportional to pvar (strong when diverged, minimal near lock).
     * SHA hash provides per-oscillator scale ∈ [0,1] for variability. */
    double sum_s = 0.0, sum_c = 0.0;
    for (int i = 0; i < ANG_DIMS; i++) {
        sum_s += sin(s->phases[i]);
        sum_c += cos(s->phases[i]);
    }
    double circ_mean = atan2(sum_s / ANG_DIMS, sum_c / ANG_DIMS);
    if (circ_mean < 0.0) circ_mean += 2.0 * ANG_PI;

    double pvar = analog8_phase_var(s);
    /* Contraction strength: up to 60% per SHA step when heavily diverged.
     * Scales down to 3% floor near lock to avoid disrupting stable state.  */
    double contraction = fmin(pvar / 8.0, 1.0) * 0.80;
    contraction = fmax(contraction, 0.03);

    for (int i = 0; i < ANG_DIMS; i++) {
        uint32_t val;
        memcpy(&val, hash + (i * 4) % 32, 4);
        double scale = (double)val / (double)UINT32_MAX;  /* [0, 1] */

        /* Shortest angular path to circular mean */
        double diff = s->phases[i] - circ_mean;
        while (diff >  ANG_PI) diff -= 2.0 * ANG_PI;
        while (diff < -ANG_PI) diff += 2.0 * ANG_PI;

        /* Contract: reduce angular distance to mean by scale × contraction */
        double delta = -diff * scale * contraction;
        s->phases[i] = fmod(s->phases[i] + delta, 2.0 * ANG_PI);
        if (s->phases[i] < 0.0) s->phases[i] += 2.0 * ANG_PI;
    }

    s->sha_feedback_count++;
}

/* ── Adaptive phase label ────────────────────────────────────────────────── */
static const char *PHASE_NAMES[4] = {"Pluck", "Sustain", "Fine Tune", "Lock"};

const char *analog8_phase_name(const AnalogState8D *s) {
    int p = (int)s->aphase;
    return PHASE_NAMES[(p >= 0 && p <= 3) ? p : 0];
}

const char *analog8_phase_name_from(APhase phase) {
    int p = (int)phase;
    return PHASE_NAMES[(p >= 0 && p <= 3) ? p : 0];
}

/* ── Brainwave band label ────────────────────────────────────────────────── */
const char *analog8_bw_band_name(const AnalogState8D *s) {
    static const char *bands[5] = {"Delta","Theta","Alpha","Beta","Gamma"};
    int b = (int)s->bw_band;
    return bands[(b >= 0 && b <= 4) ? b : 0];
}
