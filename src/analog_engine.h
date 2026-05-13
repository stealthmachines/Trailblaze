/*
 * analog_engine.h — 8D Kuramoto oscillator for token-conditioned sequence state
 *
 * Extracts the RK4 evolution core from:
 *   stealthmachines/AnalogContainer1 (analog_codec_native.c)
 *   https://github.com/stealthmachines/AnalogContainer1
 *
 * Bridged to HDGL-28 token routing:
 *   - Each input word → HDGL-routed expert_id → TokenEntry{k, γ, φ}
 *   - TokenEntry perturbs one of 8 Kuramoto oscillators
 *   - System evolves via RK4 until phase-lock (consensus)
 *   - Phase-lock score ranks retrieved passages (lower pvar = better resonance)
 *
 * Spiral8 geometry already present in HDGL (SPIRAL8_GEOMETRIES = 8 dims here).
 * https://github.com/stealthmachines/spiral8plus
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* ── Dimensions match Spiral8 geometry count in HDGL ─────────────────────── */
#define ANG_DIMS          8
#define ANG_PHASE_HIST   200
#define ANG_LOCK_WINDOW   50
#define ANG_LOCK_CV       0.05   /* coefficient of variation threshold   */
#define ANG_MAX_STEPS    4096   /* hard cap on RK4 iterations per query  */
#define ANG_DT            0.01  /* RK4 timestep (matches AnalogContainer1) */
#define ANG_SAT           1e6   /* saturation magnitude limit             */

#define ANG_PHI           1.6180339887498948
#define ANG_INV_PHI       0.6180339887498948
#define ANG_PI            3.14159265358979323846

/* ── Adaptive phase control (V4.0 Wu Wei harmonics — corrected K/γ ratios) ──
 * Source: WU_WEI_ANALYSIS.md / harmonics2/hdgl_bridge_v40.py
 * The CRITICAL insight: K/γ ratio drives convergence speed.
 * Pluck must be 1000:1 (not 25:1) to drive rapid consensus formation.
 * ─────────────────────────────────────────────────────────────────────────── */
#define ANG_SHA_INTERVAL       8   /* SHA-256 feedback correction every N steps */

/* Phase transition thresholds (variance-based with trend fallback) */
#define ANG_EMERGENCY_VAR   10.0   /* var > 10 → emergency hard damping          */
#define ANG_CV_TO_SUSTAIN   0.50   /* CV < 50% → advance Pluck → Sustain         */
#define ANG_CV_TO_FINETUNE  0.30   /* CV < 30% → advance Sustain → Fine Tune     */
#define ANG_CV_TO_LOCK      0.10   /* CV < 10% → advance Fine Tune → Lock        */

typedef enum {
    APHASE_PLUCK    = 0,   /* startup: K/γ=1000:1, excite the system        */
    APHASE_SUSTAIN  = 1,   /* oscillating: K/γ=375:1, maintain              */
    APHASE_FINETUNE = 2,   /* approaching equilibrium: K/γ=200:1            */
    APHASE_LOCK     = 3    /* stable consensus: K/γ=150:1                   */
} APhase;

/* ── Brainwave frequency bands ─────────────────────────────────────────────
 * Maps the mean Kuramoto oscillator frequency to standard EEG bands.
 * The natural frequencies freqs[i] are in units of radians/step at dt=0.01.
 * To target a specific band, seed freqs[i] to band_hz * 2π * dt.
 *   Alpha 10 Hz  → freqs[i] ≈ 10 * 2π * 0.01 ≈ 0.628  (resonant retrieval)
 *   Gamma 40 Hz  → freqs[i] ≈ 40 * 2π * 0.01 ≈ 2.513  (novel routing/insight)
 * ─────────────────────────────────────────────────────────────────────────── */
typedef enum {
    BWB_DELTA = 0,  /* < 4 Hz    (0.251 rad/step)  deep processing           */
    BWB_THETA = 1,  /* 4–8 Hz    (0.251–0.503)     memory consolidation      */
    BWB_ALPHA = 2,  /* 8–13 Hz   (0.503–0.817)     relaxed focus / retrieval */
    BWB_BETA  = 3,  /* 13–30 Hz  (0.817–1.885)     active reasoning          */
    BWB_GAMMA = 4   /* 30–100 Hz (1.885–6.283)     binding / novel routing   */
} BWBand;

/* ── 8D complex Kuramoto state ──────────────────────────────────────────────
 * Each dimension is a complex oscillator with its own natural frequency.
 * Phase coupling drives them toward consensus (all locked = low phase_var).
 * Direct port of analog_state_t from AnalogContainer1/analog_codec_native.c
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    double re[ANG_DIMS];          /* real part  of each oscillator       */
    double im[ANG_DIMS];          /* imaginary part                      */
    double freqs[ANG_DIMS];       /* natural frequency ωᵢ                */
    double phases[ANG_DIMS];      /* instantaneous phase θᵢ              */
    double phase_vels[ANG_DIMS];  /* dθᵢ/dt scratch                     */

    uint64_t step_count;          /* total RK4 steps taken               */
    double   phase_var;           /* current cross-oscillator phase var  */
    int      locked;              /* 1 = consensus lock achieved         */
    uint64_t lock_step;           /* step at which lock was first seen   */

    double phase_history[ANG_PHASE_HIST]; /* rolling window for CV check */
    int    hist_idx;
    int    hist_count;

    double gamma;      /* global damping coefficient γ                   */
    double k_coupling; /* global coupling constant K                     */

    /* V4.0 additions ─────────────────────────────────────────────────────── */
    APhase   aphase;             /* current adaptive phase (Pluck..Lock)    */
    uint64_t sha_feedback_count; /* total SHA-256 corrections applied       */    int      sha_disabled;       /* 1 = skip SHA feedback (for A/B testing) */
    /* Brainwave band tracking ──────────────────────────────────────────── */
    BWBand   bw_band;            /* dominant EEG band (Alpha/Gamma/etc.)     */
    double   mean_freq;          /* mean of freqs[] for band detection       */
    double   prev_phase_var;     /* previous-step pvar for trend detection   */
} AnalogState8D;

/* ── Token → Kuramoto entry ─────────────────────────────────────────────────
 * Encodes a (token_id, expert_id) pair as initial-condition perturbation.
 * Derived via φ-spiral mapping (consistent with hdgl_corpus_seeder.c).
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    double k;      /* coupling perturbation from φ-spiral of token_id  */
    double gamma;  /* damping offset from expert_id / num_experts       */
    double phase;  /* phase from φ-hash of token_id                     */
} TokenEntry;

/* ── Public API ──────────────────────────────────────────────────────────── */

/* Initialise or re-initialise the 8D state with deterministic seed values.
 * seed=0 gives the AnalogContainer1 default initial conditions.          */
void analog8_init(AnalogState8D *s, uint64_t seed, double gamma, double k_coupling);

/* Single RK4 step — advances the Kuramoto system by dt.
 * Mirrors rk4_step() from AnalogContainer1/analog_codec_native.c.       */
void analog8_rk4_step(AnalogState8D *s, double dt);

/* Compute cross-oscillator phase variance (lower = more synchronised).   */
double analog8_phase_var(const AnalogState8D *s);

/* Returns 1 when the rolling-window CV of phase_var falls below
 * ANG_LOCK_CV (consensus lock as per AnalogContainer1 PHASE_LOCK).      */
int analog8_is_locked(AnalogState8D *s);

/* Perturb dimension (token_idx % ANG_DIMS) with a TokenEntry.
 * Used to encode each input word into the oscillator initial conditions. */
void analog8_apply_token(AnalogState8D *s, const TokenEntry *e, int token_idx);

/* Evolve until lock or max_steps; returns final phase_var (lower = better). */
double analog8_score_until_lock(AnalogState8D *s, int max_steps);

/* Map (token_id, expert_id) → TokenEntry via φ-spiral encoding.
 * phi-spiral mapping consistent with hdgl_corpus_seeder phi_fourier_encode. */
TokenEntry token_to_analog_entry(uint32_t token_id, uint32_t expert_id, int num_experts);

/* ── V4.0 additions ──────────────────────────────────────────────────────── */

/* Apply one SHA-256 hybrid feedback correction to the oscillator phases.
 * Encodes state to 128 bytes, hashes, decodes back to phase perturbations.
 * Implements: S_{n+1} = H( D_n(r) ⊕ R_n ) from the "Defeating Shannon" doc. */
void analog8_sha_feedback(AnalogState8D *s);

/* Return a string label for the current adaptive phase ("Pluck" et al.).     */
const char *analog8_phase_name(const AnalogState8D *s);

/* Return a phase label from a plain APhase enum value (not a state pointer). */
const char *analog8_phase_name_from(APhase phase);

/* Return the EEG band name for the current oscillator mean frequency.        */
const char *analog8_bw_band_name(const AnalogState8D *s);

/* Retune natural frequencies toward target_hz (brainwave training mode).
 * Rescales freqs[] so their mean centres on target_hz while preserving
 * relative spread. Typical targets: Alpha=10.5, Gamma=40.0 Hz.            */
void analog8_tune_band(AnalogState8D *s, double target_hz);

/* Phase parameter tables — Wu Wei K/γ ratios:
 *   Pluck:    γ=0.005  K=5.0  → K/γ=1000:1 (excite)
 *   Sustain:  γ=0.008  K=3.0  → K/γ= 375:1 (maintain)
 *   FineTune: γ=0.010  K=2.0  → K/γ= 200:1 (refine)
 *   Lock:     γ=0.012  K=1.8  → K/γ= 150:1 (hold)    */
extern const double APHASE_GAMMA[4];     /* {0.005, 0.008, 0.010, 0.012}      */
extern const double APHASE_COUPLING[4];  /* {5.0,   3.0,   2.0,   1.8  }      */
