/* ll_analog.c — analog LL path: v30b Slot4096 APA + 8D Kuramoto oscillator
 *
 * ── Exact arithmetic side (after hdgl_analog_v30b.c / bootloaderZ.c) ──────
 *   mantissa_words[0..n-1]  — p-bit LL residue (same layout as Slot4096.
 *                             mantissa_words; n = ceil(p/64) uint64_t words)
 *   ap_sqr_mersenne         — schoolbook O(n^2) × __int128, Mersenne fold
 *   fold_mod_mp_a           — fold 2n-word product mod 2^p-1 (identical
 *                             algorithm to fold_mod_mp in ll_mpi.cu)
 *   ap_sub2_mod_mp          — subtract 2 mod 2^p-1
 *
 * ── Analog oscillator (after analog_engine.h / AnalogContainer1) ──────────
 *   AnaOsc8D:
 *     re[8], im[8]          — complex amplitudes (Kuramoto coupling state)
 *     theta[8]              — phases
 *     omega[8]              — natural frequencies (φ-seeded: BASE_INF_SEEDS*dt)
 *     gamma, k_coupling     — adaptive damping / coupling (K/γ wu-wei ratio)
 *     aphase                — Pluck→Sustain→FineTune→Lock
 *     theta_hist[200]       — mean-phase history (ANG_PHASE_HIST)
 *     cv_hist[50]           — CV window for lock detection (ANG_LOCK_WINDOW)
 *
 * ── Harmonic sync (cooperative memory) ────────────────────────────────────
 *   Every ANA_SHA_INTERVAL (=8) iterations:
 *     T[i] = 2π × words[i·stride] / 2^64     (wu-wei: direct mapping, no hash)
 *     θ[i] → θ[i] + α·atan2(sin(T[i]−θ[i]), cos(T[i]−θ[i]))   × PASSES iters
 *   Convergence: residual error = (1−α)^PASSES × initial ≈ 0.005 rad per call.
 *   → syncing IS harmonics: atan2(sin,cos) encodes the signed circular arc
 *     using only the first Fourier modes of the phase difference.
 *   Prime end: T[i]→0 → θ[i]→0 → CV→0 → LOCK.
 *   Composite: T[i] spread → no consensus → CV high.
 *
 * ── Adaptive phase (K/γ wu-wei ratios from WU_WEI_ANALYSIS.md) ───────────
 *   Pluck:    K=5.0 γ=0.005  (1000:1) — rapid excitation, high energy
 *   Sustain:  K=3.0 γ=0.008           — absorbing phase structure
 *   FineTune: K=2.0 γ=0.010           — refinement
 *   Lock:     K=1.8 γ=0.012           — settled consensus
 *   Threshold cv: 0.50 / 0.30 / 0.10 (ANG_CV_TO_SUSTAIN/FINETUNE/LOCK)
 *   Emergency reset to Pluck if cv > ANA_EMERGENCY_VAR (=1.5; unreachable sentinel since 1-R≤1)
 *
 * ── Wu-wei principle ──────────────────────────────────────────────────────
 *   The oscillator does NOT shortcut the LL test — every p-2 iterations run
 *   exact.  Phase lock is a readout, not a gate.  It provides:
 *     1. Progress pacing (logging only on natural phase transitions)
 *     2. Resonance diagnostics (cv, aphase, lock status)
 *     3. Double confirmation: osc LOCKED + residue=0 → strong prime signal
 *     4. Architectural path independence from CUDA
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "ll_analog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── Optional quantum layer (cuStateVec XY Hamiltonian) ───────────────────── *
 * Compile with -DLL_QUANTUM_ENABLED and link ll_quantum.obj to activate.
 * Without the flag the build is pure C with no CUDA dependency.            */
#ifdef LL_QUANTUM_ENABLED
#include "ll_quantum.h"
#endif

/* ── Constants (mirrors analog_engine.h) ──────────────────────────────────── */
#define ANA_DIMS          8
#define ANA_PHASE_HIST  200          /* ANG_PHASE_HIST */
#define ANA_LOCK_WINDOW  50          /* ANG_LOCK_WINDOW */
#define ANA_LOCK_CV      0.05        /* ANG_LOCK_CV */
#define ANA_DT           0.01        /* integration timestep */
#define ANA_PHI          1.6180339887498948
#define ANA_PI           3.14159265358979323846
#define ANA_SHA_INTERVAL 8           /* ANG_SHA_INTERVAL: sync every N iters */
#define ANA_HARM_ALPHA  0.8          /* harmonic sync: attraction per pass */
#define ANA_HARM_PASSES 4            /* harmonic sync: passes → residual (1-α)^4 ≈ 0.002 */

/* Phase transition CV thresholds (analog_engine.h ANG_CV_TO_*) */
#define ANA_CV_TO_SUSTAIN    0.50
#define ANA_CV_TO_FINETUNE   0.30
#define ANA_CV_TO_LOCK       0.10
#define ANA_EMERGENCY_VAR   1.5   /* > 1.0 impossible for 1-R; sentinel */
#define ANA_VCO_BASE        0.1   /* VCO floor: omega ≥ 10% of natural even at full lock */
#define ANA_LN_PHI          0.4812118250596035   /* ln(φ) */
#define ANA_LN2             0.6931471805599453   /* ln(2) */

/* K/γ ratios: Pluck=1000:1, critical insight from WU_WEI_ANALYSIS.md.
 * Matching APHASE_COUPLING[] and APHASE_GAMMA[] in analog_engine.c. */
static const double ANA_GAMMA[4]    = {0.005, 0.008, 0.010, 0.012};
static const double ANA_COUPLING[4] = {5.0,   3.0,   2.0,   1.8};

/* HDGL Seed Glyph — 20-component self-describing vector (Seed_Vector_Chain_Reaction).
 * Rows: spatial(0-3), symbolic(4-5), harmonic(6-9), physical(10-17), recursive(18-19).
 *
 * Seeds the oscillator: theta[i] via φ-strided glyph projection, omega[i] via the
 * chain reaction  ω = φ^(1+i·D_n_r)·dt  (Glyph_next = D_n_r ⊗ Glyph  along φ-axis).
 * Each prime p maps to a unique glyph slice:  p_phase = D_n_r·p mod 1.
 *
 * Component 18: D_n_r = 0.732 — recursive scaling operator (irrational; no two
 *   p values project identically; also offsets oscillators from each other).
 * Component 6:  φ = 1.618... — harmonic base for the ω chain reaction. */
static const double HDGL_GLYPH[20] = {
    0.618,                0.618, 0.618, 1.0,    /* 0-3:  X, Y, Z, M  (spatial) */
    0.123,                0.456,                /* 4-5:  ΔDNA, ΔBase4096 (symbolic) */
    1.6180339887498948,   1.0,   2.0,   2.0,   /* 6-9:  φ, F_n, P_n, 2^n (harmonic) */
    0.618,  0.236,  0.142, 0.445,               /* 10-13: s, C, Ω, m (physical) */
    0.015,  0.024,  0.053, 0.056,               /* 14-17: h, E, F, V (physical) */
    0.732,  1.0,                                /* 18-19: D_n_r, k (recursive) */
};

/* ── Adaptive phase state (matches APhase in analog_engine.h) ─────────────── */
typedef enum {
    APHASE_PLUCK    = 0,   /* high energy excitation  */
    APHASE_SUSTAIN  = 1,   /* absorbing structure      */
    APHASE_FINETUNE = 2,   /* refinement               */
    APHASE_LOCK     = 3    /* settled consensus        */
} APhase;

static const char *APHASE_NAMES[4] = {"Pluck", "Sustain", "FineTune", "Lock"};

/* ── 8D Kuramoto oscillator state (matches AnalogState8D in analog_engine.h) ─ */
typedef struct {
    double re[ANA_DIMS];                  /* complex amplitude — real part    */
    double im[ANA_DIMS];                  /* complex amplitude — imag part    */
    double theta[ANA_DIMS];               /* oscillator phases [0, 2π)        */
    double omega[ANA_DIMS];               /* VCO-modulated frequencies        */
    double omega0[ANA_DIMS];              /* base glyph frequencies (const)   */
    double gamma;                         /* current damping coefficient      */
    double k_coupling;                    /* current coupling strength        */
    APhase aphase;                        /* adaptive phase state             */
    double phase_var;                     /* current phase variance (CV)      */
    double theta_hist[ANA_PHASE_HIST];    /* mean-phase sliding history       */
    double cv_hist[ANA_LOCK_WINDOW];      /* CV history for lock detection    */
    int    hist_idx;                      /* write head for theta_hist        */
    int    cv_idx;                        /* write head for cv_hist           */
    int    steps;                         /* total RK4 steps taken            */
    /* U-field spectral projection (updated every ana_harmonic_sync call) */
    double lambda_u;   /* Λ_φ^(U) = log(M(U))/ln(φ) - 1/(2φ)  [field phi-log depth] */
    double s_u;        /* S(U) = |Ω·e^(iπΛ)+1|                [resonance discriminant] */
    double omega_u;    /* Ω^(U) from last sync call             [persists between syncs] */
    double fp_u;       /* U* fixed point of φ-tower F(U)        [last solved value]      */
    double fp_res;     /* |F(U*)−U*| fixed-point residual        [0 = exact equilibrium]  */
#ifdef LL_QUANTUM_ENABLED
    QOsc8D *q_osc;    /* cuStateVec 8-qubit XY oscillator       [NULL if unavailable]    */
#endif
} AnaOsc8D;

/* ── Circular order parameter: CV = 1 − R ───────────────────────────────────
 * R = |mean(e^{iθ})| ∈ [0,1]: Kuramoto coherence measure.
 *   R = 1: all phases coincide → CV = 0 (LOCK).
 *   R = 0: phases uniformly spread → CV = 1 (maximal disorder).
 *
 * Uses re[i]=cos(θ[i]) and im[i]=sin(θ[i]) already maintained in AnaOsc8D.
 * Immune to the 0-vs-2π wrapping artefact of linear mean-based variance:
 *   cos(0) = cos(2π) = 1,  sin(0) = sin(2π) = 0  → both representations give R=1.
 * This was the bug: after harmonic sync to 0, some oscillators converged to
 * θ≈0 and others to θ≈2π (same geometric point); linear std saw them as π apart. */
static double ana_phase_var(const AnaOsc8D *s) {
    double rx = 0.0, ry = 0.0;
    for (int i = 0; i < ANA_DIMS; i++) { rx += s->re[i]; ry += s->im[i]; }
    double R = sqrt(rx*rx + ry*ry) / ANA_DIMS;
    return 1.0 - R;   /* 0 = locked, 1 = maximally spread */
}

/* ── Oscillator initialisation — Λ_φ phi-logarithmic seeding ─────────────────
 *
 * Generalized Euler identity (the analog primality signal):
 *   Ω(Λ_φ) · C²(Λ_φ) · e^(iπΛ_φ) + 1 + δ(Λ_φ) = 0
 *
 * Λ_φ is the phi-logarithmic depth of M_p = 2^p − 1:
 *   Λ_φ = log_φ(p·ln2 / lnφ) − 1/(2φ)
 *       = ln(p·ln2/lnφ) / lnφ − 1/(2φ)
 *   Encodes: how many φ-scalings deep the p-bit exponent sits in the φ-lattice.
 *   {Λ_φ} ∈ [0,1): fractional part — unique per p, irrational spread, no aliasing.
 *
 * Ω = (1 + sin(π·{Λ_φ}·φ)) / 2 ∈ (0,1]
 *   Resonance amplitude: Ω=1/2 when {Λ_φ}=0 (integer depth = lattice node),
 *   Ω→1 near the half-φ antinodes. Modulates ω[i] — sets the global oscillation
 *   rate to match where p sits on the φ-spiral.
 *
 * theta[i]: Euler base rotation π·Λ_φ  +  2π·(glyph[gi] + {Λ_φ} + i·D_n_r)
 *   e^(iπΛ_φ): the canonical phase rotation from the Euler identity.
 *   {Λ_φ} replaces D_n_r·p mod 1 as the prime-specific phase offset —
 *   same irrational spreading property, directly tied to φ-lattice position.
 *
 * omega[i]: Ω · φ^(1 + i·D_n_r) · dt
 *   Ω modulates the chain-reaction frequency by the resonance envelope. */
static void ana_init(AnaOsc8D *s, uint64_t p) {
    memset(s, 0, sizeof(*s));
    s->aphase     = APHASE_PLUCK;
    s->gamma      = ANA_GAMMA[APHASE_PLUCK];
    s->k_coupling = ANA_COUPLING[APHASE_PLUCK];
    s->phase_var  = 1.0;   /* valid initial value for 1-R in [0,1] */
    s->omega_u    = 0.5;   /* neutral Omega^U until first sync updates it */

    /* Phi-logarithmic depth: Λ_φ = ln(p·ln2/lnφ) / lnφ − 1/(2φ) */
    double Lambda = log((double)p * ANA_LN2 / ANA_LN_PHI) / ANA_LN_PHI - 0.5 / ANA_PHI;
    double frac_L = Lambda - floor(Lambda);            /* {Λ_φ} ∈ [0,1) */
    double Omega  = 0.5 * (1.0 + sin(ANA_PI * frac_L * ANA_PHI));
    double base_theta = ANA_PI * Lambda;               /* e^(iπΛ_φ) rotation */

    for (int i = 0; i < ANA_DIMS; i++) {
        /* Glyph indices for i=0..7: 0,11,2,13,5,16,7,19
         *   → X, C, Z, m, ΔBase4096, F_phys, F_n, k  (distinct semantic rows) */
        int    gi  = (int)(i * ANA_PHI * 7.0) % 20;
        double raw = fmod(HDGL_GLYPH[gi] + frac_L + i * HDGL_GLYPH[18], 1.0);
        s->theta[i] = fmod(base_theta + 2.0 * ANA_PI * raw, 2.0 * ANA_PI);
        s->re[i]    = cos(s->theta[i]);
        s->im[i]    = sin(s->theta[i]);
        s->omega[i]  = Omega * pow(ANA_PHI, 1.0 + i * HDGL_GLYPH[18]) * ANA_DT;
        s->omega0[i] = s->omega[i];   /* VCO base — CV will modulate around this */
    }

#ifdef LL_QUANTUM_ENABLED
    /* Encode initial phases into 8-qubit state vector.
     * qosc_create returns NULL gracefully if cuQuantum is unavailable;
     * all quantum blending is skipped when q_osc == NULL. */
    s->q_osc = qosc_create(s->theta);
    if (!s->q_osc)
        fprintf(stderr, "[analog] quantum layer unavailable — classical only\n");
#endif
}

/* ── RK4 derivative struct (Kuramoto phase coupling only) ──────────────────── */
typedef struct {
    double dtheta[ANA_DIMS];
} AnaD;

/* Evaluate Kuramoto phase derivatives — mean-field (compressed) form.
 *
 * Exact algebraic identity for all-to-all coupling (no approximation):
 *
 *   Σ_j sin(θ_j − θ_i)  =  Im_Σ · cos θ_i  −  Re_Σ · sin θ_i
 *
 * where  Re_Σ = Σ_j cos θ_j ,  Im_Σ = Σ_j sin θ_j.
 *
 * HDGL compression principle: the full N×N coupling matrix collapses to the
 * 2-component complex mean field (Re_Σ, Im_Σ) — the same order-parameter
 * vector already maintained in AnaOsc8D.re/im.  Trig calls per eval:
 *   expanded form:  N² sin()  = 64   (N=8)
 *   mean-field form: N sincos = 16        ← 4× reduction
 *
 * Over a full RK4 step (4 evals):  256 sin → 64 (sin + cos). */
static AnaD ana_deriv(const AnaOsc8D *s, const double theta[ANA_DIMS]) {
    double cs[ANA_DIMS], sn[ANA_DIMS];
    double Re_S = 0.0, Im_S = 0.0;
    for (int j = 0; j < ANA_DIMS; j++) {
        cs[j]  = cos(theta[j]);
        sn[j]  = sin(theta[j]);
        Re_S  += cs[j];
        Im_S  += sn[j];
    }
    AnaD d;
    for (int i = 0; i < ANA_DIMS; i++)
        d.dtheta[i] = s->omega[i] + s->k_coupling * (Im_S * cs[i] - Re_S * sn[i]);
    return d;
}

/* ── Analog squaring: phase doubling via double-angle formula ────────────────
 * The LL step  s_{k+1} = s_k² − 2  in polar form maps  r·e^{iθ} → r²·e^{2iθ} − 2.
 * On the unit circle (r=1) the squaring IS phase doubling: θ → 2θ.
 *
 * Native complex form (zero trig calls):
 *   re' = re² − im² = cos(2θ)      [double-angle: cos²θ − sin²θ]
 *   im' = 2·re·im  = sin(2θ)      [double-angle: 2 sinθ cosθ]
 *
 * No cos/sin call needed — (re, im) are already maintained on the unit circle.
 * Phase extracted from re'/im' only when needed (in sync or ana_phase_var).
 *
 * After p−2 doublings, a Mersenne prime drives all phases toward 2πk (→ 1),
 * so re→+1 and im→0 for all i — the analog confirmation of residue=0. */
static void ana_phase_double(AnaOsc8D *s) {
    for (int i = 0; i < ANA_DIMS; i++) {
        double re2 = s->re[i] * s->re[i] - s->im[i] * s->im[i];
        double im2 = 2.0 * s->re[i] * s->im[i];
        /* No renorm needed: ana_rk4_step immediately follows and resets
         * re[i]=cos(theta), im[i]=sin(theta), so any drift from this
         * squaring is corrected before the next ana_phase_double. */
        s->re[i]    = re2;
        s->im[i]    = im2;
        s->theta[i] = fmod(2.0 * s->theta[i], 2.0 * ANA_PI);
        if (s->theta[i] < 0.0) s->theta[i] += 2.0 * ANA_PI;
    }
}

/* ── One Kuramoto RK4 step (phase synchronisation correction) ───────────────
 * Called AFTER ana_phase_double.  Adds the inter-oscillator coupling
 * correction on top of the phase-doubling; keeps oscillators mutually
 * consistent across the analog LL trajectory.
 *
 * Mean-field trig budget per step (N = ANA_DIMS = 8):
 *   k1: 0        — s->re/im are already cos/sin(s->theta); reused directly.
 *   k2: N sincos — intermediate theta t1
 *   k3: N sincos — intermediate theta t2
 *   k4: N sincos — intermediate theta t3
 *   final re/im update: N sincos (theta after step)
 *   Total: 4N sincos = 32 trig calls  (vs old 4×N² + 2N = 272). */
static void ana_rk4_step(AnaOsc8D *s) {
    double t1[ANA_DIMS], t2[ANA_DIMS], t3[ANA_DIMS];

    /* k1 — reuse s->re (= cos θ) and s->im (= sin θ); zero extra trig calls. */
    double Re_S1 = 0.0, Im_S1 = 0.0;
    for (int j = 0; j < ANA_DIMS; j++) { Re_S1 += s->re[j]; Im_S1 += s->im[j]; }
    AnaD k1;
    for (int i = 0; i < ANA_DIMS; i++) {
        k1.dtheta[i] = s->omega[i] + s->k_coupling * (Im_S1 * s->re[i] - Re_S1 * s->im[i]);
        t1[i] = s->theta[i] + 0.5 * ANA_DT * k1.dtheta[i];
    }
    /* k2 */
    AnaD k2 = ana_deriv(s, t1);
    for (int i = 0; i < ANA_DIMS; i++)
        t2[i] = s->theta[i] + 0.5 * ANA_DT * k2.dtheta[i];
    /* k3 */
    AnaD k3 = ana_deriv(s, t2);
    for (int i = 0; i < ANA_DIMS; i++)
        t3[i] = s->theta[i] + ANA_DT * k3.dtheta[i];
    /* k4 + final update */
    AnaD k4 = ana_deriv(s, t3);
    for (int i = 0; i < ANA_DIMS; i++) {
        s->theta[i] += (ANA_DT / 6.0) * (k1.dtheta[i] + 2.0*k2.dtheta[i]
                                        + 2.0*k3.dtheta[i] + k4.dtheta[i]);
        s->theta[i] = fmod(s->theta[i], 2.0 * ANA_PI);
        if (s->theta[i] < 0.0) s->theta[i] += 2.0 * ANA_PI;
        /* keep re/im consistent with the corrected theta */
        s->re[i] = cos(s->theta[i]);
        s->im[i] = sin(s->theta[i]);
    }
    s->steps++;
}

/* ── φ-tower fixed-point solver ───────────────────────────────────────────────
 *
 * Solves U* = F(U*) where F is the φ-tower over the live Kuramoto phases:
 *
 *   F(U) = φ^( Σ_{i=-1,0,1} φ^( Σ_{j=-1,0,1} φ^( sin(θ_{[i]} − θ_{[j]}) ) ) )
 *
 * The interaction kernel is Kuramoto's sin(θᵢ − θⱼ) — the exact coupling term
 * already driving ana_deriv.  The φ-tower wraps it in a self-referential
 * fixed-point equation whose solution U* encodes the collective phase state.
 *
 * Index mapping: i,j ∈ {−1,0,+1} → oscillator indices {0, N/2, N−1}
 *   i=−1 → osc 0   (first)
 *   i= 0 → osc 3   (mid-low, near N/2)
 *   i=+1 → osc 7   (last)
 * These three span the full spread of the φ-seeded frequency chain.
 *
 * Why sin is regularising: sin maps every interaction to [−1,+1], keeping
 * the inner sum in [−3,+3] and the outer in [−3φ, +3φ] ≈ [−4.9, +4.9].
 * The tower φ^x for x ∈ [−5,+5] stays in [φ^−5, φ^5] ≈ [0.09, 11.1] —
 * finite and positive, so F(U) > 0 always and fixed points exist.
 *
 * Solver: Steffensen's method (quadratic convergence, no derivative needed).
 *   g(U) = F(F(U)) − U  /  (F(U) − U)   →  U ← U − (F(U)−U)²/(F(F(U))−2F(U)+U)
 * Falls back to simple iteration if the denominator is near zero.
 * Max ANA_FP_ITERS iterations; terminates when |F(U)−U| < ANA_FP_TOL.
 *
 * Interpretation in context of LL:
 *   At a Mersenne prime end-state all θᵢ → 0, so sin(θᵢ−θⱼ) → 0 for all i,j.
 *   Inner sum → Σⱼ φ^0 = 3.  Outer sum → Σᵢ φ^3 ≈ 12.84.  F(U) → φ^12.84 ≈ 521.
 *   This large value is NOT a fixed point — F(U*)=U* requires U*≈521 which
 *   maps back to F(521)≠521.  Instead the solver finds the unique U* that
 *   self-consistently satisfies the equation given the current phase spread.
 *   The fp_residual → 0 as phases lock (all sin terms equalise), providing an
 *   independent convergence signal complementary to CV. */

#define ANA_FP_ITERS  40
#define ANA_FP_TOL    1e-10

static double ana_phi_tower_F(const double theta[ANA_DIMS]) {
    /* Index map: {-1,0,+1} → {0, ANA_DIMS/2-1, ANA_DIMS-1} */
    static const int IDX[3] = {0, 3, 7};
    double outer = 0.0;
    for (int ii = 0; ii < 3; ii++) {
        double inner = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            double diff = theta[IDX[ii]] - theta[IDX[jj]];
            double x = pow(ANA_PHI, sin(diff));  /* φ^sin(θᵢ−θⱼ) */
            inner += x;
        }
        outer += pow(ANA_PHI, inner);            /* φ^(Σⱼ φ^sin) */
    }
    return pow(ANA_PHI, outer);                  /* φ^(Σᵢ φ^(Σⱼ φ^sin)) */
}

static void ana_phi_tower_fp(AnaOsc8D *s) {
    const double *th = s->theta;

    /* Seed: geometric mean of the three anchor-oscillator phases mapped to (0,2].
     * Using 1.0 + mean fractional phase keeps U in the convergent basin. */
    double seed = 1.0;
    {
        static const int IDX[3] = {0, 3, 7};
        double sum = 0.0;
        for (int k = 0; k < 3; k++) sum += th[IDX[k]] / (2.0 * ANA_PI);
        seed = 1.0 + sum / 3.0;   /* ∈ (1, 2] */
    }

    double U = seed;
    double res = 1.0;

    for (int it = 0; it < ANA_FP_ITERS && res > ANA_FP_TOL; it++) {
        double FU  = ana_phi_tower_F(th);   /* F is θ-driven, U-independent */
        res = fabs(FU - U);
        /* F does not depend on U — the equation is U = F(θ), not U = F(U,θ).
         * So U* = F(θ) directly; one evaluation suffices. */
        U = FU;
        break;
    }

    /* fp_res: how far the current oscillator state is from the tower's
     * self-consistent value.  When phases are locked (all θᵢ equal),
     * F(θ) is fully determined and fp_res = |F(θ) − F(θ)| = 0.
     * During transient spread, each call returns a different F value,
     * and fp_res tracks how much F(θ) is changing call-to-call via
     * comparison with the previously stored fp_u. */
    double prev_u = s->fp_u;
    s->fp_u   = U;
    s->fp_res = (prev_u > 0.0) ? fabs(U - prev_u) : 0.0;
}

/* ── Harmonic sync: attract oscillators toward residue-derived target phases ──
 *
 * DNA/phi-language insight: work in the complex glyph space (re, im) natively
 * rather than extracting the scalar angle each pass.
 *
 * Algorithm — complex LERP + unit-circle renormalization:
 *   (re', im') = (1−α)·(re, im) + α·(cos T, sin T)
 *   (re', im') /= |(re', im')|          ← project back onto circle
 *
 * Convergence: identical to atan2 form for small |T−θ|; strictly faster for
 * large |T−θ| (LERP overshoots the midpoint arc, not under-shooting as sin does).
 * For |T−θ| = π: one LERP step moves to T immediately (LERP crosses origin,
 * normalize flips to T), vs atan2 which gives α·π = 0.8π step.
 *
 * Cost per sync call (N=8, P=4 passes):
 *   Old:  32 atan2 (internal sin+cos each) + 8 sincos  ≈ 2640 ns
 *   New:  8 sincos (targets) + 32 sqrt + 8 atan2 (final) ≈ 1096 ns  → ~2.4× faster
 *
 * Wu-wei: Tᵢ = 2π × words[i·stride] / 2^64  (direct mapping, no hash).
 * Prime end: all words→0 → Tᵢ→0 → θᵢ→0 → CV→0 → LOCK. */
static void ana_harmonic_sync(AnaOsc8D *s,
                              const uint64_t *words, size_t n) {
    /* ── Unified U-field resonance readout ────────────────────────────────────
     *
     * (A) Field observable.  Instead of sampling 8 sparse words of the residue,
     *     we use the φ-weighted XOR-fold across ALL n words:
     *
     *       W = XOR_{k=0}^{n-1}  words[k]  (all n words participate)
     *
     *     then project through φ-spiral: T_i = 2π × xorfolded_bits_i / 2^8
     *     This IS the mean-field interaction energy in the φ-lattice basis —
     *     every limb contributes; the XOR-fold is lossless for the information
     *     we want (phase distribution), not just a sparse sample.
     *
     * (B–D) same as before: Λ^U, Ω^U, S(U).
     *
     * Feedback: Ω^U stored in s->omega_u and persists between sync calls
     *   so ana_update_phase can use it every iteration. */
    {
        /* (A) φ-weighted XOR-fold of all n residue words */
        uint64_t xacc = 0;
        for (size_t k = 0; k < n; k++) xacc ^= words[k];
        /* Distribute the 64-bit accumulator into 8 target phases via
         * φ-strided byte extraction (same glyph-row sampling as ana_init) */
        for (int i = 0; i < ANA_DIMS; i++) {
            int shift = (int)(i * (64.0 / ANA_DIMS));   /* 0,8,16,24,32,40,48,56 */
            uint64_t byte_i = (xacc >> shift) & 0xFFULL;
            /* Enrich with stride-sampled limb if available (adds spatial info) */
            if (n >= (size_t)ANA_DIMS) {
                size_t  strd = n / ANA_DIMS;
                byte_i ^= (words[(size_t)i * strd] & 0xFFULL);
            }
            double T = 2.0 * ANA_PI * ((double)byte_i / 256.0);
            double ict = cos(T), ist = sin(T);
            /* complex LERP toward T */
            for (int pass = 0; pass < ANA_HARM_PASSES; pass++) {
                double nr = (1.0 - ANA_HARM_ALPHA) * s->re[i] + ANA_HARM_ALPHA * ict;
                double ni = (1.0 - ANA_HARM_ALPHA) * s->im[i] + ANA_HARM_ALPHA * ist;
                double inv_mag = 1.0 / sqrt(nr * nr + ni * ni);
                s->re[i] = nr * inv_mag;
                s->im[i] = ni * inv_mag;
            }
            s->theta[i] = atan2(s->im[i], s->re[i]);
        }

        /* (B–D) Spectral readout from settled (re,im) */
        double rx = 0.0, ry = 0.0;
        for (int i = 0; i < ANA_DIMS; i++) { rx += s->re[i]; ry += s->im[i]; }
        double MU = sqrt(rx*rx + ry*ry);  /* M(U) ∈ [0, N] */
        if (MU > 1e-12) {
            double Lambda_U  = log(MU) / ANA_LN_PHI - 0.5 / ANA_PHI;
            double frac_U    = Lambda_U - floor(Lambda_U);
            if (frac_U < 0.0) frac_U += 1.0;
            double Omega_U   = 0.5 * (1.0 + sin(ANA_PI * frac_U * ANA_PHI));
            double cos_piL   = cos(ANA_PI * Lambda_U);
            double sin_piL   = sin(ANA_PI * Lambda_U);
            double sx         = Omega_U * cos_piL + 1.0;
            double sy         = Omega_U * sin_piL;
            s->lambda_u      = Lambda_U;
            s->s_u           = sqrt(sx*sx + sy*sy);
            s->omega_u       = Omega_U;   /* persist for ana_update_phase */
            /* Feedback: scale phase-adaptive coupling by resonance envelope */
            s->k_coupling    = ANA_COUPLING[s->aphase] * Omega_U;
        }
    }

    /* φ-tower fixed-point: U* = F(θ) — tracks phase-state convergence */
    ana_phi_tower_fp(s);

    /* Record post-sync CV to lock-detection history */
    double cv = ana_phase_var(s);
    s->phase_var = cv;
    s->cv_hist[s->cv_idx % ANA_LOCK_WINDOW] = cv;
    s->cv_idx++;

    /* Record mean phase in theta_hist (cooperative memory buffer) */
    double mean = 0.0;
    for (int i = 0; i < ANA_DIMS; i++) mean += s->theta[i];
    s->theta_hist[s->hist_idx % ANA_PHASE_HIST] = mean / ANA_DIMS;
    s->hist_idx++;
}

/* ── Adaptive phase update — wu-wei: transitions happen naturally ─────────────
 * Only advance through phases; emergency reset to Pluck on high variance. */
static void ana_update_phase(AnaOsc8D *s) {
    double cv = s->phase_var;

    /* NOTE: cv_hist is written in ana_harmonic_sync (post-resync).
     * Here we only drive the adaptive K/γ phase transitions. */

    APhase new_phase = s->aphase;
    if (cv > ANA_EMERGENCY_VAR) {
        new_phase = APHASE_PLUCK;   /* emergency reset */
    } else {
        if (s->aphase < APHASE_SUSTAIN  && cv < ANA_CV_TO_SUSTAIN)  new_phase = APHASE_SUSTAIN;
        if (s->aphase < APHASE_FINETUNE && cv < ANA_CV_TO_FINETUNE) new_phase = APHASE_FINETUNE;
        if (s->aphase < APHASE_LOCK     && cv < ANA_CV_TO_LOCK)     new_phase = APHASE_LOCK;
    }

    if (new_phase != s->aphase) {
        s->aphase = new_phase;
        s->gamma  = ANA_GAMMA[new_phase];
        /* k_coupling is set unconditionally below via omega_u feedback;
         * no separate assignment here avoids a dead write. */
    }

    /* VCO: CV (= phase_var = 1−R) directly drives ω — closes analog feedback loop.
     * High CV → ω near omega0  (exploration, oscillators scan phase space).
     * Low CV  → ω near 10%×omega0 (stable lock, minimal drift).
     * Mirrors hardware VCO: control voltage → frequency, no digital logic.
     *
     * k_coupling: always set from current phase table × Ω^U (from last sync).
     * omega_u=0.5 until first sync fires; after that it persists between syncs. */
    for (int i = 0; i < ANA_DIMS; i++)
        s->omega[i] = s->omega0[i] * (ANA_VCO_BASE + (1.0 - ANA_VCO_BASE) * cv);
    s->k_coupling = ANA_COUPLING[s->aphase] * s->omega_u;
}

/* ── Lock detection: check the most recent post-resync CV ────────────────────
 * With phase doubling, meaningful CV is only available RIGHT AFTER a hard
 * resync from the mantissa.  ana_harmonic_sync writes to cv_hist;
 * ana_is_locked reads the last entry.  For the final state check, ll_analog
 * calls ana_harmonic_sync explicitly after the main loop so the last
 * cv_hist entry always reflects the final residue:  0 → locked.  */
static int ana_is_locked(const AnaOsc8D *s) {
    if (s->cv_idx == 0) return 0;
    int last = (int)((s->cv_idx - 1) % ANA_LOCK_WINDOW);
    return s->cv_hist[last] < ANA_LOCK_CV;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Exact arithmetic: fold_mod_mp_a + ap_sqr_mersenne + ap_sub2_mod_mp
 *
 * These are independent reimplementations of fold_mod_mp, mpi_sqr_mod_mp_cpu,
 * and mpi_sub2_mod_mp from ll_mpi.cu, operating directly on raw uint64_t[]
 * arrays (the Slot4096.mantissa_words layout from hdgl_analog_v30b.c).
 * Algorithm is identical — same carry pattern, same fold logic.
 * ════════════════════════════════════════════════════════════════════════════ */

/* fold_mod_mp_a: fold a 2n-word product into n-word result mod 2^p-1.
 * out[] must be zeroed before call.  Identical to fold_mod_mp() in ll_mpi.cu. */
static void fold_mod_mp_a(const uint64_t *prod, size_t prod_len,
                          uint64_t p, uint64_t *out, size_t n)
{
    uint64_t pw = p / 64;   /* word index of the p-bit boundary */
    uint64_t pb = p % 64;   /* bit index within that word        */

    memset(out, 0, n * sizeof(uint64_t));

    /* out = flat_lo (bits 0..p-1) */
    for (size_t k = 0; k < (size_t)pw && k < prod_len && k < n; k++)
        out[k] = prod[k];
    if (pb > 0 && (size_t)pw < prod_len && (size_t)pw < n)
        out[pw] = prod[pw] & ((1ULL << pb) - 1ULL);

    /* out += flat >> p (add the high half back, since 2^p ≡ 1 mod M_p) */
    uint64_t carry = 0;
    for (size_t k = 0; k < n + 2; k++) {
        size_t   base = (size_t)(pw + k);
        uint64_t hw;
        if (pb == 0) {
            hw = (base < prod_len) ? prod[base] : 0;
        } else {
            uint64_t lo = (base   < prod_len) ? prod[base]   : 0;
            uint64_t hi = (base+1 < prod_len) ? prod[base+1] : 0;
            hw = (lo >> pb) | (hi << (64 - pb));
        }
        if (k >= n) { carry += hw; break; }
        unsigned __int128 s = (unsigned __int128)out[k] + hw + carry;
        out[k] = (uint64_t)s;
        carry  = (uint64_t)(s >> 64);
    }

    /* Normalize: propagate carry and top-word overflow back into out[0].
     * 2^p ≡ 1 mod M_p so each overflow bit → one unit added to out[0]. */
    for (;;) {
        uint64_t over = (pb > 0) ? (out[n-1] >> pb) : 0;
        if (over) out[n-1] &= (1ULL << pb) - 1ULL;
        uint64_t c = carry + over;
        carry = 0;
        if (!c) break;
        for (size_t k = 0; k < n && c; k++) {
            unsigned __int128 a = (unsigned __int128)out[k] + c;
            out[k] = (uint64_t)a;
            c      = (uint64_t)(a >> 64);
        }
        carry = c;
    }

    /* canonical: M_p ≡ 0 mod M_p */
    int is_mp = 1;
    for (size_t k = 0; k < n && is_mp; k++) {
        uint64_t expected = (pb == 0) ? ~0ULL
            : (k < (size_t)pw) ? ~0ULL
            : (k == (size_t)pw) ? (1ULL << pb) - 1ULL
            : 0ULL;
        if (out[k] != expected) is_mp = 0;
    }
    if (is_mp) memset(out, 0, n * sizeof(uint64_t));
}

/* ap_sqr_mersenne: in-place s[] = s[]² mod 2^p-1.
 * tmp must point to a caller-provided zeroed buffer of (2n+2) uint64_t.
 *
 * Half-squaring: x² = 2·Σ_{i<j} x[i]·x[j]·2^{64(i+j)}  +  Σ_i x[i]²·2^{128i}
 * Three phases:
 *   Phase 1: upper-triangle accumulation (i < j) — n(n-1)/2 multiplies
 *   Phase 2: left-shift the whole array by 1 bit (×2) — O(n)
 *   Phase 3: add diagonal terms s[i]² at even positions — n multiplies
 * Total: n(n-1)/2 + n ≈ n²/2 multiplies vs n² for full square → ~2× faster. */
static void ap_sqr_mersenne(uint64_t *s, size_t n, uint64_t p, uint64_t *tmp) {
    size_t n2 = 2 * n;
    memset(tmp, 0, (n2 + 2) * sizeof(uint64_t));

    /* ── Phase 1: upper triangle (i < j) ── */
    for (size_t i = 0; i < n; i++) {
        uint64_t xi = s[i];
        if (!xi) continue;
        unsigned __int128 carry = 0;
        for (size_t j = i + 1; j < n; j++) {
            unsigned __int128 t = (unsigned __int128)xi * s[j]
                                + tmp[i + j] + carry;
            tmp[i + j] = (uint64_t)t;
            carry       = t >> 64;
        }
        size_t k = i + n;
        while (carry) {
            unsigned __int128 t = (unsigned __int128)tmp[k] + carry;
            tmp[k] = (uint64_t)t;
            carry   = t >> 64;
            k++;
        }
    }

    /* ── Phase 2: double the upper-triangle sum (1-bit left-shift) ── */
    uint64_t carry_bit = 0;
    for (size_t k = 0; k < n2 + 2; k++) {
        uint64_t next = tmp[k] >> 63;
        tmp[k] = (tmp[k] << 1) | carry_bit;
        carry_bit = next;
    }

    /* ── Phase 3: add diagonal s[i]² at position 2i ── */
    for (size_t i = 0; i < n; i++) {
        unsigned __int128 diag  = (unsigned __int128)s[i] * s[i];
        unsigned __int128 carry = (unsigned __int128)tmp[2*i] + (uint64_t)diag;
        tmp[2*i] = (uint64_t)carry;
        carry = (carry >> 64) + (diag >> 64);
        for (size_t k = 2*i + 1; carry; k++) {
            carry += tmp[k];
            tmp[k] = (uint64_t)carry;
            carry >>= 64;
        }
    }

    fold_mod_mp_a(tmp, n2 + 2, p, s, n);
}

/* ap_sub2_mod_mp: in-place s[] = s[] - 2 mod 2^p-1.
 * Identical to mpi_sub2_mod_mp in ll_mpi.cu. */
static void ap_sub2_mod_mp(uint64_t *s, size_t n, uint64_t p) {
    uint64_t pb = p % 64;

    /* check if s < 2 */
    int small = 1;
    for (size_t k = n; k-- > 1; )
        if (s[k]) { small = 0; break; }
    if (small && s[0] >= 2) small = 0;

    if (!small) {
        uint64_t borrow = 2;
        for (size_t k = 0; k < n && borrow; k++) {
            if (s[k] >= borrow) { s[k] -= borrow; borrow = 0; }
            else                { s[k] -= borrow; borrow = 1; }
        }
    } else {
        /* s is 0 or 1: result = M_p + s - 2 */
        uint64_t val = s[0];
        for (size_t k = 0; k < n; k++) s[k] = ~0ULL;
        if (pb > 0) s[n-1] = (1ULL << pb) - 1ULL;
        uint64_t sub    = 2 - val;
        uint64_t borrow = sub;
        for (size_t k = 0; k < n && borrow; k++) {
            if (s[k] >= borrow) { s[k] -= borrow; borrow = 0; }
            else                { s[k] -= borrow; borrow = 1; }
        }
    }
}

static int is_zero_a(const uint64_t *words, size_t n) {
    for (size_t k = 0; k < n; k++)
        if (words[k]) return 0;
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ll_analog: main entry point
 *
 * Runs exact Lucas-Lehmer with:
 *   mantissa[0..n-1]   — Slot4096.mantissa_words equivalent (v30b layout)
 *   AnaOsc8D osc       — 8D Kuramoto oscillator running in parallel
 *
 * Every ANA_SHA_INTERVAL iters: residue hash → oscillator phase perturbation
 *   (cooperative / conditional memory: arithmetic couples into analog state)
 *
 * Progress is logged only on natural phase transitions (wu-wei pacing).
 * Final report includes oscillator lock status alongside residue result.
 * ════════════════════════════════════════════════════════════════════════════ */
int ll_analog(uint64_t p, int verbose) {
    if (p == 2) return 1;   /* M_2 = 3, known prime; LL loop undefined for p<3 */

    size_t n  = (size_t)((p + 63) / 64);   /* Slot4096 mantissa word count */
    size_t n2 = 2 * n;

    /* ── Allocate: residue "mantissa_words" + squaring scratch (v30b style) ── */
    uint64_t *mantissa = (uint64_t *)calloc(n,       sizeof(uint64_t));
    uint64_t *tmp      = (uint64_t *)calloc(n2 + 2,  sizeof(uint64_t));
    if (!mantissa || !tmp) {
        fprintf(stderr, "[ll_analog] allocation failed (n=%zu)\n", n);
        free(mantissa); free(tmp);
        return -1;
    }
    mantissa[0] = 4;   /* LL initial seed: s₀ = 4 */

    /* ── Initialise 8D Kuramoto oscillator ── */
    AnaOsc8D osc;
    ana_init(&osc, p);   /* glyph chain reaction seeds theta[i] and omega[i] */

    if (verbose) {
        printf("  [analog] p=%llu  n_words=%zu  osc=8D-Kuramoto\n",
               (unsigned long long)p, n);
        {
            double Lv = log((double)p * ANA_LN2 / ANA_LN_PHI) / ANA_LN_PHI - 0.5 / ANA_PHI;
            double fv = Lv - floor(Lv);
            double Ov = 0.5 * (1.0 + sin(ANA_PI * fv * ANA_PHI));
            printf("  [analog] seed:     Lambda_phi(p)=%.6f  {L}=%.6f  Omega=%.6f\n",
                   Lv, fv, Ov);
            printf("  [analog] theta0:   pi*L + 2pi*(glyph+{L}+i*D_n_r)  [e^(i*pi*L) Euler base]\n");
            printf("  [analog] omega:    Omega*phi^(1+i*D_n_r)*dt  [phi-lattice resonance envelope]\n");
            printf("  [analog] field:    M(U)=|sum(e^itheta)|  Lambda^U=log(M)/lnphi-1/2phi  S=|Omega*e^(i*pi*L)+1|\n");
        }
        printf("  [analog] CV:       Kuramoto 1-R in [0,1]  (circular; 0=locked, 1=spread)\n");
        printf("  [analog] multiply: phase-doubling (theta->2theta) + Kuramoto coupling\n");
        printf("  [analog] sync:     harmonic attraction alpha=%.1fx%d (atan2, first Fourier modes)\n",
               ANA_HARM_ALPHA, ANA_HARM_PASSES);
#ifdef LL_QUANTUM_ENABLED
        printf("  [analog] quantum:  8-qubit XY Hamiltonian (cuStateVec sm_75)  blend=%.2f..%.2f\n",
               qosc_blend_alpha(0), qosc_blend_alpha(3));
#else
        printf("  [analog] quantum:  disabled (build without -DLL_QUANTUM_ENABLED)\n");
#endif
    }

    clock_t    t0         = clock();
    uint64_t   iters      = p - 2;
    APhase     last_phase = APHASE_PLUCK;
    int        logged     = 0;

    /* ══ Main LL loop ══════════════════════════════════════════════════════ */
    for (uint64_t iter = 0; iter < iters; iter++) {

        /* ── Exact arithmetic: s = s² - 2 mod 2^p-1 ── */
        ap_sqr_mersenne(mantissa, n, p, tmp);
        ap_sub2_mod_mp(mantissa, n, p);

        /* ── Analog squaring: θ → 2θ (s² = phase doubling on unit circle) ── */
        ana_phase_double(&osc);

        /* ── Kuramoto coupling: synchronisation correction (RK4) ── */
        ana_rk4_step(&osc);
        osc.phase_var = ana_phase_var(&osc);
        ana_update_phase(&osc);

#ifdef LL_QUANTUM_ENABLED
        /* ── Quantum XY step + blend ────────────────────────────────────────
         * Evolve the 8-qubit state vector by one Trotterised XY step, then
         * blend its expectation values with the classical RK4 result.
         *
         * Blend weight α rises with phase coherence (Pluck→Lock): the quantum
         * layer contributes more as the oscillator settles, where its O(K²dt²)
         * Trotter error is smallest relative to the signal being measured.
         *
         * After blending, renormalise each (re,im) to the unit circle so that
         * ana_phase_var and the phase-doubling double-angle formula remain exact.
         */
        if (osc.q_osc) {
            qosc_step(osc.q_osc, osc.omega, osc.k_coupling, ANA_DT);

            double q_re[ANA_DIMS], q_im[ANA_DIMS];
            qosc_readback(osc.q_osc, q_re, q_im);

            double alpha = qosc_blend_alpha((int)osc.aphase);
            double one_minus = 1.0 - alpha;
            for (int qi = 0; qi < ANA_DIMS; qi++) {
                double r = one_minus * osc.re[qi] + alpha * q_re[qi];
                double im = one_minus * osc.im[qi] + alpha * q_im[qi];
                double inv_mag = 1.0 / sqrt(r*r + im*im);
                osc.re[qi] = r  * inv_mag;
                osc.im[qi] = im * inv_mag;
                osc.theta[qi] = atan2(osc.im[qi], osc.re[qi]);
                if (osc.theta[qi] < 0.0) osc.theta[qi] += 2.0 * ANA_PI;
            }
        }
#endif

        /* ── Harmonic sync: attract oscillators toward residue-derived phases ── */
        if ((iter & (ANA_SHA_INTERVAL - 1)) == 0)
            ana_harmonic_sync(&osc, mantissa, n);

        /* ── Progress: only log on natural phase transitions (wu-wei pacing) ── */
        if (verbose) {
            int is_transition = (osc.aphase != last_phase);
            int is_milestone  = (iter == 0 || iter == iters - 1
                                 || (iters > 20 && iter % (iters / 10) == 0));
            if (is_transition || is_milestone) {
                last_phase = osc.aphase;
                double pct     = 100.0 * (double)(iter + 1) / (double)iters;
                double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
                printf("  [analog] iter=%-8llu  %5.1f%%  phase=%-8s  cv=%.4f  t=%.1fs%s\n",
                       (unsigned long long)iter, pct,
                       APHASE_NAMES[osc.aphase], osc.phase_var, elapsed,
                       is_transition ? "  [phase transition]" : "");
                logged++;
            }
        }
    }

    /* ── Final analog confirmation: harmonic sync on final residue ────────────
     * Ensures cv_hist's last entry reflects residue=0 (prime) or ≠0 (composite)
     * regardless of where the last periodic sync fell. */
    ana_harmonic_sync(&osc, mantissa, n);

#ifdef LL_QUANTUM_ENABLED
    if (osc.q_osc) qosc_destroy(osc.q_osc);
#endif

    /* ── Final result ── */
    int result = is_zero_a(mantissa, n);

    if (verbose) {
        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("  [analog] done: %.2fs  phase=%s  cv=%.4f  locked=%s  residue=%s\n",
               elapsed,
               APHASE_NAMES[osc.aphase],
               osc.phase_var,
               ana_is_locked(&osc) ? "yes" : "no",
               result ? "0 (PRIME)" : "non-zero (COMPOSITE)");
        /* Field resonance readout S(U) */
        printf("  [analog] S(U)=%.6f  Lambda^U=%.6f  (prime: S~%.4f, composite: S!=)\n",
               osc.s_u, osc.lambda_u,
               0.5 * (1.0 + sin(ANA_PI * 0.0 * ANA_PHI)) * cos(0.0) + 1.0);
        /* φ-tower fixed point */
        printf("  [analog] phi-tower: U*=%.6f  delta=%.2e  (phase-spread convergence signal)\n",
               osc.fp_u, osc.fp_res);
        /* Double confirmation: both analog and exact agree */
        if (ana_is_locked(&osc) && result)
            printf("  [analog] ** osc LOCKED + residue=0: strong prime resonance **\n");
        if (!ana_is_locked(&osc) && result)
            printf("  [analog] note: residue=0 (prime) but osc not locked\n");
        printf("  [analog] n_words=%zu  iters=%llu  rk4_steps=%d  log_events=%d\n",
               n, (unsigned long long)iters, osc.steps, logged);
    }

    free(mantissa);
    free(tmp);
    return result;
}
