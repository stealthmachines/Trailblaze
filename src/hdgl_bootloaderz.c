// hdgl_bootloaderz.c - HDGL-28 v2.0
// BootloaderZ V6.0 APA lattice + spiral8 geometry + RK4 Kuramoto dynamics
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "hdgl_bootloaderz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PHI           BLZ_PHI
#define MAX_INSTANCES BLZ_MAX_INSTANCES
#define CHUNK_SIZE    BLZ_CHUNK_SIZE
#define MSB_MASK      BLZ_MSB_MASK
#define MPI_INITIAL_WORDS 1

// -- Spiral8 geometry table (spiral3.py) --------------------------------------
const Spiral8Geometry SPIRAL8_TABLE[SPIRAL8_GEOMETRIES] = {
    {1, 'C',  0.015269,  1, "Point"},
    {2, 'D',  0.008262,  2, "Line"},
    {3, 'E',  0.110649,  3, "Triangle"},
    {4, 'F', -0.083485,  4, "Tetrahedron"},
    {5, 'G',  0.025847,  5, "Pentachoron"},
    {6, 'A', -0.045123, 12, "Hexacross"},
    {7, 'B',  0.067891, 14, "Heptacube"},
    {8, 'C',  0.012345, 16, "Octacube"},
};

static const float fib_table[]  = {1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const float prime_table[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
static const int   fib_len   = 16;
static const int   prime_len = 16;

/* Continuous Binet Fibonacci: fib_real(n) = (φ^n - φ^{-n}·cos(πn)) / √5
 * Matches integer Fibonacci at n∈ℤ, interpolates smoothly for fractional n.
 * URF §1: required for sub-10^{-12}% domain fits vs discrete table lookup. */
static inline double fib_real(double n) {
    static const double SQRT5 = 2.2360679774997896;
    double phi_inv = 1.0 / PHI;
    return (pow(PHI, n) - pow(phi_inv, n) * cos(M_PI * n)) / SQRT5;
}

static double get_normalized_rand(void) { return (double)rand() / RAND_MAX; }
#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | (uint64_t)rand())

static Slot4096 APA_CONST_PHI;
static Slot4096 APA_CONST_PI;
static int       s_apa_constants_initialized = 0;

HDGLLattice *g_hdgl_lattice = NULL;

/* URF §1 fractional β — settable before seeding, default h-domain 0.1 */
static double g_prismatic_beta = 0.1;
void   lattice_set_beta(double beta) { g_prismatic_beta = beta; }
double lattice_get_beta(void)        { return g_prismatic_beta; }

// -- Forward declarations ------------------------------------------------------
static void ap_normalize_legacy(Slot4096 *slot);
static void ap_add_legacy(Slot4096 *A, const Slot4096 *B);
static void ap_shift_right_legacy(uint64_t *mw, size_t nw, int64_t shift);

// -- MPI - full implementation -------------------------------------------------
void mpi_init(MPI *m, size_t n) {
    m->words = calloc(n, sizeof(uint64_t));
    m->num_words = n;
    m->sign = 0;
}

void mpi_free(MPI *m) {
    if (m->words) free(m->words);
    m->words = NULL;
    m->num_words = 0;
}

void mpi_copy(MPI *dest, const MPI *src) {
    mpi_free(dest);
    dest->num_words = src->num_words;
    dest->words = malloc(src->num_words * sizeof(uint64_t));
    if (src->words && dest->words)
        memcpy(dest->words, src->words, src->num_words * sizeof(uint64_t));
    dest->sign = src->sign;
}

void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) {
    if (m->words) m->words[0] = value;
    m->sign = sign;
}

void mpi_resize(MPI *m, size_t new_words) {
    if (new_words == m->num_words) return;
    uint64_t *nw = calloc(new_words, sizeof(uint64_t));
    if (!nw) return;
    size_t copy_n = new_words < m->num_words ? new_words : m->num_words;
    if (m->words) memcpy(nw, m->words, copy_n * sizeof(uint64_t));
    free(m->words);
    m->words = nw;
    m->num_words = new_words;
}

int mpi_compare(const MPI *A, const MPI *B) {
    if (A->sign != B->sign) return A->sign ? -1 : 1;
    size_t n = A->num_words > B->num_words ? A->num_words : B->num_words;
    for (size_t i = n; i-- > 0; ) {
        uint64_t wa = (i < A->num_words && A->words) ? A->words[i] : 0;
        uint64_t wb = (i < B->num_words && B->words) ? B->words[i] : 0;
        if (wa != wb) return (wa > wb) ? (A->sign ? -1 : 1) : (A->sign ? 1 : -1);
    }
    return 0;
}

void mpi_add(MPI *A, const MPI *B) {
    if (A->sign == B->sign) {
        size_t n = A->num_words > B->num_words ? A->num_words : B->num_words;
        mpi_resize(A, n + 1);
        uint64_t carry = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t bw = (i < B->num_words && B->words) ? B->words[i] : 0;
            uint64_t s  = A->words[i] + bw + carry;
            carry = (s < A->words[i] || (s == A->words[i] && carry)) ? 1 : 0;
            A->words[i] = s;
        }
        A->words[n] = carry;
    } else {
        MPI tmp;
        mpi_init(&tmp, B->num_words);
        mpi_copy(&tmp, B);
        tmp.sign = A->sign;
        if (mpi_compare(A, &tmp) >= 0) {
            uint64_t borrow = 0;
            for (size_t i = 0; i < A->num_words; i++) {
                uint64_t bw = (i < tmp.num_words && tmp.words) ? tmp.words[i] : 0;
                uint64_t d  = A->words[i] - bw - borrow;
                borrow = (A->words[i] < bw + borrow) ? 1 : 0;
                A->words[i] = d;
            }
        } else {
            mpi_resize(&tmp, tmp.num_words + 1);
            uint64_t borrow = 0;
            for (size_t i = 0; i < tmp.num_words; i++) {
                uint64_t aw = (i < A->num_words && A->words) ? A->words[i] : 0;
                uint64_t d  = tmp.words[i] - aw - borrow;
                borrow = (tmp.words[i] < aw + borrow) ? 1 : 0;
                tmp.words[i] = d;
            }
            A->sign = B->sign;
            mpi_free(A);
            *A = tmp;
            tmp.words = NULL;
            tmp.num_words = 0;
            mpi_free(&tmp);
            return;
        }
        mpi_free(&tmp);
    }
}

void mpi_subtract(MPI *A, const MPI *B) {
    MPI tmp;
    mpi_init(&tmp, B->num_words);
    mpi_copy(&tmp, B);
    tmp.sign ^= 1;
    mpi_add(A, &tmp);
    mpi_free(&tmp);
}

size_t mpi_get_effective_words(const MPI *m) {
    if (!m->words) return 0;
    size_t n = m->num_words;
    while (n > 1 && m->words[n-1] == 0) n--;
    return n;
}

int mpi_count_leading_zeros(const MPI *m) {
    if (!m->words || m->num_words == 0) return 64;
    uint64_t top = m->words[m->num_words - 1];
    if (top == 0) return 64;
    int z = 0;
    while (!(top & (1ULL << 63))) { top <<= 1; z++; }
    return z;
}

// -- Slot4096 -----------------------------------------------------------------
Slot4096 slot_init_apa(int bits_mant, int bits_exp) {
    Slot4096 slot = {0};
    slot.bits_mant = bits_mant;
    slot.bits_exp  = bits_exp;
    slot.num_words = (size_t)((bits_mant + 63) / 64);
    slot.mantissa_words = calloc(slot.num_words, sizeof(uint64_t));
    mpi_init(&slot.exponent_mpi,       MPI_INITIAL_WORDS);
    mpi_init(&slot.num_words_mantissa, MPI_INITIAL_WORDS);
    mpi_init(&slot.source_of_infinity, MPI_INITIAL_WORDS);
    if (!slot.mantissa_words) { fprintf(stderr, "Error: mantissa alloc.\n"); return slot; }
    if (slot.num_words > 0) {
        slot.mantissa_words[0] = GET_RANDOM_UINT64() | MSB_MASK;
    }
    slot.exponent = (int64_t)(rand() % 17) - 8;
    slot.base     = (float)(PHI + get_normalized_rand() * 0.01);
    slot.exponent_base = 4096;
    slot.phase     = get_normalized_rand() * 2.0 * M_PI;
    slot.phase_vel = 0.0;
    slot.freq      = 1.0 + 0.5 * get_normalized_rand();
    slot.amp_im    = 0.1 * get_normalized_rand();
    mpi_set_value(&slot.exponent_mpi,       (uint64_t)llabs(slot.exponent), slot.exponent < 0 ? 1 : 0);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words, 0);
    return slot;
}

void ap_free(Slot4096 *slot) {
    if (!slot) return;
    if (slot->mantissa_words) { free(slot->mantissa_words); slot->mantissa_words = NULL; }
    mpi_free(&slot->exponent_mpi);
    mpi_free(&slot->num_words_mantissa);
    mpi_free(&slot->source_of_infinity);
    slot->num_words = 0;
}

void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);
    *dest = *src;
    dest->exponent_mpi.words       = NULL; dest->exponent_mpi.num_words       = 0;
    dest->num_words_mantissa.words = NULL; dest->num_words_mantissa.num_words = 0;
    dest->source_of_infinity.words = NULL; dest->source_of_infinity.num_words = 0;
    dest->mantissa_words = malloc(src->num_words * sizeof(uint64_t));
    if (!dest->mantissa_words) { fprintf(stderr, "Error: ap_copy alloc.\n"); dest->num_words = 0; return; }
    memcpy(dest->mantissa_words, src->mantissa_words, src->num_words * sizeof(uint64_t));
    mpi_copy(&dest->exponent_mpi,       &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);
}

double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words == 0 || !slot->mantissa_words) return 0.0;
    return ((double)slot->mantissa_words[0] / (double)UINT64_MAX)
           * pow(2.0, (double)slot->exponent);
}

Slot4096 *ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 temp = slot_init_apa(bits_mant, bits_exp);
    Slot4096 *slot = malloc(sizeof(Slot4096));
    if (!slot) { ap_free(&temp); return NULL; }
    *slot = temp;
    if (value == 0.0) return slot;
    int exp_offset;
    double mant_val = frexp(value, &exp_offset);
    slot->mantissa_words[0] = (uint64_t)(mant_val * (double)UINT64_MAX);
    slot->exponent = (int64_t)exp_offset;
    mpi_update_from_legacy(&slot->exponent_mpi, slot->exponent);
    return slot;
}

static void ap_shift_right_legacy(uint64_t *mw, size_t nw, int64_t shift) {
    if (shift <= 0 || nw == 0) return;
    if (shift >= (int64_t)nw * 64) { memset(mw, 0, nw * sizeof(uint64_t)); return; }
    int64_t ws = shift / 64;
    int bs = (int)(shift % 64);
    if (ws > 0) {
        for (size_t i = nw; i-- > (size_t)ws; ) mw[i] = mw[i - ws];
        memset(mw, 0, (size_t)ws * sizeof(uint64_t));
    }
    if (bs > 0) {
        int rs = 64 - bs;
        for (size_t i = nw; i-- > 0; ) {
            uint64_t carry = (i > 0) ? (mw[i-1] << rs) : 0;
            mw[i] = (mw[i] >> bs) | carry;
        }
    }
}

void ap_shift_right(uint64_t *mw, size_t nw, int64_t shift) { ap_shift_right_legacy(mw, nw, shift); }

static void ap_normalize_legacy(Slot4096 *slot) {
    if (slot->num_words == 0) return;
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent <= -(1LL << (slot->bits_exp - 1))) { slot->state_flags |= APA_FLAG_GUZ; break; }
        uint64_t carry = 0;
        for (size_t i = slot->num_words; i-- > 0; ) {
            uint64_t nc = (slot->mantissa_words[i] & MSB_MASK) ? 1 : 0;
            slot->mantissa_words[i] = (slot->mantissa_words[i] << 1) | carry;
            carry = nc;
        }
        slot->exponent--;
    }
    if (slot->mantissa_words[0] == 0) slot->exponent = 0;
}

void ap_normalize(Slot4096 *slot) { ap_normalize_legacy(slot); }

static void ap_add_legacy(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) {
        fprintf(stderr, "Error: APA add word count mismatch.\n");
        return;
    }
    Slot4096 B_aligned = {0};
    ap_copy(&B_aligned, B);
    int64_t ed = A->exponent - B_aligned.exponent;
    if      (ed > 0) { ap_shift_right(B_aligned.mantissa_words, B_aligned.num_words, ed);  B_aligned.exponent = A->exponent; }
    else if (ed < 0) { ap_shift_right(A->mantissa_words, A->num_words, -ed); A->exponent = B_aligned.exponent; }
    uint64_t carry = 0;
    size_t nw = A->num_words;
    for (size_t i = nw; i-- > 0; ) {
        uint64_t s = A->mantissa_words[i] + B_aligned.mantissa_words[i] + carry;
        carry = (s < A->mantissa_words[i] || (s == A->mantissa_words[i] && carry)) ? 1 : 0;
        A->mantissa_words[i] = s;
    }
    if (carry) {
        if (A->exponent >= (1LL << (A->bits_exp - 1))) { A->state_flags |= APA_FLAG_GOI; }
        else { A->exponent++; ap_shift_right(A->mantissa_words, nw, 1); A->mantissa_words[0] |= MSB_MASK; }
    }
    ap_normalize(A);
    mpi_update_from_legacy(&A->exponent_mpi, A->exponent);
    ap_free(&B_aligned);
}

void ap_add(Slot4096 *A, const Slot4096 *B) { ap_add_legacy(A, B); }

// -- RK4 Kuramoto phase coupling ----------------------------------------------
typedef struct { double A_re, A_im, phase, phase_vel; } ComplexState;

static ComplexState compute_derivatives(ComplexState state, double omega,
                                         const AnalogLink *nb, int n_nb) {
    ComplexState d = {0};
    d.A_re = -HDGL_GAMMA * state.A_re;
    d.A_im = -HDGL_GAMMA * state.A_im;
    double sum_sin = 0.0;
    for (int k = 0; k < n_nb; k++) {
        double dp = nb[k].potential - state.phase;
        sum_sin  += sin(dp);
        d.A_re   += HDGL_K_COUPLING * nb[k].coupling * cos(dp);
        d.A_im   += HDGL_K_COUPLING * nb[k].coupling * sin(dp);
    }
    d.phase_vel = omega + HDGL_K_COUPLING * sum_sin;
    d.phase     = state.phase_vel;
    return d;
}

void rk4_step(Slot4096 *slot, double dt, const AnalogLink *nb, int n_nb) {
    ComplexState st = { ap_to_double(slot), slot->amp_im, slot->phase, slot->phase_vel };
    ComplexState k1 = compute_derivatives(st, slot->freq, nb, n_nb);
    ComplexState tmp;
    tmp = st; tmp.A_re += dt/2*k1.A_re; tmp.A_im += dt/2*k1.A_im;
    tmp.phase += dt/2*k1.phase; tmp.phase_vel += dt/2*k1.phase_vel;
    ComplexState k2 = compute_derivatives(tmp, slot->freq, nb, n_nb);
    tmp = st; tmp.A_re += dt/2*k2.A_re; tmp.A_im += dt/2*k2.A_im;
    tmp.phase += dt/2*k2.phase; tmp.phase_vel += dt/2*k2.phase_vel;
    ComplexState k3 = compute_derivatives(tmp, slot->freq, nb, n_nb);
    tmp = st; tmp.A_re += dt*k3.A_re; tmp.A_im += dt*k3.A_im;
    tmp.phase += dt*k3.phase; tmp.phase_vel += dt*k3.phase_vel;
    ComplexState k4 = compute_derivatives(tmp, slot->freq, nb, n_nb);

    st.A_re += dt/6*(k1.A_re + 2*k2.A_re + 2*k3.A_re + k4.A_re);
    st.A_im += dt/6*(k1.A_im + 2*k2.A_im + 2*k3.A_im + k4.A_im);
    st.phase += dt/6*(k1.phase + 2*k2.phase + 2*k3.phase + k4.phase);
    st.phase_vel += dt/6*(k1.phase_vel + 2*k2.phase_vel + 2*k3.phase_vel + k4.phase_vel);

    double A = sqrt(st.A_re*st.A_re + st.A_im*st.A_im);
    A *= exp(-HDGL_LAMBDA * dt);
    if (A > HDGL_SAT_LIMIT) A = HDGL_SAT_LIMIT;
    A += HDGL_NOISE_SIGMA * (2.0*get_normalized_rand() - 1.0);
    double norm = sqrt(st.A_re*st.A_re + st.A_im*st.A_im);
    if (norm > 1e-10) { st.A_re = st.A_re/norm*A; st.A_im = st.A_im/norm*A; }
    st.phase = fmod(st.phase, 2.0*M_PI);
    if (st.phase < 0) st.phase += 2.0*M_PI;

    Slot4096 *na = ap_from_double(st.A_re, slot->bits_mant, slot->bits_exp);
    if (na) { ap_copy(slot, na); ap_free(na); free(na); }
    slot->amp_im    = st.A_im;
    slot->phase     = st.phase;
    slot->phase_vel = st.phase_vel;
}

// -- HDGL Lattice --------------------------------------------------------------
HDGLLattice *lattice_init(int num_instances, int slots_per_instance) {
    HDGLLattice *lat = malloc(sizeof(HDGLLattice));
    if (!lat) return NULL;
    lat->num_instances = num_instances;
    lat->slots_per_instance = slots_per_instance;
    lat->omega = lat->time = lat->phase_var = 0.0;
    lat->consensus_steps = 0;
    int total = num_instances * slots_per_instance;
    lat->num_chunks = (total + CHUNK_SIZE - 1) / CHUNK_SIZE;
    lat->chunks = calloc((size_t)lat->num_chunks, sizeof(HDGLChunk *));
    if (!lat->chunks) { free(lat); return NULL; }
    return lat;
}

HDGLChunk *lattice_get_chunk(HDGLLattice *lat, int idx) {
    if (idx >= lat->num_chunks) return NULL;
    if (!lat->chunks[idx]) {
        HDGLChunk *ch = malloc(sizeof(HDGLChunk));
        if (!ch) return NULL;
        ch->allocated = CHUNK_SIZE;
        ch->slots = calloc(CHUNK_SIZE, sizeof(Slot4096));
        if (!ch->slots) { free(ch); return NULL; }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int bm = 4096 + (i % 8) * 64;
            int be = 16 + (i % 8) * 2;
            ch->slots[i] = slot_init_apa(bm, be);
        }
        lat->chunks[idx] = ch;
    }
    return lat->chunks[idx];
}

Slot4096 *lattice_get_slot(HDGLLattice *lat, int idx) {
    if (!lat || idx < 0) return NULL;
    int total = lat->num_instances * lat->slots_per_instance;
    if (idx >= total) return NULL;
    HDGLChunk *ch = lattice_get_chunk(lat, idx / CHUNK_SIZE);
    return ch ? &ch->slots[idx % CHUNK_SIZE] : NULL;
}

double prismatic_recursion(HDGLLattice *lat, int idx, double val) {
    /* URF §1: continuous slot coordinate n_cont = (idx%16) + β.
     * β is settable via lattice_set_beta(); default 0.1 (h-domain fit). */
    double n_cont = (double)(idx % fib_len) + g_prismatic_beta;
    double ph = pow(PHI, (double)(idx % 16));
    double fh = fib_real(n_cont);                       /* continuous Binet Fibonacci */
    double dy = pow(10000.0, n_cont / (double)fib_len); /* URF base b=10000 vs dyadic 2^i */
    double pr = prime_table[idx % prime_len];
    double om = 0.5 + 0.5 * sin(lat->time + idx * 0.01);
    double rd = pow(fabs(val), (double)((idx % 7) + 1) / 8.0);
    return sqrt(ph * fh * dy * pr * om) * rd;
}

void detect_harmonic_consensus(HDGLLattice *lat) {
    int total = lat->num_instances * lat->slots_per_instance;
    double mean_phase = 0.0;
    int count = 0;
    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (s && !(s->state_flags & APA_FLAG_CONSENSUS)) { mean_phase += s->phase; count++; }
    }
    if (!count) return;
    mean_phase /= count;
    double sum_var = 0.0;
    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (s && !(s->state_flags & APA_FLAG_CONSENSUS)) {
            double d = s->phase - mean_phase;
            if (d >  M_PI) d -= 2*M_PI;
            if (d < -M_PI) d += 2*M_PI;
            sum_var += d*d;
        }
    }
    lat->phase_var = sqrt(sum_var / count);
    if (lat->phase_var < HDGL_CONSENSUS_EPS) {
        if (++lat->consensus_steps >= HDGL_CONSENSUS_N) {
            for (int i = 0; i < total; i++) {
                Slot4096 *s = lattice_get_slot(lat, i);
                if (s && !(s->state_flags & APA_FLAG_CONSENSUS)) {
                    s->state_flags |= APA_FLAG_CONSENSUS;
                    s->phase_vel = 0.0;
                }
            }
            lat->consensus_steps = 0;
        }
    } else {
        lat->consensus_steps = 0;
    }
}

void lattice_integrate_rk4(HDGLLattice *lat, double dt) {
    int total = lat->num_instances * lat->slots_per_instance;
    for (int i = 0; i < total; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (!slot || (slot->state_flags & (APA_FLAG_GOI | APA_FLAG_IS_NAN | APA_FLAG_CONSENSUS)))
            continue;

        AnalogLink nb[8] = {0};
        int ni[8] = {
            (i-1+total)%total, (i+1)%total,
            (i-lat->slots_per_instance+total)%total, (i+lat->slots_per_instance)%total,
            (i-lat->slots_per_instance-1+total)%total, (i-lat->slots_per_instance+1+total)%total,
            (i+lat->slots_per_instance-1+total)%total, (i+lat->slots_per_instance+1)%total,
        };
        for (int j = 0; j < 8; j++) {
            Slot4096 *n = lattice_get_slot(lat, ni[j]);
            if (n) {
                double nv = ap_to_double(n), sv = ap_to_double(slot);
                nb[j].charge    = nv;
                nb[j].charge_im = n->amp_im;
                nb[j].tension   = (nv - sv) / (dt > 0 ? dt : 1e-6);
                nb[j].potential = n->phase - slot->phase;
                nb[j].coupling  = HDGL_K_COUPLING * exp(-fabs(1.0 - fabs(nv) / (fabs(sv) + 1e-10)));
            }
        }
        rk4_step(slot, dt, nb, 8);
    }
    detect_harmonic_consensus(lat);
    lat->omega += 0.01 * dt;
    lat->time  += dt;
}

void lattice_step_cpu(HDGLLattice *lat, double tick) { lattice_integrate_rk4(lat, tick); }

void lattice_fold(HDGLLattice *lat) {
    int oi = lat->num_instances, ni = oi*2;
    if (ni > MAX_INSTANCES) return;
    int ot = oi * lat->slots_per_instance;
    int nc = ((ni*lat->slots_per_instance)+CHUNK_SIZE-1)/CHUNK_SIZE;
    HDGLChunk **np = realloc(lat->chunks, (size_t)nc * sizeof(HDGLChunk *));
    if (!np) return;
    lat->chunks = np;
    for (int i = lat->num_chunks; i < nc; i++) lat->chunks[i] = NULL;
    for (int i = 0; i < ot; i++) {
        Slot4096 *os = lattice_get_slot(lat, i), *ns = lattice_get_slot(lat, ot+i);
        if (os && ns) {
            ap_copy(ns, os);
            Slot4096 *p = ap_from_double(fib_real((double)(i % fib_len) + 0.1) * 0.01, ns->bits_mant, ns->bits_exp);
            if (p) { ap_add(ns, p); ap_free(p); free(p); }
            ns->base  += (float)(get_normalized_rand() * 0.001);
            ns->phase += SPIRAL8_GOLDEN_DEG * (M_PI/180.0);
            ns->phase  = fmod(ns->phase, 2.0*M_PI);
        }
    }
    lat->num_instances = ni;
    lat->num_chunks = nc;
}

void lattice_free(HDGLLattice *lat) {
    if (!lat) return;
    free_apa_constants();
    if (lat == g_hdgl_lattice) g_hdgl_lattice = NULL;
    for (int i = 0; i < lat->num_chunks; i++) {
        if (lat->chunks[i]) {
            for (size_t j = 0; j < CHUNK_SIZE; j++) ap_free(&lat->chunks[i]->slots[j]);
            free(lat->chunks[i]->slots);
            free(lat->chunks[i]);
        }
    }
    free(lat->chunks);
    free(lat);
}

// -- Slot state for router -----------------------------------------------------
HDGL_SlotState *hdgl_get_slot_state(int lattice_id) {
    static HDGL_SlotState state;
    Slot4096 *s = lattice_get_slot(g_hdgl_lattice, lattice_id);
    if (!s || !s->mantissa_words) return NULL;
    uint64_t w = s->mantissa_words[0];
    state.charge  = (uint32_t)(w & 0xFFFFFFFF);
    state.entropy = (uint32_t)(w >> 32);
    state.tension = (uint32_t)(s->exponent & 0xFFFFFFFF);
    return &state;
}

// -- Closed-loop Hebbian feedback ---------------------------------------------
#define HDGL_FEEDBACK_TICK 0.0005

void hdgl_lattice_feedback(int layer_idx __attribute__((unused)),
                           int hdgl_expert, int agreed, float moe_magnitude) {
    if (!g_hdgl_lattice || hdgl_expert < 0) return;
    int total = g_hdgl_lattice->num_instances * g_hdgl_lattice->slots_per_instance;
    if (total <= 0) return;
    int slot_idx = hdgl_expert % total;
    Slot4096 *slot = lattice_get_slot(g_hdgl_lattice, slot_idx);
    if (!slot || (slot->state_flags & (APA_FLAG_GOI | APA_FLAG_IS_NAN))) return;
    if (!isfinite(ap_to_double(slot))) return;
    float mag = (moe_magnitude > 1.0f) ? 1.0f : (moe_magnitude < 0.0f) ? 0.0f : moe_magnitude;
    if (agreed) {
        double cv = ap_to_double(slot);
        if (cv == 0.0 || !isfinite(cv)) cv = HDGL_FEEDBACK_TICK;
        double tick = cv * HDGL_FEEDBACK_TICK * (double)mag;
        Slot4096 *d = ap_from_double(tick, slot->bits_mant, slot->bits_exp);
        if (d) { ap_add(slot, d); ap_free(d); free(d); }
        slot->phase += 0.01 * (double)mag;
        slot->phase  = fmod(slot->phase, 2.0*M_PI);
    } else {
        if (slot->num_words > 0 && slot->mantissa_words) {
            ap_shift_right(slot->mantissa_words, slot->num_words, 1);
            if (slot->exponent > -(1LL << (slot->bits_exp - 1))) slot->exponent--;
            ap_normalize(slot);
        }
        slot->phase += M_PI * (double)mag;
        slot->phase  = fmod(slot->phase, 2.0*M_PI);
    }
    mpi_update_from_legacy(&slot->exponent_mpi, slot->exponent);
}

// -- Bootloader integration ----------------------------------------------------
void init_apa_constants(void) {
    APA_CONST_PHI = slot_init_apa(4096, 16);
    APA_CONST_PI  = slot_init_apa(4096, 16);
    Slot4096 *tp = ap_from_double(1.6180339887, APA_CONST_PHI.bits_mant, APA_CONST_PHI.bits_exp);
    ap_copy(&APA_CONST_PHI, tp); ap_free(tp); free(tp);
    Slot4096 *tpi = ap_from_double(3.1415926535, APA_CONST_PI.bits_mant, APA_CONST_PI.bits_exp);
    ap_copy(&APA_CONST_PI, tpi); ap_free(tpi); free(tpi);
    s_apa_constants_initialized = 1;
    printf("[Bootloader] PHI, PI initialized (v%s).\n", HDGL_VERSION_STR);
}

void free_apa_constants(void) {
    if (!s_apa_constants_initialized) return;
    ap_free(&APA_CONST_PHI);
    ap_free(&APA_CONST_PI);
    s_apa_constants_initialized = 0;
}

int hdgl_load_lattice(HDGLLattice *lat, const char *path) {
    if (!lat || !path) return 0;

    typedef struct {
        uint64_t mantissa_word0;
        int64_t  exponent;
        double   phase;
        double   freq;
        uint32_t state_flags;
        uint32_t strand_idx;
    } SlotRecord;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[HDGL-28] ERROR: cannot open lattice file: %s\n", path);
        return 0;
    }

    char magic[4] = {0};
    uint32_t version = 0;
    uint32_t num_instances = 0;
    uint32_t slots_per_instance = 0;
    double time_v = 0.0, omega_v = 0.0, phase_var_v = 0.0;

    if (fread(magic, 1, 4, f) != 4 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&num_instances, sizeof(uint32_t), 1, f) != 1 ||
        fread(&slots_per_instance, sizeof(uint32_t), 1, f) != 1 ||
        fread(&time_v, sizeof(double), 1, f) != 1 ||
        fread(&omega_v, sizeof(double), 1, f) != 1 ||
        fread(&phase_var_v, sizeof(double), 1, f) != 1) {
        fprintf(stderr, "[HDGL-28] ERROR: truncated lattice header: %s\n", path);
        fclose(f);
        return 0;
    }

    if (memcmp(magic, "HDGL", 4) != 0) {
        fprintf(stderr, "[HDGL-28] ERROR: invalid lattice magic: %s\n", path);
        fclose(f);
        return 0;
    }

    if (version != 0x00020000u) {
        fprintf(stderr, "[HDGL-28] ERROR: unsupported lattice version 0x%08x: %s\n", version, path);
        fclose(f);
        return 0;
    }

    if ((int)num_instances != lat->num_instances || (int)slots_per_instance != lat->slots_per_instance) {
        fprintf(stderr,
                "[HDGL-28] ERROR: lattice shape mismatch file=%ux%u runtime=%dx%d\n",
                num_instances, slots_per_instance,
                lat->num_instances, lat->slots_per_instance);
        fclose(f);
        return 0;
    }

    int total = lat->num_instances * lat->slots_per_instance;
    for (int i = 0; i < total; i++) {
        SlotRecord rec;
        if (fread(&rec, sizeof(rec), 1, f) != 1) {
            fprintf(stderr, "[HDGL-28] ERROR: truncated lattice slot data at %d\n", i);
            fclose(f);
            return 0;
        }

        Slot4096 *s = lattice_get_slot(lat, i);
        if (!s || !s->mantissa_words || s->num_words == 0) continue;
        s->mantissa_words[0] = rec.mantissa_word0;
        s->exponent = rec.exponent;
        s->phase = rec.phase;
        s->freq = rec.freq;
        s->state_flags = rec.state_flags;
        mpi_update_from_legacy(&s->exponent_mpi, s->exponent);
    }

    lat->time = time_v;
    lat->omega = omega_v;
    lat->phase_var = phase_var_v;
    lat->consensus_steps = 0;

    fclose(f);
    return 1;
}

void bootloader_init_lattice(HDGLLattice *lat, int steps) {
    printf("[Bootloader] HDGL lattice v%s initializing...\n", HDGL_VERSION_STR);
    if (!lat) { printf("[Bootloader] ERROR: NULL lattice.\n"); return; }
    init_apa_constants();
    printf("[Bootloader] %d instances x %d slots = %d total\n",
           lat->num_instances, lat->slots_per_instance,
           lat->num_instances * lat->slots_per_instance);
    for (int i = 0; i < steps; i++) lattice_step_cpu(lat, 0.01);
    printf("[Bootloader] Seeded %d steps. Omega=%.6f Time=%.6f PhaseVar=%.6f\n",
           steps, lat->omega, lat->time, lat->phase_var);
}
