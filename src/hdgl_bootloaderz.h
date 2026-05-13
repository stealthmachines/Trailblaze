// hdgl_bootloaderz.h - HDGL-28 v2.0: BootloaderZ V6.0 + spiral8 geometry + RK4 Kuramoto
// Full APA/HDGL lattice with:
//   - Controlled slot initialization (no random overflow exponents)
//   - Complete MPI arithmetic (no stubs)
//   - RK4 Kuramoto phase coupling (from hdgl_analog_v30b.c)
//   - Spiral8 double-strand phi-tau routing (replaces XOR)
//   - GOI/GUZ saturation-safe feedback
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

/* MSVC does not support __attribute__; neutralise it for portability */
#ifdef _MSC_VER
#  ifndef __attribute__
#    define __attribute__(x)
#  endif
#endif

#define HDGL_VERSION_MAJOR 2
#define HDGL_VERSION_MINOR 0
#define HDGL_VERSION_STR   "2.0.0-spiral8"

// -- APA state flags -----------------------------------------------------------
#define APA_FLAG_SIGN_NEG   (1 << 0)
#define APA_FLAG_IS_NAN     (1 << 1)
#define APA_FLAG_GOI        (1 << 2)
#define APA_FLAG_GUZ        (1 << 3)
#define APA_FLAG_CONSENSUS  (1 << 4)

// -- System constants ----------------------------------------------------------
#define BLZ_PHI             1.6180339887498948
#define BLZ_MAX_INSTANCES   8388608
#define BLZ_SLOTS_PER_INST  4
#define BLZ_CHUNK_SIZE      1048576
#define BLZ_MSB_MASK        (1ULL << 63)

// -- Spiral8: 8 polytope geometries (from spiral3.py) -------------------------
#define SPIRAL8_GEOMETRIES  8
#define SPIRAL8_PERIOD      13.057
#define SPIRAL8_GOLDEN_DEG  (360.0 / (BLZ_PHI * BLZ_PHI))
#define SPIRAL8_ECHO_SCALE  0.8

typedef struct {
    int    dim;
    char   note;
    double alpha;
    int    vertex_count;
    const char *name;
} Spiral8Geometry;

extern const Spiral8Geometry SPIRAL8_TABLE[SPIRAL8_GEOMETRIES];

// -- Analog coupling constants (hdgl_analog_v30b.c) ---------------------------
#define HDGL_GAMMA         0.02
#define HDGL_LAMBDA        0.05
#define HDGL_SAT_LIMIT     1e6
#define HDGL_NOISE_SIGMA   0.01
#define HDGL_K_COUPLING    1.0
#define HDGL_CONSENSUS_EPS 1e-6
#define HDGL_CONSENSUS_N   100

// -- MPI ----------------------------------------------------------------------
typedef struct {
    uint64_t *words;
    size_t    num_words;
    uint8_t   sign;
} MPI;

// -- Slot4096 with RK4 phase state --------------------------------------------
typedef struct {
    uint64_t  *mantissa_words;
    MPI        num_words_mantissa;
    MPI        exponent_mpi;
    uint16_t   exponent_base;
    uint32_t   state_flags;
    MPI        source_of_infinity;
    size_t     num_words;
    int64_t    exponent;
    float      base;
    int        bits_mant;
    int        bits_exp;
    // RK4 Kuramoto phase (hdgl_analog_v30b.c)
    double     phase;
    double     phase_vel;
    double     freq;
    double     amp_im;
} Slot4096;

// -- Analog neighbor link ------------------------------------------------------
typedef struct {
    double charge, charge_im, tension, potential, coupling;
} AnalogLink;

// -- HDGLChunk / HDGLLattice ---------------------------------------------------
typedef struct { Slot4096 *slots; size_t allocated; } HDGLChunk;

typedef struct {
    HDGLChunk **chunks;
    int num_chunks, num_instances, slots_per_instance;
    double omega, time;
    double phase_var;
    int    consensus_steps;
} HDGLLattice;

// -- Router types --------------------------------------------------------------
typedef struct { char *text; int id; } Token;

typedef struct {
    uint32_t last_feedback;
    int      last_expert_id;
    double   primary_phase;
    double   mirror_phase;
    int      strand_idx;
} HDGL_History;

typedef struct { uint32_t charge, entropy, tension; } HDGL_SlotState;
typedef struct { uint32_t in_dim, out_dim; } HDGL_ShaderDims;

// -- MPI full API --------------------------------------------------------------
void   mpi_init(MPI *m, size_t initial_words);
void   mpi_free(MPI *m);
void   mpi_copy(MPI *dest, const MPI *src);
void   mpi_set_value(MPI *m, uint64_t value, uint8_t sign);
void   mpi_resize(MPI *m, size_t new_words);
int    mpi_compare(const MPI *A, const MPI *B);
void   mpi_add(MPI *A, const MPI *B);
void   mpi_subtract(MPI *A, const MPI *B);
size_t mpi_get_effective_words(const MPI *m);
int    mpi_count_leading_zeros(const MPI *m);

static inline void mpi_update_from_legacy(MPI *m, int64_t e) {
    mpi_set_value(m, (uint64_t)llabs(e), e < 0 ? 1 : 0);
}

// -- APA utility API -----------------------------------------------------------
Slot4096  slot_init_apa(int bits_mant, int bits_exp);
void      ap_free(Slot4096 *slot);
void      ap_copy(Slot4096 *dest, const Slot4096 *src);
void      ap_normalize(Slot4096 *slot);
void      ap_add(Slot4096 *A, const Slot4096 *B);
Slot4096 *ap_from_double(double value, int bits_mant, int bits_exp);
double    ap_to_double(const Slot4096 *slot);
void      ap_shift_right(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount);

// -- RK4 Kuramoto evolution ----------------------------------------------------
void rk4_step(Slot4096 *slot, double dt, const AnalogLink *neighbors, int num_neigh);

// -- Lattice API ---------------------------------------------------------------
HDGLLattice *lattice_init(int num_instances, int slots_per_instance);
HDGLChunk   *lattice_get_chunk(HDGLLattice *lat, int chunk_idx);
Slot4096    *lattice_get_slot(HDGLLattice *lat, int idx);
void         lattice_integrate_rk4(HDGLLattice *lat, double dt);
void         lattice_step_cpu(HDGLLattice *lat, double tick);
void         lattice_fold(HDGLLattice *lat);
void         lattice_free(HDGLLattice *lat);
double       prismatic_recursion(HDGLLattice *lat, int idx, double val);
void         detect_harmonic_consensus(HDGLLattice *lat);

HDGL_SlotState *hdgl_get_slot_state(int lattice_id);
void            hdgl_lattice_feedback(int layer_idx __attribute__((unused)),
                                      int hdgl_expert,
                                      int agreed, float moe_magnitude);

void init_apa_constants(void);
void free_apa_constants(void);
void bootloader_init_lattice(HDGLLattice *lat, int steps);
int  hdgl_load_lattice(HDGLLattice *lat, const char *path);

/* URF §1: fractional β coordinate injected into prismatic_recursion.
 * Default 0.1 (h-domain fit).  Call lattice_set_beta() before seeding
 * to override per the URF per-domain (n, β, Ω) table. */
void   lattice_set_beta(double beta);
double lattice_get_beta(void);

extern HDGLLattice *g_hdgl_lattice;
