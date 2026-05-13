// hdgl_router.c - HDGL-28 v2.0: spiral8 double-strand phi-tau routing
// Replaces XOR-based routing with:
//   - phi-tau path encoding (from hdgl_fileswap.py _phi_tau)
//   - Spiral8 double-strand counter-rotation (spiral3.py golden angle)
//   - Primary strand (positive rotation) + mirror strand (negative rotation)
//   - Echo scale factor for mirror: SPIRAL8_ECHO_SCALE = 0.8
//   - Alpha-aware expert selection: stable strands (alpha < 0) -> sticky routing
//   - Phase-coupled feedback: history encodes primary + mirror phase accumulators
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "hdgl_router.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// -- Module state --------------------------------------------------------------
static HDGLLattice *s_lattice     = NULL;
static int          s_num_experts = 0;

void hdgl_router_init(HDGLLattice *lat, int num_experts) {
    s_lattice     = lat;
    s_num_experts = num_experts;
}

// -- phi-tau encoding (hdgl_fileswap.py _phi_tau) -----------------------------
// Maps arbitrary text -> continuous tau via phi-weighted depth accumulation.
// Deterministic, no hash table, no lookup - geometry decides.
static double phi_tau(const char *text) {
    if (!text) return 0.0;
    // Split on ':' to get depth-separated segments (token key is "layer:pos")
    double tau = 0.0;
    int depth = 0;
    const char *p = text;
    while (*p) {
        // Accumulate intra-segment character sum for this depth
        double intra = 0.0;
        int seg_len = 0;
        while (*p && *p != ':') { intra += (double)(unsigned char)*p; p++; seg_len++; }
        if (seg_len > 0) {
            intra = fmod(intra, 1000.0) / 1000.0;
            tau += pow(BLZ_PHI, (double)depth) * ((double)depth + intra);
            depth++;
        }
        if (*p == ':') p++;
    }
    return tau;
}

// -- Spiral8 strand selection --------------------------------------------------
// tau -> strand index [0-7], then primary/mirror angles via golden ratio rotation
static int strand_for_tau(double tau) {
    return (int)fmin(tau, (double)(SPIRAL8_GEOMETRIES - 1));
}

// Primary strand angle: +i x golden_angle (deg -> rad)
static double primary_angle(int strand_idx) {
    return (double)strand_idx * SPIRAL8_GOLDEN_DEG * (M_PI / 180.0);
}

// Mirror strand angle: -i x golden_angle (counter-rotating)
static double mirror_angle(int strand_idx) {
    return -(double)strand_idx * SPIRAL8_GOLDEN_DEG * (M_PI / 180.0);
}

// Alpha-weighted spiral radius: r = exp(alpha * tau_frac) per spiral3.py
static double spiral_radius(int strand_idx, double tau_frac) {
    return exp(SPIRAL8_TABLE[strand_idx].alpha * fmod(tau_frac, SPIRAL8_PERIOD));
}

// -- Project slot via spiral geometry -> lattice coordinate --------------------
static int spiral_project(const char *text, HDGL_History *H, int use_mirror) {
    if (!s_lattice) return 0;
    int total = s_lattice->num_instances * s_lattice->slots_per_instance;
    if (total <= 0) return 0;

    double tau       = phi_tau(text);
    int    strand    = strand_for_tau(tau);
    double tau_frac  = tau - (double)strand;
    double r         = spiral_radius(strand, tau_frac);

    // Advance phase accumulator along primary or mirror strand
    double angle;
    if (!use_mirror) {
        H->primary_phase += primary_angle(strand) + r * 0.01;
        angle = H->primary_phase;
    } else {
        H->mirror_phase += mirror_angle(strand) + r * 0.01 * SPIRAL8_ECHO_SCALE;
        angle = H->mirror_phase;
    }

    H->strand_idx = strand;

    // Project: combine radius and phase into lattice coordinate
    double coord = r * fabs(cos(angle)) * (double)total;
    return (int)(fmod(fabs(coord), (double)total));
}

// -- Slot state -> feedback signal (replaces XOR) -----------------------------
static uint32_t spiral_feedback(int lattice_id, int strand_idx) {
    HDGL_SlotState *state = hdgl_get_slot_state(lattice_id);
    if (!state) return 0;

    // Alpha-weighted combination: stable strands (alpha < 0) contribute more
    double alpha = SPIRAL8_TABLE[strand_idx % SPIRAL8_GEOMETRIES].alpha;
    double stability = (alpha < 0) ? 2.0 : 1.0;

    uint32_t raw = state->charge ^ state->entropy ^ state->tension;
    int rot = (int)(stability * 7.0) % 32;
    return (raw << rot) | (raw >> (32 - rot));
}

// -- Main routing function -----------------------------------------------------
int route_token_recursive(Token t, HDGL_History *H) {
    if (s_num_experts <= 0) return 0;

    // Primary strand projection
    int primary_lattice = spiral_project(t.text, H, 0);
    uint32_t primary_fb = spiral_feedback(primary_lattice, H->strand_idx);

    // Mirror strand projection (counter-rotating, echo scale)
    int mirror_lattice  = spiral_project(t.text, H, 1);
    uint32_t mirror_fb  = spiral_feedback(mirror_lattice, H->strand_idx);

    // Combine primary + echo mirror (echo weighted by SPIRAL8_ECHO_SCALE)
    uint32_t combined = primary_fb
                      + (uint32_t)((double)mirror_fb * SPIRAL8_ECHO_SCALE)
                      + H->last_feedback;

    int expert_id = (int)((primary_lattice + (int)(combined % (uint32_t)s_num_experts))
                          % s_num_experts);
    if (expert_id < 0) expert_id += s_num_experts;

    H->last_feedback  = primary_fb;
    H->last_expert_id = expert_id;

    return expert_id;
}

// -- Cold-start primary-strand slot projection (for hdgl_corpus_seeder) ------
// Equivalent to spiral_project(text, &H_zero, 0) where H_zero = {0}.
// Does not require a live s_lattice — pure geometry.
int hdgl_router_key_to_slot(const char *text, int total_slots) {
    if (!text || total_slots <= 0) return 0;
    double tau      = phi_tau(text);
    int    strand   = strand_for_tau(tau);
    double tau_frac = tau - (double)strand;
    double r        = spiral_radius(strand, tau_frac);
    // Cold-start: primary_phase starts at 0, accumulates primary_angle + r*0.01
    double angle    = primary_angle(strand) + r * 0.01;
    double coord    = r * fabs(cos(angle)) * (double)total_slots;
    int    idx      = (int)(fmod(fabs(coord), (double)total_slots));
    return (idx < 0) ? 0 : idx;
}

// -- Batch routing -------------------------------------------------------------
void route_tokens_recursive(Token *tokens, int *expert_ids, int batch_size) {
    HDGL_History H = {0, 0, 0.0, 0.0, 0};
    for (int i = 0; i < batch_size; ++i)
        expert_ids[i] = route_token_recursive(tokens[i], &H);
}

// -- Shader utility ------------------------------------------------------------
HDGL_ShaderDims hdgl_get_packed_dims(int in_dim, int out_dim) {
    HDGL_ShaderDims d;
    d.in_dim = (uint32_t)in_dim;
    d.out_dim = (uint32_t)out_dim;
    return d;
}
