/*
 * zchg_lattice.h - Phi-Spiral Geometry & Strand Routing
 *
 * Pure ZCHG strand architecture:
 * - 8 geometric strands (A-H): Point, Line, Triangle, Tetrahedron, etc.
 * - Phi-spiral weighting (golden ratio φ = 1.618...)
 * - EMA latency smoothing
 * - Path → Strand mapping via phi-tau hash
 */

#ifndef ZCHG_LATTICE_H
#define ZCHG_LATTICE_H

#include "zchg_core.h"

/* ============================================================================
 * EMA (Exponential Moving Average) Computation
 * ============================================================================ */

#define ZCHG_EMA_ALPHA              0.3    /* Smoothing factor */
#define ZCHG_EMA_INITIAL            50.0   /* Starting estimate (ms) */

/* Update EMA with new measurement */
double zchg_ema_update(double current_ema, double new_value);

/* ============================================================================
 * Phi-Spiral Weight Computation
 * ============================================================================ */

/* Amplify raw weight using phi-spiral function */
uint8_t zchg_compute_strand_weight(double latency_ema, double storage_available);

/* Phi-spiral function: w(x) = x^1.2 (phi-weighted amplification) */
double zchg_phi_amplify(double x);

/* ============================================================================
 * Phi-Tau Hash (Path → Strand Routing)
 * ============================================================================ */

/* Compute phi-tau hash for a path (deterministic) */
uint64_t zchg_compute_phi_tau(const char *path, size_t path_len);

/* Map phi-tau hash to strand ID (0-7) */
uint8_t zchg_phi_tau_to_strand(uint64_t phi_tau);

/* Map phi-tau hash to authority node */
uint32_t zchg_phi_tau_to_authority(zchg_lattice_t *lattice, uint64_t phi_tau);

/* ============================================================================
 * Lattice Updates (Gossip Driven)
 * ============================================================================ */

/* Update lattice with gossip from peer */
int zchg_lattice_apply_gossip(zchg_lattice_t *lattice, uint32_t peer_ip, zchg_gossip_msg_t *msg);

/* Compute full lattice state (provisioner pass) */
int zchg_lattice_recompute(zchg_lattice_t *lattice);

/* Get current authority for a strand */
uint32_t zchg_lattice_get_strand_authority(zchg_lattice_t *lattice, uint8_t strand_id);

/* Update self strand weights based on local metrics */
int zchg_lattice_update_self_metrics(zchg_lattice_t *lattice, double latency_ms, double storage_available);

/* ============================================================================
 * Cluster Fingerprint (Convergence Indicator)
 * ============================================================================ */

/* Compute 32-bit cluster fingerprint from lattice state */
uint32_t zchg_lattice_compute_fingerprint(zchg_lattice_t *lattice);

/* Hamming distance between fingerprints (0-32 bits) */
uint32_t zchg_fingerprint_hamming_distance(uint32_t fp1, uint32_t fp2);

/* ============================================================================
 * PROVISIONER Pass (EMA → SCALE → PHASESHIFT → OMEGAMULT → ENERGY → FOLD256)
 * ============================================================================ */

/* NORM: normalize latencies to [0, 1] range */
void zchg_provisioner_norm(zchg_lattice_t *lattice);

/* SCALE: phi-spiral amplification */
void zchg_provisioner_scale(zchg_lattice_t *lattice);

/* PHASESHIFT: rotate strand authority based on cycle */
void zchg_provisioner_phaseshift(zchg_lattice_t *lattice, uint64_t cycle);

/* OMEGAMULT: fibonacci-weighted stabilization */
void zchg_provisioner_omegamult(zchg_lattice_t *lattice);

/* ENERGY: compute authority energy per strand */
void zchg_provisioner_energy(zchg_lattice_t *lattice);

/* FOLD256: final fold to 32-bit fingerprint */
void zchg_provisioner_fold256(zchg_lattice_t *lattice);

/* Full provisioner pipeline */
int zchg_provisioner_run(zchg_lattice_t *lattice, uint64_t cycle);

/* ============================================================================
 * Strand Assignment (My Strands)
 * ============================================================================ */

/* Determine which strands this node is authority for */
int zchg_lattice_compute_my_strands(zchg_lattice_t *lattice, uint8_t *out_strands, uint8_t *out_count);

/* ============================================================================
 * Omega-TTL Caching Model
 * ============================================================================ */

/* Compute strand-aware TTL using alpha model */
uint32_t zchg_compute_omega_ttl(uint8_t strand_id, uint64_t cycle);

#endif /* zchg_LATTICE_H */
