/*
 * zchg_lattice.c - Phi-Spiral Geometry Implementation
 *
 * Pure ZCHG strand architecture in C
 * - EMA-based latency smoothing
 * - Phi-spiral weight computation
 * - Phi-tau deterministic path routing
 * - Provisioner pipeline (NORM → SCALE → PHASESHIFT → OMEGAMULT → ENERGY → FOLD256)
 */

#include "zchg_lattice.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define zchg_PHI_VALUE              1.618033988749894848204586834365638
#define zchg_FIBONACCI_SCALE        1.6180339887      /* For OMEGAMULT */

/* ============================================================================
 * EMA (Exponential Moving Average)
 * ============================================================================ */

double zchg_ema_update(double current_ema, double new_value) {
    if (current_ema < 0.001) {
        return new_value;  /* First measurement */
    }
    return ZCHG_EMA_ALPHA * new_value + (1.0 - ZCHG_EMA_ALPHA) * current_ema;
}

/* ============================================================================
 * Phi-Spiral Weight Computation
 * ============================================================================ */

/* Phi-spiral amplification: w(x) = x^1.2 */
double zchg_phi_amplify(double x) {
    if (x <= 0.0) return 0.0;
    return pow(x, 1.2);
}

/* Convert raw EMA/storage to strand weight (1-100) */
uint8_t zchg_compute_strand_weight(double latency_ema, double storage_available) {
    /* Lower latency = higher weight
     * Higher storage = higher weight
     * Combined using phi-spiral function */

    double latency_norm = 1.0 / (1.0 + latency_ema / 50.0);  /* Normalize latency to [0, 1] */
    double storage_norm = fmin(storage_available / (1024.0 * 1024.0 * 1024.0), 1.0);  /* Storage in GB */

    double combined = (latency_norm * 0.6) + (storage_norm * 0.4);  /* 60% latency, 40% storage */
    double amplified = zchg_phi_amplify(combined);

    uint8_t weight = (uint8_t)(amplified * 100.0);
    if (weight < 1) weight = 1;  /* Floor to 1 (every healthy node gets traffic) */
    if (weight > 100) weight = 100;

    return weight;
}

/* ============================================================================
 * Phi-Tau Hash (Path → Strand Routing)
 * ============================================================================ */

/* FNV-1a hash with phi spiral */
uint64_t zchg_compute_phi_tau(const char *path, size_t path_len) {
    uint64_t hash = 0xcbf29ce484222325ULL;  /* FNV offset basis */
    const uint64_t phi_mult = 0x100000001b3ULL;  /* FNV prime */

    for (size_t i = 0; i < path_len; i++) {
        hash ^= (uint64_t)path[i];
        hash *= phi_mult;
        /* Phi-spiral rotation: rotate hash by golden angle */
        hash = ((hash << 13) | (hash >> 51)) ^ (uint64_t)(zchg_PHI_VALUE * 1e9);
    }

    return hash;
}

/* Map phi-tau to strand (0-7) */
uint8_t zchg_phi_tau_to_strand(uint64_t phi_tau) {
    /* Take lowest 3 bits: 0-7 */
    return (uint8_t)(phi_tau & 0x07);
}

/* Map phi-tau to authority node (round-robin over healthy peers) */
uint32_t zchg_phi_tau_to_authority(zchg_lattice_t *lattice, uint64_t phi_tau) {
    uint8_t strand = zchg_phi_tau_to_strand(phi_tau);
    return zchg_lattice_get_strand_authority(lattice, strand);
}

/* ============================================================================
 * Strand Authority Computation
 * ============================================================================ */

/* Get current authority for strand (highest weight peer) */
uint32_t zchg_lattice_get_strand_authority(zchg_lattice_t *lattice, uint8_t strand_id) {
    if (strand_id >= zchg_STRAND_COUNT) return 0;

    uint32_t best_peer = 0;
    uint8_t best_weight = 0;

    for (uint32_t i = 0; i < lattice->peer_count; i++) {
        zchg_peer_t *peer = &lattice->peers[i];
        if (!peer->is_healthy) continue;

        uint8_t weight = peer->strands[strand_id].authority_weight;
        if (weight > best_weight) {
            best_weight = weight;
            best_peer = peer->ip_addr;
        }
    }

    return best_peer;
}

/* ============================================================================
 * Lattice Updates
 * ============================================================================ */

/* Apply gossip message from peer */
int zchg_lattice_apply_gossip(zchg_lattice_t *lattice, uint32_t peer_ip, zchg_gossip_msg_t *msg) {
    /* Find or create peer entry */
    zchg_peer_t *peer = NULL;
    for (uint32_t i = 0; i < lattice->peer_count; i++) {
        if (lattice->peers[i].ip_addr == peer_ip) {
            peer = &lattice->peers[i];
            break;
        }
    }

    if (!peer && lattice->peer_count < zchg_MAX_PEERS) {
        peer = &lattice->peers[lattice->peer_count++];
        peer->ip_addr = peer_ip;
        peer->port = 8090;
        peer->is_healthy = 1;
    }

    if (!peer) return -1;  /* Too many peers */

    /* Update strand weights from gossip */
    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        peer->strands[i].authority_weight = msg->strand_weights[i];
    }

    peer->cluster_fingerprint = msg->cluster_fingerprint;
    peer->last_gossip_in = time(NULL);
    peer->failed_checks = 0;

    return 0;
}

/* Update self metrics */
int zchg_lattice_update_self_metrics(zchg_lattice_t *lattice, double latency_ms, double storage_available) {
    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        lattice->my_strands[i].latency_ema = zchg_ema_update(lattice->my_strands[i].latency_ema, latency_ms);
        lattice->my_strands[i].storage_available = storage_available;
        lattice->my_strands[i].authority_weight = zchg_compute_strand_weight(latency_ms, storage_available);
        lattice->my_strands[i].last_update = time(NULL);
    }
    return 0;
}

/* ============================================================================
 * Cluster Fingerprint
 * ============================================================================ */

uint32_t zchg_lattice_compute_fingerprint(zchg_lattice_t *lattice) {
    uint32_t fp = 0xFFFF0000;  /* Initial mask */

    /* XOR in strand weights */
    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        uint8_t weight = lattice->my_strands[i].authority_weight;
        fp ^= ((uint32_t)weight << (i * 4));
    }

    /* XOR in peer fingerprints */
    for (uint32_t i = 0; i < lattice->peer_count; i++) {
        fp ^= lattice->peers[i].cluster_fingerprint;
    }

    lattice->cluster_fingerprint = fp;
    return fp;
}

uint32_t zchg_fingerprint_hamming_distance(uint32_t fp1, uint32_t fp2) {
    uint32_t xor = fp1 ^ fp2;
    uint32_t distance = 0;
    while (xor) {
        distance += xor & 1;
        xor >>= 1;
    }
    return distance;
}

/* ============================================================================
 * PROVISIONER Pipeline
 * ============================================================================ */

void zchg_provisioner_norm(zchg_lattice_t *lattice) {
    /* Normalize latencies to [0, 1] range */
    double min_latency = 999999.0, max_latency = 0.0;

    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        double lat = lattice->my_strands[i].latency_ema;
        if (lat < min_latency) min_latency = lat;
        if (lat > max_latency) max_latency = lat;
    }

    if (max_latency <= min_latency) return;

    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        double normalized = (lattice->my_strands[i].latency_ema - min_latency) / (max_latency - min_latency);
        lattice->my_strands[i].latency_ema = normalized;
    }
}

void zchg_provisioner_scale(zchg_lattice_t *lattice) {
    /* Apply phi-spiral amplification */
    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        double amplified = zchg_phi_amplify(lattice->my_strands[i].latency_ema);
        lattice->my_strands[i].authority_weight = (uint8_t)(amplified * 100.0);
    }
}

void zchg_provisioner_phaseshift(zchg_lattice_t *lattice, uint64_t cycle) {
    (void)lattice;
    /* Rotate strand authority based on cycle (round-robin effect) */
    uint8_t shift = (cycle % zchg_STRAND_COUNT);
    /* Shift can be applied here if needed for dynamic rebalancing */
    (void)shift;
}

void zchg_provisioner_omegamult(zchg_lattice_t *lattice) {
    /* Fibonacci-weighted stabilization */
    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        double fib_factor = (i % 2 == 0) ? 1.0 : zchg_FIBONACCI_SCALE;
        lattice->my_strands[i].authority_weight = (uint8_t)(lattice->my_strands[i].authority_weight * fib_factor);
    }
}

void zchg_provisioner_energy(zchg_lattice_t *lattice) {
    (void)lattice;
    /* Energy is already computed as authority_weight */
}

void zchg_provisioner_fold256(zchg_lattice_t *lattice) {
    /* Fold to 32-bit fingerprint */
    zchg_lattice_compute_fingerprint(lattice);
}

int zchg_provisioner_run(zchg_lattice_t *lattice, uint64_t cycle) {
    zchg_provisioner_norm(lattice);
    zchg_provisioner_scale(lattice);
    zchg_provisioner_phaseshift(lattice, cycle);
    zchg_provisioner_omegamult(lattice);
    zchg_provisioner_energy(lattice);
    zchg_provisioner_fold256(lattice);
    return 0;
}

/* ============================================================================
 * My Strands (Authority Assignment)
 * ============================================================================ */

int zchg_lattice_compute_my_strands(zchg_lattice_t *lattice, uint8_t *out_strands, uint8_t *out_count) {
    uint8_t count = 0;

    for (uint8_t i = 0; i < zchg_STRAND_COUNT; i++) {
        uint32_t authority = zchg_lattice_get_strand_authority(lattice, i);
        if (authority == lattice->local_ip) {
            out_strands[count++] = i;
        }
    }

    *out_count = count;
    return count > 0 ? 0 : -1;
}

/* ============================================================================
 * Omega-TTL Model
 * ============================================================================ */

uint32_t zchg_compute_omega_ttl(uint8_t strand_id, uint64_t cycle) {
    (void)cycle;
    /* TTL_k = TTL_BASE * exp(-alpha_k * SPIRAL_PERIOD)
     * Contracting strands (alpha < 0) cache longer
     * Expanding strands (alpha > 0) refresh faster */

    #define zchg_TTL_BASE 3600  /* 1 hour */

    double alpha = -0.1 + (strand_id * 0.025);  /* Vary by strand */
    double ttl = zchg_TTL_BASE * exp(-alpha * zchg_SPIRAL_PERIOD);

    return (uint32_t)ttl;
}
