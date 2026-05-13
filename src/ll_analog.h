/* ll_analog.h — analog LL path: v30b Slot4096 APA + 8D Kuramoto oscillator
 *
 * Exact arithmetic:
 *   Slot4096.mantissa_words  — p-bit LL residue, arbitrary precision
 *   ap_sqr_mersenne          — schoolbook O(n^2) squaring + Mersenne fold
 *   ap_sub2_mod_mp           — subtract 2 mod 2^p-1
 *
 * Analog oscillator (after analog_engine.h / AnalogContainer1):
 *   AnaOsc8D                 — 8D Kuramoto RK4, φ-seeded natural frequencies
 *   Adaptive phase           — Pluck→Sustain→FineTune→Lock (K/γ wu-wei ratios)
 *   Cooperative memory       — XOR-fold residue hash → phase perturbation every
 *                              ANA_SHA_INTERVAL iters (conditional memory loop)
 *
 * The oscillator does NOT shortcut correctness. Every p-2 iterations run exact.
 * Phase lock is a readout, not a gate — wu-wei.
 *
 * Build: ll_analog.c compiles as plain C (-O2); link with ll_mpi.cu via clang.
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Main entry point: Lucas-Lehmer test via v30b APA + 8D Kuramoto.
 * Returns 1 if M_p is prime, 0 if composite, -1 on allocation failure. */
int ll_analog(uint64_t p, int verbose);

#ifdef __cplusplus
}
#endif
