#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef struct { double phase_var; int32_t epoch; } TB_PhiLattice;
static inline TB_PhiLattice* tb_lattice_create(uint32_t slots, uint64_t seed) {
    (void)slots; (void)seed;
    TB_PhiLattice *l = (TB_PhiLattice*)calloc(1, sizeof(TB_PhiLattice));
    if (l) { l->phase_var = 0.1; l->epoch = 1; }
    return l;
}
static inline void tb_lattice_advance(TB_PhiLattice *l, int n) { (void)l; (void)n; }
static inline void tb_lattice_destroy(TB_PhiLattice *l) { free(l); }
static inline void tb_lattice_describe(TB_PhiLattice *l, char *buf, int sz) {
    (void)l; snprintf(buf, sz, "stub-lattice"); }
static inline void tb_lattice_s_u_resonance(TB_PhiLattice *l, double *M, double *L, double *S) {
    (void)l; *M = 1.0; *L = 0.5; *S = 0.5; }
