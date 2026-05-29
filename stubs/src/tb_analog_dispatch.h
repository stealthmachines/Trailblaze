#pragma once
#include <stdint.h>
#include <stdio.h>
typedef struct { double phase_var; double s_u; double lambda_u; double omega_u; int aphase; int steps; } TBOscSnapshot;
typedef struct { int has_avx2; int has_avx512; int has_neon; } TBCpuCaps;
typedef struct { void *lattice; } TB_BackendRegistry;
static inline void tb_dispatch_detect_caps(TBCpuCaps *c) { (void)c; }
static inline void tb_dispatch_context_set(TBOscSnapshot *o, TBCpuCaps *c) { (void)o; (void)c; }
static inline void tb_registry_init(TB_BackendRegistry *r, void *lat) { (void)r; (void)lat; }
