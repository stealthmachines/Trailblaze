#pragma once
#include <stdlib.h>
typedef struct { int dim; float *mem; } TB_HopfieldMemory;
static inline TB_HopfieldMemory* tb_hopfield_alloc(int dim) {
    TB_HopfieldMemory *h = (TB_HopfieldMemory*)calloc(1, sizeof(TB_HopfieldMemory));
    if (h) { h->dim = dim; h->mem = (float*)calloc((size_t)dim*dim, sizeof(float)); }
    return h;
}
static inline void tb_hopfield_free(TB_HopfieldMemory *h) {
    if (!h) return; free(h->mem); free(h);
}
