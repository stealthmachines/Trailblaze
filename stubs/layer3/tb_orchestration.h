#pragma once
#include <stdlib.h>
typedef struct { int dummy; } TB_CognitionTree;
static inline TB_CognitionTree* tb_tree_create(void *lat, const char *name, void *cfg) {
    (void)lat; (void)name; (void)cfg;
    return (TB_CognitionTree*)calloc(1, sizeof(TB_CognitionTree));
}
static inline void tb_tree_destroy(TB_CognitionTree *t) { free(t); }
static inline void tb_tree_cell_commit(TB_CognitionTree *t, int id, const char *data, const char *tag) {
    (void)t; (void)id; (void)data; (void)tag;
}
