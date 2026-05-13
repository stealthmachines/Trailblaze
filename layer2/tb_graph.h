/*
 * tb_graph.h — TRAILBLAZE Layers 2+3: Graph Engine + Cognition Substrate (C)
 *
 * TB_Node / TB_Graph — phi-lattice addressed execution graph, Spiral8 routing,
 * Kuramoto scheduler, operator fusion.
 *
 * TB_ERLEntry / TB_ERLLedger — hash-chained audit ledger (ported from
 * MCP server.js, implemented in C using phi_fold_hash32 instead of SHA).
 *
 * TB_CognitionCell / TB_CognitionTree — branch-aware persistent state with
 * COW KV cache fork, epoch ratchet, 4-tier memory, phi_stream sealing.
 *
 * TB_StrandStore — wraps zchg_store API for Layer 3 Tier 3 persistence.
 */

#pragma once
#ifndef TB_GRAPH_H
#define TB_GRAPH_H

#include <stdint.h>
#include <stddef.h>
#include "../layer0/tb_phi_lattice.h"
#include "../layer1/tb_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Layer 2: Pass types + Spiral8 assignment
 * ============================================================================ */

typedef enum {
    TB_PASS_FETCH     = 0,  TB_PASS_SHELL     = 1,  TB_PASS_CODE      = 2,
    TB_PASS_TRANSFORM = 3,  TB_PASS_STORE     = 4,  TB_PASS_RECALL    = 5,
    TB_PASS_BROWSE    = 6,  TB_PASS_NOTIFY    = 7,  TB_PASS_RESPOND   = 8,
    TB_PASS_EMBED     = 10, TB_PASS_ATTEND    = 11, TB_PASS_FFN       = 12,
    TB_PASS_SAMPLE    = 13, TB_PASS_ROUTE     = 14, TB_PASS_NORM      = 15,
    TB_PASS_PROJ      = 16, TB_PASS_BRANCH    = 20, TB_PASS_MERGE     = 21,
    TB_PASS_SPECULATE = 22,
} TB_PassType;

/* Spiral8 dim and wave_mode per pass type (from tb_phi_lattice.h TB_SPIRAL8) */
static const int TB_PASS_DIM[32] = {
    1, 8, 3, 4, 5, 2, 7, 6, 6,   /* FETCH..RESPOND */
    0, 0,                          /* padding for 9 */
    1, 2, 3, 6, 7, 4, 5,          /* EMBED..PROJ */
    0, 0, 0,                       /* padding */
    1, 8, 3                        /* BRANCH..SPECULATE */
};
static const int TB_PASS_WAVE[32] = {
    1, 0, 0, 0, -1, 1, 1, -1, -1,
    0, 0,
    1, 0, 0, -1, 1, 0, 1,
    0, 0, 0,
    1, -1, 0
};

/* ============================================================================
 * TB_Node — phi-lattice addressed graph node
 * ============================================================================ */

#define TB_MAX_DEPS 16
#define TB_MAX_FUSED 8
#define TB_NODE_NAME_LEN 64

typedef struct TB_Node {
    uint8_t         id[16];          /* phi-lattice u128 address */
    TB_PassType     pass_type;
    char            name[TB_NODE_NAME_LEN];
    int             spiral8_dim;
    int             wave_mode;

    /* Op function: returns 0 on success, -1 on error */
    int (*op_fn)(struct TB_Node *node, void *store, void *ctx);
    void *op_userdata;

    struct TB_Node *deps[TB_MAX_DEPS];
    int             n_deps;

    struct TB_Node *fused_into;
    struct TB_Node *fused_nodes[TB_MAX_FUSED];
    int             n_fused;

    TB_Backend      backend;
    int32_t         epoch;
    void           *result;
    double          exec_ms;
    int             layer_idx;
} TB_Node;

/* ============================================================================
 * TB_Graph
 * ============================================================================ */

#define TB_GRAPH_MAX_NODES 256

typedef struct {
    char          name[64];
    TB_PhiLattice *lattice;
    TB_Node       *nodes[TB_GRAPH_MAX_NODES];
    int            n_nodes;
    int            topo_dirty;
    TB_Node       *topo_order[TB_GRAPH_MAX_NODES];
    int            n_topo;
} TB_Graph;

/* ============================================================================
 * HDGL Router
 * ============================================================================ */

typedef struct {
    TB_PhiLattice    *lattice;
    TB_BackendRegistry *registry;
    double            dn_fast_threshold;   /* default 3.0 */
    double            cv_lock;             /* default 0.3 */
    double            cv_scatter;          /* default 0.8 */
} TB_HDGLRouter;

/* ============================================================================
 * ERL Ledger
 * ============================================================================ */

#define TB_ERL_MAX_ENTRIES 65536
#define TB_ERL_HASH_LEN    32
#define TB_ERL_DATA_LEN    256

typedef enum {
    TB_ERL_CELL_COMMIT   = 0,
    TB_ERL_BRANCH_CREATE = 1,
    TB_ERL_BRANCH_MERGE  = 2,
    TB_ERL_EPOCH_ADVANCE = 3,
    TB_ERL_TASK_START    = 4,
    TB_ERL_TASK_COMPLETE = 5,
    TB_ERL_TOOL_CALL     = 6,
    TB_ERL_ERROR         = 7,
    TB_ERL_AGENT_DELEGATE= 8,
} TB_ERLType;

typedef struct {
    int32_t         seq;
    TB_ERLType      type;
    int             branch_id;
    int32_t         epoch;
    int64_t         timestamp_ms;
    char            data[TB_ERL_DATA_LEN];     /* JSON snippet */
    uint8_t         parent_hash[TB_ERL_HASH_LEN];
    uint8_t         entry_hash [TB_ERL_HASH_LEN];
} TB_ERLEntry;

typedef struct {
    TB_ERLEntry     *entries;
    int32_t          n_entries;
    int32_t          capacity;
    TB_PhiLattice   *lattice;
    char            *persist_path;    /* NULL = in-memory only */
    int              db_fd;           /* -1 = no SQLite */
} TB_ERLLedger;

/* ============================================================================
 * CognitionCell
 * ============================================================================ */

#define TB_CELL_VALUE_LEN 512

typedef struct {
    uint8_t  id[16];           /* phi-lattice u128 */
    int      branch_id;
    uint8_t  parent_id[16];
    int32_t  epoch;
    char     domain[32];
    char     value[TB_CELL_VALUE_LEN];
    double   dn_amplitude;
    uint32_t flags;
    int64_t  timestamp_ms;
    uint8_t  audit_hash[32];
} TB_CognitionCell;

/* ============================================================================
 * BranchHead
 * ============================================================================ */

#define TB_BRANCH_MAX_FLOW_KEYS 32
#define TB_FLOW_KEY_LEN 64
#define TB_FLOW_VAL_LEN 512

typedef struct {
    int         id;
    uint8_t     tip[16];
    uint8_t     fork_point[16];
    int         parent_branch_id;
    int32_t     epoch_created;
    TB_KVCache *kv_cache;
    char        flow_keys[TB_BRANCH_MAX_FLOW_KEYS][TB_FLOW_KEY_LEN];
    char        flow_vals[TB_BRANCH_MAX_FLOW_KEYS][TB_FLOW_VAL_LEN];
    int         n_flow;
    int         merged;
    int         merged_into;
} TB_BranchHead;

/* ============================================================================
 * CognitionTree
 * ============================================================================ */

#define TB_MAX_BRANCHES 256
#define TB_MAX_CELLS    65536
#define TB_SESSION_MAX  64

typedef struct {
    TB_PhiLattice    *lattice;
    char              name[64];
    TB_CognitionCell *cells[TB_MAX_CELLS];
    int               n_cells;
    TB_BranchHead    *branches[TB_MAX_BRANCHES];
    int               n_branches;
    int               next_branch_id;
    TB_ERLLedger     *ledger;
    /* Session memory (Tier 1) */
    char              session_keys[TB_SESSION_MAX][TB_FLOW_KEY_LEN];
    char              session_vals[TB_SESSION_MAX][TB_FLOW_VAL_LEN];
    int               n_session;
    /* Notes dir (Tier 2) */
    char              persist_dir[512];
} TB_CognitionTree;

/* ============================================================================
 * API: Graph
 * ============================================================================ */

TB_Graph* tb_graph_create(const char *name, TB_PhiLattice *lat);
void      tb_graph_destroy(TB_Graph *g);

TB_Node*  tb_graph_add_node(TB_Graph *g, TB_PassType pt, const char *name,
                             TB_Node **deps, int n_deps);
int       tb_graph_topo_sort(TB_Graph *g);
int       tb_graph_fuse(TB_Graph *g);      /* returns n_fusions */

/* ============================================================================
 * API: HDGL Router
 * ============================================================================ */

void       tb_router_init(TB_HDGLRouter *r, TB_PhiLattice *lat,
                           TB_BackendRegistry *reg);
TB_Backend tb_router_route_node(TB_HDGLRouter *r, TB_Node *node);
uint32_t   tb_router_route_server(TB_HDGLRouter *r,
                                   const char *request_key,
                                   uint32_t *server_pool, int pool_size);

/* ============================================================================
 * API: ERL Ledger
 * ============================================================================ */

TB_ERLLedger* tb_erl_create(TB_PhiLattice *lat, const char *persist_path);
void          tb_erl_destroy(TB_ERLLedger *L);
TB_ERLEntry*  tb_erl_append(TB_ERLLedger *L, TB_ERLType type,
                             int branch_id, const char *data_json);
int           tb_erl_verify_chain(TB_ERLLedger *L, int *out_broken_seq);
int           tb_erl_n_entries(TB_ERLLedger *L);

/* ============================================================================
 * API: CognitionTree
 * ============================================================================ */

TB_CognitionTree* tb_tree_create(TB_PhiLattice *lat, const char *name,
                                  const char *persist_dir);
void              tb_tree_destroy(TB_CognitionTree *tree);

int32_t           tb_tree_epoch_advance(TB_CognitionTree *tree, int steps);
TB_CognitionCell* tb_tree_cell_commit(TB_CognitionTree *tree, int branch_id,
                                       const char *value, const char *domain);
TB_CognitionCell* tb_tree_cell_get_by_id(TB_CognitionTree *tree,
                                          const uint8_t id[16]);
int               tb_tree_cell_verify(TB_CognitionTree *tree,
                                       TB_CognitionCell *cell);

int               tb_tree_branch_create(TB_CognitionTree *tree,
                                         int from_branch_id);
int               tb_tree_branch_merge(TB_CognitionTree *tree,
                                        int src_id, int dst_id);

void              tb_tree_memory_set(TB_CognitionTree *tree,
                                      const char *key, const char *val);
const char*       tb_tree_memory_get(TB_CognitionTree *tree, const char *key);
void              tb_tree_notes_write(TB_CognitionTree *tree,
                                       const char *filename, const char *content);
char*             tb_tree_notes_read(TB_CognitionTree *tree,
                                      const char *filename);  /* caller frees */
void              tb_tree_flow_set(TB_CognitionTree *tree, int branch_id,
                                    const char *key, const char *val);
const char*       tb_tree_flow_get(TB_CognitionTree *tree, int branch_id,
                                    const char *key);
void              tb_tree_record_tool_call(TB_CognitionTree *tree,
                                            int branch_id, const char *tool,
                                            const char *args_json);

/* Sealing (epoch-bound, forward-secret) */
size_t tb_tree_seal  (TB_CognitionTree *tree, const uint8_t *pt, size_t pt_len,
                       const char *domain, uint8_t *out, size_t out_cap);
size_t tb_tree_unseal(TB_CognitionTree *tree, const uint8_t *env, size_t env_len,
                       const char *domain, uint8_t *out, size_t out_cap);

/* Describe (JSON into buf) */
int tb_tree_describe(TB_CognitionTree *tree, char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* TB_GRAPH_H */
