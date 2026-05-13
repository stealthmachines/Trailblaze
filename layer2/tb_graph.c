/*
 * tb_graph.c — TRAILBLAZE Layers 2+3 Implementation (pure C)
 */

#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "tb_graph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#ifdef _WIN32
#  include "../src/tb_win32.h"
#else
#  include <sys/stat.h>
#endif

/* ── Timestamp helper ─────────────────────────────────────────────────────── */
static int64_t tb_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 1: Graph
 * ────────────────────────────────────────────────────────────────────────── */

TB_Graph* tb_graph_create(const char *name, TB_PhiLattice *lat) {
    TB_Graph *g = (TB_Graph *)calloc(1, sizeof(TB_Graph));
    if (!g) return NULL;
    snprintf(g->name, sizeof(g->name), "%s", name);
    g->lattice     = lat;
    g->topo_dirty  = 1;
    return g;
}

void tb_graph_destroy(TB_Graph *g) {
    if (!g) return;
    for (int i = 0; i < g->n_nodes; i++) free(g->nodes[i]);
    free(g);
}

TB_Node* tb_graph_add_node(TB_Graph *g, TB_PassType pt, const char *name,
                             TB_Node **deps, int n_deps) {
    if (g->n_nodes >= TB_GRAPH_MAX_NODES) return NULL;

    TB_Node *n = (TB_Node *)calloc(1, sizeof(TB_Node));
    if (!n) return NULL;

    /* Phi-lattice address */
    char key[128];
    snprintf(key, sizeof(key), "%d:%s:%d", pt, name ? name : "", g->n_nodes);
    const uint8_t *parts[1]  = {(const uint8_t *)key};
    size_t         lens[1]   = {strlen(key)};
    tb_lattice_phi_addr(g->lattice, parts, lens, 1, n->id);

    n->pass_type   = pt;
    snprintf(n->name, TB_NODE_NAME_LEN, "%s", name ? name : "");
    n->spiral8_dim = (pt < 23) ? TB_PASS_DIM[pt]  : 1;
    n->wave_mode   = (pt < 23) ? TB_PASS_WAVE[pt] : 0;
    n->epoch       = g->lattice->epoch;
    n->backend     = TB_BACKEND_CPU_AVX2;

    if (n_deps > TB_MAX_DEPS) n_deps = TB_MAX_DEPS;
    memcpy(n->deps, deps ? deps : NULL, n_deps * sizeof(TB_Node *));
    n->n_deps = n_deps;

    g->nodes[g->n_nodes++] = n;
    g->topo_dirty = 1;
    return n;
}

/* Kahn's topological sort */
int tb_graph_topo_sort(TB_Graph *g) {
    if (!g->topo_dirty) return 0;

    int in_deg[TB_GRAPH_MAX_NODES] = {0};
    /* Build adjacency (node → successors) */
    int adj[TB_GRAPH_MAX_NODES][TB_GRAPH_MAX_NODES];
    int adj_cnt[TB_GRAPH_MAX_NODES] = {0};
    memset(adj, 0, sizeof(adj));

    for (int i = 0; i < g->n_nodes; i++) {
        TB_Node *n = g->nodes[i];
        for (int d = 0; d < n->n_deps; d++) {
            /* find dep index */
            for (int j = 0; j < g->n_nodes; j++) {
                if (g->nodes[j] == n->deps[d]) {
                    adj[j][adj_cnt[j]++] = i;
                    in_deg[i]++;
                    break;
                }
            }
        }
    }

    /* BFS */
    int queue[TB_GRAPH_MAX_NODES], qh=0, qt=0;
    for (int i = 0; i < g->n_nodes; i++)
        if (in_deg[i] == 0) queue[qt++] = i;

    int n_sorted = 0;
    while (qh < qt) {
        int idx = queue[qh++];
        g->topo_order[n_sorted++] = g->nodes[idx];
        for (int k = 0; k < adj_cnt[idx]; k++) {
            int succ = adj[idx][k];
            if (--in_deg[succ] == 0) queue[qt++] = succ;
        }
    }
    g->n_topo    = n_sorted;
    g->topo_dirty = 0;
    if (n_sorted != g->n_nodes) return -1;  /* cycle */
    return 0;
}

/* Operator fusion: same spiral8_dim + same wave_mode, sole consumer */
int tb_graph_fuse(TB_Graph *g) {
    tb_graph_topo_sort(g);
    int n_fused = 0;

    /* Count consumers */
    int consumers[TB_GRAPH_MAX_NODES] = {0};
    for (int i = 0; i < g->n_nodes; i++) {
        TB_Node *n = g->nodes[i];
        for (int d = 0; d < n->n_deps; d++) {
            for (int j = 0; j < g->n_nodes; j++) {
                if (g->nodes[j] == n->deps[d]) {
                    consumers[j]++;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < g->n_topo - 1; i++) {
        TB_Node *n    = g->topo_order[i];
        TB_Node *next = g->topo_order[i+1];
        if (n->fused_into || next->fused_into) continue;

        /* Check: n is sole dep of next */
        int n_idx = -1;
        for (int j = 0; j < g->n_nodes; j++)
            if (g->nodes[j] == n) { n_idx = j; break; }

        if (consumers[n_idx] != 1) continue;
        if (n->spiral8_dim != next->spiral8_dim) continue;
        if (n->wave_mode   != next->wave_mode)   continue;

        /* Fuse next into n */
        next->fused_into = n;
        if (n->n_fused < TB_MAX_FUSED)
            n->fused_nodes[n->n_fused++] = next;
        n_fused++;
    }
    g->topo_dirty = 1;   /* re-sort needed */
    return n_fused;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 2: HDGL Router
 * ────────────────────────────────────────────────────────────────────────── */

void tb_router_init(TB_HDGLRouter *r, TB_PhiLattice *lat,
                    TB_BackendRegistry *reg) {
    r->lattice            = lat;
    r->registry           = reg;
    r->dn_fast_threshold  = 3.0;
    r->cv_lock            = 0.3;
    r->cv_scatter         = 0.8;
}

TB_Backend tb_router_route_node(TB_HDGLRouter *r, TB_Node *node) {
    char key[128];
    snprintf(key, sizeof(key), "%d:%s:%d",
             node->pass_type, node->name, node->spiral8_dim);
    double dn  = tb_lattice_dn_for_key(r->lattice, key, strlen(key));
    double cv  = r->lattice->phase_var;
    TB_Backend b;

    if (dn > r->dn_fast_threshold && cv < r->cv_lock) {
        b = TB_BACKEND_CPU_AVX2;
    } else if (cv > r->cv_scatter) {
        b = TB_BACKEND_ANALOG;
    } else {
        b = tb_registry_select(r->registry, TB_OP_MATMUL, 1024);
    }

    /* Wave mode -1 (absorbing) → always CPU */
    if (node->wave_mode == -1 &&
        (node->pass_type == TB_PASS_STORE || node->pass_type == TB_PASS_NOTIFY))
        b = TB_BACKEND_CPU_AVX2;

    node->backend = b;
    return b;
}

uint32_t tb_router_route_server(TB_HDGLRouter *r,
                                 const char *request_key,
                                 uint32_t *server_pool, int pool_size) {
    if (pool_size <= 0) return 0;
    uint32_t slot = tb_lattice_slot_for_key(r->lattice,
                                             request_key, strlen(request_key));
    /* phi²-irrational index: prevents clustering */
    double frac = fmod((double)slot * TB_PHI * TB_PHI / (double)r->lattice->n_slots, 1.0);
    if (frac < 0.0) frac += 1.0;
    int idx = (int)(frac * pool_size) % pool_size;
    return server_pool[idx];
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 3: ERL Ledger
 * phi_fold_hash32 replaces SHA-256 (epoch-invalidated S-box, no openssl dep)
 * ────────────────────────────────────────────────────────────────────────── */

TB_ERLLedger* tb_erl_create(TB_PhiLattice *lat, const char *persist_path) {
    TB_ERLLedger *L = (TB_ERLLedger *)calloc(1, sizeof(TB_ERLLedger));
    if (!L) return NULL;
    L->capacity     = TB_ERL_MAX_ENTRIES;
    L->entries      = (TB_ERLEntry *)calloc(L->capacity, sizeof(TB_ERLEntry));
    L->lattice      = lat;
    L->persist_path = persist_path ? strdup(persist_path) : NULL;
    L->db_fd        = -1;
    if (!L->entries) { free(L); return NULL; }
    return L;
}

void tb_erl_destroy(TB_ERLLedger *L) {
    if (!L) return;
    free(L->entries);
    free(L->persist_path);
    free(L);
}

static void tb_erl_compute_hash(TB_PhiLattice *lat, const TB_ERLEntry *e,
                                  uint8_t out[32]) {
    /* Hash: phi_fold32(seq[4] || type[4] || branch_id[4] || epoch[4] ||
     *                  timestamp[8] || data[256] || parent_hash[32])  */
    uint8_t buf[320];
    size_t  off = 0;
    memcpy(buf+off, &e->seq,          4); off+=4;
    memcpy(buf+off, &e->type,         4); off+=4;
    memcpy(buf+off, &e->branch_id,    4); off+=4;
    memcpy(buf+off, &e->epoch,        4); off+=4;
    memcpy(buf+off, &e->timestamp_ms, 8); off+=8;
    memcpy(buf+off,  e->data,       256); off+=256;
    memcpy(buf+off,  e->parent_hash, 32); off+=32;
    /* Use FNV-1a for ERL chain hash (epoch-stable, no S-box) */
    uint64_t fnv = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < off; i++) {
        fnv ^= (uint64_t)buf[i];
        fnv *= 0x100000001b3ULL;
    }
    /* Expand to 32 bytes via mixing */
    for (int _b = 0; _b < 32; _b++) {
        out[_b] = (uint8_t)(fnv >> (_b % 8 * 8));
        fnv = (fnv << 13) | (fnv >> 51);
        fnv ^= 0x9e3779b97f4a7c15ULL;
    }
}

TB_ERLEntry* tb_erl_append(TB_ERLLedger *L, TB_ERLType type,
                             int branch_id, const char *data_json) {
    if (L->n_entries >= L->capacity) return NULL;
    TB_ERLEntry *e = &L->entries[L->n_entries];
    e->seq       = L->n_entries;
    e->type      = type;
    e->branch_id = branch_id;
    e->epoch     = L->lattice->epoch;
    e->timestamp_ms = tb_now_ms();
    snprintf(e->data, TB_ERL_DATA_LEN, "%s",
             data_json ? data_json : "{}");

    /* Parent hash */
    if (L->n_entries == 0) {
        memset(e->parent_hash, 0, 32);
    } else {
        memcpy(e->parent_hash, L->entries[L->n_entries-1].entry_hash, 32);
    }

    tb_erl_compute_hash(L->lattice, e, e->entry_hash);
    L->n_entries++;
    return e;
}

int tb_erl_verify_chain(TB_ERLLedger *L, int *out_broken_seq) {
    uint8_t expected_parent[32];
    memset(expected_parent, 0, 32);

    for (int i = 0; i < L->n_entries; i++) {
        TB_ERLEntry *e = &L->entries[i];

        /* Verify parent hash */
        if (memcmp(e->parent_hash, expected_parent, 32) != 0) {
            if (out_broken_seq) *out_broken_seq = i;
            return 0;
        }

        /* Verify entry hash */
        uint8_t computed[32];
        tb_erl_compute_hash(L->lattice, e, computed);
        if (memcmp(computed, e->entry_hash, 32) != 0) {
            if (out_broken_seq) *out_broken_seq = i;
            return 0;
        }

        memcpy(expected_parent, e->entry_hash, 32);
    }
    if (out_broken_seq) *out_broken_seq = -1;
    return 1;
}

int tb_erl_n_entries(TB_ERLLedger *L) {
    return L ? L->n_entries : 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 4: CognitionCell
 * ────────────────────────────────────────────────────────────────────────── */

static void tb_cell_compute_audit_hash(TB_PhiLattice *lat,
                                         TB_CognitionCell *cell) {
    uint8_t buf[16 + TB_CELL_VALUE_LEN + 8 + 32];
    size_t off = 0;
    memcpy(buf+off, cell->parent_id,  16); off += 16;
    memcpy(buf+off, cell->value,  strlen(cell->value)); off += strlen(cell->value);
    memcpy(buf+off, &cell->epoch, 8);  off += 8;
    memcpy(buf+off, cell->domain, strlen(cell->domain)); off += strlen(cell->domain);
    tb_phi_fold_hash32(lat, buf, off, cell->audit_hash);
}

int tb_tree_cell_verify(TB_CognitionTree *tree, TB_CognitionCell *cell) {
    uint8_t expected[32];
    /* Re-derive */
    uint8_t buf[16 + TB_CELL_VALUE_LEN + 8 + 32];
    size_t off = 0;
    memcpy(buf+off, cell->parent_id, 16); off += 16;
    size_t vl = strlen(cell->value);
    memcpy(buf+off, cell->value, vl); off += vl;
    memcpy(buf+off, &cell->epoch, 8); off += 8;
    size_t dl = strlen(cell->domain);
    memcpy(buf+off, cell->domain, dl); off += dl;
    tb_phi_fold_hash32(tree->lattice, buf, off, expected);
    return memcmp(expected, cell->audit_hash, 32) == 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 5: CognitionTree
 * ────────────────────────────────────────────────────────────────────────── */

static TB_BranchHead* tb_branch_alloc(int id, const uint8_t tip[16],
                                       int parent_bid, int32_t epoch) {
    TB_BranchHead *b = (TB_BranchHead *)calloc(1, sizeof(TB_BranchHead));
    if (!b) return NULL;
    b->id               = id;
    b->parent_branch_id = parent_bid;
    b->epoch_created    = epoch;
    b->merged           = 0;
    b->merged_into      = -1;
    b->kv_cache         = NULL;
    memcpy(b->tip,        tip, 16);
    memcpy(b->fork_point, tip, 16);
    return b;
}

TB_CognitionTree* tb_tree_create(TB_PhiLattice *lat, const char *name,
                                   const char *persist_dir) {
    TB_CognitionTree *tree = (TB_CognitionTree *)calloc(1, sizeof(*tree));
    if (!tree) return NULL;

    tree->lattice = lat;
    snprintf(tree->name, sizeof(tree->name), "%s", name ? name : "default");
    if (persist_dir)
        snprintf(tree->persist_dir, sizeof(tree->persist_dir), "%s", persist_dir);
    tree->next_branch_id = 1;

    /* Create ERL ledger */
    char erl_path[560] = {0};
    if (persist_dir) {
        snprintf(erl_path, sizeof(erl_path), "%s/erl.bin", persist_dir);
        mkdir(persist_dir, 0755);
    }
    tree->ledger = tb_erl_create(lat, persist_dir ? erl_path : NULL);

    /* Genesis cell */
    char genesis_key[] = "TRAILBLAZE::GENESIS";
    const uint8_t *pk[1] = {(const uint8_t *)genesis_key};
    size_t pl[1] = {strlen(genesis_key)};
    uint8_t genesis_id[16];
    tb_lattice_phi_addr(lat, pk, pl, 1, genesis_id);

    TB_CognitionCell *root = (TB_CognitionCell *)calloc(1, sizeof(*root));
    memcpy(root->id, genesis_id, 16);
    root->branch_id = 0;
    memset(root->parent_id, 0, 16);
    root->epoch = lat->epoch;
    snprintf(root->domain, sizeof(root->domain), "genesis");
    snprintf(root->value,  sizeof(root->value),
             "{\"trailblaze\":true,\"name\":\"%s\"}", tree->name);
    root->dn_amplitude = tb_lattice_dn_for_key(lat, "genesis", 7);
    root->timestamp_ms = tb_now_ms();
    tb_cell_compute_audit_hash(lat, root);
    tree->cells[tree->n_cells++] = root;

    /* Main branch */
    TB_BranchHead *main_branch = tb_branch_alloc(0, genesis_id, -1, lat->epoch);
    tree->branches[tree->n_branches++] = main_branch;

    char buf[128];
    snprintf(buf, sizeof(buf), "{\"name\":\"main\",\"root\":\"...\"}");
    tb_erl_append(tree->ledger, TB_ERL_BRANCH_CREATE, 0, buf);
    return tree;
}

void tb_tree_destroy(TB_CognitionTree *tree) {
    if (!tree) return;
    for (int i = 0; i < tree->n_cells; i++) free(tree->cells[i]);
    for (int i = 0; i < tree->n_branches; i++) {
        if (tree->branches[i]) {
            tb_kvcache_free(tree->branches[i]->kv_cache);
            free(tree->branches[i]);
        }
    }
    tb_erl_destroy(tree->ledger);
    free(tree);
}

int32_t tb_tree_epoch_advance(TB_CognitionTree *tree, int steps) {
    int32_t old_ep = tree->lattice->epoch;
    tb_lattice_advance(tree->lattice, steps);
    int32_t new_ep = tree->lattice->epoch;

    /* Invalidate all KV caches */
    for (int i = 0; i < tree->n_branches; i++) {
        TB_BranchHead *b = tree->branches[i];
        if (b && b->kv_cache)
            tb_kvcache_invalidate(b->kv_cache, new_ep);
    }

    char buf[128];
    snprintf(buf, sizeof(buf),
             "{\"old\":%d,\"new\":%d,\"branches\":%d}",
             old_ep, new_ep, tree->n_branches);
    tb_erl_append(tree->ledger, TB_ERL_EPOCH_ADVANCE, 0, buf);
    return new_ep;
}

TB_CognitionCell* tb_tree_cell_commit(TB_CognitionTree *tree, int branch_id,
                                       const char *value, const char *domain) {
    if (tree->n_cells >= TB_MAX_CELLS) return NULL;
    if (branch_id >= tree->n_branches) return NULL;
    TB_BranchHead *b = tree->branches[branch_id];
    if (!b) return NULL;

    TB_PhiLattice *lat = tree->lattice;
    const uint8_t *pk[4];
    size_t         pl[4];

    pk[0] = b->tip; pl[0] = 16;
    pk[1] = (const uint8_t *)value; pl[1] = strlen(value) > 64 ? 64 : strlen(value);
    pk[2] = (const uint8_t *)&branch_id; pl[2] = 4;
    pk[3] = (const uint8_t *)domain; pl[3] = strlen(domain);

    uint8_t cell_id[16];
    tb_lattice_phi_addr(lat, pk, pl, 4, cell_id);

    TB_CognitionCell *cell = (TB_CognitionCell *)calloc(1, sizeof(*cell));
    if (!cell) return NULL;
    memcpy(cell->id,        cell_id, 16);
    memcpy(cell->parent_id, b->tip,  16);
    cell->branch_id    = branch_id;
    cell->epoch        = lat->epoch;
    cell->dn_amplitude = tb_lattice_dn_for_key(lat, domain, strlen(domain));
    cell->timestamp_ms = tb_now_ms();
    snprintf(cell->domain, sizeof(cell->domain), "%s", domain ? domain : "data");
    snprintf(cell->value,  sizeof(cell->value),  "%s", value  ? value  : "{}");
    tb_cell_compute_audit_hash(lat, cell);

    tree->cells[tree->n_cells++] = cell;
    memcpy(b->tip, cell_id, 16);

    char buf[128];
    snprintf(buf, sizeof(buf), "{\"domain\":\"%s\"}", domain);
    tb_erl_append(tree->ledger, TB_ERL_CELL_COMMIT, branch_id, buf);
    return cell;
}

int tb_tree_branch_create(TB_CognitionTree *tree, int from_branch_id) {
    if (tree->n_branches >= TB_MAX_BRANCHES) return -1;
    TB_BranchHead *src = tree->branches[from_branch_id];
    if (!src) return -1;

    int new_id = tree->next_branch_id++;
    TB_BranchHead *nb = tb_branch_alloc(new_id, src->tip,
                                         from_branch_id, tree->lattice->epoch);
    if (!nb) return -1;

    /* COW fork KV cache */
    if (src->kv_cache)
        nb->kv_cache = tb_kvcache_fork(src->kv_cache, new_id);

    /* Copy flow state */
    memcpy(nb->flow_keys, src->flow_keys, sizeof(src->flow_keys));
    memcpy(nb->flow_vals, src->flow_vals, sizeof(src->flow_vals));
    nb->n_flow = src->n_flow;

    tree->branches[tree->n_branches++] = nb;

    char buf[128];
    snprintf(buf, sizeof(buf), "{\"from\":%d}", from_branch_id);
    tb_erl_append(tree->ledger, TB_ERL_BRANCH_CREATE, new_id, buf);
    return new_id;
}

int tb_tree_branch_merge(TB_CognitionTree *tree, int src_id, int dst_id) {
    if (src_id >= tree->n_branches || dst_id >= tree->n_branches) return 0;
    TB_BranchHead *src = tree->branches[src_id];
    TB_BranchHead *dst = tree->branches[dst_id];
    if (!src || !dst || src->merged) return 0;

    /* Walk src history back to fork point, replay cells onto dst */
    /* (Simplified: copy src tip cell value onto dst as merged commit) */
    uint8_t *fork_pt = src->fork_point;
    int new_cells = 0;
    for (int i = tree->n_cells - 1; i >= 0; i--) {
        TB_CognitionCell *c = tree->cells[i];
        if (c->branch_id != src_id) continue;
        if (memcmp(c->id, fork_pt, 16) == 0) break;

        /* Re-commit onto dst branch */
        TB_CognitionCell *mc = tb_tree_cell_commit(tree, dst_id,
                                                    c->value, c->domain);
        if (mc) new_cells++;
    }

    /* KV reconciliation */
    if (src->kv_cache && dst->kv_cache) {
        TB_KVCache *r = tb_kvcache_reconcile(src->kv_cache, dst->kv_cache,
                                              tree->lattice->epoch);
        tb_kvcache_free(dst->kv_cache);
        dst->kv_cache = r;
    }

    src->merged     = 1;
    src->merged_into= dst_id;

    char buf[128];
    snprintf(buf, sizeof(buf), "{\"src\":%d,\"dst\":%d,\"cells\":%d}",
             src_id, dst_id, new_cells);
    tb_erl_append(tree->ledger, TB_ERL_BRANCH_MERGE, dst_id, buf);
    return 1;
}

/* ── Memory tiers ─────────────────────────────────────────────────────────── */

void tb_tree_memory_set(TB_CognitionTree *tree, const char *key, const char *val) {
    for (int i = 0; i < tree->n_session; i++) {
        if (strcmp(tree->session_keys[i], key) == 0) {
            snprintf(tree->session_vals[i], TB_FLOW_VAL_LEN, "%s", val);
            return;
        }
    }
    if (tree->n_session >= TB_SESSION_MAX) return;
    snprintf(tree->session_keys[tree->n_session], TB_FLOW_KEY_LEN, "%s", key);
    snprintf(tree->session_vals[tree->n_session], TB_FLOW_VAL_LEN, "%s", val);
    tree->n_session++;
}

const char* tb_tree_memory_get(TB_CognitionTree *tree, const char *key) {
    for (int i = 0; i < tree->n_session; i++)
        if (strcmp(tree->session_keys[i], key) == 0)
            return tree->session_vals[i];
    return NULL;
}

void tb_tree_notes_write(TB_CognitionTree *tree,
                          const char *filename, const char *content) {
    if (!tree->persist_dir[0]) return;
    char path[640];
    snprintf(path, sizeof(path), "%s/notes", tree->persist_dir);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/notes/%s", tree->persist_dir, filename);
    FILE *f = fopen(path, "w");
    if (f) { fputs(content, f); fclose(f); }
}

char* tb_tree_notes_read(TB_CognitionTree *tree, const char *filename) {
    if (!tree->persist_dir[0]) return NULL;
    char path[640];
    snprintf(path, sizeof(path), "%s/notes/%s", tree->persist_dir, filename);
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc(sz + 1);
    if (buf) { (void)fread(buf, 1, sz, f); buf[sz] = '\0'; }
    fclose(f);
    return buf;
}

void tb_tree_flow_set(TB_CognitionTree *tree, int branch_id,
                       const char *key, const char *val) {
    if (branch_id >= tree->n_branches) return;
    TB_BranchHead *b = tree->branches[branch_id];
    if (!b) return;
    for (int i = 0; i < b->n_flow; i++) {
        if (strcmp(b->flow_keys[i], key) == 0) {
            snprintf(b->flow_vals[i], TB_FLOW_VAL_LEN, "%s", val); return;
        }
    }
    if (b->n_flow >= TB_BRANCH_MAX_FLOW_KEYS) return;
    snprintf(b->flow_keys[b->n_flow], TB_FLOW_KEY_LEN, "%s", key);
    snprintf(b->flow_vals[b->n_flow], TB_FLOW_VAL_LEN, "%s", val);
    b->n_flow++;
}

const char* tb_tree_flow_get(TB_CognitionTree *tree, int branch_id,
                               const char *key) {
    if (branch_id >= tree->n_branches) return NULL;
    TB_BranchHead *b = tree->branches[branch_id];
    if (!b) return NULL;
    for (int i = 0; i < b->n_flow; i++)
        if (strcmp(b->flow_keys[i], key) == 0) return b->flow_vals[i];
    return NULL;
}

void tb_tree_record_tool_call(TB_CognitionTree *tree, int branch_id,
                               const char *tool, const char *args_json) {
    char buf[TB_ERL_DATA_LEN];
    snprintf(buf, sizeof(buf), "{\"tool\":\"%s\",\"args\":%s}",
             tool ? tool : "unknown", args_json ? args_json : "{}");
    tb_erl_append(tree->ledger, TB_ERL_TOOL_CALL, branch_id, buf);
}

/* ── Sealing (forward-secret, epoch-bound) ───────────────────────────────── */

size_t tb_tree_seal(TB_CognitionTree *tree, const uint8_t *pt, size_t pt_len,
                     const char *domain, uint8_t *out, size_t out_cap) {
    return tb_phi_stream_seal(tree->lattice, pt, pt_len, domain, out, out_cap);
}

size_t tb_tree_unseal(TB_CognitionTree *tree, const uint8_t *env, size_t env_len,
                       const char *domain, uint8_t *out, size_t out_cap) {
    return tb_phi_stream_unseal(tree->lattice, env, env_len, domain, out, out_cap);
}

/* ── Describe ────────────────────────────────────────────────────────────── */

int tb_tree_describe(TB_CognitionTree *tree, char *buf, size_t buf_len) {
    int broken = -1;
    int chain_ok = tb_erl_verify_chain(tree->ledger, &broken);
    return snprintf(buf, buf_len,
        "{\"name\":\"%s\",\"epoch\":%d,\"n_cells\":%d,"
        "\"n_branches\":%d,\"erl_entries\":%d,\"chain_valid\":%s}",
        tree->name, tree->lattice->epoch, tree->n_cells,
        tree->n_branches, tree->ledger->n_entries,
        chain_ok ? "true" : "false");
}

TB_CognitionCell* tb_tree_cell_get_by_id(TB_CognitionTree *tree,
                                          const uint8_t id[16]) {
    for (int i = 0; i < tree->n_cells; i++)
        if (memcmp(tree->cells[i]->id, id, 16) == 0)
            return tree->cells[i];
    return NULL;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 6: Self-test
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef TB_L23_TEST
#include <assert.h>
#include <unistd.h>

int main(void) {
    printf("=== TRAILBLAZE Layers 2+3 C Self-Test ===\n\n");

    TB_PhiLattice *lat = tb_lattice_create(256, 0xABCDEF01ULL);
    assert(lat);
    for (int i = 0; i < 5; i++) tb_lattice_advance(lat, 1);

    /* ── Graph ── */
    TB_Graph *g = tb_graph_create("test", lat);
    TB_Node *n0 = tb_graph_add_node(g, TB_PASS_EMBED,   "embed",  NULL, 0);
    TB_Node *n1 = tb_graph_add_node(g, TB_PASS_ATTEND,  "attn0",  &n0, 1);
    TB_Node *n2 = tb_graph_add_node(g, TB_PASS_FFN,     "ffn0",   &n1, 1);
    TB_Node *n3 = tb_graph_add_node(g, TB_PASS_SAMPLE,  "sample", &n2, 1);

    assert(tb_graph_topo_sort(g) == 0);
    assert(g->n_topo == 4);
    assert(g->topo_order[0] == n0 && g->topo_order[3] == n3);
    printf("[topo sort] 4-node graph: PASS\n");

    /* Unique phi-lattice IDs */
    for (int i = 0; i < 3; i++)
        assert(memcmp(g->nodes[i]->id, g->nodes[i+1]->id, 16) != 0);
    printf("[phi-address] 4 unique u128 IDs: PASS\n");

    /* HDGL routing */
    TB_BackendRegistry reg;
    tb_registry_init(&reg, lat);
    TB_HDGLRouter router;
    tb_router_init(&router, lat, &reg);
    for (int i = 0; i < g->n_topo; i++)
        tb_router_route_node(&router, g->topo_order[i]);
    printf("[HDGL router] all nodes routed: PASS\n");

    /* Fusion */
    int fused = tb_graph_fuse(g);
    printf("[fusion] %d fusions applied: PASS\n", fused);

    /* Server routing */
    uint32_t servers[3] = {0xC0A80001, 0xC0A80002, 0xC0A80003};  /* 192.168.0.1/2/3 */
    int dist[3] = {0};
    for (int i = 0; i < 30; i++) {
        char key[32]; snprintf(key, sizeof(key), "req_%d", i);
        uint32_t srv = tb_router_route_server(&router, key, servers, 3);
        for (int j = 0; j < 3; j++) if (srv == servers[j]) { dist[j]++; break; }
    }
    assert(dist[0] > 0 && dist[1] > 0 && dist[2] > 0);
    printf("[phi-server] dist: %d/%d/%d (all>0): PASS\n", dist[0], dist[1], dist[2]);
    tb_graph_destroy(g);

    /* ── ERL Ledger ── */
    TB_ERLLedger *L = tb_erl_create(lat, NULL);
    tb_erl_append(L, TB_ERL_BRANCH_CREATE, 0, "{\"name\":\"main\"}");
    tb_erl_append(L, TB_ERL_CELL_COMMIT,   0, "{\"domain\":\"task\"}");
    tb_erl_append(L, TB_ERL_TASK_START,    0, "{\"task\":\"test\"}");

    int broken;
    int ok = tb_erl_verify_chain(L, &broken);
    assert(ok && broken == -1);
    printf("\n[ERL] chain valid, %d entries: PASS\n", tb_erl_n_entries(L));
    tb_erl_destroy(L);

    /* ── CognitionTree ── */
    char tmpdir[] = "/tmp/tb_l23_XXXXXX";
    char *td = mkdtemp(tmpdir);
    assert(td);

    TB_CognitionTree *tree = tb_tree_create(lat, "test", td);
    assert(tree);
    printf("\n[tree] created: %d cells, %d branches\n",
           tree->n_cells, tree->n_branches);

    /* Cell commit + hash chain */
    TB_CognitionCell *c1 = tb_tree_cell_commit(tree, 0, "{\"task\":\"auth\"}", "task");
    TB_CognitionCell *c2 = tb_tree_cell_commit(tree, 0, "{\"step\":\"read\"}", "task");
    assert(c1 && c2);
    assert(tb_tree_cell_verify(tree, c1));
    assert(tb_tree_cell_verify(tree, c2));
    assert(memcmp(c2->parent_id, c1->id, 16) == 0);
    printf("[cells] 2 cells committed, hash chain verified: PASS\n");

    /* ERL chain */
    ok = tb_erl_verify_chain(tree->ledger, &broken);
    assert(ok);
    printf("[ERL] chain valid after cells: %d entries PASS\n",
           tb_erl_n_entries(tree->ledger));

    /* Branch fork */
    int b1 = tb_tree_branch_create(tree, 0);
    int b2 = tb_tree_branch_create(tree, 0);
    assert(b1 == 1 && b2 == 2);
    tb_tree_cell_commit(tree, b1, "{\"branch\":\"feature\"}", "task");
    tb_tree_cell_commit(tree, b2, "{\"branch\":\"hotfix\"}", "task");
    printf("[branches] b1=%d b2=%d: PASS\n", b1, b2);

    /* Branch merge */
    int merged = tb_tree_branch_merge(tree, b1, 0);
    assert(merged);
    assert(tree->branches[b1]->merged);
    ok = tb_erl_verify_chain(tree->ledger, &broken);
    assert(ok);
    printf("[merge] b1→main, ERL valid: PASS (%d entries)\n",
           tb_erl_n_entries(tree->ledger));

    /* Epoch advance + KV invalidation */
    TB_KVCache *kv = tb_kvcache_alloc(2, 4, 8, 32, b2, lat->epoch);
    tree->branches[b2]->kv_cache = kv;
    float q[32]={0}, k[32]={0}, v[32]={0.1f}, ao[32];
    tb_attention(q, k, v, kv, 0, 4, 4, 8, ao);
    assert(kv->seq_len == 1);
    int32_t old_ep = lat->epoch;
    tb_tree_epoch_advance(tree, 1);
    assert(lat->epoch == old_ep + 1);
    assert(kv->seq_len == 0 && kv->epoch == lat->epoch);
    ok = tb_erl_verify_chain(tree->ledger, &broken);
    assert(ok);
    printf("[epoch] advance %d→%d, KV cleared, ERL valid: PASS\n",
           old_ep, lat->epoch);

    /* Forward secrecy */
    const char *secret = "trailblaze forward-secret state";
    uint8_t env[256];
    size_t env_len = tb_tree_seal(tree, (const uint8_t*)secret, strlen(secret),
                                  "state", env, sizeof(env));
    assert(env_len == strlen(secret) + 40);
    uint8_t plain[256];
    size_t pl = tb_tree_unseal(tree, env, env_len, "state", plain, sizeof(plain));
    assert(pl == strlen(secret));
    assert(memcmp(plain, secret, pl) == 0);
    printf("[seal] roundtrip: PASS\n");

    tb_tree_epoch_advance(tree, 1);
    size_t bad = tb_tree_unseal(tree, env, env_len, "state", plain, sizeof(plain));
    assert(bad == 0);
    printf("[seal] forward secrecy (stale rejected): PASS\n");

    /* Memory tiers */
    tb_tree_memory_set(tree, "last_task", "analyze auth");
    assert(strcmp(tb_tree_memory_get(tree, "last_task"), "analyze auth") == 0);
    printf("[Tier1] session memory: PASS\n");

    tb_tree_notes_write(tree, "session.md", "# Session\nall operational\n");
    char *note = tb_tree_notes_read(tree, "session.md");
    assert(note && strstr(note, "operational"));
    free(note);
    printf("[Tier2] durable notes: PASS\n");

    /* Tool call recording */
    tb_tree_record_tool_call(tree, b2, "shell_exec", "{\"cmd\":\"echo ok\"}");
    ok = tb_erl_verify_chain(tree->ledger, &broken);
    assert(ok);
    printf("[tool call] ERL valid: PASS (%d entries)\n",
           tb_erl_n_entries(tree->ledger));

    /* Describe */
    char desc[512];
    tb_tree_describe(tree, desc, sizeof(desc));
    printf("\n[describe] %s\n", desc);

    tb_tree_destroy(tree);
    tb_lattice_destroy(lat);

    /* Cleanup temp dir */
    char rmcmd[640]; snprintf(rmcmd, sizeof(rmcmd), "rm -rf %s", td);
    system(rmcmd);

    printf("\n=== Layers 2+3 C PASS ===\n");
    return 0;
}
#endif /* TB_L23_TEST */
