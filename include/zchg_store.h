/*
 * zchg_store.h  —  HDGL-sql v0.2: Strand-Native Persistent Store
 *
 * v0.2 scaling additions
 * ──────────────────────
 * • Arbitrary strand count: configurable at open time (8, 16, 32, 64, 128,
 *   256 …). More strands = finer key distribution + better I/O parallelism.
 *   strand_count must be a power of 2. Default is 8 (backward-compatible).
 *   Strands 0-7 keep geometric names; strands 8+ use strand_N.hdgl.
 *
 * • Dynamic/growable in-memory index: no fixed 4096-slot cap. The index
 *   starts at index_cap_hint (default 4096) and doubles automatically when
 *   the load factor exceeds ZCHG_STORE_LOAD_FACTOR_REHASH (75%). Supports
 *   tens of millions of live records on a single node before any OOM risk.
 *
 * • Shard-aware routing: optional shard_id / shard_count for multi-node
 *   deployments. Each node opens its slice of the key space. put() returns
 *   ZCHG_ERR_WRONG_SHARD (-2) for keys that belong to another shard.
 *   zchg_store_shard_of(key, shard_count) lets callers pre-route.
 *   shard_count=1 disables sharding (all keys accepted).
 *
 * • Write buffers are per-store (not a static global), so multiple
 *   zchg_store_t instances in the same process are fully independent.
 *
 * v0.1 compatibility
 * ──────────────────
 * zchg_store_open() still works unchanged — it calls zchg_store_open_ex()
 * with defaults (strand_count=8, index_cap=4096, shard_id=0, shard_count=1).
 * v0.2 can open v0.1 store directories as-is.
 *
 * Architecture (unchanged from v0.1)
 * ────────────────────────────────────
 * • Append-only binary strand files, one per strand.
 * • Every record is a zchg_frame_t (HMAC-SHA256 signed).
 * • Addressing is phi-tau geometric; strand = phi_addr & (strand_count-1).
 * • authority_w is an analog EMA signal stored as 24.8 fixed-point.
 * • Boot-scan rebuilds the in-memory index from on-disk frames.
 *
 * Strand file layout (per file)
 * ─────────────────────────────
 *   [zchg_frame_header_t][payload_len bytes payload]  (repeat, append-only)
 *
 * Payload encoding
 * ────────────────
 *   [uint8_t type_len][type_len bytes record_type]
 *   [uint8_t ref_len] [ref_len bytes lattice_ref hex, or 0 if root]
 *   [remaining bytes: raw JSON payload]
 */

#ifndef ZCHG_STORE_H
#define ZCHG_STORE_H

#include "zchg_core.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/* Frame type for store records */
#define ZCHG_FRAME_STORE                0x08

/* Strand count — configurable at open time.  Must be a power of 2. */
#define ZCHG_STORE_STRAND_COUNT_DEFAULT 8
#define ZCHG_STORE_STRAND_COUNT_MAX     256

/* In-memory index — grows automatically; these are defaults / ceiling. */
#define ZCHG_STORE_INDEX_CAP_DEFAULT    4096
#define ZCHG_STORE_INDEX_CAP_MAX        (1u << 24)   /* 16 M slots */

/* Rehash when occupied slots exceed this percentage of capacity. */
#define ZCHG_STORE_LOAD_FACTOR_REHASH   75

/* EMA alpha — mirrors zchg_lattice.c ZCHG_EMA_ALPHA */
#define ZCHG_STORE_EMA_ALPHA            0.3

/* Max record_type length (excluding null) */
#define ZCHG_STORE_TYPE_MAX             31

/* lattice_ref is a 16-char hex phi_addr + null */
#define ZCHG_STORE_REF_LEN              17

/* Return codes */
#define ZCHG_OK                 0
#define ZCHG_ERR                (-1)
#define ZCHG_ERR_WRONG_SHARD    (-2)    /* key belongs to a different shard */
#define ZCHG_ERR_INDEX_OOM      (-3)    /* index rehash failed (OOM) */

/* ============================================================================
 * Record (in-memory representation of the latest frame for a phi_addr)
 * ============================================================================ */

typedef struct zchg_store_record {
    uint64_t    phi_addr;                        /* Geometric address (phi-tau hash) */
    uint32_t    strand_id;                       /* Strand (0 .. strand_count-1) */
    char        record_type[ZCHG_STORE_TYPE_MAX + 1];
    char        lattice_ref[ZCHG_STORE_REF_LEN]; /* Parent phi_addr hex, or "" */
    double      authority_w;                     /* EMA authority signal [0.0, 1.0] */
    char       *payload;                         /* Heap-allocated JSON payload */
    size_t      payload_len;
    uint64_t    last_ts;                         /* Milliseconds since epoch */
    struct zchg_store_record *_next;             /* Reserved (open-addressing) */
} zchg_store_record_t;

/* ============================================================================
 * Per-Strand Analog Signal
 * ============================================================================ */

typedef struct {
    uint32_t    strand_id;
    char        strand_name[32];
    double      authority_w;        /* EMA of write activity on this strand */
    uint64_t    record_count;       /* Unique phi_addrs in this strand */
    uint64_t    frame_count;        /* Total frames appended (inc. history) */
} zchg_strand_signal_t;

/* ============================================================================
 * Store  (v0.2 — all fixed arrays replaced with heap-allocated pointers)
 * ============================================================================ */

typedef struct {
    char                    store_dir[512];

    /* Strand file descriptors — heap-allocated, strand_count entries */
    int                    *strand_fd;

    /* In-memory lattice index — heap-allocated, index_cap entries, grows */
    zchg_store_record_t   **index;
    uint32_t                index_cap;      /* current capacity (power of 2) */
    uint32_t                index_used;     /* occupied slots */

    /* Per-strand signals — heap-allocated, strand_count entries */
    zchg_strand_signal_t   *strands;

    /* Scaling parameters */
    uint32_t                strand_count;   /* 8, 16, 32 … 256 (power of 2) */
    uint32_t                shard_id;       /* 0-based shard index for this node */
    uint32_t                shard_count;    /* 1 = no sharding */

    const char             *cluster_secret;
    size_t                  secret_len;
    uint64_t                total_puts;
    uint64_t                total_gets;

    /* Internal write-coalescing buffers (opaque to callers) */
    void                   *_wbuf;          /* _strand_wbuf_t[], strand_count */
} zchg_store_t;

/* ============================================================================
 * API
 * ============================================================================ */

/*
 * zchg_store_open — v0.1-compatible open. Defaults: 8 strands, 4096-slot
 *                   index, no sharding.  Calls zchg_store_open_ex internally.
 * Returns 0 on success, -1 on error.
 */
int  zchg_store_open(zchg_store_t *store,
                     const char   *store_dir,
                     const char   *cluster_secret,
                     size_t        secret_len);

/*
 * zchg_store_open_ex — full v0.2 open.
 *
 *   strand_count   : number of strand files to create/open (0 = default 8).
 *                    Must be a power of 2, max 256.
 *   index_cap_hint : initial in-memory index capacity (0 = default 4096).
 *                    Must be a power of 2.  Grows automatically.
 *   shard_id       : 0-based index of this node's shard.
 *   shard_count    : total number of shards (1 = no sharding / accept all).
 *
 * Shard routing: zchg_store_put() returns ZCHG_ERR_WRONG_SHARD (-2) for
 * keys that hash to a different shard.  Use zchg_store_shard_of() to
 * pre-check before calling put.
 */
int  zchg_store_open_ex(zchg_store_t *store,
                         const char   *store_dir,
                         const char   *cluster_secret,
                         size_t        secret_len,
                         uint32_t      strand_count,
                         uint32_t      index_cap_hint,
                         uint32_t      shard_id,
                         uint32_t      shard_count);

/*
 * zchg_store_close — flush write buffers, close strand fds, free all memory.
 */
void zchg_store_close(zchg_store_t *store);

/*
 * zchg_store_flush — flush coalescing write buffer for all strands to disk.
 */
int  zchg_store_flush(zchg_store_t *store);

/*
 * zchg_store_put — write a record into the lattice.
 * Returns ZCHG_OK (0), ZCHG_ERR (-1), or ZCHG_ERR_WRONG_SHARD (-2).
 */
int  zchg_store_put(zchg_store_t *store,
                    const char   *key,
                    const char   *record_type,
                    const char   *lattice_ref,
                    const char   *payload,
                    size_t        payload_len);

/*
 * zchg_store_get — O(1) lookup of latest record by logical key.
 * Returns pointer to in-memory record (do NOT free), or NULL if not found.
 */
zchg_store_record_t* zchg_store_get(zchg_store_t *store, const char *key);

/*
 * zchg_store_get_by_addr — lookup by phi_addr directly.
 */
zchg_store_record_t* zchg_store_get_by_addr(zchg_store_t *store, uint64_t phi_addr);

/*
 * zchg_store_scan — iterate every live record in the lattice index.
 */
int  zchg_store_scan(zchg_store_t *store,
                     void (*cb)(zchg_store_record_t *rec, void *user),
                     void *user);

/*
 * zchg_store_scan_type — iterate records matching a record_type.
 */
int  zchg_store_scan_type(zchg_store_t *store,
                           const char   *record_type,
                           void (*cb)(zchg_store_record_t *rec, void *user),
                           void *user);

/*
 * zchg_store_scan_ref — iterate records whose lattice_ref matches parent.
 */
int  zchg_store_scan_ref(zchg_store_t *store,
                          uint64_t      parent_phi_addr,
                          void (*cb)(zchg_store_record_t *rec, void *user),
                          void *user);

/*
 * zchg_store_strand_signals — fill out[8] with per-strand EMA signals.
 * For backward compat: if strand_count < 8 the remaining slots are zeroed;
 * if strand_count > 8 only the first 8 entries are returned.
 * Use zchg_store_strand_signals_n to get all strand_count entries.
 */
void zchg_store_strand_signals(zchg_store_t *store, zchg_strand_signal_t out[8]);

/*
 * zchg_store_strand_signals_n — fill out[] with all strand_count entries.
 * *out_count is set to the number of entries written.
 * out must have room for at least store->strand_count entries.
 */
int  zchg_store_strand_signals_n(zchg_store_t        *store,
                                  zchg_strand_signal_t *out,
                                  uint32_t             *out_count);

/*
 * zchg_store_phi_addr — compute phi-tau address for a logical key.
 */
uint64_t zchg_store_phi_addr(const char *key);

/*
 * zchg_store_shard_of — compute which shard owns a key.
 * Returns a value in [0, shard_count).  Use before put() to route keys.
 */
uint32_t zchg_store_shard_of(const char *key, uint32_t shard_count);

#ifdef __cplusplus
}
#endif

#endif /* ZCHG_STORE_H */
