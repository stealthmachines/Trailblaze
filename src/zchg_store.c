/*
 * zchg_store.c  —  HDGL-sql v0.2: Strand-Native Persistent Store
 *
 * v0.2 scaling additions over v0.1:
 *   • Arbitrary strand count (power of 2, 8–256) — finer key distribution
 *   • Dynamic/growable in-memory index — no 4096-slot cap; auto-rehashes
 *   • Shard-aware routing — multi-node key space partitioning
 *   • Per-store write buffers — multiple store instances are independent
 *
 * On-disk format is identical to v0.1 (backward-compatible).
 */

#define _POSIX_C_SOURCE 200809L

#include "zchg_store.h"
#include "zchg_lattice.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

/* ============================================================================
 * Per-strand coalescing write buffer (internal to this translation unit)
 * ============================================================================
 * Frames are packed here until WBUF_FLUSH_COUNT is reached or
 * zchg_store_flush() is called.  In v0.2 this is heap-allocated per-store
 * (stored in store->_wbuf) rather than a static global, so multiple store
 * instances in the same process are fully independent.
 * Raise WBUF_FLUSH_COUNT to e.g. 16 to amortise VFS overhead at the cost of
 * a small durability window. */
#define WBUF_FLUSH_COUNT  1
#define WBUF_MAX_BYTES    (65536 * 4)   /* 256 KiB safety cap per strand */

typedef struct {
    uint8_t  *data;
    size_t    len;
    size_t    cap;
    int       count;    /* frames staged */
} _strand_wbuf_t;

static int _wbuf_flush(int strand_fd, _strand_wbuf_t *wb) {
    if (wb->len == 0) return 0;
    if (lseek(strand_fd, 0, SEEK_END) < 0) return -1;
    ssize_t nw = write(strand_fd, wb->data, wb->len);
    int ok = (nw == (ssize_t)wb->len) ? 0 : -1;
    wb->len   = 0;
    wb->count = 0;
    return ok;
}

static int _wbuf_append(int strand_fd, _strand_wbuf_t *wb,
                         const void *hdr, size_t hdr_len,
                         const void *pay, size_t pay_len) {
    size_t need = hdr_len + pay_len;

    if (wb->len + need > wb->cap) {
        size_t new_cap = wb->cap ? wb->cap * 2 : 4096;
        while (new_cap < wb->len + need) new_cap *= 2;
        if (new_cap > WBUF_MAX_BYTES) {
            if (_wbuf_flush(strand_fd, wb) != 0) return -1;
            new_cap = need * 2;
        }
        uint8_t *p = (uint8_t *)realloc(wb->data, new_cap);
        if (!p) return -1;
        wb->data = p;
        wb->cap  = new_cap;
    }

    memcpy(wb->data + wb->len, hdr, hdr_len);
    wb->len += hdr_len;
    memcpy(wb->data + wb->len, pay, pay_len);
    wb->len += pay_len;
    wb->count++;

    if (wb->count >= WBUF_FLUSH_COUNT)
        return _wbuf_flush(strand_fd, wb);
    return 0;
}

/* ============================================================================
 * Strand naming
 * ============================================================================ */

static const char *_STRAND_NAMES_8[8] = {
    "Point", "Line", "Triangle", "Tetrahedron",
    "Pentachoron", "Hexacross", "Heptacube", "Octacube",
};

/* Build the strand file path for strand s.
 * Strands 0-7 use geometric names; strands 8+ use strand_N.hdgl. */
static void _strand_path(char *out, size_t out_size,
                          const char *store_dir, uint32_t s)
{
    if (s < 8) {
        static const char *names[8] = {
            "point","line","triangle","tetrahedron",
            "pentachoron","hexacross","heptacube","octacube"
        };
        snprintf(out, out_size, "%s/strand_%u_%s.hdgl", store_dir, s, names[s]);
    } else {
        snprintf(out, out_size, "%s/strand_%u.hdgl", store_dir, s);
    }
}

/* ============================================================================
 * EMA fixed-point encoding (24.8 format, range 0–1 → 0–256)
 * ============================================================================ */
#define _W_TO_FP(w)   ((uint32_t)((w) * 256.0))
#define _FP_TO_W(fp)  ((double)(fp) / 256.0)

/* phi_addr split: low 32 bits → authority_ep, high 32 bits → source_ip */
#define _ADDR_LO(a)   ((uint32_t)((a) & 0xFFFFFFFFULL))
#define _ADDR_HI(a)   ((uint32_t)(((a) >> 32) & 0xFFFFFFFFULL))
#define _ADDR_JOIN(lo, hi) (((uint64_t)(hi) << 32) | (uint64_t)(lo))

/* ============================================================================
 * Dynamic hash table helpers
 * ============================================================================ */

/* Fibonacci hash — fast, good distribution */
static uint32_t _slot(uint64_t phi_addr, uint32_t cap) {
    uint64_t h = phi_addr * 0x9e3779b97f4a7c15ULL;
    return (uint32_t)((h >> 32) & (cap - 1));
}

/* Find slot for phi_addr in index[cap]. Returns pointer-to-slot or NULL if
 * the table is completely full (should only happen if cap=0 or caller forgot
 * to rehash — in normal use load factor is kept below 75%). */
static zchg_store_record_t **_index_slot(zchg_store_record_t **index,
                                          uint32_t cap, uint64_t phi_addr)
{
    if (!index || cap == 0) return NULL;
    uint32_t s = _slot(phi_addr, cap);
    for (uint32_t i = 0; i < cap; i++) {
        uint32_t idx = (s + i) & (cap - 1);
        if (index[idx] == NULL)              return &index[idx];
        if (index[idx]->phi_addr == phi_addr) return &index[idx];
    }
    return NULL;
}

static zchg_store_record_t *_index_find(zchg_store_record_t **index,
                                         uint32_t cap, uint64_t phi_addr)
{
    if (!index || cap == 0) return NULL;
    uint32_t s = _slot(phi_addr, cap);
    for (uint32_t i = 0; i < cap; i++) {
        uint32_t idx = (s + i) & (cap - 1);
        if (index[idx] == NULL)               return NULL;
        if (index[idx]->phi_addr == phi_addr) return index[idx];
    }
    return NULL;
}

/* Double the index capacity and re-probe all existing records.
 * Returns ZCHG_OK or ZCHG_ERR_INDEX_OOM. */
static int _index_rehash(zchg_store_t *store) {
    uint32_t new_cap = store->index_cap * 2;
    if (new_cap > ZCHG_STORE_INDEX_CAP_MAX || new_cap == 0) {
        return ZCHG_ERR_INDEX_OOM;
    }

    zchg_store_record_t **new_index =
        (zchg_store_record_t **)calloc(new_cap, sizeof(*new_index));
    if (!new_index) return ZCHG_ERR_INDEX_OOM;

    for (uint32_t i = 0; i < store->index_cap; i++) {
        if (!store->index[i]) continue;
        zchg_store_record_t **slot =
            _index_slot(new_index, new_cap, store->index[i]->phi_addr);
        if (!slot) {
            free(new_index);
            return ZCHG_ERR_INDEX_OOM;
        }
        *slot = store->index[i];
    }

    free(store->index);
    store->index     = new_index;
    store->index_cap = new_cap;
    return ZCHG_OK;
}

/* Check load factor and rehash if needed. Called before every insert. */
static int _index_maybe_rehash(zchg_store_t *store) {
    if (store->index_used * 100 >= store->index_cap * ZCHG_STORE_LOAD_FACTOR_REHASH)
        return _index_rehash(store);
    return ZCHG_OK;
}

/* ============================================================================
 * Strand routing — arbitrary count
 * ============================================================================ */

static uint32_t _strand_of(uint64_t phi_addr, uint32_t strand_count) {
    return (uint32_t)(phi_addr & (strand_count - 1));
}

/* ============================================================================
 * Shard routing
 * ============================================================================ */

uint32_t zchg_store_shard_of(const char *key, uint32_t shard_count) {
    if (!key || shard_count <= 1) return 0;
    uint64_t phi = zchg_compute_phi_tau(key, strlen(key));
    /* High 32 bits of phi_addr for shard routing (low bits used for strand) */
    return (uint32_t)((phi >> 32) % shard_count);
}

/* ============================================================================
 * Payload encoding / decoding (unchanged from v0.1)
 * ============================================================================
 *  Layout: [type_len:1][type:N][ref_len:1][ref:M][json payload:rest]
 */

static uint8_t *_encode_payload(const char *record_type,
                                 const char *lattice_ref,
                                 const char *json, size_t json_len,
                                 size_t *out_len)
{
    size_t type_len = record_type ? strlen(record_type) : 0;
    if (type_len > ZCHG_STORE_TYPE_MAX) type_len = ZCHG_STORE_TYPE_MAX;
    size_t ref_len  = lattice_ref ? strlen(lattice_ref) : 0;
    if (ref_len > 16) ref_len = 16;
    size_t total    = 1 + type_len + 1 + ref_len + json_len;

    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return NULL;

    uint8_t *p = buf;
    *p++ = (uint8_t)type_len;
    if (type_len) { memcpy(p, record_type, type_len); p += type_len; }
    *p++ = (uint8_t)ref_len;
    if (ref_len)  { memcpy(p, lattice_ref, ref_len);  p += ref_len; }
    if (json_len) { memcpy(p, json, json_len); }

    *out_len = total;
    return buf;
}

static int _decode_payload(const uint8_t *buf, size_t buf_len,
                            char *out_type,
                            char *out_ref,
                            char **out_json, size_t *out_json_len)
{
    if (!buf || buf_len < 2) return -1;
    const uint8_t *p   = buf;
    const uint8_t *end = buf + buf_len;

    uint8_t type_len = *p++;
    if (p + type_len >= end) return -1;
    memset(out_type, 0, ZCHG_STORE_TYPE_MAX + 1);
    if (type_len) {
        size_t copy = type_len <= ZCHG_STORE_TYPE_MAX ? type_len : ZCHG_STORE_TYPE_MAX;
        memcpy(out_type, p, copy);
    }
    p += type_len;

    uint8_t ref_len = *p++;
    if (p + ref_len > end) return -1;
    memset(out_ref, 0, ZCHG_STORE_REF_LEN);
    if (ref_len) {
        size_t copy = ref_len <= 16 ? ref_len : 16;
        memcpy(out_ref, p, copy);
    }
    p += ref_len;

    size_t json_len = (size_t)(end - p);
    char *json = NULL;
    if (json_len > 0) {
        json = (char *)malloc(json_len + 1);
        if (!json) return -1;
        memcpy(json, p, json_len);
        json[json_len] = '\0';
    } else {
        json = (char *)calloc(1, 1);
    }
    *out_json     = json;
    *out_json_len = json_len;
    return 0;
}

/* ============================================================================
 * Millisecond timestamp
 * ============================================================================ */

static uint64_t _now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

/* ============================================================================
 * Boot scan — read strand file, reconstruct lattice index
 * ============================================================================ */

static void _boot_scan_strand(zchg_store_t *store, uint32_t strand_id, int fd) {
    if (lseek(fd, 0, SEEK_SET) < 0) return;

    size_t  hdr_size = sizeof(zchg_frame_header_t);
    uint8_t hdr_buf[sizeof(zchg_frame_header_t)];
    uint64_t frame_count = 0;

    for (;;) {
        ssize_t nr = read(fd, hdr_buf, hdr_size);
        if (nr == 0) break;
        if (nr != (ssize_t)hdr_size) break;

        zchg_frame_header_t hdr;
        memcpy(&hdr, hdr_buf, hdr_size);

        if (hdr.version != zchg_FRAME_VERSION || hdr.type != ZCHG_FRAME_STORE) {
            if (hdr.payload_len > 0 && hdr.payload_len < zchg_FRAME_MAX_PAYLOAD)
                lseek(fd, (off_t)hdr.payload_len, SEEK_CUR);
            continue;
        }

        if (hdr.payload_len == 0 || hdr.payload_len > zchg_FRAME_MAX_PAYLOAD)
            continue;

        uint8_t *payload_buf = (uint8_t *)malloc(hdr.payload_len);
        if (!payload_buf) break;

        nr = read(fd, payload_buf, hdr.payload_len);
        if (nr != (ssize_t)hdr.payload_len) { free(payload_buf); break; }

        /* Verify HMAC */
        zchg_frame_t frame;
        memset(&frame, 0, sizeof(frame));
        frame.header      = hdr;
        frame.payload     = payload_buf;
        frame.payload_len = hdr.payload_len;

        if (store->cluster_secret && store->secret_len > 0) {
            if (zchg_hmac_verify_frame(&frame, store->cluster_secret,
                                       store->secret_len) != 0) {
                free(payload_buf);
                continue;
            }
        }

        frame_count++;

        char    rec_type[ZCHG_STORE_TYPE_MAX + 1];
        char    rec_ref[ZCHG_STORE_REF_LEN];
        char   *rec_json     = NULL;
        size_t  rec_json_len = 0;

        if (_decode_payload(payload_buf, hdr.payload_len,
                            rec_type, rec_ref,
                            &rec_json, &rec_json_len) != 0) {
            free(payload_buf);
            continue;
        }
        free(payload_buf);

        uint64_t phi_addr  = _ADDR_JOIN(hdr.authority_ep, hdr.source_ip);
        double   authority = _FP_TO_W(hdr.reserved);

        /* Rehash if needed before inserting */
        if (_index_maybe_rehash(store) != ZCHG_OK) { free(rec_json); continue; }

        zchg_store_record_t **slot =
            _index_slot(store->index, store->index_cap, phi_addr);
        if (!slot) { free(rec_json); continue; }

        if (*slot == NULL) {
            *slot = (zchg_store_record_t *)calloc(1, sizeof(zchg_store_record_t));
            if (!*slot) { free(rec_json); continue; }
            (*slot)->phi_addr  = phi_addr;
            (*slot)->strand_id = strand_id;
            store->strands[strand_id].record_count++;
            store->index_used++;
        } else {
            if ((*slot)->payload) free((*slot)->payload);
            authority = ZCHG_STORE_EMA_ALPHA * 1.0
                        + (1.0 - ZCHG_STORE_EMA_ALPHA) * (*slot)->authority_w;
        }

        memcpy((*slot)->record_type, rec_type, ZCHG_STORE_TYPE_MAX + 1);
        memcpy((*slot)->lattice_ref, rec_ref,  ZCHG_STORE_REF_LEN);
        (*slot)->authority_w = authority;
        (*slot)->payload     = rec_json;
        (*slot)->payload_len = rec_json_len;
        (*slot)->last_ts     = hdr.timestamp;
    }

    store->strands[strand_id].frame_count += frame_count;
}

/* ============================================================================
 * Utility: round up to power of 2
 * ============================================================================ */

static uint32_t _pow2_ceil(uint32_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* ============================================================================
 * zchg_store_open_ex  (full v0.2 implementation)
 * ============================================================================ */

int zchg_store_open_ex(zchg_store_t *store,
                        const char   *store_dir,
                        const char   *cluster_secret,
                        size_t        secret_len,
                        uint32_t      strand_count,
                        uint32_t      index_cap_hint,
                        uint32_t      shard_id,
                        uint32_t      shard_count)
{
    if (!store || !store_dir) return -1;
    memset(store, 0, sizeof(*store));

    /* Resolve and validate strand_count */
    if (strand_count == 0) strand_count = ZCHG_STORE_STRAND_COUNT_DEFAULT;
    strand_count = _pow2_ceil(strand_count);
    if (strand_count > ZCHG_STORE_STRAND_COUNT_MAX) {
        fprintf(stderr, "[HDGL-sql] strand_count %u exceeds max %u\n",
                strand_count, ZCHG_STORE_STRAND_COUNT_MAX);
        return -1;
    }

    /* Resolve and validate index_cap_hint */
    if (index_cap_hint == 0) index_cap_hint = ZCHG_STORE_INDEX_CAP_DEFAULT;
    index_cap_hint = _pow2_ceil(index_cap_hint);
    if (index_cap_hint > ZCHG_STORE_INDEX_CAP_MAX) index_cap_hint = ZCHG_STORE_INDEX_CAP_MAX;

    /* Shard routing */
    if (shard_count == 0) shard_count = 1;
    if (shard_id >= shard_count) {
        fprintf(stderr, "[HDGL-sql] shard_id %u >= shard_count %u\n",
                shard_id, shard_count);
        return -1;
    }

    strncpy(store->store_dir, store_dir, sizeof(store->store_dir) - 1);
    store->cluster_secret = cluster_secret;
    store->secret_len     = secret_len;
    store->strand_count   = strand_count;
    store->shard_id       = shard_id;
    store->shard_count    = shard_count;
    store->index_cap      = index_cap_hint;
    store->index_used     = 0;

    /* Allocate heap arrays */
    store->strand_fd = (int *)calloc(strand_count, sizeof(int));
    store->strands   = (zchg_strand_signal_t *)calloc(strand_count,
                            sizeof(zchg_strand_signal_t));
    store->index     = (zchg_store_record_t **)calloc(index_cap_hint,
                            sizeof(zchg_store_record_t *));
    store->_wbuf     = calloc(strand_count, sizeof(_strand_wbuf_t));

    if (!store->strand_fd || !store->strands ||
        !store->index     || !store->_wbuf) {
        fprintf(stderr, "[HDGL-sql] OOM allocating store structures\n");
        goto fail;
    }

    /* Initialise strand signal metadata */
    for (uint32_t s = 0; s < strand_count; s++) {
        store->strand_fd[s] = -1;
        store->strands[s].strand_id = s;
        if (s < 8)
            strncpy(store->strands[s].strand_name, _STRAND_NAMES_8[s], 31);
        else
            snprintf(store->strands[s].strand_name, 32, "Strand%u", s);
    }

    /* Create store directory if needed */
    mkdir(store_dir, 0755);

    /* Open/create strand files */
    for (uint32_t s = 0; s < strand_count; s++) {
        char path[768];
        _strand_path(path, sizeof(path), store_dir, s);
        int fd = open(path, O_RDWR | O_CREAT, 0644);
        if (fd < 0) {
            fprintf(stderr, "[HDGL-sql] Cannot open strand file %s: %s\n",
                    path, strerror(errno));
            goto fail;
        }
        store->strand_fd[s] = fd;
    }

    /* Boot scan */
    for (uint32_t s = 0; s < strand_count; s++) {
        _boot_scan_strand(store, s, store->strand_fd[s]);
    }

    /* Boot summary */
    fprintf(stderr, "[HDGL-sql] Lattice store: %s  strands=%u  index_cap=%u",
            store_dir, strand_count, store->index_cap);
    if (shard_count > 1)
        fprintf(stderr, "  shard=%u/%u", shard_id, shard_count);
    fprintf(stderr, "\n[HDGL-sql] Strand lattice:");
    for (uint32_t s = 0; s < strand_count; s++) {
        fprintf(stderr, " %s(%llu|w=%.3f)",
                store->strands[s].strand_name,
                (unsigned long long)store->strands[s].record_count,
                store->strands[s].authority_w);
    }
    fprintf(stderr, "\n");

    return 0;

fail:
    free(store->strand_fd); store->strand_fd = NULL;
    free(store->strands);   store->strands   = NULL;
    free(store->index);     store->index     = NULL;
    free(store->_wbuf);     store->_wbuf     = NULL;
    return -1;
}

/* ============================================================================
 * zchg_store_open  (v0.1-compatible wrapper)
 * ============================================================================ */

int zchg_store_open(zchg_store_t *store,
                    const char   *store_dir,
                    const char   *cluster_secret,
                    size_t        secret_len)
{
    return zchg_store_open_ex(store, store_dir, cluster_secret, secret_len,
                               0, 0, 0, 1);
}

/* ============================================================================
 * zchg_store_close
 * ============================================================================ */

void zchg_store_close(zchg_store_t *store) {
    if (!store) return;

    _strand_wbuf_t *wbuf = (_strand_wbuf_t *)store->_wbuf;

    if (wbuf && store->strand_fd) {
        for (uint32_t s = 0; s < store->strand_count; s++) {
            if (store->strand_fd[s] > 0)
                _wbuf_flush(store->strand_fd[s], &wbuf[s]);
        }
        for (uint32_t s = 0; s < store->strand_count; s++) {
            if (wbuf[s].data) { free(wbuf[s].data); wbuf[s].data = NULL; }
            if (store->strand_fd[s] > 0) {
                close(store->strand_fd[s]);
                store->strand_fd[s] = -1;
            }
        }
    }

    if (store->index) {
        for (uint32_t i = 0; i < store->index_cap; i++) {
            if (store->index[i]) {
                if (store->index[i]->payload) free(store->index[i]->payload);
                free(store->index[i]);
                store->index[i] = NULL;
            }
        }
        free(store->index);
        store->index = NULL;
    }

    free(store->strand_fd); store->strand_fd = NULL;
    free(store->strands);   store->strands   = NULL;
    free(store->_wbuf);     store->_wbuf     = NULL;
    store->index_cap  = 0;
    store->index_used = 0;
}

/* ============================================================================
 * zchg_store_flush
 * ============================================================================ */

int zchg_store_flush(zchg_store_t *store) {
    if (!store || !store->_wbuf) return -1;
    _strand_wbuf_t *wbuf = (_strand_wbuf_t *)store->_wbuf;
    int rc = 0;
    for (uint32_t s = 0; s < store->strand_count; s++) {
        if (store->strand_fd[s] > 0)
            if (_wbuf_flush(store->strand_fd[s], &wbuf[s]) != 0) rc = -1;
    }
    return rc;
}

/* ============================================================================
 * zchg_store_put
 * ============================================================================ */

int zchg_store_put(zchg_store_t *store,
                   const char   *key,
                   const char   *record_type,
                   const char   *lattice_ref,
                   const char   *payload,
                   size_t        payload_len)
{
    if (!store || !key || !payload) return ZCHG_ERR;

    /* Shard check */
    if (store->shard_count > 1) {
        uint32_t target = zchg_store_shard_of(key, store->shard_count);
        if (target != store->shard_id) return ZCHG_ERR_WRONG_SHARD;
    }

    uint64_t phi_addr = zchg_compute_phi_tau(key, strlen(key));
    uint32_t strand   = _strand_of(phi_addr, store->strand_count);
    int      fd       = store->strand_fd[strand];
    _strand_wbuf_t *wbuf = &((_strand_wbuf_t *)store->_wbuf)[strand];

    /* EMA continuation from existing record */
    zchg_store_record_t *existing =
        _index_find(store->index, store->index_cap, phi_addr);
    double prev_w = existing ? existing->authority_w : 0.0;
    double new_w  = prev_w < 0.001
                    ? 1.0
                    : ZCHG_STORE_EMA_ALPHA * 1.0
                      + (1.0 - ZCHG_STORE_EMA_ALPHA) * prev_w;

    /* Encode payload */
    size_t   enc_len = 0;
    uint8_t *enc = _encode_payload(record_type, lattice_ref,
                                    payload, payload_len, &enc_len);
    if (!enc) return ZCHG_ERR;

    /* Build frame */
    zchg_frame_t frame;
    memset(&frame, 0, sizeof(frame));
    frame.header.version      = zchg_FRAME_VERSION;
    frame.header.type         = ZCHG_FRAME_STORE;
    frame.header.strand_id    = (uint8_t)(strand & 0xFF);
    frame.header.reserved     = _W_TO_FP(new_w);
    frame.header.authority_ep = _ADDR_LO(phi_addr);
    frame.header.source_ip    = _ADDR_HI(phi_addr);
    frame.header.payload_len  = (uint32_t)enc_len;
    frame.header.timestamp    = _now_ms();
    frame.payload             = enc;
    frame.payload_len         = enc_len;

    if (store->cluster_secret && store->secret_len > 0)
        zchg_hmac_sign_frame(&frame, store->cluster_secret, store->secret_len);

    int rc = _wbuf_append(fd, wbuf,
                           &frame.header, sizeof(frame.header),
                           enc, enc_len);
    free(enc);
    if (rc != 0) return ZCHG_ERR;

    /* Grow index if needed */
    if (_index_maybe_rehash(store) != ZCHG_OK) return ZCHG_ERR_INDEX_OOM;

    /* Upsert in-memory index */
    zchg_store_record_t **slot =
        _index_slot(store->index, store->index_cap, phi_addr);
    if (!slot) return ZCHG_ERR;

    if (*slot == NULL) {
        *slot = (zchg_store_record_t *)calloc(1, sizeof(zchg_store_record_t));
        if (!*slot) return ZCHG_ERR;
        (*slot)->phi_addr  = phi_addr;
        (*slot)->strand_id = strand;
        store->strands[strand].record_count++;
        store->index_used++;
    } else {
        if ((*slot)->payload) free((*slot)->payload);
    }

    memset((*slot)->record_type, 0, sizeof((*slot)->record_type));
    memset((*slot)->lattice_ref, 0, sizeof((*slot)->lattice_ref));
    if (record_type) strncpy((*slot)->record_type, record_type, ZCHG_STORE_TYPE_MAX);
    if (lattice_ref) strncpy((*slot)->lattice_ref, lattice_ref, 16);

    (*slot)->authority_w = new_w;
    (*slot)->payload     = (char *)malloc(payload_len + 1);
    if ((*slot)->payload) {
        memcpy((*slot)->payload, payload, payload_len);
        (*slot)->payload[payload_len] = '\0';
    }
    (*slot)->payload_len = payload_len;
    (*slot)->last_ts     = frame.header.timestamp;

    zchg_strand_signal_t *sig = &store->strands[strand];
    sig->authority_w = ZCHG_STORE_EMA_ALPHA * 1.0
                       + (1.0 - ZCHG_STORE_EMA_ALPHA) * sig->authority_w;
    sig->frame_count++;
    store->total_puts++;
    return ZCHG_OK;
}

/* ============================================================================
 * zchg_store_get / zchg_store_get_by_addr
 * ============================================================================ */

zchg_store_record_t* zchg_store_get(zchg_store_t *store, const char *key) {
    if (!store || !key) return NULL;
    uint64_t phi_addr = zchg_compute_phi_tau(key, strlen(key));
    store->total_gets++;
    return _index_find(store->index, store->index_cap, phi_addr);
}

zchg_store_record_t* zchg_store_get_by_addr(zchg_store_t *store, uint64_t phi_addr) {
    if (!store) return NULL;
    store->total_gets++;
    return _index_find(store->index, store->index_cap, phi_addr);
}

/* ============================================================================
 * zchg_store_phi_addr / zchg_store_shard_of
 * ============================================================================ */

uint64_t zchg_store_phi_addr(const char *key) {
    if (!key) return 0;
    return zchg_compute_phi_tau(key, strlen(key));
}

/* ============================================================================
 * Scan helpers
 * ============================================================================ */

int zchg_store_scan(zchg_store_t *store,
                    void (*cb)(zchg_store_record_t *, void *),
                    void *user)
{
    if (!store || !cb) return -1;
    for (uint32_t i = 0; i < store->index_cap; i++) {
        if (store->index[i]) cb(store->index[i], user);
    }
    return 0;
}

int zchg_store_scan_type(zchg_store_t *store,
                          const char *record_type,
                          void (*cb)(zchg_store_record_t *, void *),
                          void *user)
{
    if (!store || !cb || !record_type) return -1;
    for (uint32_t i = 0; i < store->index_cap; i++) {
        if (store->index[i] &&
            strncmp(store->index[i]->record_type, record_type,
                    ZCHG_STORE_TYPE_MAX) == 0) {
            cb(store->index[i], user);
        }
    }
    return 0;
}

int zchg_store_scan_ref(zchg_store_t *store,
                         uint64_t      parent_phi_addr,
                         void (*cb)(zchg_store_record_t *, void *),
                         void *user)
{
    if (!store || !cb) return -1;
    char ref_hex[17];
    snprintf(ref_hex, sizeof(ref_hex), "%016llx",
             (unsigned long long)parent_phi_addr);
    for (uint32_t i = 0; i < store->index_cap; i++) {
        if (store->index[i] &&
            strncmp(store->index[i]->lattice_ref, ref_hex, 16) == 0) {
            cb(store->index[i], user);
        }
    }
    return 0;
}

/* ============================================================================
 * Strand signal accessors
 * ============================================================================ */

void zchg_store_strand_signals(zchg_store_t *store, zchg_strand_signal_t out[8]) {
    if (!store || !out) return;
    uint32_t n = store->strand_count < 8 ? store->strand_count : 8;
    memcpy(out, store->strands, n * sizeof(zchg_strand_signal_t));
    /* Zero any remainder if strand_count < 8 */
    if (n < 8)
        memset(out + n, 0, (8 - n) * sizeof(zchg_strand_signal_t));
}

int zchg_store_strand_signals_n(zchg_store_t        *store,
                                 zchg_strand_signal_t *out,
                                 uint32_t             *out_count)
{
    if (!store || !out || !out_count) return -1;
    memcpy(out, store->strands, store->strand_count * sizeof(zchg_strand_signal_t));
    *out_count = store->strand_count;
    return 0;
}

