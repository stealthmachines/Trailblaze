/*
 * zchg_core.h - ZCHG v0.6 Core Structures & Constants
 *
 * Pure C implementation of Hypergeometric Distributed Geometry Layer
 * End-to-end zchg://: strand routing, phi-spiral geometry, async I/O transport
 *
 * Performance targets:
 * - Throughput: 200K+ req/sec
 * - Latency P99: <1ms
 * - Memory: <10MB per 10K concurrent
 * - Concurrency: 500K+ connections
 */

#ifndef ZCHG_CORE_H
#define ZCHG_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <sys/types.h>

/* ============================================================================
 * Frame Protocol (Binary, 52 bytes minimum)
 * ============================================================================ */

#define zchg_FRAME_VERSION          1
#define zchg_FRAME_HEADER_SIZE      52      /* Fixed header size in bytes */
#define zchg_FRAME_MAX_PAYLOAD      (1024 * 1024)  /* 1MB max payload */

/* Compatibility aliases for mixed macro casing in v0.6-c sources */
#define ZCHG_FRAME_VERSION          zchg_FRAME_VERSION
#define ZCHG_FRAME_HEADER_SIZE      zchg_FRAME_HEADER_SIZE
#define ZCHG_FRAME_MAX_PAYLOAD      zchg_FRAME_MAX_PAYLOAD

/* Frame types */
typedef enum {
    zchg_FRAME_INFO       = 0x01,
    zchg_FRAME_GOSSIP     = 0x02,
    zchg_FRAME_FETCH      = 0x03,
    zchg_FRAME_HEALTH     = 0x04,
    zchg_FRAME_FILESWAP   = 0x05,
    zchg_FRAME_ACK        = 0x06,
    zchg_FRAME_ERROR      = 0x07
} zchg_frame_type_t;

/* Frame header (binary packed, no padding) */
typedef struct {
    uint8_t     version;           /* Frame format version */
    uint8_t     type;              /* Frame type (zchg_frame_type_t) */
    uint32_t    strand_id;         /* Target strand (0-7 = A-H) */
    uint32_t    reserved;          /* Reserved for future use */
    uint32_t    authority_ep;      /* Authority endpoint hash */
    uint32_t    source_ip;         /* Source IP (network byte order) */
    uint32_t    payload_len;       /* Payload length in bytes */
    uint64_t    timestamp;         /* Milliseconds since epoch */
    uint8_t     hmac[32];          /* HMAC-SHA256 (20 bytes reserved in header) */
} __attribute__((packed)) zchg_frame_header_t;

/* Frame (header + payload) */
typedef struct {
    zchg_frame_header_t header;
    uint8_t            *payload;
    size_t              payload_len;
    time_t              created_at;
} zchg_frame_t;

/* ============================================================================
 * Strand Geometry (Phi-Spiral, 8 Strands)
 * ============================================================================ */

#define zchg_STRAND_COUNT           8
#define zchg_STRAND_A               0
#define zchg_STRAND_B               1
#define zchg_STRAND_C               2
#define zchg_STRAND_D               3
#define zchg_STRAND_E               4
#define zchg_STRAND_F               5
#define zchg_STRAND_G               6
#define zchg_STRAND_H               7

/* Strand names (Point, Line, Triangle, Tetrahedron, Pentachoron, etc.) */
static const char *zchg_STRAND_NAMES[] __attribute__((unused)) = {
    "Point", "Line", "Triangle", "Tetrahedron",
    "Pentachoron", "Hexacross", "Heptacube", "Octacube"
};

/* Phi-spiral parameters */
#define zchg_PHI                    1.618033988749894848204586834365638
#define zchg_SPIRAL_PERIOD          8       /* One rotation = 8 strands */

#define ZCHG_STRAND_COUNT           zchg_STRAND_COUNT
#define ZCHG_PHI                    zchg_PHI
#define ZCHG_SPIRAL_PERIOD          zchg_SPIRAL_PERIOD

/* Strand weight (EMA-based) */
typedef struct {
    uint32_t    strand_id;
    double      latency_ema;        /* Exponential moving average latency (ms) */
    double      storage_available;  /* Available storage (bytes) */
    uint32_t    authority_weight;   /* Phi-weighted authority (1-100) */
    time_t      last_update;
} zchg_strand_weight_t;

/* ============================================================================
 * Phi-Tau Routing (Path → Strand Mapping)
 * ============================================================================ */

/* Phi-tau hash: deterministic path → strand routing */
typedef struct {
    char       *path;               /* Logical file/request path */
    uint64_t    hash_value;         /* PHI_TAU hash */
    uint8_t     strand_id;          /* Determined strand (0-7) */
    uint32_t    authority_node;     /* Best authority for this strand */
} zchg_phi_tau_route_t;

/* Cache entry for O(1) phi_tau lookups */
typedef struct {
    char       *path;
    uint8_t     strand_id;
    uint32_t    authority_node;
    time_t      cached_at;
    uint32_t    hit_count;          /* For cache statistics */
} zchg_routing_cache_entry_t;

/* ============================================================================
 * Cluster State (Lattice)
 * ============================================================================ */

#define zchg_CLUSTER_FINGERPRINT_SIZE   4   /* 32-bit fingerprint */
#define zchg_MAX_PEERS                  256

#define ZCHG_CLUSTER_FINGERPRINT_SIZE   zchg_CLUSTER_FINGERPRINT_SIZE
#define ZCHG_MAX_PEERS                  zchg_MAX_PEERS

/* Per-peer state */
typedef struct {
    uint32_t    ip_addr;            /* Peer IP (network byte order) */
    uint16_t    port;               /* Peer port */
    zchg_strand_weight_t strands[zchg_STRAND_COUNT];
    uint32_t    cluster_fingerprint;
    time_t      last_gossip_in;
    time_t      last_gossip_out;
    uint32_t    failed_checks;
    int         is_healthy;
} zchg_peer_t;

/* Local lattice state */
typedef struct {
    uint32_t    local_ip;           /* This node's IP */
    uint16_t    port;               /* This node's port (8090) */
    zchg_peer_t peers[zchg_MAX_PEERS];
    uint32_t    peer_count;
    zchg_strand_weight_t my_strands[zchg_STRAND_COUNT];
    uint32_t    cluster_fingerprint;
    uint64_t    cycle_number;
    time_t      last_cycle;
} zchg_lattice_t;

/* ============================================================================
 * Connection Pool (Per-Peer)
 * ============================================================================ */

#define zchg_MAX_POOL_SIZE          32
#define zchg_KEEP_ALIVE_TTL         60.0    /* Seconds */
#define zchg_POOL_REUSE_LIMIT       64      /* Requests per connection */

#define ZCHG_MAX_POOL_SIZE          zchg_MAX_POOL_SIZE
#define ZCHG_KEEP_ALIVE_TTL         zchg_KEEP_ALIVE_TTL
#define ZCHG_POOL_REUSE_LIMIT       zchg_POOL_REUSE_LIMIT

/* Pooled connection state */
typedef struct {
    int         fd;                 /* Socket file descriptor */
    uint32_t    peer_ip;
    uint16_t    peer_port;
    time_t      created_at;
    time_t      last_used;
    uint32_t    request_count;      /* Requests on this connection */
    uint32_t    error_count;        /* Consecutive errors */
    int         is_valid;           /* Connection is usable */
} zchg_pooled_conn_t;

/* Per-peer connection pool */
typedef struct {
    uint32_t    peer_ip;
    zchg_pooled_conn_t connections[zchg_MAX_POOL_SIZE];
    uint32_t    conn_count;
    uint32_t    total_reused;
    uint32_t    total_new;
    time_t      created_at;
} zchg_connection_pool_t;

/* ============================================================================
 * Frame Pool (Object Reuse)
 * ============================================================================ */

#define zchg_FRAME_POOL_SIZE        1024

typedef struct {
    zchg_frame_t    frames[zchg_FRAME_POOL_SIZE];
    uint8_t         in_use[zchg_FRAME_POOL_SIZE];
    uint32_t        reused_count;
    uint32_t        allocated_count;
} zchg_frame_pool_t;

/* ============================================================================
 * Transport Server State
 * ============================================================================ */

typedef struct {
    int         listen_fd;          /* Server listening socket */
    uint32_t    local_ip;
    uint16_t    port;
    char       *cluster_secret;     /* For HMAC signing */
    size_t      secret_len;

    /* Connection management */
    zchg_connection_pool_t *peer_pools;
    uint32_t    pool_count;

    /* Frame management */
    zchg_frame_pool_t frame_pool;

    /* Routing cache */
    zchg_routing_cache_entry_t *route_cache;
    uint32_t    cache_size;
    uint32_t    cache_hits;
    uint32_t    cache_misses;

    /* Cluster state */
    zchg_lattice_t lattice;

    /* Metrics */
    uint64_t    total_frames_sent;
    uint64_t    total_frames_recv;
    uint64_t    total_bytes_sent;
    uint64_t    total_bytes_recv;
    uint64_t    active_connections;
    uint64_t    total_connections;
    time_t      started_at;
} zchg_transport_server_t;

/* ============================================================================
 * Gossip Protocol (Binary Encoded)
 * ============================================================================ */

#define zchg_GOSSIP_PORT            8090
#define zchg_GOSSIP_INTERVAL        30      /* Seconds between gossip cycles */

#define ZCHG_GOSSIP_PORT            zchg_GOSSIP_PORT
#define ZCHG_GOSSIP_INTERVAL        zchg_GOSSIP_INTERVAL

/* Gossip message (packed binary, ~16 bytes) */
typedef struct {
    uint32_t    source_ip;
    uint8_t     strand_weights[8];          /* Phi-weighted authority per strand */
    uint32_t    storage_available;
    uint32_t    cluster_fingerprint;
} __attribute__((packed)) zchg_gossip_msg_t;

/* ============================================================================
 * Fileswap (Strand-Addressed Distributed FS)
 * ============================================================================ */

#define zchg_FILESWAP_ROOT          "/opt/zchg_swap"
#define zchg_FILESWAP_MAX_SIZE_GB   7       /* Max fileswap size */

#define ZCHG_FILESWAP_ROOT          zchg_FILESWAP_ROOT
#define ZCHG_FILESWAP_MAX_SIZE_GB   zchg_FILESWAP_MAX_SIZE_GB

/* Replay window is shared by frame/transport paths. */
#define zchg_REPLAY_WINDOW_SEC      30
#define ZCHG_REPLAY_WINDOW_SEC      zchg_REPLAY_WINDOW_SEC

/* File route (path → strand → authority node) */
typedef struct {
    char       *logical_path;
    uint8_t     strand_id;
    uint32_t    authority_node_ip;
    char       *physical_path;      /* Local cache path */
    time_t      cached_at;
    uint64_t    file_size;
} zchg_file_route_t;

/* ============================================================================
 * Configuration (from environment / site_config.json)
 * ============================================================================ */

typedef struct {
    char       *local_node_ip;
    char       *peer_ips[zchg_MAX_PEERS];
    uint32_t    peer_count;
    char       *cluster_secret;
    char       *primary_domain;
    char       *fileswap_root;
    int         dry_run;
    int         simulation_mode;
} zchg_config_t;

/* ============================================================================
 * Function Declarations
 * ============================================================================ */

/* Core initialization */
zchg_transport_server_t* zchg_server_create(zchg_config_t *cfg);
void zchg_server_destroy(zchg_transport_server_t *server);

/* Phi-tau routing */
uint8_t zchg_compute_strand_id(const char *path);
uint32_t zchg_compute_phi_tau_hash(const char *path);

/* Frame operations */
zchg_frame_t* zchg_frame_alloc(zchg_frame_pool_t *pool);
void zchg_frame_free(zchg_frame_pool_t *pool, zchg_frame_t *frame);
int zchg_frame_serialize(zchg_frame_t *frame, uint8_t **out_buf, size_t *out_len);
int zchg_frame_deserialize(uint8_t *buf, size_t len, zchg_frame_t *out_frame);

/* Connection pooling */
int zchg_pool_get_connection(zchg_connection_pool_t *pool, int *out_fd);
void zchg_pool_return_connection(zchg_connection_pool_t *pool, int fd);
void zchg_pool_invalidate_connection(zchg_connection_pool_t *pool, int fd);

/* Lattice / cluster state */
void zchg_lattice_update_strand_weight(zchg_lattice_t *lattice, uint8_t strand_id, double latency_ms);
uint32_t zchg_lattice_compute_fingerprint(zchg_lattice_t *lattice);
uint32_t zchg_lattice_get_authority(zchg_lattice_t *lattice, uint8_t strand_id);

/* HMAC / security */
int zchg_hmac_sign_frame(zchg_frame_t *frame, const char *secret, size_t secret_len);
int zchg_hmac_verify_frame(zchg_frame_t *frame, const char *secret, size_t secret_len);

#endif /* zchg_CORE_H */
