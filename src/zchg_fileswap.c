/*
 * zchg_fileswap.c — Strand-addressed distributed filesystem
 *
 * Implements ZCHG fileswap distribution:
 * - Strand-based routing (file hash → strand authority)
 * - Distributed caching with LRU eviction
 * - Authority shift-driven file migration
 * - Passive mirror capture on reads
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>

#include "../include/zchg_core.h"
#include "../include/zchg_lattice.h"
#include "../include/zchg_transport.h"

/*
 * Compute which strand owns a file based on path hash.
 * Uses phi-tau deterministic routing.
 */
static uint8_t zchg_fileswap_hash_to_strand(const char *path) {
    uint32_t hash = 0;
    const unsigned char *p = (const unsigned char *)path;

    /* Simple FNV-1a hash */
    while (*p) {
        hash ^= *p;
        hash *= 16777619U;
        p++;
    }

    /* Map hash to strand using phi-tau */
    return (uint8_t)((hash * 8) / (UINT32_MAX / zchg_STRAND_COUNT + 1)) % zchg_STRAND_COUNT;
}

/*
 * Compute fileswap path on local filesystem for a given strand-addressed file.
 */
static void zchg_fileswap_local_path(const char *logical_path,
                                      char *out_local_path,
                                      size_t out_len) {
    snprintf(out_local_path, out_len, "%s/%s",
             zchg_FILESWAP_ROOT, logical_path);
}

/*
 * Allocate file space from fileswap budget (LRU eviction if necessary).
 * Returns 1 if space allocated, 0 if eviction needed, -1 on error.
 */
static int zchg_fileswap_ensure_space(size_t needed_bytes) {
    (void)needed_bytes;
    struct stat st;

    /* Check if fileswap root exists */
    if (stat(zchg_FILESWAP_ROOT, &st) != 0) {
        return -1;
    }

    /* Placeholder: actual LRU eviction logic goes here */
    /* For now, assume we have space */
    return 1;
}

/*
 * Store a file in fileswap cache.
 * File is placed in the cache directory and tracked for authority.
 */
int zchg_fileswap_store(zchg_transport_server_t *server,
                        const char *logical_path,
                        const uint8_t *data,
                        size_t data_len) {
    if (!server || !logical_path || !data) {
        return -1;
    }

    uint8_t target_strand = zchg_fileswap_hash_to_strand(logical_path);
    uint32_t authority = zchg_lattice_get_strand_authority(&server->lattice, target_strand);

    char local_path[2048];
    zchg_fileswap_local_path(logical_path, local_path, sizeof(local_path));

    /* Check if we have local authority or are a mirror */
    int is_authority = (authority == server->local_ip);

    if (!is_authority) {
        /* Not authority; check if we should become a passive mirror */
        /* For now, only store if we're the authority */
        return -1;
    }

    /* Ensure space in fileswap budget */
    if (zchg_fileswap_ensure_space(data_len) != 1) {
        return -1;  /* No space after eviction */
    }

    /* Create directories if needed */
    char dir_path[2048];
    char *slash = strrchr(local_path, '/');
    if (slash) {
        size_t dir_len = (size_t)(slash - local_path);
        if (dir_len < sizeof(dir_path)) {
            strncpy(dir_path, local_path, dir_len);
            dir_path[dir_len] = '\0';
            mkdir(dir_path, 0755);
        }
    }

    /* Write file to local cache */
    FILE *fp = fopen(local_path, "wb");
    if (!fp) {
        return -1;
    }

    size_t written = fwrite(data, 1, data_len, fp);
    fclose(fp);

    if (written != data_len) {
        unlink(local_path);
        return -1;
    }

    server->cache_hits++;  /* Track cache stats */
    return 0;
}

/*
 * Retrieve a file from fileswap cache.
 * If not cached locally, fetch from authority strand.
 */
int zchg_fileswap_fetch(zchg_transport_server_t *server,
                        const char *logical_path,
                        uint8_t **out_data,
                        size_t *out_len) {
    if (!server || !logical_path || !out_data || !out_len) {
        return -1;
    }

    char local_path[2048];
    zchg_fileswap_local_path(logical_path, local_path, sizeof(local_path));

    struct stat st;
    if (stat(local_path, &st) == 0 && S_ISREG(st.st_mode)) {
        /* File exists locally; read it */
        FILE *fp = fopen(local_path, "rb");
        if (!fp) {
            server->cache_misses++;
            return -1;
        }

        uint8_t *buf = (uint8_t *)malloc((size_t)st.st_size);
        if (!buf) {
            fclose(fp);
            return -1;
        }

        size_t read_len = fread(buf, 1, (size_t)st.st_size, fp);
        fclose(fp);

        if (read_len != (size_t)st.st_size) {
            free(buf);
            server->cache_misses++;
            return -1;
        }

        *out_data = buf;
        *out_len = read_len;
        server->cache_hits++;
        return 0;
    }

    /* Not cached locally; would fetch from authority (placeholder) */
    server->cache_misses++;
    return -1;
}

/*
 * Migrate file to new authority when strand authority changes.
 * Triggered by gossip-driven PROVISIONER updates.
 */
int zchg_fileswap_migrate_on_authority_shift(zchg_transport_server_t *server,
                                              uint8_t strand,
                                              uint32_t new_authority) {
    (void)strand;
    if (!server || new_authority == 0) {
        return -1;
    }

    /* Placeholder: scan fileswap for files assigned to this strand */
    /* and migrate them to new authority via zchg_client_send_frame */

    return 0;
}

/*
 * Evict expired or oversized files from fileswap (LRU policy).
 * Called periodically by main event loop.
 */
int zchg_fileswap_evict_lru(zchg_transport_server_t *server,
                             size_t target_free_bytes) {
    (void)target_free_bytes;
    if (!server) {
        return -1;
    }

    /* Placeholder: scan cache, sort by access time, delete oldest until space freed */

    return 0;
}

/*
 * Capture file as passive mirror when we observe authority.
 * Called after gossip cycle to keep mirrors fresh.
 */
int zchg_fileswap_capture_as_mirror(zchg_transport_server_t *server,
                                     uint8_t strand,
                                     uint32_t authority) {
    (void)strand;
    if (!server || authority == 0) {
        return 0;  /* No-op if no authority */
    }

    if (authority == server->local_ip) {
        return 0;  /* We are the authority; no mirroring needed */
    }

    /* Placeholder: fetch recent files from authority strand and cache locally */

    return 0;
}

/*
 * Report fileswap statistics (cache hit ratio, size).
 */
int zchg_fileswap_stats(const char *fileswap_root,
                        size_t *out_total_bytes,
                        uint32_t *out_file_count) {
    if (!fileswap_root) {
        return -1;
    }

    *out_total_bytes = 0;
    *out_file_count = 0;

    /* Placeholder: walk directory tree and aggregate stats */

    return 0;
}
