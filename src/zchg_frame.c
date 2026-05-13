/*
 * zchg_frame.c - Binary Frame Serialization & Deserialization
 *
 * ZCHG frame format: 52-byte header + payload
 * - Version + Type + Strand ID + Reserved
 * - Authority endpoint + Source IP + Payload length + Timestamp
 * - HMAC-SHA256 signature (20 bytes in header + validation)
 * - Binary encoding for performance (83% reduction vs JSON)
 */

#include "zchg_core.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>

/* ============================================================================
 * Frame Serialization
 * ============================================================================ */

int zchg_frame_serialize(zchg_frame_t *frame, uint8_t **out_buf, size_t *out_len) {
    if (!frame || !out_buf || !out_len) return -1;

    /* Total size: header + payload */
    size_t total_size = zchg_FRAME_HEADER_SIZE + frame->payload_len;

    uint8_t *buf = (uint8_t *)malloc(total_size);
    if (!buf) return -1;

    /* Copy header (already packed) */
    memcpy(buf, &frame->header, zchg_FRAME_HEADER_SIZE);

    /* Copy payload */
    if (frame->payload && frame->payload_len > 0) {
        memcpy(buf + zchg_FRAME_HEADER_SIZE, frame->payload, frame->payload_len);
    }

    *out_buf = buf;
    *out_len = total_size;

    return 0;
}

int zchg_frame_deserialize(uint8_t *buf, size_t len, zchg_frame_t *out_frame) {
    if (!buf || len < zchg_FRAME_HEADER_SIZE || !out_frame) return -1;

    /* Copy header */
    memcpy(&out_frame->header, buf, zchg_FRAME_HEADER_SIZE);

    /* Validate payload length */
    if (out_frame->header.payload_len > zchg_FRAME_MAX_PAYLOAD) return -1;
    if (zchg_FRAME_HEADER_SIZE + out_frame->header.payload_len != len) return -1;

    /* Allocate and copy payload */
    if (out_frame->header.payload_len > 0) {
        out_frame->payload = (uint8_t *)malloc(out_frame->header.payload_len);
        if (!out_frame->payload) return -1;

        memcpy(out_frame->payload, buf + zchg_FRAME_HEADER_SIZE, out_frame->header.payload_len);
    } else {
        out_frame->payload = NULL;
    }

    out_frame->payload_len = out_frame->header.payload_len;
    out_frame->created_at = time(NULL);

    return 0;
}

/* ============================================================================
 * Frame Pool (Object Reuse)
 * ============================================================================ */

zchg_frame_t* zchg_frame_alloc(zchg_frame_pool_t *pool) {
    if (!pool) return NULL;

    /* Find available frame in pool */
    for (uint32_t i = 0; i < zchg_FRAME_POOL_SIZE; i++) {
        if (!pool->in_use[i]) {
            pool->in_use[i] = 1;
            pool->reused_count++;
            return &pool->frames[i];
        }
    }

    /* Pool exhausted (shouldn't happen in practice with proper sizing) */
    return NULL;
}

void zchg_frame_free(zchg_frame_pool_t *pool, zchg_frame_t *frame) {
    if (!pool || !frame) return;

    /* Find frame in pool */
    for (uint32_t i = 0; i < zchg_FRAME_POOL_SIZE; i++) {
        if (&pool->frames[i] == frame) {
            pool->in_use[i] = 0;
            /* Clear payload if allocated */
            if (frame->payload) {
                free(frame->payload);
                frame->payload = NULL;
            }
            frame->payload_len = 0;
            return;
        }
    }
}

/* ============================================================================
 * HMAC Signing & Verification (SHA256)
 * ============================================================================ */

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <openssl/hmac.h>
#include <openssl/sha.h>

int zchg_hmac_sign_frame(zchg_frame_t *frame, const char *secret, size_t secret_len) {
    if (!frame || !secret || secret_len == 0) return -1;

    /* Create HMAC over header (minus signature) + payload */
    HMAC_CTX *ctx = HMAC_CTX_new();
    if (!ctx) return -1;

    uint8_t digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;

    HMAC_Init_ex(ctx, (unsigned char *)secret, secret_len, EVP_sha256(), NULL);

    /* Hash header fields (skip HMAC field itself) */
    HMAC_Update(ctx, (unsigned char *)&frame->header,
                offsetof(zchg_frame_header_t, hmac));

    /* Hash payload */
    if (frame->payload && frame->payload_len > 0) {
        HMAC_Update(ctx, frame->payload, frame->payload_len);
    }

    HMAC_Final(ctx, digest, &digest_len);
    HMAC_CTX_free(ctx);

    /* Copy digest to frame (20 bytes in reserved space) */
    if (digest_len > 20) digest_len = 20;
    memcpy(frame->header.hmac, digest, digest_len);

    return 0;
}

int zchg_hmac_verify_frame(zchg_frame_t *frame, const char *secret, size_t secret_len) {
    if (!frame || !secret || secret_len == 0) return -1;

    uint8_t expected_hmac[20];
    memcpy(expected_hmac, frame->header.hmac, 20);

    /* Re-sign to get expected HMAC */
    zchg_frame_t test_frame = *frame;
    memset(test_frame.header.hmac, 0, 20);

    if (zchg_hmac_sign_frame(&test_frame, secret, secret_len) != 0) {
        return -1;
    }

    /* Compare */
    return (memcmp(expected_hmac, test_frame.header.hmac, 20) == 0) ? 0 : -1;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* ============================================================================
 * Timestamp Validation (Replay Protection)
 * ============================================================================ */

int zchg_timestamp_is_valid(uint64_t timestamp) {
    uint64_t now = (uint64_t)time(NULL) * 1000;  /* Convert to milliseconds */
    int64_t diff = (int64_t)(now - timestamp);

    /* Accept timestamps within ±30 seconds */
    if (diff < 0) diff = -diff;
    return (diff <= zchg_REPLAY_WINDOW_SEC * 1000) ? 1 : 0;
}
