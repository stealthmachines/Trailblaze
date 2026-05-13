/*
 * tb_semantic_os.h — TRAILBLAZE Layer 5: Semantic OS
 *
 * Three subsystems:
 *   1. Capability Authority — phi-lattice backed capability tokens (unforgeable
 *      because the token hash is derived from the lattice slot + dn amplitude).
 *
 *   2. WuWei Codec — adaptive, lattice-aware stream encoder.  Strategy is chosen
 *      per-call by tb_wuwei_select() based on phase_var and S-U resonance, then
 *      applied identically in compress/decompress (all strategies are self-inverse
 *      or carry their own inverse).
 *
 *   3. Semantic Context / Compressor — phi-decay relevance ranking of cognition
 *      cells; low-relevance cell values are compressed in-place with WuWei to
 *      free KV-cache space while staying auditable.
 *
 * Ported from v0.1/semantic_os.py — pure C11, no heap except where noted.
 */

#pragma once
#ifndef TB_SEMANTIC_OS_H
#define TB_SEMANTIC_OS_H

#include <stdint.h>
#include <stddef.h>
#include "../layer0/tb_phi_lattice.h"
#include "../layer2/tb_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Capability Authority
 * ============================================================================ */

typedef enum {
    TB_CAP_NONE          = 0,
    TB_CAP_SHELL_EXEC    = 1,
    TB_CAP_FILE_READ     = 2,
    TB_CAP_FILE_WRITE    = 3,
    TB_CAP_NET_FETCH     = 4,
    TB_CAP_MEMORY_SET    = 5,
    TB_CAP_MEMORY_GET    = 6,
    TB_CAP_BRANCH_CREATE = 7,
    TB_CAP_BRANCH_MERGE  = 8,
    TB_CAP_EPOCH_ADVANCE = 9,
    TB_CAP_DELEGATE      = 10,
    TB_CAP_TOOL_REGISTER = 11,
} TB_Capability;

#define TB_CAP_GRANTEE_LEN   64
#define TB_CAP_AUTH_MAX_TOKENS 128

typedef struct {
    TB_Capability capability;
    char          grantee[TB_CAP_GRANTEE_LEN];
    int32_t       epoch_granted;
    int           slot;             /* lattice slot index at grant time */
    float         dn_amplitude;     /* lattice dn amplitude at grant time */
    uint32_t      token_hash;       /* phi_fold_hash32(cap|grantee|epoch|slot) */
    int64_t       expires_ms;       /* 0 = never */
} TB_CapToken;

typedef struct {
    TB_CognitionTree *tree;
    TB_CapToken       tokens[TB_CAP_AUTH_MAX_TOKENS];
    int               n_tokens;
} TB_CapabilityAuthority;

/* Grant a capability — populates *tok and registers it in the authority.
 * Returns 0 on success, -1 if authority is full. */
int tb_cap_grant(TB_CapabilityAuthority *auth, TB_Capability cap,
                 const char *grantee, int64_t expires_ms, TB_CapToken *tok);

/* Verify a previously-issued token against the authority.
 * Returns 1 if valid, 0 if revoked / hash mismatch / expired. */
int tb_cap_verify(TB_CapabilityAuthority *auth, const TB_CapToken *tok);

/* Create / destroy the authority (tree must outlive the authority). */
TB_CapabilityAuthority* tb_cap_authority_create(TB_CognitionTree *tree);
void                    tb_cap_authority_destroy(TB_CapabilityAuthority *auth);

/* ============================================================================
 * 2. WuWei Codec
 * ============================================================================ */

typedef enum {
    TB_WUWEI_DELTA_FOLD   = 0,  /* XOR with lattice slot stream (self-inverse) */
    TB_WUWEI_PHI_COMPRESS = 1,  /* phi-threshold RLE */
    TB_WUWEI_SPIRAL_PACK  = 2,  /* bit-rotate by Spiral8 dimension */
    TB_WUWEI_RESONANCE    = 3,  /* amplitude-sorted substitution cipher */
    TB_WUWEI_RAW          = 4,  /* identity (high phase_var fall-through) */
} TB_WuWeiStrategy;

typedef struct {
    TB_PhiLattice *lattice;
    uint8_t        perm[256];      /* RESONANCE substitution table */
    uint8_t        inv_perm[256];  /* inverse for decompress */
    int32_t        perm_epoch;     /* epoch when perm was last rebuilt */
} TB_WuWeiCodec;

/* Initialise a codec (does not allocate — pass stack-allocated struct). */
void tb_wuwei_init(TB_WuWeiCodec *c, TB_PhiLattice *lat);

/* Choose strategy based on current lattice resonance state. */
TB_WuWeiStrategy tb_wuwei_select(const TB_WuWeiCodec *c);

/* Encode `in_len` bytes from `in` into `out` (must have space for at least
 * in_len + in_len/14 + 4 bytes).  Returns bytes written, -1 on error. */
int tb_wuwei_compress(TB_WuWeiCodec *c, TB_WuWeiStrategy strat,
                      const uint8_t *in, int in_len,
                      uint8_t *out, int out_cap);

/* Decode `in_len` bytes from `in` into `out`.
 * Returns bytes written, -1 on error. */
int tb_wuwei_decompress(TB_WuWeiCodec *c, TB_WuWeiStrategy strat,
                        const uint8_t *in, int in_len,
                        uint8_t *out, int out_cap);

/* ============================================================================
 * 3. Semantic Context
 * ============================================================================ */

#define TB_SEMCTX_MAX_RANKED 256

typedef struct {
    int   cell_index;   /* index into tree->cells[] */
    float relevance;
} TB_RankedCell;

typedef struct {
    TB_CognitionTree *tree;
    int               branch_id;
    float             relevance_decay; /* per-step decay factor, default 0.953 */
    int               focus_cell_idx;  /* -1 = none */
    TB_RankedCell     ranked[TB_SEMCTX_MAX_RANKED];
    int               n_ranked;
    int               cache_dirty;
} TB_SemanticContext;

/* Create / destroy. */
TB_SemanticContext* tb_semctx_create(TB_CognitionTree *tree, int branch_id,
                                      float decay);
void                tb_semctx_destroy(TB_SemanticContext *sc);

/* Rebuild the ranked cell list (call after new cells are committed). */
void tb_semctx_rebuild(TB_SemanticContext *sc);

/* Get the top-N most relevant cells (pointers into tree->cells[]).
 * Returns actual count written (<= n). */
int  tb_semctx_top_n(TB_SemanticContext *sc, int n, TB_CognitionCell **out);

/* Set focus cell by tree index; pass -1 to clear. */
void tb_semctx_set_focus(TB_SemanticContext *sc, int cell_idx);

/* ============================================================================
 * 4. Semantic Compressor
 * ============================================================================ */

/* Minimum cell value length before compression is considered. */
#define TB_SEMCOMP_MIN_LEN 32

/* Relevance threshold below which a cell value is eligible for compression. */
#define TB_SEMCOMP_THRESHOLD 0.3f

typedef struct {
    TB_SemanticContext *sc;
    TB_WuWeiCodec       codec;
} TB_SemanticCompressor;

/* Create / destroy. */
TB_SemanticCompressor* tb_semcomp_create(TB_SemanticContext *sc);
void                   tb_semcomp_destroy(TB_SemanticCompressor *comp);

/* Compress low-relevance cell values in branch_id in-place.
 * Compressed values are replaced with a JSON blob:
 *   {"__ww":true,"s":<strategy>,"d":"<hex>"}
 * Returns number of cells compressed. */
int tb_semcomp_compress_branch(TB_SemanticCompressor *comp, float threshold);

/* Decompress a single compressed value string (heap-allocated, caller frees).
 * Returns NULL if the string is not a compressed blob. */
char* tb_semcomp_decompress_value(TB_SemanticCompressor *comp,
                                   const char *value);

#ifdef __cplusplus
}
#endif

#endif /* TB_SEMANTIC_OS_H */
