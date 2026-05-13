/*
 * tb_semantic_os.c — TRAILBLAZE Layer 5: Semantic OS implementation
 *
 * CapabilityAuthority  — phi-lattice hash tokens, branch-0 root trust.
 * WuWeiCodec           — DELTA_FOLD / PHI_COMPRESS / SPIRAL_PACK / RESONANCE / RAW.
 * SemanticContext      — phi-decay ranked cell relevance, focus boost.
 * SemanticCompressor   — in-place cell compression for KV space reclaim.
 */

#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif
#include "tb_semantic_os.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#  include "../src/tb_win32.h"
#endif

/* ── Millisecond wall clock ───────────────────────────────────────────────── */
static int64_t tb_sem_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* ── Rotate byte left/right ──────────────────────────────────────────────── */
static inline uint8_t rotl8(uint8_t b, int n) {
    n &= 7;
    return (uint8_t)((b << n) | (b >> (8 - n)));
}
static inline uint8_t rotr8(uint8_t b, int n) {
    n &= 7;
    return (uint8_t)((b >> n) | (b << (8 - n)));
}

/* ── Simple hex encode/decode helpers ──────────────────────────────────────── */
static void hex_encode(const uint8_t *in, int n, char *out) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        out[i*2]   = h[in[i] >> 4];
        out[i*2+1] = h[in[i] & 0xf];
    }
    out[n*2] = '\0';
}

static int hex_decode(const char *s, uint8_t *out, int max_out) {
    int n = 0;
    while (s[0] && s[1] && n < max_out) {
        uint8_t hi = (s[0] >= 'a') ? (uint8_t)(s[0]-'a'+10) :
                     (s[0] >= 'A') ? (uint8_t)(s[0]-'A'+10) :
                                     (uint8_t)(s[0]-'0');
        uint8_t lo = (s[1] >= 'a') ? (uint8_t)(s[1]-'a'+10) :
                     (s[1] >= 'A') ? (uint8_t)(s[1]-'A'+10) :
                                     (uint8_t)(s[1]-'0');
        out[n++] = (uint8_t)((hi << 4) | lo);
        s += 2;
    }
    return n;
}

/* ============================================================================
 * SECTION 1: Capability Authority
 * ============================================================================ */

TB_CapabilityAuthority* tb_cap_authority_create(TB_CognitionTree *tree) {
    TB_CapabilityAuthority *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->tree = tree;
    return a;
}

void tb_cap_authority_destroy(TB_CapabilityAuthority *auth) {
    free(auth);
}

int tb_cap_grant(TB_CapabilityAuthority *auth, TB_Capability cap,
                 const char *grantee, int64_t expires_ms, TB_CapToken *tok) {
    if (!auth || !tok || auth->n_tokens >= TB_CAP_AUTH_MAX_TOKENS) return -1;

    TB_PhiLattice *lat = auth->tree->lattice;

    /* Derive slot and amplitude from grantee string. */
    int slot = (int)tb_lattice_slot_for_key(lat, (const char*)grantee,
                                             strlen(grantee));
    float dn  = (float)tb_lattice_dn_for_key(lat, (const char*)grantee,
                                              strlen(grantee));

    /* Build composite seed string and hash it. */
    char seed[128];
    int slen = snprintf(seed, sizeof(seed), "%d|%s|%d|%d",
                        (int)cap, grantee, (int)lat->epoch, slot);
    uint8_t hash_out[32];
    tb_phi_fold_hash32(lat, (const uint8_t*)seed, (size_t)slen, hash_out);
    uint32_t token_hash = (uint32_t)hash_out[0]
                        | ((uint32_t)hash_out[1] << 8)
                        | ((uint32_t)hash_out[2] << 16)
                        | ((uint32_t)hash_out[3] << 24);

    tok->capability    = cap;
    tok->epoch_granted = lat->epoch;
    tok->slot          = slot;
    tok->dn_amplitude  = dn;
    tok->token_hash    = token_hash;
    tok->expires_ms    = expires_ms;
    snprintf(tok->grantee, TB_CAP_GRANTEE_LEN, "%s", grantee);

    auth->tokens[auth->n_tokens++] = *tok;
    return 0;
}

int tb_cap_verify(TB_CapabilityAuthority *auth, const TB_CapToken *tok) {
    if (!auth || !tok) return 0;

    /* Branch-0 root is always trusted (internal kernel). */
    if (tok->slot == 0 && tok->epoch_granted == 0) return 1;

    /* Check expiry. */
    if (tok->expires_ms > 0 && tb_sem_now_ms() > tok->expires_ms) return 0;

    /* Recompute hash from the same seed string used in grant. */
    TB_PhiLattice *lat = auth->tree->lattice;
    char seed[128];
    int slen = snprintf(seed, sizeof(seed), "%d|%s|%d|%d",
                        (int)tok->capability, tok->grantee,
                        (int)tok->epoch_granted, tok->slot);
    uint8_t hash_out[32];
    tb_phi_fold_hash32(lat, (const uint8_t*)seed, (size_t)slen, hash_out);
    uint32_t expected = (uint32_t)hash_out[0]
                      | ((uint32_t)hash_out[1] << 8)
                      | ((uint32_t)hash_out[2] << 16)
                      | ((uint32_t)hash_out[3] << 24);

    return expected == tok->token_hash;
}

/* ============================================================================
 * SECTION 2: WuWei Codec
 * ============================================================================ */

void tb_wuwei_init(TB_WuWeiCodec *c, TB_PhiLattice *lat) {
    memset(c, 0, sizeof(*c));
    c->lattice     = lat;
    c->perm_epoch  = -1;  /* force rebuild on first use */
    /* Default to identity permutation. */
    for (int i = 0; i < 256; i++) c->perm[i] = c->inv_perm[i] = (uint8_t)i;
}

TB_WuWeiStrategy tb_wuwei_select(const TB_WuWeiCodec *c) {
    TB_PhiLattice *lat = c->lattice;
    double M = 0.0, L = 0.0, S = 0.0;
    tb_lattice_s_u_resonance(lat, &M, &L, &S);

    if (lat->phase_var > 0.8)   return TB_WUWEI_RAW;
    if (S > 1.5)                 return TB_WUWEI_RESONANCE;
    if (S > 1.0)                 return TB_WUWEI_PHI_COMPRESS;
    if (fabs(fmod(L, 1.0)) > 0.6) return TB_WUWEI_SPIRAL_PACK;
    return TB_WUWEI_DELTA_FOLD;
}

/* ── RESONANCE: rebuild permutation table when epoch changed ─────────────── */
static void resonance_rebuild(TB_WuWeiCodec *c) {
    TB_PhiLattice *lat = c->lattice;
    if (c->perm_epoch == lat->epoch) return;

    /* Collect (index, amplitude) for first 256 slots (or fewer). */
    int n = lat->n_slots < 256 ? lat->n_slots : 256;
    /* Build indices sorted by decreasing dn_amplitude using insertion sort. */
    uint8_t order[256];
    for (int i = 0; i < n; i++) order[i] = (uint8_t)i;
    for (int i = 1; i < n; i++) {
        uint8_t key = order[i];
        double  amp = lat->slots[key].dn_amplitude;
        int j = i - 1;
        while (j >= 0 && lat->slots[order[j]].dn_amplitude < amp) {
            order[j+1] = order[j];
            j--;
        }
        order[j+1] = key;
    }
    /* Fill remaining 256 - n with identity. */
    for (int i = n; i < 256; i++) order[i] = (uint8_t)i;

    memcpy(c->perm, order, 256);
    /* Build inverse. */
    for (int i = 0; i < 256; i++) c->inv_perm[c->perm[i]] = (uint8_t)i;
    c->perm_epoch = lat->epoch;
}

int tb_wuwei_compress(TB_WuWeiCodec *c, TB_WuWeiStrategy strat,
                      const uint8_t *in, int in_len,
                      uint8_t *out, int out_cap) {
    if (strat == TB_WUWEI_RAW) {
        if (in_len > out_cap) return -1;
        memcpy(out, in, in_len);
        return in_len;
    }

    if (strat == TB_WUWEI_DELTA_FOLD) {
        if (in_len > out_cap) return -1;
        TB_PhiLattice *lat = c->lattice;
        int ns = lat->n_slots;
        for (int i = 0; i < in_len; i++)
            out[i] = in[i] ^ (uint8_t)(lat->slots[i % ns].value * 255.999);
        return in_len;
    }

    if (strat == TB_WUWEI_PHI_COMPRESS) {
        /* RLE with threshold = 14 (≈ floor(PHI * 256) % 16).
         * Sentinel: 0xFF  → always encoded as escape triple {0xFF, b, run}
         * even when run < THRESH, so 0xFF is never emitted as a literal byte.
         * Worst-case expansion: 3× (input all-0xFF); typical: 1× or less.   */
        const int THRESH = 14;
        int wi = 0;
        int i  = 0;
        while (i < in_len) {
            uint8_t b = in[i];
            int run = 1;
            while (i + run < in_len && in[i + run] == b && run < 255) run++;
            if (run >= THRESH || b == 0xFF) {
                if (wi + 3 > out_cap) return -1;
                out[wi++] = 0xFF;
                out[wi++] = b;
                out[wi++] = (uint8_t)run;
            } else {
                for (int r = 0; r < run; r++) {
                    if (wi >= out_cap) return -1;
                    out[wi++] = b;
                }
            }
            i += run;
        }
        return wi;
    }

    if (strat == TB_WUWEI_SPIRAL_PACK) {
        if (in_len > out_cap) return -1;
        for (int i = 0; i < in_len; i++) {
            int dim = TB_PASS_DIM[(i / 8) % 8] & 7;
            out[i] = rotl8(in[i], dim);
        }
        return in_len;
    }

    if (strat == TB_WUWEI_RESONANCE) {
        resonance_rebuild(c);
        if (in_len > out_cap) return -1;
        for (int i = 0; i < in_len; i++)
            out[i] = c->perm[in[i]];
        return in_len;
    }

    return -1;
}

int tb_wuwei_decompress(TB_WuWeiCodec *c, TB_WuWeiStrategy strat,
                        const uint8_t *in, int in_len,
                        uint8_t *out, int out_cap) {
    if (strat == TB_WUWEI_RAW) {
        if (in_len > out_cap) return -1;
        memcpy(out, in, in_len);
        return in_len;
    }

    if (strat == TB_WUWEI_DELTA_FOLD) {
        /* XOR is self-inverse. */
        return tb_wuwei_compress(c, TB_WUWEI_DELTA_FOLD, in, in_len,
                                  out, out_cap);
    }

    if (strat == TB_WUWEI_PHI_COMPRESS) {
        const int THRESH = 14;
        int wi = 0, i = 0;
        while (i < in_len) {
            if (in[i] == 0xFF && i + 2 < in_len) {
                uint8_t b   = in[i+1];
                int     run = (int)in[i+2];
                if (wi + run > out_cap) return -1;
                for (int r = 0; r < run; r++) out[wi++] = b;
                i += 3;
            } else {
                if (wi >= out_cap) return -1;
                out[wi++] = in[i++];
            }
        }
        (void)THRESH;  /* used in compress, not needed here */
        return wi;
    }

    if (strat == TB_WUWEI_SPIRAL_PACK) {
        if (in_len > out_cap) return -1;
        for (int i = 0; i < in_len; i++) {
            int dim = TB_PASS_DIM[(i / 8) % 8] & 7;
            out[i] = rotr8(in[i], dim);
        }
        return in_len;
    }

    if (strat == TB_WUWEI_RESONANCE) {
        resonance_rebuild(c);
        if (in_len > out_cap) return -1;
        for (int i = 0; i < in_len; i++)
            out[i] = c->inv_perm[in[i]];
        return in_len;
    }

    return -1;
}

/* ============================================================================
 * SECTION 3: Semantic Context
 * ============================================================================ */

TB_SemanticContext* tb_semctx_create(TB_CognitionTree *tree, int branch_id,
                                      float decay) {
    TB_SemanticContext *sc = calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->tree           = tree;
    sc->branch_id      = branch_id;
    sc->relevance_decay = (decay > 0.0f && decay < 1.0f) ? decay : 0.953f;
    sc->focus_cell_idx = -1;
    sc->cache_dirty    = 1;
    return sc;
}

void tb_semctx_destroy(TB_SemanticContext *sc) {
    free(sc);
}

void tb_semctx_rebuild(TB_SemanticContext *sc) {
    TB_CognitionTree *tree = sc->tree;
    sc->n_ranked = 0;

    /* Collect cells belonging to this branch (in reverse insertion order). */
    /* First pass: gather indices in reverse. */
    int indices[TB_SEMCTX_MAX_RANKED];
    int n = 0;
    for (int i = tree->n_cells - 1; i >= 0 && n < TB_SEMCTX_MAX_RANKED; i--) {
        if (tree->cells[i]->branch_id == sc->branch_id)
            indices[n++] = i;
    }

    /* Assign relevance: indices[0] is most recent → age 0 → relevance 1.0 */
    for (int a = 0; a < n; a++) {
        float rel = powf(sc->relevance_decay, (float)a);

        /* Focus boost: cells at focus get full relevance. */
        if (sc->focus_cell_idx >= 0) {
            int dist = abs(indices[a] - sc->focus_cell_idx);
            if (dist == 0) {
                rel = 1.0f;
            } else {
                float focus_factor = powf(sc->relevance_decay, (float)(dist - 1));
                if (focus_factor > rel) rel = focus_factor;
            }
        }

        sc->ranked[sc->n_ranked].cell_index = indices[a];
        sc->ranked[sc->n_ranked].relevance  = rel;
        sc->n_ranked++;
    }

    sc->cache_dirty = 0;
}

int tb_semctx_top_n(TB_SemanticContext *sc, int n, TB_CognitionCell **out) {
    if (sc->cache_dirty) tb_semctx_rebuild(sc);
    int count = sc->n_ranked < n ? sc->n_ranked : n;
    for (int i = 0; i < count; i++)
        out[i] = sc->tree->cells[sc->ranked[i].cell_index];
    return count;
}

void tb_semctx_set_focus(TB_SemanticContext *sc, int cell_idx) {
    sc->focus_cell_idx = cell_idx;
    sc->cache_dirty    = 1;
}

/* ============================================================================
 * SECTION 4: Semantic Compressor
 * ============================================================================ */

TB_SemanticCompressor* tb_semcomp_create(TB_SemanticContext *sc) {
    TB_SemanticCompressor *comp = calloc(1, sizeof(*comp));
    if (!comp) return NULL;
    comp->sc = sc;
    tb_wuwei_init(&comp->codec, sc->tree->lattice);
    return comp;
}

void tb_semcomp_destroy(TB_SemanticCompressor *comp) {
    free(comp);
}

int tb_semcomp_compress_branch(TB_SemanticCompressor *comp, float threshold) {
    TB_SemanticContext *sc = comp->sc;
    if (sc->cache_dirty) tb_semctx_rebuild(sc);

    TB_WuWeiStrategy strat = tb_wuwei_select(&comp->codec);
    int n_compressed = 0;

    for (int i = 0; i < sc->n_ranked; i++) {
        TB_RankedCell    *rc   = &sc->ranked[i];
        TB_CognitionCell *cell = sc->tree->cells[rc->cell_index];

        if (rc->relevance >= threshold) continue;
        int vlen = (int)strlen(cell->value);
        if (vlen < TB_SEMCOMP_MIN_LEN) continue;

        /* Already compressed? */
        if (strncmp(cell->value, "{\"__ww\":", 8) == 0) continue;

        /* Compress the value. */
        uint8_t  buf[TB_CELL_VALUE_LEN * 2];
        int clen = tb_wuwei_compress(&comp->codec, strat,
                                      (const uint8_t*)cell->value, vlen,
                                      buf, (int)sizeof(buf));
        if (clen <= 0) continue;

        /* Encode as hex. */
        char hex[TB_CELL_VALUE_LEN * 4];
        hex_encode(buf, clen, hex);

        /* Build JSON blob and write back into cell->value. */
        snprintf(cell->value, TB_CELL_VALUE_LEN,
                 "{\"__ww\":true,\"s\":%d,\"d\":\"%.*s\"}",
                 (int)strat, TB_CELL_VALUE_LEN - 40, hex);

        n_compressed++;
    }

    return n_compressed;
}

char* tb_semcomp_decompress_value(TB_SemanticCompressor *comp,
                                   const char *value) {
    if (!value || strncmp(value, "{\"__ww\":", 8) != 0) return NULL;

    /* Parse {"__ww":true,"s":N,"d":"hex"} */
    int strat_int = 0;
    const char *sp = strstr(value, "\"s\":");
    if (sp) {
        sp += 4;
        while (*sp == ' ') sp++;
        strat_int = (int)strtol(sp, NULL, 10);
    }

    const char *dp = strstr(value, "\"d\":\"");
    if (!dp) return NULL;
    dp += 5;  /* skip "d":"  */
    const char *end = strchr(dp, '"');
    if (!end) return NULL;

    int hex_len = (int)(end - dp);
    uint8_t *encoded = malloc(hex_len / 2 + 1);
    if (!encoded) return NULL;
    int enc_len = hex_decode(dp, encoded, hex_len / 2 + 1);

    uint8_t *decoded = malloc(enc_len * 2 + 4);
    if (!decoded) { free(encoded); return NULL; }

    int dlen = tb_wuwei_decompress(&comp->codec, (TB_WuWeiStrategy)strat_int,
                                    encoded, enc_len, decoded, enc_len * 2 + 4);
    free(encoded);

    if (dlen < 0) { free(decoded); return NULL; }
    decoded[dlen] = '\0';
    return (char*)decoded;
}
