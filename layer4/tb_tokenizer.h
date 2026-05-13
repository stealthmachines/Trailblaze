/*
 * tb_tokenizer.h — TRAILBLAZE Tokenizer + Long-Term Cognition
 *
 * BPE tokenizer (conscious tokenizer.h) + φ-Fourier context encoding
 * (vector_container.c) + ERL v3 matching MCP-0.11 schema.
 */
#pragma once
#ifndef TB_TOKENIZER_H
#define TB_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include "../layer0/tb_phi_lattice.h"
#include "../layer1/tb_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BPE_SYM_MAX 256

/* ── BPE merge rule ─────────────────────────────────────────────────────── */
typedef struct { char *a, *b; int priority; } TB_BPEMerge;

/* ── Tokenizer ──────────────────────────────────────────────────────────── */
typedef struct {
    char    **vocab;       /* token strings */
    float    *scores;
    int      *token_type;  /* 1=normal 2=byte 3=control */
    int       vocab_size;
    int       bos_id, eos_id, pad_id, nl_id;
    TB_BPEMerge *merges;
    int          n_merges;
    /* Vocab hash table (open addressing, FNV-1a) */
    char    **ht_keys;
    int      *ht_ids;
    uint32_t  ht_mask;
    int       ht_cap;
    /* Added tokens */
    int       n_added;
} TB_Tokenizer;

int   tb_tokenizer_from_gguf_vocab(TB_Tokenizer *tok,
                                    const char **vocab_strings,
                                    const float *scores,
                                    const int   *token_types,
                                    int          vocab_size,
                                    const char **merge_strings,
                                    int          n_merges,
                                    const char  *arch);
void  tb_tokenizer_free(TB_Tokenizer *tok);
int   tb_tokenizer_token_to_id(const TB_Tokenizer *tok, const char *s);
int   tb_tokenizer_encode(const TB_Tokenizer *tok, const char *text,
                           int add_bos, int *out_ids, int max_ids);
char* tb_tokenizer_decode(const TB_Tokenizer *tok, const int *ids, int n_ids);

/* ── φ-Fourier context vector (long-term cognition) ─────────────────────── */
#define TB_CTX_FOURIER_N 12
typedef struct {
    double cos_coeff[TB_CTX_FOURIER_N];
    double sin_coeff[TB_CTX_FOURIER_N];
    double mean, scale, temporal_phase;
    uint64_t n_tokens_encoded;
    int32_t  epoch;
} TB_ContextVector;

void tb_context_encode    (TB_ContextVector *cv, const int *token_ids,
                            int n_tokens, TB_PhiLattice *lat);
void tb_context_decode_dist(const TB_ContextVector *cv, double *out_dist, int n);
int  tb_context_to_json   (const TB_ContextVector *cv, char *buf, size_t buf_len);
int  tb_context_from_json (TB_ContextVector *cv, const char *json);

/* ── ERL v3 (matches MCP-0.11 erl-ledger.json exactly) ─────────────────── */
typedef struct {
    char  id[65], parent_id[65], branch[64], timestamp[32], role[16];
    char *content;
    char  session_id[64];
} TB_ERLEntry_v3;

typedef struct {
    char             version[8], created_at[32];
    TB_ERLEntry_v3 **entries;
    int              n_entries, cap;
    char             branch_names[32][64];
    char             branch_tips [32][65];
    int              n_branches;
    char            *persist_path;
} TB_ERL_v3;

TB_ERL_v3*     tb_erl_v3_create(const char *persist_path);
void           tb_erl_v3_free  (TB_ERL_v3 *L);
TB_ERLEntry_v3* tb_erl_v3_append(TB_ERL_v3 *L, const char *branch,
                                  const char *role, const char *content,
                                  const char *session_id);
int            tb_erl_v3_verify(TB_ERL_v3 *L, const char *branch);
int            tb_erl_v3_save  (TB_ERL_v3 *L);

/* ── Long-term cognition (context clearing protocol) ────────────────────── */
typedef struct {
    TB_ContextVector context_vec;
    TB_ERL_v3       *ledger;
    char             session_id[64];
    char             branch[64];
    int              tokens_since_compress;
    int              compress_threshold;
} TB_LongTermCognition;

TB_LongTermCognition* tb_ltc_create(const char *session_id,
                                     const char *erl_path,
                                     int compress_threshold);
void tb_ltc_free           (TB_LongTermCognition *ltc);
int  tb_ltc_absorb         (TB_LongTermCognition *ltc, const int *tok_ids,
                             int n, TB_PhiLattice *lat);
int  tb_ltc_compress_and_clear(TB_LongTermCognition *ltc, TB_PhiLattice *lat,
                                TB_KVCache **kvs, int n_kvs);
int  tb_ltc_restore        (TB_LongTermCognition *ltc, const char *erl_json_path);

#ifdef __cplusplus
}
#endif
#endif
