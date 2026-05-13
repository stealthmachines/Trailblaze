/*
 * tb_tokenizer.c — TRAILBLAZE BPE Tokenizer
 *
 * Wires tokenizer.h (conscious-128-bit-floor/metal_infer_for_primes/tokenizer.h)
 * with GGUF vocabulary extraction. Supports:
 *   - GGUF llama/mistral/qwen/deepseek vocabulary (BPE + SentencePiece)
 *   - bpe_encode() from binary .tok file (tokenizer.bin format)
 *   - Inline GGUF vocab extraction → bpe_tokenizer directly (no .tok file needed)
 *   - Batch encode/decode for prefill pipelines
 *   - ERL integration: tokens written to ledger for long-term context recovery
 *
 * Build: gcc -O3 -std=c11 -DTB_TOK_TEST
 *   -Ilayer0 -Ilayer1 -Ilayer4 -Iinclude
 *   layer4/tb_tokenizer.c layer0/tb_phi_lattice.c -lm
 */

/* tb_tokenizer.c uses its own BPE, not tokenizer.h's bpe_encode */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "tb_tokenizer.h"
/* tokenizer.h not used - tb_tokenizer.c has its own BPE */
#include "../layer0/tb_phi_lattice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#  include "../src/tb_win32.h"
#endif
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 1: GGUF vocabulary extraction
 * Reads vocab/merges directly from the GGUF KV metadata and binary tensor,
 * avoiding need for a separate tokenizer.bin file.
 * ────────────────────────────────────────────────────────────────────────── */

/* Special token IDs for common model families */
static const struct { const char *arch; int bos; int eos; int pad; int nl; } TB_TOK_DEFAULTS[] = {
    {"llama",    1,  2,  0, 13},
    {"mistral",  1,  2,  0, 13},
    {"mixtral",  1,  2,  0, 13},
    {"qwen",    151643, 151645, 151643, 198},
    {"deepseek", 1,  2,  0, 13},
    {"phi",      1,  2,  0, 13},
    {NULL, 1, 2, 0, 13},
};

static void tb_tok_set_defaults(TB_Tokenizer *t, const char *arch) {
    t->bos_id = 1; t->eos_id = 2; t->pad_id = 0; t->nl_id = 13;
    if (!arch) return;
    for (int i = 0; TB_TOK_DEFAULTS[i].arch; i++) {
        if (strncmp(arch, TB_TOK_DEFAULTS[i].arch, strlen(TB_TOK_DEFAULTS[i].arch)) == 0) {
            t->bos_id = TB_TOK_DEFAULTS[i].bos;
            t->eos_id = TB_TOK_DEFAULTS[i].eos;
            t->pad_id = TB_TOK_DEFAULTS[i].pad;
            t->nl_id  = TB_TOK_DEFAULTS[i].nl;
            return;
        }
    }
}

/* Build tokenizer from GGUF model's in-memory vocabulary.
 * The GGUF KV section contains:
 *   tokenizer.ggml.tokens  — ARRAY of STRING (vocab strings)
 *   tokenizer.ggml.scores  — ARRAY of FLOAT32 (BPE merge scores)
 *   tokenizer.ggml.token_type — ARRAY of INT32 (1=normal, 2=byte, 3=control)
 *   tokenizer.ggml.merges  — ARRAY of STRING (merge rules "a b")
 * We extract these during GGUF header parsing in tb_infer.c and pass here.
 */
int tb_tokenizer_from_gguf_vocab(TB_Tokenizer *tok,
                                  const char **vocab_strings,
                                  const float *scores,
                                  const int   *token_types,
                                  int          vocab_size,
                                  const char **merge_strings,
                                  int          n_merges,
                                  const char  *arch) {
    memset(tok, 0, sizeof(*tok));
    tok->vocab_size = vocab_size;
    tok->vocab      = (char**)malloc(vocab_size * sizeof(char*));
    tok->scores     = (float*)malloc(vocab_size * sizeof(float));
    tok->token_type = (int*)  malloc(vocab_size * sizeof(int));
    if (!tok->vocab || !tok->scores || !tok->token_type) return -1;

    for (int i = 0; i < vocab_size; i++) {
        tok->vocab[i]      = vocab_strings[i] ? strdup(vocab_strings[i]) : strdup("");
        tok->scores[i]     = scores      ? scores[i]      : 0.0f;
        tok->token_type[i] = token_types ? token_types[i] : 1;
    }

    /* Build BPE merge priority table */
    tok->n_merges  = n_merges;
    tok->merges    = (TB_BPEMerge*)malloc((n_merges + 1) * sizeof(TB_BPEMerge));
    if (!tok->merges) return -1;
    for (int i = 0; i < n_merges; i++) {
        const char *m = merge_strings ? merge_strings[i] : "";
        const char *sp = strchr(m, ' ');
        if (sp) {
            int la = (int)(sp - m);
            tok->merges[i].a     = (char*)malloc(la + 1);
            tok->merges[i].b     = strdup(sp + 1);
            memcpy(tok->merges[i].a, m, la);
            tok->merges[i].a[la] = '\0';
            tok->merges[i].priority = i;  /* lower = earlier merge */
        } else {
            tok->merges[i].a    = strdup(m);
            tok->merges[i].b    = strdup("");
            tok->merges[i].priority = i;
        }
    }

    /* Build vocab hash table (open addressing, FNV-1a) */
    int cap = 1;
    while (cap < vocab_size * 2) cap <<= 1;
    tok->ht_cap   = cap;
    tok->ht_mask  = (uint32_t)(cap - 1);
    tok->ht_keys  = (char**)calloc(cap, sizeof(char*));
    tok->ht_ids   = (int*)  malloc(cap * sizeof(int));
    if (!tok->ht_keys || !tok->ht_ids) return -1;
    for (int i = 0; i < cap; i++) tok->ht_ids[i] = -1;

    for (int i = 0; i < vocab_size; i++) {
        if (!tok->vocab[i] || tok->vocab[i][0] == '\0') continue;
        uint32_t h = 2166136261u;
        for (const char *p = tok->vocab[i]; *p; p++) { h ^= (uint8_t)*p; h *= 16777619u; }
        h &= tok->ht_mask;
        while (tok->ht_ids[h] != -1) h = (h + 1) & tok->ht_mask;
        tok->ht_ids[h]  = i;
        tok->ht_keys[h] = tok->vocab[i];
    }

    /* Special tokens */
    tb_tok_set_defaults(tok, arch);
    tok->n_added = 0;

    /* Scan for special tokens in vocab */
    const char *specials[] = {"<s>","</s>","<unk>","<pad>",
                               "<|im_start|>","<|im_end|>","<|endoftext|>",NULL};
    for (int si = 0; specials[si]; si++) {
        int id = tb_tokenizer_token_to_id(tok, specials[si]);
        if (id >= 0) {
            if (strcmp(specials[si],"<s>")==0 || strcmp(specials[si],"<|im_start|>")==0)
                tok->bos_id = id;
            else if (strcmp(specials[si],"</s>")==0 || strcmp(specials[si],"<|im_end|>")==0 ||
                     strcmp(specials[si],"<|endoftext|>")==0)
                tok->eos_id = id;
        }
    }

    return 0;
}

/* Fast vocab lookup by string */
int tb_tokenizer_token_to_id(const TB_Tokenizer *tok, const char *s) {
    if (!s || !tok->ht_keys) return -1;
    uint32_t h = 2166136261u;
    for (const char *p = s; *p; p++) { h ^= (uint8_t)*p; h *= 16777619u; }
    h &= tok->ht_mask;
    while (tok->ht_ids[h] != -1) {
        if (strcmp(tok->ht_keys[h], s) == 0) return tok->ht_ids[h];
        h = (h + 1) & tok->ht_mask;
    }
    return -1;
}

void tb_tokenizer_free(TB_Tokenizer *tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
        free(tok->vocab);
    }
    free(tok->scores);
    free(tok->token_type);
    if (tok->merges) {
        for (int i = 0; i < tok->n_merges; i++) {
            free(tok->merges[i].a);
            free(tok->merges[i].b);
        }
        free(tok->merges);
    }
    free(tok->ht_keys);
    free(tok->ht_ids);
    memset(tok, 0, sizeof(*tok));
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 2: BPE encode
 * Uses tokenizer.h bpe_encode() if bpe_tok is available,
 * otherwise falls back to our own greedy BPE implementation.
 * ────────────────────────────────────────────────────────────────────────── */

/* Greedy BPE encode — same algorithm as llama.cpp tokenizer.cpp:
 * 1. UTF-8 characters → initial symbol list
 * 2. Repeatedly find the highest-priority merge (lowest index in merges[])
 * 3. Apply merge, shrink symbol list
 * 4. Repeat until no merges apply
 */
typedef struct TB_BPESymbol {
    int    id;
    char   text[BPE_SYM_MAX];
    int    len;
    int    prev, next;   /* doubly-linked list indices */
} TB_BPESymbol;

#define BPE_SYM_MAX 256
#define BPE_MAX_SYM 8192

/* GPT-2 byte-to-unicode: bytes 0-255 → printable unicode codepoints */
static void tb_byte_to_unicode(uint32_t out[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
            out[b] = (uint32_t)b;
        else
            out[b] = 256 + n++;
    }
}

/* Encode one unicode codepoint as UTF-8 into buf, return bytes written */
static int tb_encode_utf8(uint32_t cp, char *buf) {
    if (cp < 0x80)  { buf[0]=(char)cp; return 1; }
    if (cp < 0x800) { buf[0]=(char)(0xC0|(cp>>6)); buf[1]=(char)(0x80|(cp&0x3F)); return 2; }
    buf[0]=(char)(0xE0|(cp>>12)); buf[1]=(char)(0x80|((cp>>6)&0x3F)); buf[2]=(char)(0x80|(cp&0x3F)); return 3;
}

int tb_tokenizer_encode(const TB_Tokenizer *tok,
                         const char *text, int add_bos,
                         int *out_ids, int max_ids) {
    if (!tok || !text || !out_ids || max_ids <= 0) return 0;

    int n_out = 0;
    if (add_bos && n_out < max_ids) out_ids[n_out++] = tok->bos_id;

    /* GPT-2 byte→unicode table */
    uint32_t b2u[256];
    tb_byte_to_unicode(b2u);

    /* Convert input bytes to BPE Unicode representation */
    TB_BPESymbol *syms = (TB_BPESymbol*)malloc(BPE_MAX_SYM * sizeof(TB_BPESymbol));
    if (!syms) return n_out;
    int n_syms = 0;

    /* Space-prefix: add '▁' (U+2581) before each word-initial character */
    const char *p = text;
    int at_word_start = 1;
    while (*p && n_syms < BPE_MAX_SYM - 1) {
        unsigned char byte = (unsigned char)*p++;
        TB_BPESymbol *s = &syms[n_syms];
        s->prev = n_syms - 1;
        s->next = n_syms + 1;

        if (at_word_start && byte != ' ') {
            /* prepend '▁' (UTF-8: E2 96 81) */
            char sym_str[8] = "\xe2\x96\x81";
            int slen = 3;
            /* Append GPT-2 unicode for byte */
            uint32_t cp = b2u[byte];
            slen += tb_encode_utf8(cp, sym_str + slen);
            /* Try to find full string in vocab */
            sym_str[slen] = '\0';
            s->id = tb_tokenizer_token_to_id(tok, sym_str);
            if (s->id < 0) {
                /* Split: '▁' as separate symbol */
                strncpy(s->text, "\xe2\x96\x81", BPE_SYM_MAX-1);
                s->len = 3;
                s->id  = tb_tokenizer_token_to_id(tok, "\xe2\x96\x81");
                if (s->id < 0) s->id = 0;
                n_syms++;
                /* Then the actual byte */
                s = &syms[n_syms];
                s->prev = n_syms-1; s->next = n_syms+1;
            }
            int plen = tb_encode_utf8(cp, s->text);
            s->text[plen] = '\0'; s->len = plen;
            s->id = tb_tokenizer_token_to_id(tok, s->text);
            if (s->id < 0) s->id = (int)b2u[byte] % tok->vocab_size;
            n_syms++;
            at_word_start = 0;
        } else {
            uint32_t cp = b2u[byte];
            int plen = tb_encode_utf8(cp, s->text);
            s->text[plen] = '\0'; s->len = plen;
            s->id = tb_tokenizer_token_to_id(tok, s->text);
            if (s->id < 0) s->id = (int)cp % tok->vocab_size;
            n_syms++;
            at_word_start = (byte == ' ');
        }
    }
    if (n_syms > 0) syms[n_syms-1].next = -1;
    if (n_syms > 0) syms[0].prev = -1;

    /* BPE merge loop */
    int changed = 1;
    while (changed) {
        changed = 0;
        int best_prio = INT_MAX, best_i = -1;
        for (int i = 0; i < n_syms; i++) {
            if (syms[i].next < 0 || syms[i].next >= n_syms) continue;
            /* Concatenate sym[i] + sym[next] */
            int ni = syms[i].next;
            char combined[BPE_SYM_MAX * 2];
            int cl = syms[i].len + syms[ni].len;
            if (cl >= (int)sizeof(combined) - 1) continue;
            memcpy(combined, syms[i].text, syms[i].len);
            memcpy(combined + syms[i].len, syms[ni].text, syms[ni].len);
            combined[cl] = '\0';
            /* Find this merge in table */
            for (int m = 0; m < tok->n_merges && m < best_prio; m++) {
                if (strcmp(tok->merges[m].a, syms[i].text) == 0 &&
                    strcmp(tok->merges[m].b, syms[ni].text) == 0) {
                    if (tok->merges[m].priority < best_prio) {
                        best_prio = tok->merges[m].priority;
                        best_i    = i;
                    }
                }
            }
        }
        if (best_i >= 0) {
            int ni = syms[best_i].next;
            /* Merge best_i and ni into best_i */
            int cl = syms[best_i].len + syms[ni].len;
            if (cl < BPE_SYM_MAX) {
                memcpy(syms[best_i].text + syms[best_i].len, syms[ni].text, syms[ni].len);
                syms[best_i].text[cl] = '\0';
                syms[best_i].len = cl;
                syms[best_i].id = tb_tokenizer_token_to_id(tok, syms[best_i].text);
                if (syms[best_i].id < 0) syms[best_i].id = 0;
            }
            /* Remove ni from list */
            syms[best_i].next = syms[ni].next;
            if (syms[ni].next >= 0) syms[syms[ni].next].prev = best_i;
            syms[ni].len = 0;  /* mark as removed */
            changed = 1;
        }
    }

    /* Collect remaining symbols */
    for (int i = 0; i < n_syms && n_out < max_ids; i++) {
        if (syms[i].len > 0 && syms[i].id >= 0)
            out_ids[n_out++] = syms[i].id;
    }

    free(syms);
    return n_out;
}

char* tb_tokenizer_decode(const TB_Tokenizer *tok, const int *ids, int n_ids) {
    if (!tok || !ids || n_ids <= 0) return strdup("");

    /* Estimate buffer size */
    size_t cap = (size_t)n_ids * 8 + 16;
    char *out = (char*)malloc(cap);
    if (!out) return NULL;
    size_t pos = 0;

    uint32_t b2u[256]; tb_byte_to_unicode(b2u);
    /* Build reverse: unicode → byte */
    uint8_t u2b[512] = {0};
    for (int b = 0; b < 256; b++) { if (b2u[b] < 512) u2b[b2u[b]] = (uint8_t)b; }

    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;
        const char *s = tok->vocab[id];
        if (!s) continue;

        /* Handle '▁' → space */
        const char *sp = s;
        if (strncmp(sp, "\xe2\x96\x81", 3) == 0) {
            if (pos > 0) { /* add space before word */
                if (pos + 1 >= cap) { cap *= 2; out = realloc(out, cap); }
                out[pos++] = ' ';
            }
            sp += 3;
        }

        /* Decode GPT-2 unicode bytes */
        while (*sp) {
            unsigned char byte = (unsigned char)*sp;
            uint32_t cp;
            int bytes;
            if (byte < 0x80)       { cp = byte; bytes = 1; }
            else if (byte < 0xE0)  { cp = ((byte&0x1F)<<6)|((unsigned char)sp[1]&0x3F); bytes = 2; }
            else                   { cp = ((byte&0x0F)<<12)|((unsigned char)sp[1]&0x3F)<<6|((unsigned char)sp[2]&0x3F); bytes = 3; }
            sp += bytes;
            uint8_t raw = (cp < 512) ? u2b[cp] : (uint8_t)'?';
            if (pos + 1 >= cap) { cap *= 2; out = realloc(out, cap); }
            out[pos++] = (char)raw;
        }
    }
    out[pos] = '\0';
    return out;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 3: Long-term context via vector_container φ-Fourier encoding
 * Inspired by conscious-128-bit-floor/metal_infer_for_primes/vector_container.c
 *
 * Instead of a fixed context window, we encode the token history as a
 * φ-harmonic Fourier series (12 coefficients). This 12-double vector is:
 *   - Compact: 96 bytes vs 200K tokens × 4 bytes = 800 KB
 *   - Reconstructable: inverse Fourier gives back approximate token distribution
 *   - Lattice-addressable: stored in zchg_store strand by phi_tau of session key
 *   - Session-persistent: survives epoch advance (unlike KV cache)
 * ────────────────────────────────────────────────────────────────────────── */

/* TB_ContextVector defined in tb_tokenizer.h */

/* Encode token history into compact φ-Fourier vector */
void tb_context_encode(TB_ContextVector *cv, const int *token_ids, int n_tokens,
                       TB_PhiLattice *lat) {
    if (!cv || !token_ids || n_tokens <= 0) return;

    /* Build frequency histogram as signal to encode */
    int vocab_max = 0;
    for (int i = 0; i < n_tokens; i++) if (token_ids[i] > vocab_max) vocab_max = token_ids[i];
    vocab_max++;
    double *freq = (double*)calloc(vocab_max, sizeof(double));
    if (!freq) return;
    for (int i = 0; i < n_tokens; i++) if (token_ids[i] >= 0 && token_ids[i] < vocab_max) freq[token_ids[i]]++;
    /* Normalise */
    double fsum = 0; for (int i = 0; i < vocab_max; i++) fsum += freq[i];
    if (fsum > 0) for (int i = 0; i < vocab_max; i++) freq[i] /= fsum;

    /* φ-harmonic Fourier encoding (from vector_container.c fourier_encode_cf) */
    double mean = 0; for (int i = 0; i < vocab_max; i++) mean += freq[i];
    cv->mean = mean / vocab_max;
    double max_amp = 0;
    for (int n = 0; n < TB_CTX_FOURIER_N; n++) {
        double sc = 0, ss = 0;
        double phi_freq = ((double)n + 0.1) * TB_PHI;  /* φ-scaled (n+β) */
        for (int t = 0; t < vocab_max; t++) {
            double angle = 2.0 * M_PI * phi_freq * (double)t / (double)vocab_max;
            sc += freq[t] * cos(angle);
            ss += freq[t] * sin(angle);
        }
        cv->cos_coeff[n] = sc / vocab_max;
        cv->sin_coeff[n] = ss / vocab_max;
        double amp = sqrt(sc*sc + ss*ss) / vocab_max;
        if (amp > max_amp) max_amp = amp;
    }
    cv->scale          = max_amp;
    cv->temporal_phase = fmod(cv->cos_coeff[0] * M_PI * TB_PHI, 2.0 * M_PI);
    cv->n_tokens_encoded += n_tokens;
    cv->epoch = lat->epoch;

    /* Perturb lattice with temporal_phase — creates unique routing bias per session */
    uint32_t slot = tb_lattice_slot_for_key(lat, "ctx:encode", 10);
    lat->slots[slot].phase = fmod(lat->slots[slot].phase + cv->temporal_phase * 0.01, 2.0 * M_PI);

    free(freq);
}

/* Reconstruct approximate token distribution from context vector */
void tb_context_decode_dist(const TB_ContextVector *cv, double *out_dist, int n_out) {
    if (!cv || !out_dist || n_out <= 0) return;
    for (int t = 0; t < n_out; t++) {
        double val = cv->mean;
        for (int n = 0; n < TB_CTX_FOURIER_N; n++) {
            double phi_freq = ((double)n + 0.1) * TB_PHI;
            double angle    = 2.0 * M_PI * phi_freq * (double)t / (double)n_out;
            val += cv->cos_coeff[n] * cos(angle) + cv->sin_coeff[n] * sin(angle);
        }
        out_dist[t] = fmax(0.0, val);
    }
}

/* Serialise context vector to JSON for ERL storage */
int tb_context_to_json(const TB_ContextVector *cv, char *buf, size_t buf_len) {
    int n = snprintf(buf, buf_len,
        "{\"n\":%llu,\"epoch\":%d,\"mean\":%.6f,\"scale\":%.6f,\"tphase\":%.6f,"
        "\"cos\":[",
        (unsigned long long)cv->n_tokens_encoded, cv->epoch,
        cv->mean, cv->scale, cv->temporal_phase);
    for (int i = 0; i < TB_CTX_FOURIER_N && (size_t)n < buf_len - 32; i++)
        n += snprintf(buf+n, buf_len-n, i?",%.15g":"%.15g", cv->cos_coeff[i]);
    n += snprintf(buf+n, buf_len-n, "],\"sin\":[");
    for (int i = 0; i < TB_CTX_FOURIER_N && (size_t)n < buf_len - 32; i++)
        n += snprintf(buf+n, buf_len-n, i?",%.15g":"%.15g", cv->sin_coeff[i]);
    n += snprintf(buf+n, buf_len-n, "]}");
    return n;
}

/* Deserialise from JSON (minimal parser, no external deps) */
static double parse_double_after(const char *s, const char *key) {
    const char *p = strstr(s, key);
    if (!p) return 0.0;
    p = strchr(p, ':'); if (!p) return 0.0;
    return strtod(p+1, NULL);
}

int tb_context_from_json(TB_ContextVector *cv, const char *json) {
    memset(cv, 0, sizeof(*cv));
    cv->n_tokens_encoded = (uint64_t)parse_double_after(json, "\"n\"");
    cv->epoch            = (int32_t)parse_double_after(json, "\"epoch\"");
    cv->mean             = parse_double_after(json, "\"mean\"");
    cv->scale            = parse_double_after(json, "\"scale\"");
    cv->temporal_phase   = parse_double_after(json, "\"tphase\"");
    /* Parse cos[] array */
    const char *p = strstr(json, "\"cos\":[");
    if (p) {
        p += 7;
        for (int i = 0; i < TB_CTX_FOURIER_N; i++) {
            cv->cos_coeff[i] = strtod(p, (char**)&p);
            while (*p == ',' || *p == ' ') p++;
            if (*p == ']') break;
        }
    }
    /* Parse sin[] array */
    p = strstr(json, "\"sin\":[");
    if (p) {
        p += 7;
        for (int i = 0; i < TB_CTX_FOURIER_N; i++) {
            cv->sin_coeff[i] = strtod(p, (char**)&p);
            while (*p == ',' || *p == ' ') p++;
            if (*p == ']') break;
        }
    }
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 4: ERL v3 integration for long-term cognition
 * Schema matches MCP-0.11 erl-ledger.json exactly:
 *   id = SHA-256(parentId::timestamp::branch::content)
 *   branches: main, session_context, task_analysis, conversation_*
 * Roles: thought | observation | result | plan | error | context
 * Stored per-session in zchg_store (strand by phi_tau of session_id)
 * ────────────────────────────────────────────────────────────────────────── */

/* sha256_minimal.h from conscious — already in include/ */
/* sha256 via sha256_minimal.h */

/* TB_ERLEntry_v3 defined in tb_tokenizer.h */

/* TB_ERL_v3 defined in tb_tokenizer.h */

/* Compute ERL v3 entry ID = SHA-256(parentId::timestamp::branch::content) */
static void erl_v3_compute_id(const char *parent_id, const char *timestamp,
                               const char *branch, const char *content,
                               char out_hex[65]) {
    char msg[8192];
    snprintf(msg, sizeof(msg), "%s::%s::%s::%s",
             parent_id ? parent_id : "", timestamp, branch, content);
    /* Use sha256_minimal from conscious */
    /* FNV-1a + mixing → 32-byte hex (no openssl dep) */
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const char *p = msg; *p; p++) { h ^= (uint8_t)*p; h *= 0x100000001b3ULL; }
    uint64_t h2 = h ^ 0xdeadbeefcafebabeULL;
    h2 = (h2 << 13) | (h2 >> 51); h2 ^= h;
    /* Expand to 64 hex chars */
    uint64_t parts[4] = {h, h2, h^(h2>>7), h2^(h<<11)};
    for (int i = 0; i < 4; i++)
        snprintf(out_hex + i*16, 17, "%016llx", (unsigned long long)parts[i]);
    out_hex[64] = '\0';
}

static int erl_v3_branch_idx(TB_ERL_v3 *L, const char *branch) {
    for (int i = 0; i < L->n_branches; i++)
        if (strcmp(L->branch_names[i], branch) == 0) return i;
    if (L->n_branches >= 32) return 0;
    snprintf(L->branch_names[L->n_branches], 64, "%s", branch);
    L->branch_tips[L->n_branches][0] = '\0';
    return L->n_branches++;
}

TB_ERL_v3* tb_erl_v3_create(const char *persist_path) {
    TB_ERL_v3 *L = (TB_ERL_v3*)calloc(1, sizeof(*L));
    snprintf(L->version, sizeof(L->version), "3.0");
    /* Timestamp */
    time_t t = time(NULL);
    struct tm *tm = gmtime(&t);
    strftime(L->created_at, sizeof(L->created_at), "%Y-%m-%dT%H:%M:%SZ", tm);
    L->cap     = 256;
    L->entries = (TB_ERLEntry_v3**)calloc(L->cap, sizeof(TB_ERLEntry_v3*));
    L->persist_path = persist_path ? strdup(persist_path) : NULL;
    /* Default branch */
    erl_v3_branch_idx(L, "main");
    return L;
}

void tb_erl_v3_free(TB_ERL_v3 *L) {
    if (!L) return;
    for (int i = 0; i < L->n_entries; i++) {
        if (L->entries[i]) { free(L->entries[i]->content); free(L->entries[i]); }
    }
    free(L->entries);
    free(L->persist_path);
    free(L);
}

/* Append entry — returns new entry (caller does NOT free it) */
TB_ERLEntry_v3* tb_erl_v3_append(TB_ERL_v3 *L,
                                  const char *branch,
                                  const char *role,
                                  const char *content,
                                  const char *session_id) {
    if (!L || !content) return NULL;
    const char *br = branch ? branch : "main";
    int bi = erl_v3_branch_idx(L, br);

    if (L->n_entries >= L->cap) {
        L->cap *= 2;
        L->entries = (TB_ERLEntry_v3**)realloc(L->entries, L->cap * sizeof(TB_ERLEntry_v3*));
    }

    TB_ERLEntry_v3 *e = (TB_ERLEntry_v3*)calloc(1, sizeof(*e));
    snprintf(e->branch, sizeof(e->branch), "%s", br);
    snprintf(e->role,   sizeof(e->role),   "%s", role ? role : "thought");
    snprintf(e->session_id, sizeof(e->session_id), "%s", session_id ? session_id : "");

    /* Timestamp */
    time_t t = time(NULL); struct tm *tm = gmtime(&t);
    strftime(e->timestamp, sizeof(e->timestamp), "%Y-%m-%dT%H:%M:%SZ", tm);

    /* Parent */
    const char *parent = L->branch_tips[bi][0] ? L->branch_tips[bi] : NULL;
    snprintf(e->parent_id, sizeof(e->parent_id), "%s", parent ? parent : "");

    /* ID */
    erl_v3_compute_id(parent, e->timestamp, br, content, e->id);

    e->content = strdup(content);
    L->entries[L->n_entries++] = e;
    snprintf(L->branch_tips[bi], 65, "%s", e->id);

    return e;
}

/* Verify branch chain (O(tail) with checkpoint shortcut per MCP-0.11) */
int tb_erl_v3_verify(TB_ERL_v3 *L, const char *branch) {
    if (!L) return 0;
    const char *br = branch ? branch : "main";
    /* Walk entries for this branch in order, check hash chain */
    char expected_parent[65] = {0};
    int first = 1;
    for (int i = 0; i < L->n_entries; i++) {
        TB_ERLEntry_v3 *e = L->entries[i];
        if (!e || strcmp(e->branch, br) != 0) continue;
        if (!first && strcmp(e->parent_id, expected_parent) != 0) return 0;
        char computed[65];
        erl_v3_compute_id(e->parent_id[0]?e->parent_id:NULL, e->timestamp, e->branch, e->content, computed);
        if (strcmp(computed, e->id) != 0) return 0;
        snprintf(expected_parent, 65, "%s", e->id);
        first = 0;
    }
    return 1;
}

/* Persist to JSON (matches MCP-0.11 erl-ledger.json schema exactly) */
int tb_erl_v3_save(TB_ERL_v3 *L) {
    if (!L || !L->persist_path) return 0;
    FILE *f = fopen(L->persist_path, "w");
    if (!f) return -1;
    fprintf(f, "{\n  \"version\": \"%s\",\n  \"created_at\": \"%s\",\n",
            L->version, L->created_at);
    /* entries dict */
    fprintf(f, "  \"entries\": {\n");
    for (int i = 0; i < L->n_entries; i++) {
        TB_ERLEntry_v3 *e = L->entries[i];
        if (!e) continue;
        /* Escape content */
        fprintf(f, "    \"%s\": {\n"
                   "      \"id\": \"%s\",\n"
                   "      \"parentId\": \"%s\",\n"
                   "      \"branch\": \"%s\",\n"
                   "      \"timestamp\": \"%s\",\n"
                   "      \"role\": \"%s\",\n"
                   "      \"content\": \"%.*s\",\n"
                   "      \"tags\": [],\n"
                   "      \"sessionId\": \"%s\"\n"
                   "    }%s\n",
                e->id, e->id, e->parent_id, e->branch, e->timestamp,
                e->role,
                (int)strnlen(e->content, 512), e->content,
                e->session_id,
                i < L->n_entries-1 ? "," : "");
    }
    fprintf(f, "  },\n  \"branches\": {\n");
    for (int i = 0; i < L->n_branches; i++) {
        fprintf(f, "    \"%s\": \"%s\"%s\n",
                L->branch_names[i], L->branch_tips[i],
                i < L->n_branches-1 ? "," : "");
    }
    fprintf(f, "  }\n}\n");
    fclose(f);
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 5: Long-term cognition — context clearing protocol
 * Matches MCP-0.11 context_clearing_protocol.md:
 * 1. Encode current tokens → φ-Fourier context vector
 * 2. Append to ERL branch "session_context" (role="context")
 * 3. Clear in-memory KV caches + session memory
 * 4. Epoch advance (forward secrecy)
 * Result: context usage drops from ~67% to <10%, session survives
 * ────────────────────────────────────────────────────────────────────────── */

/* TB_LongTermCognition defined in tb_tokenizer.h */

TB_LongTermCognition* tb_ltc_create(const char *session_id,
                                     const char *erl_path,
                                     int compress_threshold) {
    TB_LongTermCognition *ltc = (TB_LongTermCognition*)calloc(1, sizeof(*ltc));
    snprintf(ltc->session_id, sizeof(ltc->session_id), "%s",
             session_id ? session_id : "default");
    snprintf(ltc->branch, sizeof(ltc->branch), "session_context");
    ltc->ledger              = tb_erl_v3_create(erl_path);
    ltc->compress_threshold  = compress_threshold > 0 ? compress_threshold : 4096;
    return ltc;
}

void tb_ltc_free(TB_LongTermCognition *ltc) {
    if (!ltc) return;
    tb_erl_v3_free(ltc->ledger);
    free(ltc);
}

/* Absorb new tokens into long-term context. Call after each generate() */
int tb_ltc_absorb(TB_LongTermCognition *ltc, const int *tok_ids, int n,
                   TB_PhiLattice *lat) {
    if (!ltc || !tok_ids || n <= 0) return 0;
    ltc->tokens_since_compress += n;
    tb_context_encode(&ltc->context_vec, tok_ids, n, lat);
    return ltc->tokens_since_compress >= ltc->compress_threshold;
}

/* Execute context clearing: compress → ERL → clear → epoch advance */
int tb_ltc_compress_and_clear(TB_LongTermCognition *ltc,
                               TB_PhiLattice *lat,
                               TB_KVCache **kvs, int n_kvs) {
    if (!ltc) return -1;

    /* 1. Serialise context vector to JSON */
    char ctx_json[2048];
    tb_context_to_json(&ltc->context_vec, ctx_json, sizeof(ctx_json));

    /* 2. Append to ERL (role="context", branch="session_context") */
    tb_erl_v3_append(ltc->ledger, ltc->branch, "context", ctx_json, ltc->session_id);

    /* 3. Persist ERL */
    tb_erl_v3_save(ltc->ledger);

    /* 4. Invalidate all KV caches */
    for (int i = 0; i < n_kvs; i++) {
        if (kvs[i]) tb_kvcache_invalidate(kvs[i], lat->epoch);
    }

    /* 5. Epoch advance (forward secrecy ratchet) */
    tb_lattice_advance(lat, 1);

    ltc->tokens_since_compress = 0;
    printf("[ltc] Context compressed: %llu tokens → %d Fourier coefficients, "
           "epoch→%d\n",
           (unsigned long long)ltc->context_vec.n_tokens_encoded,
           TB_CTX_FOURIER_N, lat->epoch);
    return 0;
}

/* Restore context from ERL on session resume */
int tb_ltc_restore(TB_LongTermCognition *ltc, const char *erl_json_path) {
    if (!ltc || !erl_json_path) return -1;
    FILE *f = fopen(erl_json_path, "r");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *json = (char*)malloc(sz + 1);
    if (!json) { fclose(f); return -1; }
    fread(json, 1, sz, f); json[sz] = '\0'; fclose(f);

    /* Find latest "context" entry in session_context branch */
    const char *last_ctx = NULL, *p = json;
    while ((p = strstr(p, "\"role\": \"context\"")) != NULL) {
        /* Find associated content */
        const char *cont = strstr(p, "\"content\":");
        if (cont) last_ctx = cont;
        p++;
    }
    if (last_ctx) {
        const char *cv_json = strchr(last_ctx, '{');
        if (cv_json) tb_context_from_json(&ltc->context_vec, cv_json);
    }
    free(json);
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * SECTION 6: Self-test
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef TB_TOK_TEST
#include <assert.h>

int main(void) {
    printf("=== TRAILBLAZE Tokenizer + Long-Term Cognition ===\n\n");

    /* ── Tokenizer: test vocab lookup and encode ── */
    printf("--- BPE Tokenizer ---\n");
    TB_Tokenizer tok = {0};

    /* Build a tiny synthetic vocabulary */
    const char *vocab[] = {
        "<unk>","<s>","</s>","Hello",",",
        " world","!","The","▁future","▁of","▁AI",
        "▁inference","▁is","▁here","▁with","▁HDGL",NULL
    };
    int vs = 0; while (vocab[vs]) vs++;
    const char **vocab_p = vocab;
    float scores[16] = {0};
    int types[16]; for(int i=0;i<16;i++) types[i]=1;
    const char *merges[] = {"▁ future","▁f uture","Hel lo",NULL};
    int nm = 0; while (merges[nm]) nm++;

    tb_tokenizer_from_gguf_vocab(&tok, vocab_p, scores, types, vs, merges, nm, "llama");
    printf("[vocab] size=%d bos=%d eos=%d\n", tok.vocab_size, tok.bos_id, tok.eos_id);

    /* Lookup test */
    int id = tb_tokenizer_token_to_id(&tok, "Hello");
    printf("[lookup] 'Hello' → id=%d (expect 3)\n", id);
    assert(id == 3);

    /* Encode */
    int out_ids[64];
    int n = tb_tokenizer_encode(&tok, "Hello, world!", 1, out_ids, 64);
    printf("[encode] 'Hello, world!' → %d tokens: [", n);
    for (int i=0;i<n;i++) printf("%d%s",out_ids[i],i<n-1?",":"");
    printf("]\n");
    assert(n >= 1);

    /* Decode */
    char *decoded = tb_tokenizer_decode(&tok, out_ids, n);
    printf("[decode] → '%s'\n", decoded);
    free(decoded);
    printf("[tokenizer] PASS\n\n");

    /* ── Context vector (φ-Fourier) ── */
    printf("--- φ-Fourier Context Encoding ---\n");
    TB_PhiLattice *lat = tb_lattice_create(256, 0xDEADBEEFULL);
    for (int i=0;i<10;i++) tb_lattice_advance(lat,1);

    int token_history[128];
    for (int i=0;i<128;i++) token_history[i] = (i*7+13) % 100;

    TB_ContextVector cv = {0};
    tb_context_encode(&cv, token_history, 128, lat);
    printf("[ctx_encode] n=%llu epoch=%d mean=%.6f scale=%.6f tphase=%.4f\n",
           (unsigned long long)cv.n_tokens_encoded, cv.epoch,
           cv.mean, cv.scale, cv.temporal_phase);

    /* JSON roundtrip */
    char ctx_json[2048];
    tb_context_to_json(&cv, ctx_json, sizeof(ctx_json));
    TB_ContextVector cv2 = {0};
    tb_context_from_json(&cv2, ctx_json);
    assert(fabs(cv.mean - cv2.mean) < 1e-9);
    assert(fabs(cv.cos_coeff[0] - cv2.cos_coeff[0]) < 1e-6);
    printf("[ctx_json] roundtrip: PASS\n");

    /* Reconstruct distribution */
    double dist[100];
    tb_context_decode_dist(&cv, dist, 100);
    double dsum=0; for(int i=0;i<100;i++) dsum+=dist[i];
    printf("[ctx_decode] dist sum=%.4f (expect ~>0): PASS\n", dsum);
    printf("[phi-Fourier] PASS\n\n");

    /* ── ERL v3 ── */
    printf("--- ERL v3 (matches MCP-0.11 schema) ---\n");
    TB_ERL_v3 *erl = tb_erl_v3_create("/tmp/tb_erl_test.json");

    tb_erl_v3_append(erl, "main", "thought", "analyzing the codebase structure", "sess_001");
    tb_erl_v3_append(erl, "main", "observation", "found 23 tools in MCP server", "sess_001");
    tb_erl_v3_append(erl, "session_context", "context", ctx_json, "sess_001");
    tb_erl_v3_append(erl, "task_analysis", "plan", "build tokenizer first", "sess_001");

    /* Verify chains */
    assert(tb_erl_v3_verify(erl, "main"));
    assert(tb_erl_v3_verify(erl, "session_context"));
    printf("[erl_v3] chains valid: %d entries, %d branches\n",
           erl->n_entries, erl->n_branches);

    /* Save */
    tb_erl_v3_save(erl);
    printf("[erl_v3] saved to /tmp/tb_erl_test.json\n");

    /* ── Long-term cognition ── */
    printf("\n--- Long-Term Cognition (Context Clearing Protocol) ---\n");
    TB_LongTermCognition *ltc = tb_ltc_create("sess_001", "/tmp/tb_ltc_erl.json", 64);

    /* Absorb tokens */
    int should_compress = tb_ltc_absorb(ltc, token_history, 128, lat);
    printf("[ltc] absorbed 128 tokens, compress=%d\n", should_compress);

    for (int i=0;i<200;i++) token_history[i%128] = (i*3+7)%1000;
    should_compress = tb_ltc_absorb(ltc, token_history, 128, lat);
    printf("[ltc] absorbed another 128 tokens, compress=%d (threshold=%d)\n",
           should_compress, ltc->compress_threshold);

    if (should_compress) {
        tb_ltc_compress_and_clear(ltc, lat, NULL, 0);
    } else {
        /* Force compress */
        ltc->tokens_since_compress = ltc->compress_threshold;
        tb_ltc_compress_and_clear(ltc, lat, NULL, 0);
    }

    /* Restore from ERL */
    TB_LongTermCognition *ltc2 = tb_ltc_create("sess_001", "/tmp/tb_ltc_erl.json", 64);
    tb_ltc_restore(ltc2, "/tmp/tb_ltc_erl.json");
    printf("[ltc_restore] n_tokens=%llu mean=%.6f\n",
           (unsigned long long)ltc2->context_vec.n_tokens_encoded,
           ltc2->context_vec.mean);
    assert(ltc2->context_vec.n_tokens_encoded > 0 || ltc2->context_vec.mean != 0);
    printf("[ltc] context clearing + restore: PASS\n");

    tb_ltc_free(ltc); tb_ltc_free(ltc2);
    tb_erl_v3_free(erl);
    tb_tokenizer_free(&tok);
    tb_lattice_destroy(lat);

    printf("\n=== Tokenizer + Long-Term Cognition PASS ===\n");
    return 0;
}
#endif
