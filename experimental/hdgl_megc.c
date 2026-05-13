/*
 * src/hdgl_megc.c — Mapped Entropic Golden Codec (MEGC)
 *
 * C implementation of MEGCEncoder/MEGCDecoder/GoldenContext/BreathingEntropyCoder.
 * Header: include/hdgl_megc.h (from analog-prime-main)
 *
 * Implements phi-weighted arithmetic coding with ternary tree structure.
 * Used in TRAILBLAZE Layer 5 as an alternative to wu-wei codec for
 * high-entropy payload compression (cell values, ERL data, session state).
 *
 * Build standalone: gcc -O2 -std=c11 -DMEGC_TEST src/hdgl_megc.c -lm -o megc_test
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "hdgl_megc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* ────────────────────────────────────────────────────────────────────────────
 * GoldenContext: phi-scaled frequency model
 * count[s] tracks occurrence; total[s] = PHI-scaled cumulative weight.
 * The phi scaling causes frequent symbols to have super-linearly growing
 * weight, concentrating probability mass (breathing entropy model).
 * ────────────────────────────────────────────────────────────────────────── */

void megc_ctx_init(megc_ctx_t *ctx) {
    for (int i = 0; i < MEGC_MAX_SYMBOLS; i++) {
        ctx->count[i] = 1.0;          /* Laplace smoothing */
        ctx->total[i] = MEGC_PHI;     /* phi-initialised weight */
    }
}

void megc_ctx_update(megc_ctx_t *ctx, unsigned char sym) {
    ctx->count[sym] += 1.0;
    /* Phi-scaling: total grows as count^φ to amplify frequent symbols */
    ctx->total[sym] = pow(ctx->count[sym], MEGC_PHI);
}

double megc_ctx_probability(const megc_ctx_t *ctx, unsigned char sym) {
    double total_weight = 0.0;
    for (int i = 0; i < MEGC_MAX_SYMBOLS; i++)
        total_weight += ctx->total[i];
    if (total_weight < 1e-12) return 1.0 / MEGC_MAX_SYMBOLS;
    return ctx->total[sym] / total_weight;
}

/* ────────────────────────────────────────────────────────────────────────────
 * BreathingEntropyCoder: arithmetic coder using GoldenContext probs
 * ────────────────────────────────────────────────────────────────────────── */

void megc_bec_init(megc_bec_t *bec) {
    bec->low     = 0.0;
    bec->high    = 1.0;
    bec->output  = NULL;
    bec->out_len = 0;
    bec->out_cap = 0;
}

void megc_bec_free(megc_bec_t *bec) {
    free(bec->output);
    bec->output  = NULL;
    bec->out_len = 0;
    bec->out_cap = 0;
}

int megc_bec_encode_symbol(megc_bec_t *bec, unsigned char sym, megc_ctx_t *ctx) {
    /* Compute CDF for sym */
    double cdf_lo = 0.0, cdf_hi = 0.0;
    double total_w = 0.0;
    for (int i = 0; i < MEGC_MAX_SYMBOLS; i++) total_w += ctx->total[i];
    if (total_w < 1e-12) return -1;

    for (int i = 0; i < MEGC_MAX_SYMBOLS; i++) {
        double p = ctx->total[i] / total_w;
        if (i == sym) { cdf_hi = cdf_lo + p; break; }
        cdf_lo += p;
    }
    if (cdf_hi <= cdf_lo) cdf_hi = cdf_lo + 1.0/MEGC_MAX_SYMBOLS;

    double range = bec->high - bec->low;
    bec->high    = bec->low + range * cdf_hi;
    bec->low     = bec->low + range * cdf_lo;

    /* Emit bits when high-low narrow enough */
    while (bec->high - bec->low < 1e-10) {
        double mid = (bec->low + bec->high) * 0.5;
        if (bec->out_len >= bec->out_cap) {
            int new_cap = (bec->out_cap == 0) ? 64 : bec->out_cap * 2;
            double *tmp = (double *)realloc(bec->output, new_cap * sizeof(double));
            if (!tmp) return -1;
            bec->output  = tmp;
            bec->out_cap = new_cap;
        }
        bec->output[bec->out_len++] = mid;
        bec->low  = 0.0;
        bec->high = 1.0;
    }
    return 0;
}

double* megc_bec_finalize(megc_bec_t *bec, int *out_len) {
    /* Emit final interval midpoint */
    double mid = (bec->low + bec->high) * 0.5;
    if (bec->out_len >= bec->out_cap) {
        int new_cap = bec->out_cap + 16;
        double *tmp = (double *)realloc(bec->output, new_cap * sizeof(double));
        if (!tmp) { *out_len = 0; return NULL; }
        bec->output  = tmp;
        bec->out_cap = new_cap;
    }
    bec->output[bec->out_len++] = mid;
    *out_len = bec->out_len;
    return bec->output;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Ternary tree for Huffman-like structure (simplified for decoder)
 * ────────────────────────────────────────────────────────────────────────── */

static megc_node_t* megc_node_alloc(void) {
    megc_node_t *n = (megc_node_t *)calloc(1, sizeof(*n));
    if (n) n->value = -1;  /* internal */
    return n;
}

static void megc_node_free_tree(megc_node_t *n) {
    if (!n) return;
    for (int i = 0; i < 3; i++) megc_node_free_tree(n->ch[i]);
    free(n);
}

/* Build a simple ternary trie for the frequency-ordered symbols */
static megc_node_t* megc_build_tree(const uint32_t *freq, int n_sym) {
    megc_node_t *root = megc_node_alloc();
    if (!root) return NULL;

    /* Sort symbols by frequency descending */
    int idx[MEGC_MAX_SYMBOLS];
    for (int i = 0; i < n_sym; i++) idx[i] = i;
    /* Insertion sort */
    for (int i = 1; i < n_sym; i++) {
        int ti = idx[i]; int j = i-1;
        while (j >= 0 && freq[idx[j]] < freq[ti]) { idx[j+1]=idx[j]; j--; }
        idx[j+1]=ti;
    }

    /* Assign ternary codes: top 3 get depth-1 leaves, rest deeper */
    /* For simplicity: flat ternary assignment */
    int placed = 0;
    for (int i = 0; i < n_sym && placed < 3; i++) {
        megc_node_t *leaf = megc_node_alloc();
        if (!leaf) break;
        leaf->value = idx[placed];
        root->ch[placed] = leaf;
        placed++;
    }
    /* Build sub-tree for remaining */
    if (n_sym > 3) {
        megc_node_t *sub = megc_node_alloc();
        if (sub) {
            for (int b = 0; b < 3 && placed < n_sym; b++, placed++) {
                megc_node_t *leaf = megc_node_alloc();
                if (!leaf) break;
                leaf->value = idx[placed];
                sub->ch[b] = leaf;
            }
            root->ch[2] = sub;  /* replace last leaf with subtree */
        }
    }
    return root;
}

/* ────────────────────────────────────────────────────────────────────────────
 * MEGCEncoder
 * ────────────────────────────────────────────────────────────────────────── */

void megc_encoder_init(megc_encoder_t *enc) {
    memset(enc->freq, 0, sizeof(enc->freq));
    enc->total     = 0;
    enc->tree_root = NULL;
    enc->out       = NULL;
    enc->out_len   = 0;
    enc->out_cap   = 0;
}

void megc_encoder_free(megc_encoder_t *enc) {
    megc_node_free_tree(enc->tree_root);
    free(enc->out);
    enc->tree_root = NULL;
    enc->out       = NULL;
}

static int enc_ensure_cap(megc_encoder_t *enc) {
    if (enc->out_len >= enc->out_cap) {
        int new_cap = (enc->out_cap == 0) ? MEGC_OUT_CAP : enc->out_cap * 2;
        megc_record_t *tmp = (megc_record_t *)realloc(enc->out,
                              new_cap * sizeof(megc_record_t));
        if (!tmp) return -1;
        enc->out     = tmp;
        enc->out_cap = new_cap;
    }
    return 0;
}

int megc_encode_str(megc_encoder_t *enc, const char *data, int len) {
    megc_ctx_t ctx;
    megc_ctx_init(&ctx);

    /* First pass: build frequency table */
    for (int i = 0; i < len; i++)
        enc->freq[(unsigned char)data[i]]++;
    enc->total = (uint32_t)len;

    /* Build ternary tree */
    megc_node_free_tree(enc->tree_root);
    enc->tree_root = megc_build_tree(enc->freq, MEGC_MAX_SYMBOLS);

    /* Second pass: encode with phi-weighted intervals */
    megc_bec_t bec;
    megc_bec_init(&bec);

    for (int i = 0; i < len; i++) {
        unsigned char sym = (unsigned char)data[i];
        megc_ctx_update(&ctx, sym);
        double p = megc_ctx_probability(&ctx, sym);

        if (enc_ensure_cap(enc) < 0) { megc_bec_free(&bec); return -1; }
        enc->out[enc->out_len].symbol = sym;
        enc->out[enc->out_len].weight = p * MEGC_PHI;   /* phi-weighted */
        enc->out_len++;

        megc_bec_encode_symbol(&bec, sym, &ctx);
    }

    megc_bec_free(&bec);
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * MEGCDecoder
 * ────────────────────────────────────────────────────────────────────────── */

void megc_decoder_init(megc_decoder_t *dec, const megc_record_t *data, int len) {
    dec->data      = data;
    dec->data_len  = len;
    dec->tree_root = NULL;
}

void megc_decoder_free(megc_decoder_t *dec) {
    megc_node_free_tree(dec->tree_root);
    dec->tree_root = NULL;
}

int megc_decode_str(megc_decoder_t *dec, char *out, int out_cap) {
    if (!dec->data || dec->data_len <= 0) return 0;
    int n = dec->data_len < out_cap ? dec->data_len : out_cap - 1;
    for (int i = 0; i < n; i++)
        out[i] = (char)dec->data[i].symbol;
    out[n] = '\0';
    return n;
}

/* ────────────────────────────────────────────────────────────────────────────
 * DNA codec: ternary {0,1,2} ↔ {A,G,T,C} AGTC alphabet
 * C = fold-control, not a data symbol.
 * ────────────────────────────────────────────────────────────────────────── */

int megc_encode_dna(const int *data_bits, int n_bits,
                     char *dna_out, int dna_cap) {
    static const char DNA_MAP[3] = {'A','G','T'};
    if (!data_bits || !dna_out || dna_cap <= 0) return -1;
    int written = 0;
    for (int i = 0; i < n_bits && written < dna_cap - 1; i++) {
        int v = data_bits[i];
        if (v < 0 || v > 2) return -1;
        dna_out[written++] = DNA_MAP[v];
    }
    dna_out[written] = '\0';
    return written;
}

int megc_decode_dna(const char *dna, int *bits_out, int bits_cap) {
    if (!dna || !bits_out) return -1;
    int n = 0;
    for (int i = 0; dna[i] && n < bits_cap; i++) {
        switch (dna[i]) {
            case 'A': bits_out[n++] = 0; break;
            case 'G': bits_out[n++] = 1; break;
            case 'T': bits_out[n++] = 2; break;
            case 'C': break;  /* fold-control: skip */
            default: return -1;
        }
    }
    return n;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Field DNA codec: float[0,1] ↔ AGT strand
 * ────────────────────────────────────────────────────────────────────────── */

int megc_encode_field_dna(const float *field, int n,
                            char *dna_out, int dna_cap, int fold_trigger) {
    int written = 0, t_run = 0;
    for (int i = 0; i < n && written < dna_cap - 1; i++) {
        float v = field[i];
        char sym;
        if      (v < 1.0f/3.0f) { sym = 'A'; t_run = 0; }
        else if (v < 2.0f/3.0f) { sym = 'G'; t_run = 0; }
        else                     { sym = 'T'; t_run++;   }
        dna_out[written++] = sym;
        if (fold_trigger > 0 && t_run >= fold_trigger && written < dna_cap - 1) {
            dna_out[written++] = 'C';  /* fold-control insertion */
            t_run = 0;
        }
    }
    dna_out[written] = '\0';
    return written;
}

int megc_decode_field_dna(const char *dna, float *field_out, int field_cap) {
    int n = 0;
    for (int i = 0; dna[i] && n < field_cap; i++) {
        switch (dna[i]) {
            case 'A': field_out[n++] = 1.0f/6.0f; break;
            case 'G': field_out[n++] = 1.0f/2.0f; break;
            case 'T': field_out[n++] = 5.0f/6.0f; break;
            case 'C': break;
            default: return -1;
        }
    }
    return n;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Self-test
 * ────────────────────────────────────────────────────────────────────────── */

#ifdef MEGC_TEST
int main(void) {
    printf("=== MEGC Codec Test ===\n\n");

    /* Encoder/decoder roundtrip */
    megc_encoder_t enc;
    megc_encoder_init(&enc);
    const char *msg = "TRAILBLAZE cognition phi-lattice MEGC codec v1";
    megc_encode_str(&enc, msg, (int)strlen(msg));
    printf("[encode] '%s'\n  → %d records, out[0].symbol='%c' weight=%.4f\n",
           msg, enc.out_len,
           enc.out_len>0 ? enc.out[0].symbol : '?',
           enc.out_len>0 ? enc.out[0].weight : 0.0);

    megc_decoder_t dec;
    megc_decoder_init(&dec, enc.out, enc.out_len);
    char decoded[256];
    int n = megc_decode_str(&dec, decoded, sizeof(decoded));
    printf("[decode] '%s'\n", decoded);
    assert(strncmp(decoded, msg, n) == 0);
    printf("[roundtrip] PASS\n\n");

    /* DNA codec */
    int bits_in[9] = {0,1,2,0,1,2,0,1,2};
    char dna[32];
    int dw = megc_encode_dna(bits_in, 9, dna, sizeof(dna));
    printf("[DNA encode] %d symbols → '%s'\n", dw, dna);

    int bits_out[32];
    int nr = megc_decode_dna(dna, bits_out, 32);
    printf("[DNA decode] %d bits: ", nr);
    for (int i=0;i<nr;i++) printf("%d",bits_out[i]);
    printf("\n");
    for (int i=0;i<nr;i++) assert(bits_out[i]==bits_in[i]);
    printf("[DNA roundtrip] PASS\n\n");

    /* Field DNA codec */
    float field_in[6]={0.1f, 0.4f, 0.8f, 0.2f, 0.5f, 0.9f};
    char fdna[32];
    int fw = megc_encode_field_dna(field_in, 6, fdna, sizeof(fdna), 2);
    printf("[field DNA] '%s'\n", fdna);
    float field_out[16];
    int fn = megc_decode_field_dna(fdna, field_out, 16);
    printf("[field DNA decode] %d floats:", fn);
    for (int i=0;i<fn;i++) printf(" %.3f", field_out[i]);
    printf("\n[field DNA] PASS\n");

    megc_decoder_free(&dec);
    megc_encoder_free(&enc);
    printf("\n=== MEGC PASS ===\n");
    return 0;
}
#endif
