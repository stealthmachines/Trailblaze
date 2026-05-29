#pragma once
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
    int    n_layers;
    int    n_kv_heads;
    int    head_dim;
    int    max_seq;
    int    seq_len;
    int    slot;
    float *k_cache;   /* [n_layers][max_seq][n_kv_heads * head_dim] */
    float *v_cache;
} TB_KVCache;

static inline TB_KVCache* tb_kvcache_alloc(int nl, int nkv, int hd, int mseq, int slot, int32_t epoch) {
    (void)epoch;
    TB_KVCache *kv = (TB_KVCache*)calloc(1, sizeof(TB_KVCache));
    if (!kv) return NULL;
    kv->n_layers   = nl;
    kv->n_kv_heads = nkv;
    kv->head_dim   = hd;
    kv->max_seq    = mseq;
    kv->seq_len    = 0;
    kv->slot       = slot;
    size_t sz = (size_t)nl * mseq * nkv * hd * sizeof(float);
    kv->k_cache = (float*)calloc(1, sz);
    kv->v_cache = (float*)calloc(1, sz);
    if (!kv->k_cache || !kv->v_cache) { free(kv->k_cache); free(kv->v_cache); free(kv); return NULL; }
    fprintf(stderr, "[kvcache] alloc slot=%d nl=%d nkv=%d hd=%d mseq=%d => %.1f MB\n",
            slot, nl, nkv, hd, mseq, 2.0*(double)sz/1e6);
    return kv;
}
static inline void tb_kvcache_free(TB_KVCache *kv) {
    if (!kv) return; free(kv->k_cache); free(kv->v_cache); free(kv);
}

/* GQA softmax attention.
 * q[NH*HD], k[NK*HD], v[NK*HD] for current token.
 * Writes k,v into cache at pos, then attends over [0..pos].
 * attn_out[NH*HD]. */
static inline void tb_attention(
    const float *q, const float *k, const float *v,
    TB_KVCache *kv, int layer_idx, int NH, int NK, int HD, int pos,
    float *attn_out)
{
    if (!kv) { memcpy(attn_out, q, (size_t)NH*HD*sizeof(float)); return; }
    int mseq = kv->max_seq;
    if (pos >= mseq) pos = mseq - 1;

    size_t layer_stride = (size_t)mseq * NK * HD;
    float *kc = kv->k_cache + (size_t)layer_idx * layer_stride;
    float *vc = kv->v_cache + (size_t)layer_idx * layer_stride;
    memcpy(kc + (size_t)pos * NK * HD, k, (size_t)NK * HD * sizeof(float));
    memcpy(vc + (size_t)pos * NK * HD, v, (size_t)NK * HD * sizeof(float));
    if (pos + 1 > kv->seq_len) kv->seq_len = pos + 1;

    float scale = 1.0f / sqrtf((float)HD);
    float *scores = (float*)malloc((pos + 1) * sizeof(float));
    if (!scores) { memcpy(attn_out, q, (size_t)NH*HD*sizeof(float)); return; }

    for (int h = 0; h < NH; h++) {
        const float *qh = q + h * HD;
        int kv_h = (NK > 0) ? (h * NK / NH) : 0;
        float *out_h = attn_out + h * HD;

        float max_s = -1e38f;
        for (int t = 0; t <= pos; t++) {
            const float *kt = kc + (size_t)t * NK * HD + (size_t)kv_h * HD;
            float s = 0.0f;
            for (int d = 0; d < HD; d++) s += qh[d] * kt[d];
            s *= scale;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - max_s); sum += scores[t]; }
        if (sum > 0.0f) for (int t = 0; t <= pos; t++) scores[t] /= sum;

        memset(out_h, 0, (size_t)HD * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *vt = vc + (size_t)t * NK * HD + (size_t)kv_h * HD;
            float w = scores[t];
            for (int d = 0; d < HD; d++) out_h[d] += w * vt[d];
        }
    }
    free(scores);
}

/* GQA attention with separate Q/K and V head dims.
 * Handles architectures (e.g. Qwen3.5) where HD_k != HD_v:
 *   q[NH*HD_k], k[NK_k*HD_k], v[NK_v*HD_v]  -- current token
 *   NK_k*HD_k == NK_v*HD_v  (same total KV cache slot size)
 *   attn_out[NH*HD_v]                          -- output
 */
static inline void tb_attention_split_kv(
    const float *q, const float *k, const float *v,
    TB_KVCache *kv, int layer_idx, int NH,
    int NK_k, int HD_k,
    int NK_v, int HD_v,
    int pos, float *attn_out)
{
    if (!kv) { memset(attn_out, 0, (size_t)NH*HD_v*sizeof(float)); return; }
    int mseq = kv->max_seq;
    if (pos >= mseq) pos = mseq - 1;

    /* K and V share the same per-position slot size in the cache */
    int slot = kv->n_kv_heads * kv->head_dim;   /* = NK_k*HD_k = NK_v*HD_v */
    if (slot <= 0) slot = NK_k * HD_k;
    size_t layer_stride = (size_t)mseq * slot;
    float *kc = kv->k_cache + (size_t)layer_idx * layer_stride;
    float *vc = kv->v_cache + (size_t)layer_idx * layer_stride;
    memcpy(kc + (size_t)pos * slot, k, (size_t)NK_k * HD_k * sizeof(float));
    memcpy(vc + (size_t)pos * slot, v, (size_t)NK_v * HD_v * sizeof(float));
    if (pos + 1 > kv->seq_len) kv->seq_len = pos + 1;

    float scale = 1.0f / sqrtf((float)HD_k);
    float *scores = (float*)malloc((pos + 1) * sizeof(float));
    if (!scores) { memset(attn_out, 0, (size_t)NH*HD_v*sizeof(float)); return; }

    for (int h = 0; h < NH; h++) {
        const float *qh  = q + (size_t)h * HD_k;
        int kv_k = (NK_k > 0) ? (h * NK_k / NH) : 0;
        int kv_v = (NK_v > 0) ? (h * NK_v / NH) : 0;
        float *out_h = attn_out + (size_t)h * HD_v;

        float max_s = -1e38f;
        for (int t = 0; t <= pos; t++) {
            const float *kt = kc + (size_t)t * slot + (size_t)kv_k * HD_k;
            float s = 0.0f;
            for (int d = 0; d < HD_k; d++) s += qh[d] * kt[d];
            s *= scale;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - max_s); sum += scores[t]; }
        if (sum > 0.0f) for (int t = 0; t <= pos; t++) scores[t] /= sum;

        memset(out_h, 0, (size_t)HD_v * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *vt = vc + (size_t)t * slot + (size_t)kv_v * HD_v;
            float w = scores[t];
            for (int d = 0; d < HD_v; d++) out_h[d] += w * vt[d];
        }
    }
    free(scores);
}
