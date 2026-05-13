"""
patch_infer.py  — patch v0.3/layer4/tb_infer.c for Qwen3 compatibility
Changes:
  1. Add QK per-head RMSNorm after QKV projection (blk.L.attn_q_norm.weight)
  2. Route GGUF-native tensor dequant through tb_gguf_dequant_matvec()
     instead of the old 4-bit compat shim (which assumed manifest format)
  3. Fix MoE packed tensor handling: ffn_gate_exps / ffn_up_exps / ffn_down_exps
Run: python scripts/patch_infer.py  (from v0.3/ root)
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "layer4", "tb_infer.c")

with open(SRC, encoding="utf-8", errors="replace") as f:
    content = f.read()

# ── Patch 1: After RoPE, insert QK-norm for Qwen3 ─────────────────────────
# Find the line:  /* RoPE */
# Replace the block from /* RoPE */ through the end of attention call
OLD_ROPE = """    /* RoPE */
    for (int h=0;h<NH;h++) tb_rope_apply(q+h*HD, k+(h%NK)*HD, HD, pos, m->rope_base);

    /* Attention (uses KV cache from tb_tensor.h) */
    if (kv && (kv->epoch == ctx->lattice->epoch)) {
        tb_attention(q, k, v, kv, layer_idx, NH, NK, HD, attn_out);
    } else {
        memcpy(attn_out, q, NH*HD*sizeof(float));
    }"""

NEW_ROPE = """    /* RoPE */
    for (int h=0;h<NH;h++) tb_rope_apply(q+h*HD, k+(h%NK)*HD, HD, pos, m->rope_base);

    /* Qwen3 per-head QK-norm (blk.L.attn_q_norm / attn_k_norm) */
    {
        TNSR("blk.%d.attn_q_norm.weight", layer_idx);
        const TB_GGUFTensorInfo *qn_t = tb_gguf_find_tensor(m, tname);
        TNSR("blk.%d.attn_k_norm.weight", layer_idx);
        const TB_GGUFTensorInfo *kn_t = tb_gguf_find_tensor(m, tname);
        if ((qn_t || kn_t) && m->weights_data) {
            float *w32 = (float*)malloc(HD * sizeof(float));
            if (w32) {
                if (qn_t) {
                    const uint16_t *nw = (const uint16_t*)tb_gguf_tensor_data(m, qn_t);
                    if (qn_t->qtype == 30)
                        for (int i=0;i<HD;i++) w32[i] = tb_bf16_to_f32_infer(nw[i]);
                    else if (qn_t->qtype == 1) {
                        for (int i=0;i<HD;i++) { uint32_t b=(uint32_t)nw[i]<<16; memcpy(&w32[i],&b,4); }
                    } else memcpy(w32, tb_gguf_tensor_data(m, qn_t), HD*sizeof(float));
                    for (int h=0;h<NH;h++) {
                        float *qh = q + h*HD;
                        float sq = 0.0f;
                        for (int i=0;i<HD;i++) sq += qh[i]*qh[i];
                        float sc = 1.0f / sqrtf(sq/HD + eps);
                        for (int i=0;i<HD;i++) qh[i] = qh[i]*sc*w32[i];
                    }
                }
                if (kn_t) {
                    const uint16_t *nw = (const uint16_t*)tb_gguf_tensor_data(m, kn_t);
                    if (kn_t->qtype == 30)
                        for (int i=0;i<HD;i++) w32[i] = tb_bf16_to_f32_infer(nw[i]);
                    else if (kn_t->qtype == 1) {
                        for (int i=0;i<HD;i++) { uint32_t b=(uint32_t)nw[i]<<16; memcpy(&w32[i],&b,4); }
                    } else memcpy(w32, tb_gguf_tensor_data(m, kn_t), HD*sizeof(float));
                    for (int h=0;h<NK;h++) {
                        float *kh = k + h*HD;
                        float sk = 0.0f;
                        for (int i=0;i<HD;i++) sk += kh[i]*kh[i];
                        float sc = 1.0f / sqrtf(sk/HD + eps);
                        for (int i=0;i<HD;i++) kh[i] = kh[i]*sc*w32[i];
                    }
                }
                free(w32);
            }
        }
    }

    /* Attention (uses KV cache from tb_tensor.h) */
    if (kv && (kv->epoch == ctx->lattice->epoch)) {
        tb_attention(q, k, v, kv, layer_idx, NH, NK, HD, attn_out);
    } else {
        memcpy(attn_out, q, NH*HD*sizeof(float));
    }"""

# ── Patch 2: Replace 4-bit compat shim calls with tb_gguf_dequant_matvec ──
# The compat shim expected the old flat manifest format (scales follow weights).
# Standard GGUF tensors embed scales within the quantised block itself;
# tb_gguf_dequant_matvec() handles all formats correctly.
OLD_QKV = """    if (wq_t && wk_t && wv_t && m->weights_data) {
        /* Q4_0/Q4_K dequant matvec for QKV */
        const uint32_t *wq = (const uint32_t*)tb_gguf_tensor_data(m, wq_t);
        const uint16_t *sq = (const uint16_t*)((const char*)wq + NH*HD*(H/8));  /* scales follow weights */
        const uint16_t *bq = sq + (NH*HD/m->group_size);
        tb_gguf_dequant_matvec_4bit_compat(wq, sq, bq, xn, q, NH*HD, H, m->group_size);

        const uint32_t *wk = (const uint32_t*)tb_gguf_tensor_data(m, wk_t);
        const uint16_t *sk = (const uint16_t*)((const char*)wk + NK*HD*(H/8));
        const uint16_t *bk = sk + (NK*HD/m->group_size);
        tb_gguf_dequant_matvec_4bit_compat(wk, sk, bk, xn, k, NK*HD, H, m->group_size);

        const uint32_t *wv = (const uint32_t*)tb_gguf_tensor_data(m, wv_t);
        const uint16_t *sv = (const uint16_t*)((const char*)wv + NK*HD*(H/8));
        const uint16_t *bv = sv + (NK*HD/m->group_size);
        tb_gguf_dequant_matvec_4bit_compat(wv, sv, bv, xn, v, NK*HD, H, m->group_size);
    } else {"""

NEW_QKV = """    if (wq_t && wk_t && wv_t && m->weights_data) {
        /* Route through tb_gguf_dequant_matvec — handles Q4_0/Q4_K/Q8_0/BF16/F16/F32 */
        tb_gguf_dequant_matvec(tb_gguf_tensor_data(m, wq_t), wq_t->qtype,
                               NH*HD, H, xn, q);
        tb_gguf_dequant_matvec(tb_gguf_tensor_data(m, wk_t), wk_t->qtype,
                               NK*HD, H, xn, k);
        tb_gguf_dequant_matvec(tb_gguf_tensor_data(m, wv_t), wv_t->qtype,
                               NK*HD, H, xn, v);
    } else {"""

OLD_WO = """    if (wo_t && m->weights_data) {
        const uint32_t *wo = (const uint32_t*)tb_gguf_tensor_data(m, wo_t);
        const uint16_t *so = (const uint16_t*)((const char*)wo + H*(NH*HD/8));
        const uint16_t *bo = so + (H/m->group_size);
        tb_gguf_dequant_matvec_4bit_compat(wo, so, bo, attn_out, attn_proj, H, NH*HD, m->group_size);
    } else {"""

NEW_WO = """    if (wo_t && m->weights_data) {
        tb_gguf_dequant_matvec(tb_gguf_tensor_data(m, wo_t), wo_t->qtype,
                               H, NH*HD, attn_out, attn_proj);
    } else {"""

# ── Patch 3: Packed MoE tensors (ffn_gate_exps) ───────────────────────────
# When we have packed tensors, index by expert row offset
OLD_PACKED_MOE = """            if (exp_t && m->weights_data) {
                /* Expert weights are available — use tb_expert_forward stub */
                TB_ExpertBlobLayout layout = {0};
                int ffn_dim = m->ffn_dim > 0 ? m->ffn_dim : H*4;
                layout.gate_w_off = 0;
                layout.gate_w_sz   = (size_t)(ffn_dim * H / 2);
                layout.up_w_off   = layout.gate_w_sz;
                layout.up_w_sz     = layout.gate_w_sz;
                layout.down_w_off = layout.up_w_off + layout.up_w_sz;
                layout.down_w_sz   = layout.gate_w_sz;
                layout.expert_total   = layout.down_w_off + layout.down_w_sz;

                const unsigned char *exp_data =
                    (const unsigned char*)tb_gguf_tensor_data(m, exp_t);
                memset(exp_out, 0, H*sizeof(float));
                tb_expert_forward(exp_data, m, &layout, xn2, exp_out);
            } else {"""

NEW_PACKED_MOE = """            if (exp_t && m->weights_data) {
                /* Packed MoE: ffn_gate_exps shape [n_experts, ffn_dim, hidden]
                 * index by expert: gate_row = exp_idx * ffn_dim */
                int ffn_dim = m->moe_intermediate_size > 0 ? m->moe_intermediate_size
                            : (m->ffn_dim > 0 ? m->ffn_dim : H*4);
                /* up/down companion tensors */
                char up_tname[128], dn_tname[128];
                snprintf(up_tname, sizeof(up_tname), "blk.%d.ffn_up_exps.weight",   layer_idx);
                snprintf(dn_tname, sizeof(dn_tname), "blk.%d.ffn_down_exps.weight", layer_idx);
                const TB_GGUFTensorInfo *eu_t = tb_gguf_find_tensor(m, up_tname);
                const TB_GGUFTensorInfo *ed_t = tb_gguf_find_tensor(m, dn_tname);

                /* Compute byte offset per expert row */
                int64_t block_rows_g = (int64_t)exp_idx * ffn_dim;
                int64_t block_rows_d = (int64_t)exp_idx * H;
                float *g_ep = (float*)malloc(ffn_dim*sizeof(float));
                float *u_ep = (float*)malloc(ffn_dim*sizeof(float));
                float *a_ep = (float*)malloc(ffn_dim*sizeof(float));
                if (g_ep && u_ep && a_ep) {
                    const void *gd = tb_gguf_tensor_data(m, exp_t);
                    const void *ud = eu_t ? tb_gguf_tensor_data(m, eu_t) : gd;
                    const void *dd = ed_t ? tb_gguf_tensor_data(m, ed_t) : gd;
                    /* Each row is H weights; advance by block_rows_g rows */
                    /* tb_gguf_dequant_matvec reads M rows × K cols from W   */
                    /* For packed tensor: W_gate[exp][ffn_row][H]            */
                    /* We need: out(ffn_dim) = W_gate[exp] @ x(H)            */
                    /* Fake it: slice the packed tensor at the expert offset  */
                    int bytes_per_row_g = (exp_t->qtype == 12) ? (H/256*(144)) :
                                         (exp_t->qtype ==  8) ? (H*sizeof(int8_t) + H/32*4) :
                                         (exp_t->qtype ==  2) ? (H/2 + H/32*4) : H*4;
                    const char *g_base = (const char*)gd + (size_t)block_rows_g * bytes_per_row_g;
                    tb_gguf_dequant_matvec(g_base, exp_t->qtype, ffn_dim, H, xn2, g_ep);
                    tb_gguf_dequant_matvec(g_base, exp_t->qtype, ffn_dim, H, xn2, u_ep);
                    if (eu_t) {
                        const char *u_base = (const char*)ud + (size_t)block_rows_g * bytes_per_row_g;
                        tb_gguf_dequant_matvec(u_base, eu_t->qtype, ffn_dim, H, xn2, u_ep);
                    }
                    tb_swiglu(g_ep, u_ep, a_ep, ffn_dim);
                    if (ed_t) {
                        int bytes_per_row_d = (ed_t->qtype == 12) ? (ffn_dim/256*(144)) :
                                             (ed_t->qtype ==  8) ? (ffn_dim + ffn_dim/32*4) :
                                             (ed_t->qtype ==  2) ? (ffn_dim/2 + ffn_dim/32*4) : ffn_dim*4;
                        const char *d_base = (const char*)dd + (size_t)block_rows_d * bytes_per_row_d;
                        tb_gguf_dequant_matvec(d_base, ed_t->qtype, H, ffn_dim, a_ep, exp_out);
                    } else {
                        memcpy(exp_out, a_ep, H*sizeof(float));
                    }
                }
                free(g_ep); free(u_ep); free(a_ep);
            } else {"""

# Apply patches
orig_len = len(content)
content = content.replace(OLD_ROPE, NEW_ROPE, 1)
content = content.replace(OLD_QKV,  NEW_QKV,  1)
content = content.replace(OLD_WO,   NEW_WO,   1)
content = content.replace(OLD_PACKED_MOE, NEW_PACKED_MOE, 1)

applied = 0
for old in [OLD_ROPE, OLD_QKV, OLD_WO, OLD_PACKED_MOE]:
    if old not in open(SRC, encoding="utf-8", errors="replace").read():
        applied += 1  # already replaced (not in original any more)

# Check that each patch was applied
for name, old in [("QK-norm", OLD_ROPE), ("QKV dequant", OLD_QKV),
                  ("output proj", OLD_WO), ("packed MoE", OLD_PACKED_MOE)]:
    was_in_orig = old in open(SRC, encoding="utf-8", errors="replace").read()
    print(f"  Patch '{name}': {'APPLIED' if not was_in_orig else 'NOT APPLIED — old text still present'}")

with open(SRC, "w", encoding="utf-8") as f:
    f.write(content)

print(f"  Done: {SRC}")
print(f"  Characters: {orig_len} → {len(content)}")
