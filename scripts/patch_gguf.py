"""
patch_gguf.py  — apply Qwen3 KV-key support to v0.3/layer4/tb_gguf.c
Run: python scripts/patch_gguf.py  (from v0.3/ root)
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "layer4", "tb_gguf.c")

with open(SRC, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# ── The old KV block is lines 362–412 (0-indexed: 361–411) ──────────────
# We'll replace everything from "if (strstr(key,\"general.architecture\"))"
# through "else if (strstr(key,\"attention.layer_norm_rms_epsilon\"))"
# (the closing brace of that else-if).

# Find boundaries
start_ln = None   # first line of old block
end_ln   = None   # last line of old block (inclusive)
for i, l in enumerate(lines):
    if 'if (strstr(key,"general.architecture"))' in l and start_ln is None:
        start_ln = i
    if 'attention.layer_norm_rms_epsilon' in l and start_ln is not None:
        # find the closing brace
        end_ln = i
        # skip the `}` line that follows
        if i + 2 < len(lines) and lines[i+1].strip() == "else gguf_skip_val(f,vtype);":
            end_ln = i + 3  # past the closing brace
        break

if start_ln is None or end_ln is None:
    print(f"ERROR: could not locate KV block (start={start_ln}, end={end_ln})")
    sys.exit(1)

print(f"Replacing lines {start_ln+1}–{end_ln} ({end_ln - start_ln} lines) with Qwen3-aware block")

NEW_BLOCK = """\
        if (strstr(key,"general.architecture")) {
            char *v=gguf_read_str(f); snprintf(g->arch,sizeof(g->arch),"%s",v); free(v);
        }
        else if (strstr(key,"vocab_size")||strstr(key,"n_vocab")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->vocab_size); }
            else gguf_skip_val(f,vtype);
        }
        /* hidden_dim: qwen3 uses "embedding_length" */
        else if (strstr(key,"n_embd")||strstr(key,"hidden_size")
                 ||strstr(key,"embedding_length")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->hidden_dim); }
            else gguf_skip_val(f,vtype);
        }
        /* n_layers: qwen3 uses "block_count" */
        else if (strstr(key,"n_layer")||strstr(key,"num_hidden_layers")
                 ||strstr(key,"block_count")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->n_layers); }
            else gguf_skip_val(f,vtype);
        }
        /* n_kv_heads: qwen3 uses "attention.head_count_kv" — must precede n_heads */
        else if (strstr(key,"n_head_kv")||strstr(key,"n_kv_heads")
                 ||strstr(key,"head_count_kv")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->n_kv_heads); }
            else gguf_skip_val(f,vtype);
        }
        /* n_heads: qwen3 uses "attention.head_count" */
        else if ((strstr(key,"n_head")||strstr(key,"num_attention_heads")
                  ||(strstr(key,"head_count")&&!strstr(key,"head_count_kv")))
                 && !strstr(key,"n_head_kv")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->n_heads); }
            else gguf_skip_val(f,vtype);
        }
        /* ffn_dim: qwen3 uses "feed_forward_length" */
        else if (strstr(key,"feed_forward_length")||strstr(key,"intermediate_size")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->ffn_dim); }
            else gguf_skip_val(f,vtype);
        }
        /* n_experts_per_tok: qwen3-moe uses "expert_used_count" */
        else if (strstr(key,"num_experts_per_tok")||strstr(key,"moe.top_k")
                 ||strstr(key,"expert_used_count")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->n_experts_per_tok); }
            else gguf_skip_val(f,vtype);
        }
        /* n_experts: qwen3-moe uses "expert_count" */
        else if ((strstr(key,"n_experts")||strstr(key,"num_experts")
                  ||strstr(key,"moe.num_experts")||strstr(key,"expert_count"))
                 && !strstr(key,"per_tok") && !strstr(key,"used")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->n_experts); }
            else gguf_skip_val(f,vtype);
        }
        /* rope_base: qwen3 uses 1,000,000 — value comes from GGUF */
        else if (strstr(key,"rope.freq_base")||strstr(key,"rope_theta")
                 ||strstr(key,"rope_freq_base")) {
            if (vtype==GGUF_FLOAT32) { fread(&g->rope_base,4,1,f); }
            else gguf_skip_val(f,vtype);
        }
        else if (strstr(key,"context_length")||strstr(key,"max_seq_len")
                 ||strstr(key,"max_position_embeddings")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->max_seq_len); }
            else gguf_skip_val(f,vtype);
        }
        else if (strstr(key,"rope.dimension_count")||strstr(key,"rope_dim")
                 ||strstr(key,"key_length")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->head_dim); }
            else gguf_skip_val(f,vtype);
        }
        /* norm_eps: various patterns across architectures */
        else if (strstr(key,"rms_epsilon")||strstr(key,"rms_norm_eps")
                 ||strstr(key,"layer_norm_epsilon")
                 ||strstr(key,"attention.layer_norm_rms_epsilon")) {
            if (vtype==GGUF_FLOAT32) { fread(&g->norm_eps,4,1,f); }
            else gguf_skip_val(f,vtype);
        }
        /* moe_intermediate_size: Qwen3-MoE expert FFN dim differs from dense */
        else if (strstr(key,"moe_intermediate_size")||strstr(key,"ffn_dim_exps")) {
            if (vtype==GGUF_UINT32||vtype==GGUF_INT32) { RD_UINT(g->moe_intermediate_size); }
            else gguf_skip_val(f,vtype);
        }
"""

# Replace
new_lines = lines[:start_ln] + [NEW_BLOCK] + lines[end_ln:]

with open(SRC, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"  Done: {SRC}")
print(f"  New line count: {len(new_lines)}")
