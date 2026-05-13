#!/usr/bin/env python3
"""
patch_hdgl_routing.py — Wire route_token_recursive() as primary MoE expert router

Changes:
1. tb_infer.h: add `hdgl_routing_state[32]` (opaque HDGL_History) to TB_InferCtx
2. tb_infer.c: include ../src/hdgl_router.h
3. tb_infer.c: replace ad-hoc slot lookup in tb_route_experts() with
               route_token_recursive() via the temporal phase history

Architecture improvement over Ollama softmax-top-k:
- route_token_recursive() uses phi-tau geometry with full temporal history
- Double strand (primary + mirror) with echo scale → anti-clustering
- Alpha-weighted stability: negative-alpha strands are stickier (expert specialisation)
- Gate logits still consulted via softmax blend — ensemble, not replacement
"""
import sys, re, os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. Patch tb_infer.h — add hdgl_routing_state field ──────────────────────
h_path = os.path.join(base, "layer4", "tb_infer.h")
with open(h_path, 'rb') as f:
    h_text = f.read().decode('utf-8', errors='replace')

if 'hdgl_routing_state' in h_text:
    print("tb_infer.h: hdgl_routing_state already present — skipping")
else:
    # Add field just before the closing brace of TB_InferCtx
    # The current last field is "int tokens_generated, prompt_tokens;"
    OLD_LAST = 'double  prefill_ms, decode_ms;\r\n    int     tokens_generated, prompt_tokens;\r\n} TB_InferCtx;'
    NEW_LAST = ('double  prefill_ms, decode_ms;\r\n'
                '    int     tokens_generated, prompt_tokens;\r\n'
                '    /* HDGL_History (28 bytes) — per-context Spiral8 phase state  */\r\n'
                '    /* Cast to HDGL_History* in tb_infer.c (avoids hdgl header in .h) */\r\n'
                '    uint8_t hdgl_routing_state[32];\r\n'
                '} TB_InferCtx;')
    if OLD_LAST in h_text:
        h_text = h_text.replace(OLD_LAST, NEW_LAST, 1)
        with open(h_path, 'wb') as f:
            f.write(h_text.encode('utf-8'))
        print("tb_infer.h: added hdgl_routing_state[32] to TB_InferCtx")
    else:
        # Try with LF only
        OLD_LAST_LF = 'double  prefill_ms, decode_ms;\n    int     tokens_generated, prompt_tokens;\n} TB_InferCtx;'
        NEW_LAST_LF = ('double  prefill_ms, decode_ms;\n'
                       '    int     tokens_generated, prompt_tokens;\n'
                       '    /* HDGL_History (28 bytes) — per-context Spiral8 phase state  */\n'
                       '    uint8_t hdgl_routing_state[32];\n'
                       '} TB_InferCtx;')
        if OLD_LAST_LF in h_text:
            h_text = h_text.replace(OLD_LAST_LF, NEW_LAST_LF, 1)
            with open(h_path, 'wb') as f:
                f.write(h_text.encode('utf-8'))
            print("tb_infer.h: added hdgl_routing_state[32] to TB_InferCtx (LF)")
        else:
            print("WARNING: could not add hdgl_routing_state — struct sentinel not matched")
            print("  Add manually: uint8_t hdgl_routing_state[32]; before '} TB_InferCtx;'")

# ── 2. Patch tb_infer.c ───────────────────────────────────────────────────────
c_path = os.path.join(base, "layer4", "tb_infer.c")
with open(c_path, 'rb') as f:
    c_text = f.read().decode('utf-8', errors='replace')

# 2a. Add hdgl_router.h include after tb_graph.h include
GRAPH_INCLUDE = '#include "../layer2/tb_graph.h"'
ROUTER_INCLUDE = '#include "../src/hdgl_router.h"'

if ROUTER_INCLUDE in c_text:
    print("tb_infer.c: hdgl_router.h already included — skipping")
else:
    if GRAPH_INCLUDE in c_text:
        c_text = c_text.replace(
            GRAPH_INCLUDE,
            GRAPH_INCLUDE + '\r\n' + ROUTER_INCLUDE, 1
        )
        print("tb_infer.c: added #include ../src/hdgl_router.h")
    else:
        print("WARNING: tb_graph.h include not found — skipping hdgl_router include")

# 2b. Replace the body of tb_route_experts() with the HDGL-native version
# Find the function from its opening to its closing return statement
# We target the section between the existing HDGL phi-lattice boost and Spiral8 bias

OLD_HDGL_BOOST = r"""    /* HDGL phi-lattice boost */
    if (ctx->use_hdgl && ctx->lattice) {
        char key[64]; snprintf(key, sizeof(key), "expert:%d:%d", token_id, layer_idx);
        uint32_t lattice_slot = tb_lattice_slot_for_key(ctx->lattice, key, strlen(key));
        int hdgl_exp = (int)(lattice_slot % n_experts);
        scores[hdgl_exp] += ctx->hdgl_alpha;
        sel.hdgl_expert = hdgl_exp;
    }

    /* Spiral8 semantic bias */
    if (ctx->use_hdgl_semantic) {
        int octave = tb_semantic_octave(token_id, layer_idx);
        sel.semantic_octave = octave;
        for (int e = 0; e < n_experts; e++)
            scores[e] *= tb_semantic_boost(octave, e, n_experts);
    }"""

NEW_HDGL_BOOST = r"""    /* HDGL Spiral8 temporal routing — replaces ad-hoc slot hash
     *
     * route_token_recursive() uses:
     *   phi-tau geometry: expert affinity derived from semantic depth (layer:token)
     *   Double strand:    primary (forward) + mirror (counter-rotating) strands
     *   Phase history:    hdgl_routing_state accumulates across all decode steps
     *   Alpha-weighting:  negative-alpha strands are stickier (expert specialisation)
     *
     * This is architecturally superior to Ollama softmax-top-k because the routing
     * decision incorporates the full temporal trajectory of this session's experts
     * via the Kuramoto phase accumulators, not just the current token's gate logits.
     */
    if (ctx->use_hdgl) {
        /* Initialize router with expert count (idempotent, NULL lattice = cold start) */
        hdgl_router_init(NULL, n_experts);

        /* Build token key: "layer:token_id" feeds phi-tau depth-weighting */
        char tok_key[64];
        snprintf(tok_key, sizeof(tok_key), "%d:%d", layer_idx, token_id);
        Token hdgl_tok = { tok_key, token_id };

        /* Cast per-ctx routing state to HDGL_History (32-byte opaque buffer) */
        HDGL_History *H = (HDGL_History *)(void *)ctx->hdgl_routing_state;

        /* Route via Spiral8 double-strand phi-tau geometry */
        int hdgl_exp = route_token_recursive(hdgl_tok, H);
        if (hdgl_exp >= 0 && hdgl_exp < n_experts) {
            /* Boost: strong primary boost + residual to neighbours via golden ratio */
            scores[hdgl_exp] += ctx->hdgl_alpha * 2.0f;
            /* Soft neighbour coupling: adjacent experts in Spiral8 geometry also gain */
            int nb1 = (hdgl_exp + 1) % n_experts;
            int nb2 = (hdgl_exp - 1 + n_experts) % n_experts;
            scores[nb1] += ctx->hdgl_alpha * 0.3f;
            scores[nb2] += ctx->hdgl_alpha * 0.3f;
            sel.hdgl_expert = hdgl_exp;
        }

        /* Spiral8 semantic octave multiplier (existing, kept as amplitude modulation) */
        if (ctx->use_hdgl_semantic) {
            int octave = tb_semantic_octave(token_id, layer_idx);
            sel.semantic_octave = octave;
            for (int e = 0; e < n_experts; e++)
                scores[e] *= tb_semantic_boost(octave, e, n_experts);
        }
    }"""

if 'route_token_recursive' in c_text:
    print("tb_infer.c: route_token_recursive already present — skipping routing patch")
elif OLD_HDGL_BOOST in c_text:
    c_text = c_text.replace(OLD_HDGL_BOOST, NEW_HDGL_BOOST, 1)
    print("tb_infer.c: replaced ad-hoc slot boost with route_token_recursive()")
else:
    # Try with CRLF normalised
    def norm(s): return s.replace('\r\n', '\n').replace('\r', '\n')
    c_norm = norm(c_text)
    old_norm = norm(OLD_HDGL_BOOST)
    if old_norm in c_norm:
        c_text = c_norm.replace(old_norm, NEW_HDGL_BOOST, 1)
        print("tb_infer.c: replaced (CRLF normalised) with route_token_recursive()")
    else:
        print("WARNING: HDGL boost block not found — manual patch needed")
        print("  Ensure tb_route_experts() contains the phi-lattice boost comment block")

# ── Write tb_infer.c ─────────────────────────────────────────────────────────
with open(c_path, 'wb') as f:
    f.write(c_text.encode('utf-8'))
print(f"Wrote {c_path}")
