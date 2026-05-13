#!/usr/bin/env python3
"""
patch_critic_stack.py — Wire the 3-layer routing stack into tb_infer.c

After patch_hdgl_routing.py runs, tb_route_experts() uses route_token_recursive()
for Spiral8 temporal geometry.  This patch adds two more layers:

  Layer 2 (v39 amplitude weighting):
    For each expert, weight its score by exp(SHARPNESS * cos(dphi) * sqrt(amp))
    where dphi = phase offset from the primary routed expert, amp = gate magnitude.
    This is the v39 routing_weight() formula adapted to CPU scalar code.

  Layer 3 (TD critic):
    critic_alpha_mod(features) → sigmoid-mapped scalar in [0.3, 1.0]
    Multiplies hdgl_alpha BEFORE the route_token_recursive boost.
    After top-k selection, critic_observe() trains on observed routing quality
    (max_expert_weight = proxy for routing confidence = TD reward).
    critic_update() flushes gradients every CRITIC_UPDATE_INTERVAL tokens.

Feature mapping (replaces analog-prime's primality features with routing features):
  feat[0] = inv_conf   = 1.0f - top1_raw_prob   (pre-boost model uncertainty)
  feat[1] = coherence  = tb_phase_coherence(lat) / n_exp (Kuramoto order, 0..1)
  feat[2] = mean_gate  = mean of selected expert gate scores after normalisation
  feat[3] = pos_norm   = token_pos / max_seq_len
  feat[4] = accum_norm = clamp(n_tokens_generated / 100.0, 0, 1)
"""
import os, re, sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c_path = os.path.join(base, "layer4", "tb_infer.c")

with open(c_path, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-8', errors='replace')
orig = text

changes = 0

# ── 1. Add #include "../src/hdgl_critic.h" after hdgl_router.h ───────────────
ROUTER_INC  = '#include "../src/hdgl_router.h"'
CRITIC_INC  = '#include "../src/hdgl_critic.h"'
CRITIC_IMPL = '#include "../src/hdgl_critic.c"'   # inline impl avoids Makefile changes

if CRITIC_INC in text:
    print("tb_infer.c: hdgl_critic.h already included — skipping")
elif ROUTER_INC in text:
    # Insert after router include
    ins = '\n' + CRITIC_INC + '\n' + CRITIC_IMPL
    text = text.replace(ROUTER_INC, ROUTER_INC + ins, 1)
    changes += 1
    print("tb_infer.c: added hdgl_critic.h + hdgl_critic.c includes")
else:
    # Fallback: insert after tb_graph.h
    GRAPH_INC = '#include "../layer2/tb_graph.h"'
    if GRAPH_INC in text:
        ins = '\n' + CRITIC_INC + '\n' + CRITIC_IMPL
        text = text.replace(GRAPH_INC, GRAPH_INC + ins, 1)
        changes += 1
        print("tb_infer.c: added hdgl_critic includes after tb_graph.h")
    else:
        print("WARNING: could not find include anchor — add manually")

# ── 2. Wire critic into route_token_recursive block ───────────────────────────
# The routing patch (patch_hdgl_routing.py) produced this block:
#
#   if (ctx->use_hdgl) {
#       hdgl_router_init(NULL, n_experts);
#       char tok_key[64];
#       snprintf(tok_key, sizeof(tok_key), "%d:%d", layer_idx, token_id);
#       Token hdgl_tok = { tok_key, token_id };
#       HDGL_History *H = (HDGL_History *)(void *)ctx->hdgl_routing_state;
#       int hdgl_exp = route_token_recursive(hdgl_tok, H);
#       if (hdgl_exp >= 0 && hdgl_exp < n_experts) {
#           scores[hdgl_exp] += ctx->hdgl_alpha * 2.0f;
#           int nb1 = (hdgl_exp + 1) % n_experts;
#           int nb2 = (hdgl_exp - 1 + n_experts) % n_experts;
#           scores[nb1] += ctx->hdgl_alpha * 0.3f;
#           scores[nb2] += ctx->hdgl_alpha * 0.3f;
#           sel.hdgl_expert = hdgl_exp;
#       }
#       ... Spiral8 octave block ...
#   }
#
# We replace the inner boost lines to:
#   (a) compute critic_alpha_mod before the boost
#   (b) scale the boosts by critic_alpha_mod
#   (c) after the closing brace of the HDGL block: do critic observe + periodic update

OLD_BOOST_LINES = (
    '        int hdgl_exp = route_token_recursive(hdgl_tok, H);\n'
    '        if (hdgl_exp >= 0 && hdgl_exp < n_experts) {\n'
    '            /* Boost: strong primary boost + residual to neighbours via golden ratio */\n'
    '            scores[hdgl_exp] += ctx->hdgl_alpha * 2.0f;\n'
    '            /* Soft neighbour coupling: adjacent experts in Spiral8 geometry also gain */\n'
    '            int nb1 = (hdgl_exp + 1) % n_experts;\n'
    '            int nb2 = (hdgl_exp - 1 + n_experts) % n_experts;\n'
    '            scores[nb1] += ctx->hdgl_alpha * 0.3f;\n'
    '            scores[nb2] += ctx->hdgl_alpha * 0.3f;\n'
    '            sel.hdgl_expert = hdgl_exp;\n'
    '        }'
)

NEW_BOOST_LINES = (
    '        int hdgl_exp = route_token_recursive(hdgl_tok, H);\n'
    '\n'
    '        /* ── Layer 3: critic alpha modulator ──────────────────────────────\n'
    '         * Compute pre-boost top1 probability as inverse-confidence feature.\n'
    '         * critic_alpha_mod() returns sigmoid-mapped [0.3, 1.0]:\n'
    '         *   low value  → suppress HDGL influence (model is already confident)\n'
    '         *   high value → amplify HDGL influence (model is uncertain, trust geometry)\n'
    '         * At cold start: critic weights ≈ 0, sigmoid(0) = 0.5, mod = 0.65\n'
    '         */\n'
    '        float _top1_raw = scores[0];\n'
    '        for (int _e = 1; _e < n_experts; _e++)\n'
    '            if (scores[_e] > _top1_raw) _top1_raw = scores[_e];\n'
    '        /* Raw softmax approx for top1 prob (fast) */\n'
    '        float _raw_sum = 0.0f;\n'
    '        for (int _e = 0; _e < n_experts; _e++) _raw_sum += expf(scores[_e] - _top1_raw);\n'
    '        float _top1_prob = 1.0f / _raw_sum;   /* top1_prob = exp(0) / sum */\n'
    '\n'
    '        float _crit_feat[CRITIC_IN];\n'
    '        _crit_feat[0] = 1.0f - _top1_prob;                           /* inv_conf   */\n'
    '        _crit_feat[1] = (ctx->lattice)\n'
    '                        ? (float)tb_phase_coherence(ctx->lattice) / (float)n_experts\n'
    '                        : 0.5f;                                       /* coherence  */\n'
    '        _crit_feat[2] = _top1_prob;                                   /* mean_gate  */\n'
    '        _crit_feat[3] = (ctx->model && ctx->model->max_seq_len > 0)\n'
    '                        ? (float)ctx->tokens_generated /\n'
    '                          (float)ctx->model->max_seq_len\n'
    '                        : 0.0f;                                       /* pos_norm   */\n'
    '        _crit_feat[4] = (ctx->tokens_generated > 0)\n'
    '                        ? fminf((float)ctx->tokens_generated / 100.0f, 1.0f)\n'
    '                        : 0.0f;                                       /* accum_norm */\n'
    '\n'
    '        float _alpha_mod = critic_alpha_mod(_crit_feat);  /* [0.3, 1.0] */\n'
    '        float _eff_alpha = ctx->hdgl_alpha * _alpha_mod;\n'
    '\n'
    '        if (hdgl_exp >= 0 && hdgl_exp < n_experts) {\n'
    '            /* Primary boost scaled by learned alpha modulator */\n'
    '            scores[hdgl_exp] += _eff_alpha * 2.0f;\n'
    '            /* ── Layer 2: v39 amplitude weighting ─────────────────────────\n'
    '             * routing_weight(dphi, amp) = exp(SHARPNESS * cos(dphi) * sqrt(amp))\n'
    '             * Adjacent experts get softer boosts with 60-degree phase separation.\n'
    '             * cos(±π/3) = 0.5, so neighbours see exp(0.5*sqrt(amp)) scaling.\n'
    '             */\n'
    '            int nb1 = (hdgl_exp + 1) % n_experts;\n'
    '            int nb2 = (hdgl_exp - 1 + n_experts) % n_experts;\n'
    '            float _gate_amp = _top1_prob;   /* use softmax top1 as amplitude proxy */\n'
    '            float _v39_nb   = expf(0.5f * sqrtf(_gate_amp)) - 1.0f;  /* ≥0 */\n'
    '            scores[nb1] += _eff_alpha * (0.3f + _v39_nb * 0.1f);\n'
    '            scores[nb2] += _eff_alpha * (0.3f + _v39_nb * 0.1f);\n'
    '            sel.hdgl_expert = hdgl_exp;\n'
    '\n'
    '            /* ── Critic TD update: observe routing quality ─────────────────\n'
    '             * Reward = confidence of primary expert after boosting (pre-normalise).\n'
    '             * TD target = reward + gamma * V(next).  next ≈ same features (stationary).\n'
    '             */\n'
    '            float _td_target = critic_td_target(_top1_prob, _crit_feat);\n'
    '            critic_observe(_crit_feat, _td_target);\n'
    '            /* Flush gradients every 32 routing calls to amortise cost */\n'
    '            if ((ctx->tokens_generated & 31) == 0) critic_update();\n'
    '        }'
)

def normalize(s):
    return s.replace('\r\n', '\n').replace('\r', '\n')

if 'critic_alpha_mod' in text:
    print("tb_infer.c: critic stack already wired — skipping")
elif normalize(OLD_BOOST_LINES) in normalize(text):
    text_n = normalize(text)
    old_n  = normalize(OLD_BOOST_LINES)
    text_n = text_n.replace(old_n, NEW_BOOST_LINES, 1)
    text   = text_n
    changes += 1
    print("tb_infer.c: wired critic_alpha_mod + v39 amplitude + TD observe")
else:
    print("WARNING: routing boost block not found — run patch_hdgl_routing.py first")
    print("  Then re-run this script.")

if changes > 0:
    with open(c_path, 'wb') as f:
        f.write(text.encode('utf-8'))
    print(f"Wrote {c_path}  ({changes} change(s))")
else:
    print("No changes written to tb_infer.c")
