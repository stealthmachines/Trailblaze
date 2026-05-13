/*
 * src/hdgl_critic.h — TRAILBLAZE v0.3 Routing Critic
 *
 * TD(0) mini-network that learns to modulate hdgl_alpha based on routing quality.
 * Feature vector: [inv_conf, coherence, mean_gate, pos_norm, accum_norm]
 *
 *   inv_conf   = 1.0 - top1_softmax_prob  (inverse model confidence pre-boost)
 *   coherence  = tb_phase_coherence() / n_experts  (Kuramoto order parameter)
 *   mean_gate  = mean expert gate score of selected top-k experts
 *   pos_norm   = token_pos / max_seq_len
 *   accum_norm = clamp(reward_accum / 10, 0, 1)
 *
 * Output: critic_forward() returns state value ∈ (-inf, +inf).
 * Usage:  alpha_mod = 0.3f + 0.7f * sigmoid(critic_forward(feat))  → [0.3, 1.0]
 *         effective_alpha = ctx->hdgl_alpha * (0.3 + 0.7 * alpha_mod)
 */
#pragma once
#include <stdint.h>

#define CRITIC_IN 5

#ifdef __cplusplus
extern "C" {
#endif

void  critic_init(void);
float critic_forward(const float s[CRITIC_IN]);
void  critic_observe(const float s[CRITIC_IN], float target);
void  critic_update(void);
float critic_td_target(float observed_reward, const float s_next[CRITIC_IN]);
int   critic_save(const char *path);
int   critic_load(const char *path);

#ifdef __cplusplus
}
#endif
