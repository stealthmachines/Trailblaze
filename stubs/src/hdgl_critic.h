#pragma once
#define CRITIC_IN 5
static inline float critic_alpha_mod(const float *f) { (void)f; return 0.65f; }
static inline float critic_td_target(float p, const float *f) { (void)p; (void)f; return 0.0f; }
static inline void critic_observe(const float *f, float t) { (void)f; (void)t; }
static inline void critic_update(void) {}
