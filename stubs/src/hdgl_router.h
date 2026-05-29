#pragma once
#include <stdint.h>
typedef uint8_t HDGL_History;
typedef struct { const char *key; int id; } Token;
static inline void hdgl_router_init(void *lat, int n_experts) { (void)lat; (void)n_experts; }
static inline int route_token_recursive(Token tok, HDGL_History *H) { (void)tok; (void)H; return 0; }
