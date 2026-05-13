// hdgl_router.h ? HDGL-28 recursive temporal token router public API
#pragma once
#include "hdgl_bootloaderz.h"

// Initialize router with a live lattice and the total expert count
void hdgl_router_init(HDGLLattice *lat, int num_experts);

// Route a single token using the temporal history; updates H in place
int route_token_recursive(Token t, HDGL_History *H);

// Route a batch of tokens; expert_ids[i] receives the routed expert index
void route_tokens_recursive(Token *tokens, int *expert_ids, int batch_size);

// Returns the primary-strand lattice slot index for text from a cold-start
// history (primary_phase = 0).  Mirrors spiral_project(text, &H_zero, 0)
// without requiring a live lattice pointer.  Use this in hdgl_corpus_seeder
// to guarantee the biased slot is the one the router will visit first,
// satisfying ∂p/∂α > 0 (increasing corpus bias monotonically increases
// the probability of selecting corpus-relevant experts).
int hdgl_router_key_to_slot(const char *text, int total_slots);

// Shader utility: pack in_dim/out_dim for non-square Metal buffer(3)
HDGL_ShaderDims hdgl_get_packed_dims(int in_dim, int out_dim);
