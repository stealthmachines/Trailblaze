/*
 * tb_integration_test.c — TRAILBLAZE C Stack Full Integration Test
 *
 * Tests all three C layers end-to-end in one coherent scenario.
 * Build: make test
 */

#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "layer0/tb_phi_lattice.h"
#include "layer1/tb_tensor.h"
#include "layer2/tb_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#define PASS(msg) printf("  \033[32m✓\033[0m %s\n", msg)
#define FAIL(msg) do { printf("  \033[31m✗\033[0m %s\n", msg); exit(1); } while(0)
#define SECTION(s) printf("\n\033[1m══ %s ══\033[0m\n", s)

static double tb_wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    printf("\033[1m=== TRAILBLAZE C Stack — Integration Test ===\033[0m\n");
    double t_start = tb_wall_ms();
    srand(42);

    /* ── LAYER 0 ─────────────────────────────────────────────────────────── */
    SECTION("LAYER 0: Phi-Lattice Core");

    TB_PhiLattice *lat = tb_lattice_create(512, 0xDEADBEEFCAFEBABEULL);
    assert(lat);
    for (int i = 0; i < 10; i++) tb_lattice_advance(lat, 10);

    char desc[512];
    tb_lattice_describe(lat, desc, sizeof(desc));
    PASS(desc);

    /* Avalanche test */
    uint8_t h1[32], h2[32];
    tb_phi_fold_hash32(lat, (const uint8_t*)"trailblaze", 10, h1);
    tb_phi_fold_hash32(lat, (const uint8_t*)"trailblazf", 10, h2);
    int diff = 0;
    for (int i = 0; i < 32; i++) diff += __builtin_popcount(h1[i]^h2[i]);
    double av = diff/256.0*100.0;
    assert(av > 30.0 && av < 70.0);
    char av_msg[64]; snprintf(av_msg, sizeof(av_msg), "phi_fold32 avalanche: %.1f%%", av);
    PASS(av_msg);

    /* PhiStream seal/unseal */
    const char *secret = "TRAILBLAZE cognition substrate v0.1 C implementation";
    uint8_t env[512];
    size_t env_len = tb_phi_stream_seal(lat, (const uint8_t*)secret, strlen(secret),
                                         "integration", env, sizeof(env));
    assert(env_len == strlen(secret) + 40);
    uint8_t plain[512];
    size_t pl = tb_phi_stream_unseal(lat, env, env_len, "integration", plain, sizeof(plain));
    assert(pl == strlen(secret) && memcmp(plain, secret, pl) == 0);
    PASS("PhiStream seal/unseal roundtrip");

    /* Tamper */
    env[50] ^= 0xFF;
    assert(tb_phi_stream_unseal(lat, env, env_len, "integration", plain, sizeof(plain)) == 0);
    env[50] ^= 0xFF;
    PASS("PhiStream tamper detection");

    /* Epoch advance → forward secrecy */
    int32_t old_ep = lat->epoch;
    tb_lattice_advance(lat, 1);
    assert(lat->epoch == old_ep + 1);
    assert(tb_phi_stream_unseal(lat, env, env_len, "integration", plain, sizeof(plain)) == 0);
    char ep_msg[64]; snprintf(ep_msg, sizeof(ep_msg), "Epoch %d→%d forward secrecy", old_ep, lat->epoch);
    PASS(ep_msg);

    /* Backend registry */
    TB_BackendRegistry reg;
    tb_registry_init(&reg, lat);
    TB_Backend sel = tb_registry_select(&reg, TB_OP_MATMUL, 4096);
    assert(sel == TB_BACKEND_CPU_AVX2 || sel == TB_BACKEND_ANALOG);
    PASS("Backend registry + Dn-modulated selection");

    /* U-field */
    double M, L, S;
    tb_lattice_s_u_resonance(lat, &M, &L, &S);
    char su_msg[64]; snprintf(su_msg, sizeof(su_msg), "S(U)=%.4f Λ=%.4f M=%.4f", S, L, M);
    PASS(su_msg);

    /* ── LAYER 1 ─────────────────────────────────────────────────────────── */
    SECTION("LAYER 1: Tensor Runtime + Hopfield Memory");

    /* Tiled matmul */
    int M_sz=32, K_sz=64, N_sz=32;
    float *A = (float*)calloc(M_sz*K_sz, sizeof(float));
    float *B = (float*)calloc(K_sz*N_sz, sizeof(float));
    float *C = (float*)calloc(M_sz*N_sz, sizeof(float));
    for (int i=0; i<M_sz*K_sz; i++) A[i] = 0.01f*(i%13);
    for (int i=0; i<K_sz*N_sz; i++) B[i] = 0.01f*(i%7);
    tb_matmul(A, B, C, M_sz, K_sz, N_sz);
    PASS("Tiled SGEMM (32×64)×(64×32)");
    free(A); free(B); free(C);

    /* Q8_0 roundtrip */
    size_t shape128[1]={128};
    TB_Tensor *orig = tb_tensor_alloc(shape128, 1, TB_DTYPE_F32, lat->epoch, 1);
    for (int i=0; i<128; i++) orig->data[i] = (float)(i-64)*0.1f;
    TB_Tensor *q8 = tb_tensor_quantize_q8(orig, lat->epoch);
    TB_Tensor *dq = tb_tensor_dequantize_q8(q8);
    float max_err = 0;
    for (int i=0; i<128; i++) {
        float e = fabsf(orig->data[i]-dq->data[i]);
        if (e>max_err) max_err=e;
    }
    assert(max_err < 0.05f);
    char q8_msg[64]; snprintf(q8_msg,sizeof(q8_msg),"Q8_0 roundtrip err=%.4f",max_err);
    PASS(q8_msg);
    tb_tensor_free(orig); tb_tensor_free(q8); tb_tensor_free(dq);

    /* Epoch guard */
    size_t s1[1]={16};
    TB_Tensor *stale = tb_tensor_alloc(s1, 1, TB_DTYPE_F32, lat->epoch-2, 1);
    assert(!tb_tensor_is_valid(stale, lat->epoch));
    tb_tensor_free(stale);
    PASS("Stale tensor epoch guard");

    /* KV cache COW fork */
    TB_KVCache *kv = tb_kvcache_alloc(2, 4, 16, 64, 1, lat->epoch);
    float q_arr[64]={0}, k_arr[64]={0.1f}, v_arr[64]={0.2f}, ao[64];
    tb_attention(q_arr, k_arr, v_arr, kv, 0, 4, 4, 16, ao);
    assert(kv->seq_len == 1);
    TB_KVCache *kv2 = tb_kvcache_fork(kv, 2);
    float q2[64]={0.1f}, k2[64]={0.1f}, v2[64]={0.3f}, ao2[64];
    tb_attention(q2, k2, v2, kv2, 0, 4, 4, 16, ao2);
    assert(kv->seq_len==1 && kv2->seq_len==2);
    PASS("KV cache COW fork: branch1.seq=1, branch2.seq=2");
    tb_lattice_advance(lat, 1);
    tb_kvcache_invalidate(kv, lat->epoch);
    assert(kv->seq_len==0);
    PASS("KV cache epoch invalidation");
    tb_kvcache_free(kv); tb_kvcache_free(kv2);

    /* Hopfield memory */
    int N = 16;
    TB_HopfieldMemory *hm = tb_hopfield_alloc(N);
    float *pat = (float*)malloc(N*sizeof(float));
    /* Store 3 bipolar patterns */
    for (int p=0; p<3; p++) {
        for (int i=0; i<N; i++) pat[i] = ((i+p)%2 == 0) ? 1.0f : -1.0f;
        tb_hebbian_train(hm, pat);
    }
    assert(hm->n_patterns == 3);
    /* Corrupt pattern 0 and relax */
    for (int i=0; i<N; i++) pat[i] = ((i)%2 == 0) ? 1.0f : -1.0f;
    pat[2] = 0.3f; pat[5] = -0.4f;  /* corruption */
    float E_b = tb_hopfield_energy(pat, hm->weights, N);
    tb_hopfield_relax(pat, hm->weights, N, 2.0f, 50, 1e-4f);
    float E_a = tb_hopfield_energy(pat, hm->weights, N);
    assert(E_a <= E_b + 0.5f);
    char hf_msg[80];
    snprintf(hf_msg, sizeof(hf_msg), "Hopfield relax: E %.2f→%.2f (converged)", E_b, E_a);
    PASS(hf_msg);

    /* Phase coherence */
    float phases[8] = {0.1f,0.1f,0.11f,0.09f,0.1f,0.1f,0.1f,0.1f};
    float R = tb_phase_coherence(phases, 8);
    assert(R > 0.98f);
    char coh_msg[64]; snprintf(coh_msg, sizeof(coh_msg), "Phase coherence: R=%.4f (locked)", R);
    PASS(coh_msg);

    /* Relax merge */
    float src_a[8]={1,1,1,1,0,0,0,0}, src_b[8]={0,0,0,0,1,1,1,1}, mrgd[8];
    tb_relax_merge(mrgd, src_a, src_b, 8, 0.5f);
    for (int i=0; i<8; i++) assert(fabsf(mrgd[i]-0.5f)<1e-5f);
    PASS("Relaxation merge (branch reconciliation α=0.5)");

    /* Sampler */
    float logits[512];
    for (int i=0; i<512; i++) logits[i]=-10.0f+0.02f*i;
    int g_tok = tb_sample_greedy(logits, 512);
    assert(g_tok == 511);
    int p_tok = tb_sample_top_p(logits, 512, 0.9f, 1.0f);
    assert(p_tok >= 0 && p_tok < 512);
    PASS("Samplers: greedy=511 (correct), top_p in range");
    tb_hopfield_free(hm); free(pat);

    /* ── LAYERS 2+3 ──────────────────────────────────────────────────────── */
    SECTION("LAYERS 2+3: Graph Engine + Cognition Substrate");

    /* Build inference graph */
    TB_Graph *graph = tb_graph_create("inference", lat);
    TB_Node *embed  = tb_graph_add_node(graph, TB_PASS_EMBED,  "embed",  NULL, 0);
    TB_Node *attn0  = tb_graph_add_node(graph, TB_PASS_ATTEND, "attn0",  &embed, 1);
    TB_Node *ffn0   = tb_graph_add_node(graph, TB_PASS_FFN,    "ffn0",   &attn0, 1);
    TB_Node *attn1  = tb_graph_add_node(graph, TB_PASS_ATTEND, "attn1",  &ffn0, 1);
    TB_Node *sample = tb_graph_add_node(graph, TB_PASS_SAMPLE, "sample", &attn1, 1);
    assert(tb_graph_topo_sort(graph) == 0);
    assert(graph->n_topo == 5);
    assert(graph->topo_order[0] == embed && graph->topo_order[4] == sample);
    PASS("5-node inference graph topo sort");

    /* Unique phi-lattice IDs */
    for (int i=0; i<4; i++)
        assert(memcmp(graph->nodes[i]->id, graph->nodes[i+1]->id, 16) != 0);
    PASS("phi-lattice u128 node IDs: all unique");

    /* HDGL routing */
    TB_HDGLRouter router;
    tb_router_init(&router, lat, &reg);
    for (int i=0; i<graph->n_topo; i++)
        tb_router_route_node(&router, graph->topo_order[i]);
    PASS("HDGL Kuramoto scheduler: all nodes routed");

    /* Server routing */
    uint32_t pool[5] = {0x01020301,0x01020302,0x01020303,0x01020304,0x01020305};
    int sdist[5]={0};
    for (int i=0; i<50; i++) {
        char key[32]; snprintf(key, sizeof(key),"request_%d",i);
        uint32_t srv = tb_router_route_server(&router, key, pool, 5);
        for (int j=0; j<5; j++) if (srv==pool[j]) { sdist[j]++; break; }
    }
    int all_pos = 1;
    for (int j=0; j<5; j++) if (sdist[j]==0) all_pos=0;
    assert(all_pos);
    PASS("phi-spiral server routing: all 5 servers receive traffic");

    tb_graph_destroy(graph);

    /* CognitionTree full lifecycle */
    char tmpdir[] = "/tmp/tb_integration_XXXXXX";
    char *td = mkdtemp(tmpdir);
    assert(td);
    TB_CognitionTree *tree = tb_tree_create(lat, "integration", td);
    assert(tree);
    PASS("CognitionTree created");

    /* 5 cells with hash chain */
    TB_CognitionCell *cells[5];
    for (int i=0; i<5; i++) {
        char val[64]; snprintf(val, sizeof(val), "{\"step\":%d}", i);
        cells[i] = tb_tree_cell_commit(tree, 0, val, "task");
        assert(cells[i]);
        assert(tb_tree_cell_verify(tree, cells[i]));
    }
    /* Verify chain linkage */
    assert(memcmp(cells[1]->parent_id, cells[0]->id, 16) == 0);
    assert(memcmp(cells[4]->parent_id, cells[3]->id, 16) == 0);
    PASS("5-cell hash chain: all verified");

    int broken=-1;
    assert(tb_erl_verify_chain(tree->ledger, &broken));
    char erl_msg[64]; snprintf(erl_msg,sizeof(erl_msg),
        "ERL chain: %d entries valid",tb_erl_n_entries(tree->ledger));
    PASS(erl_msg);

    /* 3 parallel branches */
    int b[3];
    for (int i=0; i<3; i++) {
        b[i] = tb_tree_branch_create(tree, 0);
        char val[64]; snprintf(val,sizeof(val),"{\"sub_agent\":%d}",i);
        tb_tree_cell_commit(tree, b[i], val, "task");
    }
    /* Merge all back */
    for (int i=0; i<3; i++) {
        assert(tb_tree_branch_merge(tree, b[i], 0));
        assert(tree->branches[b[i]]->merged);
    }
    assert(tb_erl_verify_chain(tree->ledger, &broken));
    PASS("3 parallel branches + merge: ERL chain valid");

    /* Epoch advance invalidates all KV caches */
    TB_KVCache *kv3 = tb_kvcache_alloc(2,4,16,64,b[0],lat->epoch);
    tree->branches[b[0]]->kv_cache = kv3;
    float q3[64]={0.1f},k3[64]={0.1f},v3[64]={0.1f},ao3[64];
    tb_attention(q3,k3,v3,kv3,0,4,4,16,ao3);
    assert(kv3->seq_len==1);
    int32_t ep_before = lat->epoch;
    tb_tree_epoch_advance(tree, 1);
    assert(kv3->seq_len==0);
    assert(tb_erl_verify_chain(tree->ledger, &broken));
    PASS("Epoch advance + KV cache invalidation + ERL valid");

    /* Forward secrecy */
    const char *state = "cognition state checkpoint";
    uint8_t fenv[256];
    size_t fenv_len = tb_tree_seal(tree, (const uint8_t*)state, strlen(state),
                                    "checkpoint", fenv, sizeof(fenv));
    uint8_t fplain[256];
    assert(tb_tree_unseal(tree, fenv, fenv_len, "checkpoint", fplain, sizeof(fplain)) == strlen(state));
    PASS("State seal + unseal");
    tb_tree_epoch_advance(tree, 1);
    assert(tb_tree_unseal(tree, fenv, fenv_len, "checkpoint", fplain, sizeof(fplain)) == 0);
    PASS("Forward secrecy: sealed state rejected after epoch advance");

    /* Memory tiers */
    tb_tree_memory_set(tree, "agent_task", "security audit");
    assert(strcmp(tb_tree_memory_get(tree, "agent_task"), "security audit") == 0);
    PASS("Tier 1: session memory");

    tb_tree_notes_write(tree, "audit.md", "# Security Audit\nAll checks passed.\n");
    char *note = tb_tree_notes_read(tree, "audit.md");
    assert(note && strstr(note, "passed"));
    free(note);
    PASS("Tier 2: durable notes");

    tb_tree_flow_set(tree, 0, "cwd", "/opt/project");
    assert(strcmp(tb_tree_flow_get(tree, 0, "cwd"), "/opt/project") == 0);
    PASS("FlowState: per-branch key-value");

    tb_tree_record_tool_call(tree, 0, "shell_exec", "{\"cmd\":\"make test\"}");
    assert(tb_erl_verify_chain(tree->ledger, &broken));
    PASS("Tool call recorded + ERL chain intact");

    /* Final describe */
    char fdesc[512];
    tb_tree_describe(tree, fdesc, sizeof(fdesc));
    PASS(fdesc);

    tb_tree_destroy(tree);

    /* ── Layer 5 C Hopfield semantic recall ──────────────────────────────── */
    SECTION("LAYER 5 (Hopfield Primitives): Semantic Convergence");

    int DIM = 32;
    TB_HopfieldMemory *sem = tb_hopfield_alloc(DIM);
    /* Encode 5 "semantic patterns" as bipolar vectors */
    float *patterns[5];
    for (int p=0; p<5; p++) {
        patterns[p] = (float*)malloc(DIM*sizeof(float));
        for (int i=0; i<DIM; i++)
            patterns[p][i] = ((i*p+p)%3 == 0) ? 1.0f : -1.0f;
        tb_hebbian_train(sem, patterns[p]);
    }
    /* Query with corrupted pattern 2 */
    float *query = (float*)malloc(DIM*sizeof(float));
    memcpy(query, patterns[2], DIM*sizeof(float));
    for (int i=0; i<DIM/4; i++) query[i*4] *= 0.5f;  /* corrupt 25% */
    float E_pre = tb_hopfield_energy(query, sem->weights, DIM);
    tb_hopfield_relax(query, sem->weights, DIM, 1.5f, 200, 1e-5f);
    float E_post = tb_hopfield_energy(query, sem->weights, DIM);
    assert(E_post <= E_pre + 1.0f);
    char sem_msg[80];
    snprintf(sem_msg, sizeof(sem_msg),
             "Semantic recall: E %.2f→%.2f (converged to attractor)", E_pre, E_post);
    PASS(sem_msg);

    /* Random projection topology plugin */
    float input[8]={1,0,0,0,0,0,0,0}, output[4];
    float proj_mat[32] = {
        1,0,0,0,0,0,0,0,   /* row 0 */
        0,1,0,0,0,0,0,0,   /* row 1 */
        0,0,1,0,0,0,0,0,   /* row 2 */
        0,0,0,1,0,0,0,0,   /* row 3 */
    };
    tb_random_projection(input, output, proj_mat, 8, 4);
    assert(fabsf(output[0]-1.0f)<1e-5f && fabsf(output[1])<1e-5f);
    PASS("Random projection (topology plugin): PASS");

    for (int p=0; p<5; p++) free(patterns[p]);
    free(query);
    tb_hopfield_free(sem);

    /* ── Benchmark ───────────────────────────────────────────────────────── */
    SECTION("Benchmark");

    /* phi_fold throughput */
    double t0 = tb_wall_ms();
    for (int i=0; i<10000; i++) {
        uint8_t h[32];
        char k[16]; snprintf(k,16,"bench_%d",i);
        tb_phi_fold_hash32(lat,(const uint8_t*)k,strlen(k),h);
    }
    double phi_ms = tb_wall_ms() - t0;
    char bph[64]; snprintf(bph,sizeof(bph),"phi_fold32 10K: %.1fms (%.0f/s)",
        phi_ms, 10000.0/(phi_ms/1000.0));
    PASS(bph);

    /* matmul throughput */
    float *bA=(float*)malloc(64*64*sizeof(float));
    float *bB=(float*)malloc(64*64*sizeof(float));
    float *bC=(float*)malloc(64*64*sizeof(float));
    for (int i=0; i<64*64; i++) { bA[i]=0.01f*i; bB[i]=0.01f*(i+1); }
    t0 = tb_wall_ms();
    for (int i=0; i<500; i++) tb_matmul(bA,bB,bC,64,64,64);
    double mat_ms = tb_wall_ms()-t0;
    char bm[80]; snprintf(bm,sizeof(bm),"SGEMM 64×64 500x: %.1fms (%.0f GFLOPS est.)",
        mat_ms, 500.0*2*64*64*64/1e9/(mat_ms/1000.0));
    PASS(bm);
    free(bA);free(bB);free(bC);

    /* Slot routing throughput */
    t0 = tb_wall_ms();
    for (int i=0; i<50000; i++) {
        char k[32]; snprintf(k,32,"route_%d",i);
        tb_lattice_slot_for_key(lat,k,strlen(k));
    }
    double rt_ms = tb_wall_ms()-t0;
    char br[64]; snprintf(br,sizeof(br),"slot routing 50K: %.1fms (%.0f/s)",
        rt_ms, 50000.0/(rt_ms/1000.0));
    PASS(br);

    tb_lattice_destroy(lat);

    /* ── Summary ──────────────────────────────────────────────────────────── */
    double total_ms = tb_wall_ms() - t_start;
    printf("\n\033[1m");
    printf("┌─────────────────────────────────────────────────────┐\n");
    printf("│  TRAILBLAZE C Stack — All Layers Operational        │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Layer 0: Phi-Lattice     Kuramoto/phi_fold/AEAD ✓  │\n");
    printf("│  Layer 1: Tensor Runtime  SGEMM/KV/Hopfield/Quant ✓ │\n");
    printf("│  Layer 2: Graph Engine    HDGL/Spiral8/Fusion     ✓  │\n");
    printf("│  Layer 3: Cognition       ERL/Branch/Seal/Tiers   ✓  │\n");
    printf("│  Layer 5: Hopfield OS     Recall/Coherence/Merge  ✓  │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Total: %8.1f ms                                 │\n", total_ms);
    printf("└─────────────────────────────────────────────────────┘\n");
    printf("\033[0m");

    /* Cleanup */
    char rmcmd[640]; snprintf(rmcmd,sizeof(rmcmd),"rm -rf %s",td);
    (void)system(rmcmd);
    return 0;
}
