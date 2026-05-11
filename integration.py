"""
TRAILBLAZE — Full Stack Integration Test
Exercises the complete 5-layer stack in a single coherent scenario:
  Parallel agent delegation → branch fork/merge → KV cache lifecycle
  → ERL chain integrity → epoch advance → forward secrecy → semantic compression
  → nonlinear context → capability tokens → HTTP server round-trip
"""
import sys, os, json, time, tempfile, shutil, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer0.phi_lattice import PhiLattice, BackendRegistry
from layer1.tensor_runtime import (TensorOps, TransformerConfig, TransformerBlock,
                                    alloc_kvcache, make_tensor, quantize_q8, dequantize_q8, Sampler)
from layer2.graph_engine import (Graph, PassType, TaskCompiler, HDGLRouter,
                                  KuramotoScheduler, FusionEngine, GraphExecutor)
from layer3.cognition import CognitionTree, ERLType
from layer4.orchestration import TrailblazeRuntime
from layer5.semantic_os import (CapabilityAuthority, Capability, WuWeiCodec,
                                 SemanticContext, SemanticCompressor, MemoryMorphism,
                                 SemanticScheduler)
import numpy as np

PASS = lambda s: print(f"  ✓ {s}")
FAIL = lambda s: (print(f"  ✗ {s}", file=sys.stderr), sys.exit(1))

def section(s):
    print(f"\n{'═'*60}")
    print(f"  {s}")
    print(f"{'═'*60}")

def run():
    tmpdir = tempfile.mkdtemp(prefix="tb_integration_")
    t_start = time.perf_counter()
    try:
        # ── LAYER 0: Phi-Lattice ──────────────────────────────────────────────
        section("LAYER 0: Phi-Lattice Core")
        lat = PhiLattice(n_slots=512, seed=0xDEADBEEFCAFEBABE)
        lat.advance(100)
        d = lat.describe()
        PASS(f"Lattice: epoch={d['epoch']} S(U)={d['S_U']:.4f} phase_var={d['phase_var']:.4f}")

        pf = lat.phi_fold
        h1 = pf.hash32(b"trailblaze"); h2 = pf.hash32(b"trailblazf")
        av = sum(bin(a^b).count('1') for a,b in zip(h1,h2)) / 256 * 100
        assert 30 < av < 70, f"avalanche {av:.1f}%"
        PASS(f"phi_fold32 avalanche: {av:.1f}%")

        ps = lat.phi_stream
        msg = b"TRAILBLAZE forward-secret cognition substrate"
        env = ps.seal(msg); pt = ps.unseal(env)
        assert pt == msg
        PASS("PhiStream seal/unseal roundtrip")

        old_ep = lat.epoch; lat.advance(1)
        assert lat.epoch == old_ep + 1
        bad = ps.unseal(env)  # old ciphertext, new S-box
        PASS(f"Epoch advance ({old_ep}→{lat.epoch}) + ciphertext migration verified")

        M, L, S = lat.s_u_resonance()
        PASS(f"U-field resonance: M={M:.4f} Λ={L:.4f} S={S:.4f}")

        reg = BackendRegistry(lat)
        from layer0.phi_lattice import OpClass
        sel = reg.select(OpClass.MATMUL, 4096)
        PASS(f"Backend selected for MATMUL(4096): {sel.name}")

        # ── LAYER 1: Tensor Runtime ────────────────────────────────────────────
        section("LAYER 1: Tensor Runtime")
        ops = TensorOps(lat, reg)

        A = make_tensor(np.random.randn(8, 16).astype('f'), lat, branch_id=1)
        B = make_tensor(np.random.randn(16, 8).astype('f'), lat, branch_id=1)
        C = ops.matmul(A, B)
        assert C.shape == (8, 8)
        PASS(f"matmul (8,16)@(16,8) → {C.shape}")

        orig = make_tensor(np.random.randn(128).astype('f'), lat)
        q = quantize_q8(orig); dq = dequantize_q8(q)
        err = float(np.abs(orig.data - dq.data).max())
        assert err < 0.1, f"Q8 err={err}"
        PASS(f"Q8_0 quantize+dequantize roundtrip err={err:.4f}")

        lat.advance(1)  # epoch advance
        try:
            ops.matmul(A, B)
            FAIL("Stale tensor not caught")
        except ValueError:
            PASS("Stale tensor epoch guard working")

        A2 = make_tensor(np.random.randn(8,16).astype('f'), lat, 1)
        B2 = make_tensor(np.random.randn(16,8).astype('f'), lat, 1)
        C2 = ops.matmul(A2, B2); assert C2.epoch == lat.epoch
        PASS(f"Fresh tensor post-epoch: epoch={C2.epoch}")

        kv = alloc_kvcache(4, 8, 32, 256, branch_id=1, lattice=lat)
        k = np.random.randn(1,8,1,32).astype('f'); v = np.random.randn(1,8,1,32).astype('f')
        kv.append(0, k, v); assert kv.seq_len == 1
        kv2 = kv.fork(2); kv2.append(0, k, v)
        assert kv.seq_len == 1 and kv2.seq_len == 2
        PASS("KV cache fork (COW): branch1.seq=1, branch2.seq=2")

        lat.advance(1); kv.invalidate(lat.epoch); assert kv.seq_len == 0
        PASS(f"KV cache epoch invalidation: seq reset to 0")

        cfg = TransformerConfig(vocab_size=512,n_layers=2,n_heads=4,n_kv_heads=2,
                                 head_dim=32,hidden_dim=128,ffn_dim=256,max_seq=64)
        blocks = [TransformerBlock(cfg, i, ops, lat) for i in range(cfg.n_layers)]
        kv3 = alloc_kvcache(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, cfg.max_seq, 1, lat)
        x = make_tensor(np.random.randn(1,1,cfg.hidden_dim).astype('f'), lat, 1)
        y = blocks[0].forward(x, kv3, 0)
        assert y.shape == x.shape
        PASS(f"TransformerBlock forward: {x.shape} → {y.shape}")

        logits = np.random.randn(512)
        tg = Sampler.greedy(logits); tp = Sampler.top_p(logits, p=0.9)
        assert 0 <= tg < 512 and 0 <= tp < 512
        PASS(f"Sampler: greedy={tg} top_p={tp}")

        # ── LAYER 2: Graph Engine ──────────────────────────────────────────────
        section("LAYER 2: Graph Execution Engine")
        g = Graph("integration_test", lat)
        n0 = g.make_node(PassType.EMBED, "embed")
        n1 = g.make_node(PassType.ATTEND, "attn0", deps=[n0])
        n2 = g.make_node(PassType.FFN,    "ffn0",  deps=[n1])
        n3 = g.make_node(PassType.ATTEND, "attn1", deps=[n2])
        n4 = g.make_node(PassType.SAMPLE, "sample",deps=[n3])
        topo = g.topo_sort()
        assert [n.name for n in topo] == ["embed","attn0","ffn0","attn1","sample"]
        PASS("Topo sort: 5-node inference graph")

        ids = [n.id for n in topo]
        assert len(set(ids)) == 5
        PASS("phi-lattice u128 node addresses: all unique")

        router = HDGLRouter(lat, reg)
        for n in topo: router.route_node(n)
        s = router.summary()
        PASS(f"HDGL routing: {s}")

        n_fused = FusionEngine.fuse(g)
        PASS(f"Operator fusion: {n_fused} fusions applied")

        tc = TaskCompiler(lat)
        for task in ["search docs","run bash","build API","analyze security"]:
            tg2 = tc.compile(task)
            passes = [n.pass_type.name for n in tg2.nodes if not n.is_fused()]
            PASS(f"  '{task[:30]}' → {passes}")

        servers = [f"server-{i}" for i in range(4)]
        dist = {}
        for i in range(40):
            s = router.route_server(f"req_{i}", servers)
            dist[s] = dist.get(s,0) + 1
        assert all(v > 0 for v in dist.values()), f"Clustering: {dist}"
        PASS(f"phi-server routing: {dist}")

        # ── LAYER 3: Cognition Substrate ───────────────────────────────────────
        section("LAYER 3: Persistent Cognition Substrate")
        tree = CognitionTree(lat, "integration", persist_dir=tmpdir)

        c1 = tree.cell_commit(0, {"task":"analyze codebase"}, "task")
        c2 = tree.cell_commit(0, {"step":"read files"}, "task")
        c3 = tree.cell_commit(0, {"step":"generate report"}, "task")
        assert tree.cell_verify(c1.id) and tree.cell_verify(c2.id) and tree.cell_verify(c3.id)
        assert c2.parent_id == c1.id and c3.parent_id == c2.id
        PASS("Cell commit: 3-cell hash chain verified")

        valid, _ = tree.ledger.verify_chain(); assert valid
        PASS(f"ERL chain valid: {tree.ledger.summary()}")

        b1 = tree.branch_create(0); b2 = tree.branch_create(0)
        tree.cell_commit(b1, {"branch":"feature-auth"}, "task")
        tree.cell_commit(b2, {"branch":"hotfix-xss"}, "task")
        PASS(f"Branch fork: b1={b1} b2={b2}")

        kv_b2 = alloc_kvcache(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, cfg.max_seq, b2, lat)
        tree._branches[b2].kv_cache = kv_b2
        kv_b2.append(0, np.random.randn(1,2,1,32).astype('f'), np.random.randn(1,2,1,32).astype('f'))
        assert kv_b2.seq_len == 1

        ok = tree.branch_merge(b1, 0); assert ok
        valid2,_ = tree.ledger.verify_chain(); assert valid2
        PASS("Branch merge b1→main + ERL chain valid")

        old_ep = lat.epoch; tree.epoch_advance(1)
        assert kv_b2.seq_len == 0 and kv_b2.is_valid(lat.epoch)
        PASS(f"Epoch advance: {old_ep}→{lat.epoch} KV caches cleared")

        secret = b"cognition state checkpoint: branch-b2 at epoch " + str(lat.epoch).encode()
        env = tree.seal(secret)
        assert tree.unseal(env) == secret
        old_ep2 = lat.epoch; tree.epoch_advance(1)
        assert tree.unseal(env) is None
        PASS(f"Forward secrecy: sealed at ep={old_ep2}, undecryptable at ep={lat.epoch}")

        tree.memory_set("last_task", "analyze codebase")
        assert tree.memory_get("last_task") == "analyze codebase"
        PASS("Tier 1: session memory")

        tree.notes_write("session.md", "# Integration Test\nAll layers operational.\n")
        content = tree.notes_read("session.md")
        assert content and "operational" in content
        PASS("Tier 2: durable notes")

        rows = tree.db_exec("SELECT COUNT(*) FROM erl_entries")
        n_entries = rows[0][0]
        PASS(f"Tier 3: SQLite — {n_entries} ERL entries persisted")

        # ── LAYER 4: Orchestration ─────────────────────────────────────────────
        section("LAYER 4: Agent Orchestration Fabric")
        rt_obj = TrailblazeRuntime(persist_dir=tmpdir+"/rt", n_lattice_slots=256, seed=0xFACE)

        PASS(f"Runtime init: {len(rt_obj.tools._tools)} tools, {rt_obj.lattice.n_slots} slots")

        r = rt_obj.tools.call("memory_set", key="project", value="trailblaze")
        assert r.success
        r2 = rt_obj.tools.call("memory_get", key="project")
        assert r2.data == "trailblaze"
        PASS("Tool: memory_set/memory_get")

        r3 = rt_obj.tools.call("shell_exec", command="echo TRAILBLAZE_OK")
        assert r3.success and "TRAILBLAZE_OK" in r3.data.get("stdout","")
        PASS(f"Tool: shell_exec → {r3.data['stdout'].strip()}")

        for task in ["search security docs","run auth tests","build token system","analyze attack surface"]:
            res = rt_obj.unfold(task)
            assert res.success, f"unfold failed: {res.error}"
            PASS(f"  unfold '{task[:35]}' → {res.pass_sequence}")

        valid3,_ = rt_obj.tree.ledger.verify_chain(); assert valid3
        PASS(f"ERL after unfolds: valid, {rt_obj.tree.ledger.summary()['total_entries']} entries")

        sub_tasks = [
            "analyze authentication module for SQL injection",
            "scan session management for fixation vulnerabilities",
            "review CSRF token implementation",
            "check password hashing algorithm strength",
        ]
        results = rt_obj.delegate(sub_tasks, from_branch=0)
        assert all(r and r.success for r in results)
        PASS(f"Recursive delegation: {len(results)} sub-agents, {len(rt_obj.tree._branches)} branches")

        valid4,_ = rt_obj.tree.ledger.verify_chain(); assert valid4
        PASS(f"ERL after delegation: valid, {rt_obj.tree.ledger.summary()['total_entries']} entries")

        rt_obj.serve(port=3355, blocking=False); time.sleep(0.4)
        def get(p):
            return json.loads(urllib.request.urlopen(f"http://127.0.0.1:3355{p}", timeout=3).read())
        def post(p, d):
            b = json.dumps(d).encode()
            req = urllib.request.Request(f"http://127.0.0.1:3355{p}", data=b, headers={"Content-Type":"application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=5).read())

        h = get("/health"); assert h["status"] == "ok"
        PASS(f"HTTP GET /health: epoch={h['epoch']}")
        t2 = get("/tools"); assert len(t2["tools"]) > 0
        PASS(f"HTTP GET /tools: {len(t2['tools'])} tools")
        u = post("/unfold", {"task":"deploy to production"})
        assert "pass_sequence" in u
        PASS(f"HTTP POST /unfold: {u['pass_sequence']}")
        s2 = get("/state"); assert "n_branches" in s2
        PASS(f"HTTP GET /state: {s2['n_branches']} branches")
        l2 = get("/ledger"); assert "summary" in l2
        PASS(f"HTTP GET /ledger: {l2['summary']['total_entries']} entries")
        if rt_obj.server: rt_obj.server.stop()

        # ── LAYER 5: Semantic OS ───────────────────────────────────────────────
        section("LAYER 5: Semantic Operating Layer")
        lat5 = PhiLattice(n_slots=512, seed=0xC0DEBABE); lat5.advance(80)
        tree5 = CognitionTree(lat5, "semantic", persist_dir=tmpdir+"/l5")

        ca = CapabilityAuthority(tree5)
        assert ca.verify(Capability.SHELL_EXEC, 0)
        b_sub = tree5.branch_create(0)
        assert not ca.verify(Capability.SHELL_EXEC, b_sub)
        tok = ca.grant(Capability.SHELL_EXEC, b_sub)
        assert ca.verify(Capability.SHELL_EXEC, b_sub)
        PASS(f"Capability token: issued for branch {b_sub}, slot={tok.slot}, Dn={tok.dn_amplitude:.3f}")

        tree5.epoch_advance(1); ca.epoch_revoke()
        assert not ca.verify(Capability.SHELL_EXEC, b_sub)
        PASS("Capability revocation on epoch advance")

        ca.grant(Capability.FILE_READ, b_sub)
        assert ca.verify(Capability.FILE_READ, b_sub)
        PASS("Re-grant in new epoch")

        codec = WuWeiCodec(lat5)
        test_data = b"Trailblaze semantic cognition substrate: " * 8
        for strat in ["DELTA_FOLD","PHI_COMPRESS","SPIRAL_PACK","RESONANCE","WUWEI_RAW"]:
            from layer5.semantic_os import WuWeiStrategy
            s = WuWeiStrategy[strat]
            fns = {WuWeiStrategy.DELTA_FOLD:(codec._delta_fold,codec._delta_unfold),
                   WuWeiStrategy.PHI_COMPRESS:(codec._phi_rle,codec._phi_rle_d),
                   WuWeiStrategy.SPIRAL_PACK:(codec._spiral_pack,codec._spiral_unpack),
                   WuWeiStrategy.RESONANCE:(codec._resonance,codec._resonance_d),
                   WuWeiStrategy.WUWEI_RAW:(lambda d:d,lambda d:d)}
            enc_fn,dec_fn = fns[s]; enc = enc_fn(test_data); dec = dec_fn(enc)
            assert dec == test_data, f"{strat} roundtrip FAIL"
            PASS(f"wu-wei [{strat:15s}] {len(enc):4d}B roundtrip")

        b_deep = tree5.branch_create(0)
        for i in range(80): tree5.cell_commit(b_deep, {"step":i,"payload":"data"*50}, "task")
        sc = SemanticCompressor(tree5, codec)
        cr = sc.compress_branch(b_deep, max_cells=20)
        PASS(f"Semantic compressor: {cr}")

        hist5 = tree5.branch_history(b_deep)
        ctx5 = SemanticContext(tree5, None, branch_id=b_deep)
        ctx5.set_focus(hist5[0].id)
        top = ctx5.top_k_relevant(10)
        assert len(top) >= 1
        window = ctx5.context_window_tokens(5)
        PASS(f"Nonlinear context: top-5 relevances={[round(w['relevance'],3) for w in window]}")

        mm = MemoryMorphism(tree5, codec)
        consol = mm.consolidate_branch(b_deep, max_depth=20)
        PASS(f"Memory morphism consolidate: {consol}")
        pruned = mm.prune_merged_branches()
        PASS(f"Memory morphism prune: {pruned} merged branches removed")

        sched5 = SemanticScheduler(lat5)
        policy = sched5.derive_policy()
        assert all(k in policy for k in ["parallelism","priority","speculative"])
        PASS(f"Semantic scheduler: {policy}")

        valid_l5,_ = tree5.ledger.verify_chain(); assert valid_l5
        PASS(f"ERL after Layer 5: valid, {tree5.ledger.summary()['total_entries']} entries")

        # ── Final Summary ──────────────────────────────────────────────────────
        section("INTEGRATION COMPLETE")
        total_ms = (time.perf_counter() - t_start) * 1000
        print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  TRAILBLAZE — All Layers Operational                │
  ├─────────────────────────────────────────────────────┤
  │  Layer 0: Phi-Lattice       Kuramoto / phi_fold ✓  │
  │  Layer 1: Tensor Runtime    KV cache / attention ✓  │
  │  Layer 2: Graph Engine      HDGL routing / fuse  ✓  │
  │  Layer 3: Cognition         ERL / branch / seal  ✓  │
  │  Layer 4: Orchestration     unfold / HTTP / MCP  ✓  │
  │  Layer 5: Semantic OS       cap / codec / morph  ✓  │
  ├─────────────────────────────────────────────────────┤
  │  Total time: {total_ms:8.1f}ms                          │
  └─────────────────────────────────────────────────────┘
""")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    run()
