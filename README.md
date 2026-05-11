# TRAILBLAZE — Post-CUDA Recursive Cognition Runtime
### v0.1 — Architecture + Full Stack Implementation

A next-generation local AI runtime and cognition substrate designed to eventually
replace CUDA orchestration, stateless prompt-response architectures, and linear
transformer execution assumptions.

---

## Architecture

```
LAYER 5  Semantic OS          — capability tokens, wu-wei codec, nonlinear context
LAYER 4  Agent Orchestration  — unfold() engine, MCP server, recursive delegation  
LAYER 3  Cognition Substrate  — ERL ledger, CognitionTree, branch model, sealing
LAYER 2  Graph Engine         — HDGL router, Spiral8 topology, Kuramoto scheduler
LAYER 1  Tensor Runtime       — epoch-aware tensors, KV cache, attention, Q8_0
LAYER 0  Phi-Lattice Core     — Kuramoto dynamics, phi_fold hash, PhiStream AEAD
```

## Files

```
trailblaze/
├── layer0/phi_lattice.py      # PhiLattice, PhiFold, PhiStream, BackendRegistry
├── layer1/tensor_runtime.py   # Tensor, KVCache, AttentionOp, TransformerBlock, Sampler
├── layer2/graph_engine.py     # Graph, HDGLRouter, FusionEngine, TaskCompiler
├── layer3/cognition.py        # ERLLedger, CognitionCell, CognitionTree
├── layer4/orchestration.py    # ToolRegistry, UnfoldEngine, AgentFabric, MCPServer, TrailblazeRuntime
├── layer5/semantic_os.py      # CapabilityAuthority, WuWeiCodec, SemanticContext, SemanticScheduler
├── cli/trailblaze.py          # CLI entry point
└── tests/integration.py       # Full 5-layer integration test (all green)
```

## Quick Start

```bash
# Run integration tests (all layers)
python3 tests/integration.py

# Benchmark
python3 cli/trailblaze.py bench

# Start MCP server
python3 cli/trailblaze.py serve --port 3333 --persist ./data

# Execute a task
python3 cli/trailblaze.py unfold "analyze the authentication module"

# Inspect runtime state
python3 cli/trailblaze.py inspect

# Verify ERL audit chain
python3 cli/trailblaze.py ledger --verify --tail 20

# Fork a branch
python3 cli/trailblaze.py branch create --from-branch 0

# Advance lattice epoch (forward secrecy ratchet)
python3 cli/trailblaze.py epoch advance --steps 1
```

## From Python

```python
from layer4.orchestration import TrailblazeRuntime

rt = TrailblazeRuntime(persist_dir="./data", n_lattice_slots=512)

# Execute a task (compiles → graph → executes → ERL)
result = rt.unfold("analyze the authentication module")
print(result.pass_sequence)   # ['RECALL', 'CODE', 'TRANSFORM', 'RESPOND']
print(result.erl_entries)     # ERL entries committed

# Fork a branch (COW KV cache)
branch_id = rt.tree.branch_create(from_bid=0)

# Parallel delegation
results = rt.delegate([
    "check SQL injection vectors",
    "verify CSRF tokens",
    "audit session management",
], from_branch=0)

# Epoch advance (forward secrecy: voids all KV caches + sealed state)
rt.tree.epoch_advance()

# MCP HTTP server
rt.serve(port=3333)   # /health /tools /unfold /state /ledger /sse
```

## Key Properties

| Property | Mechanism |
|---|---|
| **Branch-aware KV cache** | Each branch gets COW-forked cache, invalidated by epoch advance |
| **Forward secrecy** | `epoch_advance()` rebuilds S-box, voids sealed blobs and caches |
| **Phi-lattice addressing** | Node IDs are `phi_fold_hash64(...)[:16]` — not sequential integers |
| **HDGL routing** | Backend assignment via Dₙ amplitude + Kuramoto phase coherence |
| **Spiral8 topology** | 8 polytope geometry dims, wave modes +1/0/-1 assign execution character |
| **ERL audit chain** | Every state transition is a sha256-chained, branch-indexed ledger entry |
| **Wu-wei codec** | 5-strategy adaptive compression; strategy selected by S(U) resonance |
| **Nonlinear context** | φ-exponential relevance decay, not ring buffer |

## Benchmark Results (512-slot lattice)

```
lattice.advance()     100 steps  :    ~95ms  (~1000/s)
phi_fold.hash32()     1000 calls :    ~66ms  (~15000/s)
phi_stream.seal()     500×256B   :   ~411ms  (~1200/s)
cell_commit()         200 commits:    ~42ms  (~4700/s)
ledger.verify_chain() 200 entries:     ~2ms
unfold()              5 tasks    :     ~8ms  (~650/s)
hdgl.route_server()   5000 routes:   ~323ms  (~15000/s)
```

## Phase Roadmap

| Phase | Target | Status |
|---|---|---|
| **1** | Replace Ollama orchestration | ✅ MVK complete |
| **2** | Custom tensor kernels (Metal/CUDA) | 🔲 Backend stubs defined |
| **3** | HDGL routing live at inference level | 🔲 Router built, token-level pending |
| **4** | Post-linear cognition | 🔲 Layer 5 primitives built |

---

*Derived from: conscious-128-bit-floor (phi-lattice, Kuramoto, HDGL) + MCP-Jailbreak-0.4 (ERL, FlowState, unfold)*
