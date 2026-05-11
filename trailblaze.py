#!/usr/bin/env python3
"""
TRAILBLAZE CLI — Post-CUDA Recursive Cognition Runtime
Usage: trailblaze <cmd> [options]
  serve     Start MCP HTTP server
  unfold    Execute a task
  inspect   Show runtime state
  ledger    ERL ledger operations
  branch    Branch management
  epoch     Epoch operations
  lattice   Phi-lattice info
  tool      Call a tool directly
  bench     Benchmark suite
"""
import sys, os, json, argparse, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from layer4.orchestration import TrailblazeRuntime

def rt(args):
    return TrailblazeRuntime(
        persist_dir=getattr(args,'persist',None),
        n_lattice_slots=getattr(args,'slots',512),
        seed=getattr(args,'seed',None)
    )

def cmd_serve(args):
    r = rt(args)
    print(f"[TRAILBLAZE] {len(r.tools._tools)} tools | {r.lattice.n_slots} slots | epoch={r.lattice.epoch}")
    r.serve(host=args.host, port=args.port, blocking=True)

def cmd_unfold(args):
    r = rt(args)
    task = ' '.join(args.task)
    print(f"[unfold] '{task}' branch={args.branch}")
    res = r.unfold(task, branch_id=args.branch)
    print(f"  success:    {res.success}")
    print(f"  passes:     {res.pass_sequence}")
    print(f"  nodes:      {res.graph_nodes}")
    print(f"  exec_ms:    {res.exec_ms:.1f}")
    print(f"  erl_new:    {res.erl_entries}")
    if res.error:   print(f"  error:      {res.error}")
    if res.results: print(f"  results:    {json.dumps(res.results, default=str)[:300]}")

def cmd_inspect(args):
    print(json.dumps(rt(args).describe(), indent=2, default=str))

def cmd_ledger(args):
    r = rt(args); L = r.tree.ledger
    print(f"[ERL] entries={len(L._entries)} branches={list(L._by_branch.keys())}")
    print(f"      head={L.head_hash()[:32]}...")
    if getattr(args,'verify',False):
        ok,broken = L.verify_chain()
        print(f"      chain={'VALID ✓' if ok else f'BROKEN at seq={broken} ✗'}")
        if not ok: sys.exit(1)
    tail = getattr(args,'tail',10)
    for e in L._entries[-tail:]:
        ts = time.strftime('%H:%M:%S', time.localtime(e.timestamp))
        print(f"  [{e.seq:4d}] {ts} {e.type:20s} br={e.branch_id} ep={e.epoch} {str(e.data)[:60]}")

def cmd_branch(args):
    r = rt(args); sub = args.branch_cmd
    if sub == 'create':
        b = r.tree.branch_create(getattr(args,'from_branch',0))
        print(f"Created branch {b}  tip={hex(r.tree.branch_tip(b))}")
    elif sub == 'merge':
        ok = r.tree.branch_merge(int(args.src), int(args.dst))
        print(f"Merge branch {args.src} -> {args.dst}: {'OK' if ok else 'FAILED'}")
    elif sub == 'list':
        for bid,b in r.tree._branches.items():
            st = 'merged' if b.merged else 'active'
            kv = b.kv_cache.seq_len if b.kv_cache else 0
            print(f"  {bid:3d} [{st:6s}] tip={hex(b.tip)[:18]}... ep={b.epoch_created} kv={kv}")

def cmd_epoch(args):
    r = rt(args); old = r.lattice.epoch
    new = r.tree.epoch_advance(getattr(args,'steps',1))
    print(f"Epoch {old} -> {new}  (KV caches cleared, S-box rebuilt)")
    print(f"Lattice: {r.lattice.describe()}")

def cmd_lattice(args):
    r = rt(args); steps = getattr(args,'steps',10)
    r.lattice.advance(steps)
    d = r.lattice.describe(); M,L,S = r.lattice.s_u_resonance()
    print(json.dumps(d, indent=2))
    print(f"\nU-field: M_U={M:.4f} Λ^U={L:.4f} S(U)={S:.4f}")
    for k in ["matmul","attention","ffn","embed","sample"]:
        slot=r.lattice.slot_for_key(k); dn=r.lattice.slots[slot].dn_amplitude
        print(f"  '{k}' -> slot {slot:4d} dim={r.lattice.slots[slot].dimension} Dn={dn:.3f}")

def cmd_tool(args):
    r = rt(args)
    kwargs = json.loads(getattr(args,'args','{}') or '{}')
    res = r.tools.call(args.tool_name, branch_id=getattr(args,'branch',0), **kwargs)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    if not res.success: sys.exit(1)

def cmd_bench(args):
    import time
    r = rt(args); print("[TRAILBLAZE Benchmark]\n")

    t0=time.perf_counter()
    for _ in range(100): r.lattice.advance(1)
    dt=(time.perf_counter()-t0)*1000
    print(f"  lattice.advance()     100 steps  : {dt:7.1f}ms  ({100/dt*1000:6.0f}/s)")

    pf=r.lattice.phi_fold
    t0=time.perf_counter()
    for i in range(1000): pf.hash32(f"bench_{i}".encode())
    dt=(time.perf_counter()-t0)*1000
    print(f"  phi_fold.hash32()     1000 calls : {dt:7.1f}ms  ({1000/dt*1000:6.0f}/s)")

    ps=r.lattice.phi_stream; msg=b"x"*256
    t0=time.perf_counter()
    for _ in range(500): ps.seal(msg)
    dt=(time.perf_counter()-t0)*1000
    print(f"  phi_stream.seal()     500x256B   : {dt:7.1f}ms  ({500/dt*1000:6.0f}/s)")

    t0=time.perf_counter()
    for i in range(200): r.tree.cell_commit(0,{"i":i},"bench")
    dt=(time.perf_counter()-t0)*1000
    print(f"  cell_commit()         200 commits : {dt:7.1f}ms  ({200/dt*1000:6.0f}/s)")

    t0=time.perf_counter()
    ok,_=r.tree.ledger.verify_chain()
    dt=(time.perf_counter()-t0)*1000; n=len(r.tree.ledger._entries)
    print(f"  ledger.verify_chain() {n:4d} entries: {dt:7.1f}ms  (valid={ok})")

    tasks=["search docs","run tests","save results","analyze code","build api"]
    t0=time.perf_counter()
    for task in tasks: r.unfold(task)
    dt=(time.perf_counter()-t0)*1000
    print(f"  unfold()              {len(tasks)} tasks    : {dt:7.1f}ms  ({len(tasks)/dt*1000:6.0f}/s)")

    from layer2.graph_engine import HDGLRouter
    from layer0.phi_lattice import BackendRegistry
    router=HDGLRouter(r.lattice,BackendRegistry(r.lattice)); svrs=[f"s{i}" for i in range(5)]
    t0=time.perf_counter()
    for i in range(5000): router.route_server(f"req_{i}",svrs)
    dt=(time.perf_counter()-t0)*1000
    print(f"  hdgl.route_server()   5000 routes : {dt:7.1f}ms  ({5000/dt*1000:6.0f}/s)")

    print(f"\nLattice: {r.lattice.describe()}")

def main():
    p = argparse.ArgumentParser(prog='trailblaze', description='TRAILBLAZE — Post-CUDA Recursive Cognition Runtime')
    p.add_argument('--persist','-p', default=None)
    p.add_argument('--slots','-s', type=int, default=512)
    p.add_argument('--seed', type=lambda x: int(x,0), default=None)
    sub = p.add_subparsers(dest='cmd')

    ps=sub.add_parser('serve'); ps.add_argument('--host',default='0.0.0.0'); ps.add_argument('--port','-P',type=int,default=3333)
    pu=sub.add_parser('unfold'); pu.add_argument('task',nargs='+'); pu.add_argument('--branch','-b',type=int,default=0)
    sub.add_parser('inspect')
    pl=sub.add_parser('ledger'); pl.add_argument('--verify','-v',action='store_true'); pl.add_argument('--tail','-t',type=int,default=10)
    pb=sub.add_parser('branch'); pb.add_argument('branch_cmd',choices=['create','merge','list'])
    pb.add_argument('--from-branch',dest='from_branch',type=int,default=0)
    pb.add_argument('--src',default=None); pb.add_argument('--dst',type=int,default=0)
    pe=sub.add_parser('epoch'); pe.add_argument('epoch_cmd',choices=['advance']); pe.add_argument('--steps',type=int,default=1)
    pla=sub.add_parser('lattice'); pla.add_argument('--steps',type=int,default=10)
    pt=sub.add_parser('tool'); pt.add_argument('tool_name'); pt.add_argument('--args','-a',default='{}'); pt.add_argument('--branch','-b',type=int,default=0)
    sub.add_parser('bench')

    args = p.parse_args()
    if not args.cmd: p.print_help(); sys.exit(0)
    {'serve':cmd_serve,'unfold':cmd_unfold,'inspect':cmd_inspect,'ledger':cmd_ledger,
     'branch':cmd_branch,'epoch':cmd_epoch,'lattice':cmd_lattice,'tool':cmd_tool,'bench':cmd_bench}[args.cmd](args)

if __name__ == '__main__':
    main()
