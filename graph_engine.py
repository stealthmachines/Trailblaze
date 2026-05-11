"""TRAILBLAZE — Layer 2: Graph Execution Engine
TB_Node/TB_Graph, HDGL phi-router, Spiral8 topology, Kuramoto scheduler,
operator fusion, pass-sequence compiler. From conscious hdgl_router.c + MCP unfold().
"""
import sys, os, time, math
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer0.phi_lattice import PhiLattice, Backend, OpClass, BackendRegistry, SPIRAL8_TABLE, PHI

class PassType(IntEnum):
    FETCH=0; SHELL=1; CODE=2; TRANSFORM=3; STORE=4; RECALL=5
    BROWSE=6; NOTIFY=7; RESPOND=8; EMBED=10; ATTEND=11; FFN=12
    SAMPLE=13; ROUTE=14; NORM=15; PROJ=16; BRANCH=20; MERGE=21; SPECULATE=22

PASS_WAVE = {PassType.FETCH:1,PassType.BROWSE:1,PassType.RECALL:1,PassType.EMBED:1,
    PassType.ATTEND:0,PassType.FFN:0,PassType.NORM:0,PassType.TRANSFORM:0,
    PassType.CODE:0,PassType.SHELL:0,PassType.STORE:-1,PassType.RESPOND:-1,
    PassType.NOTIFY:-1,PassType.SAMPLE:-1,PassType.ROUTE:1,PassType.PROJ:0,
    PassType.BRANCH:1,PassType.MERGE:-1,PassType.SPECULATE:0}
PASS_DIM  = {PassType.EMBED:1,PassType.ATTEND:2,PassType.FFN:3,PassType.NORM:4,
    PassType.PROJ:5,PassType.SAMPLE:6,PassType.ROUTE:7,PassType.FETCH:1,
    PassType.RECALL:2,PassType.CODE:3,PassType.TRANSFORM:4,PassType.STORE:5,
    PassType.RESPOND:6,PassType.BROWSE:7,PassType.SHELL:8,PassType.BRANCH:1,
    PassType.MERGE:8,PassType.SPECULATE:3,PassType.NOTIFY:6}

@dataclass
class Node:
    id: int; pass_type: PassType; name: str=""
    input_names: List[str]=field(default_factory=list)
    output_names: List[str]=field(default_factory=list)
    spiral8_dim: int=1; wave_mode: int=0
    op_fn: Optional[Callable]=None
    deps: List["Node"]=field(default_factory=list)
    fused_into: Optional["Node"]=None
    fused_nodes: List["Node"]=field(default_factory=list)
    backend: Backend=Backend.CPU_NUMPY
    epoch: int=0; result: Any=None; exec_time_ms: float=0.0
    layer_idx: int=0; kwargs: dict=field(default_factory=dict)
    def is_fused(self): return self.fused_into is not None
    def spiral8_info(self): return SPIRAL8_TABLE[max(0,min(7,self.spiral8_dim-1))]
    def __repr__(self): return f"Node({self.pass_type.name}:{self.name} dim={self.spiral8_dim})"

class Graph:
    def __init__(self, name, lattice):
        self.name=name; self.lattice=lattice; self.nodes=[]
        self.tensor_store={}; self._idx={}; self._topo=None
    def add_node(self, n):
        self.nodes.append(n); self._idx[n.id]=n; self._topo=None; return n
    def make_node(self, pt, name="", deps=None, **kw):
        key=f"{pt.name}:{name}:{len(self.nodes)}"
        nid=self.lattice.phi_hash_address(key.encode())
        n=Node(id=nid,pass_type=pt,name=name or pt.name,
               spiral8_dim=PASS_DIM.get(pt,1),wave_mode=PASS_WAVE.get(pt,0),
               deps=deps or [],epoch=self.lattice.epoch,kwargs=kw)
        return self.add_node(n)
    def topo_sort(self):
        if self._topo: return self._topo
        indeg={n.id:0 for n in self.nodes}; adj={n.id:[] for n in self.nodes}
        for n in self.nodes:
            for d in n.deps:
                if d.id in adj: adj[d.id].append(n); indeg[n.id]+=1
        q=[n for n in self.nodes if indeg[n.id]==0]; res=[]
        while q:
            n=q.pop(0); res.append(n)
            for s in adj[n.id]:
                indeg[s.id]-=1
                if indeg[s.id]==0: q.append(s)
        if len(res)!=len(self.nodes): raise RuntimeError("Cycle in graph")
        self._topo=res; return res
    def describe(self):
        return [{"name":n.name,"type":n.pass_type.name,"dim":n.spiral8_dim,
                 "wave":n.wave_mode,"backend":n.backend.name,"fused":n.is_fused()} for n in self.topo_sort()]

class HDGLRouter:
    DN_FAST=3.0; CV_LOCK=0.3; CV_SCATTER=0.8
    def __init__(self, lattice, registry):
        self.lattice=lattice; self.registry=registry; self._hist=[]
    def route_node(self, node):
        lat=self.lattice; key=f"{node.pass_type.name}:{node.name}:{node.spiral8_dim}"
        slot=lat.slot_for_key(key); s=lat.slots[slot]; dn=s.dn_amplitude; cv=lat.phase_var
        op={PassType.ATTEND:OpClass.ATTENTION,PassType.FFN:OpClass.MATMUL,
            PassType.PROJ:OpClass.MATMUL,PassType.EMBED:OpClass.ELEMENTWISE,
            PassType.NORM:OpClass.ELEMENTWISE}.get(node.pass_type,OpClass.ELEMENTWISE)
        if dn>self.DN_FAST and cv<self.CV_LOCK: b=Backend.CPU_NUMPY
        elif cv>self.CV_SCATTER: b=Backend.ANALOG
        else: b=self.registry.select(op,1024)
        if node.wave_mode==-1 and node.pass_type in (PassType.STORE,PassType.NOTIFY): b=Backend.CPU_NUMPY
        node.backend=b; self._hist.append((key,b,dn)); return b
    def route_server(self, key, pool):
        if not pool: raise ValueError("empty pool")
        slot=self.lattice.slot_for_key(key)
        idx=int(math.modf(slot*PHI*PHI/self.lattice.n_slots)[0]*len(pool))
        return pool[idx%len(pool)]
    def summary(self):
        if not self._hist: return {}
        avg=sum(d for _,_,d in self._hist)/len(self._hist)
        by={}
        for _,b,_ in self._hist: by[b.name]=by.get(b.name,0)+1
        return {"routed":len(self._hist),"avg_dn":round(avg,3),"by_backend":by}

class FusionEngine:
    @staticmethod
    def fuse(graph):
        topo=graph.topo_sort(); fused=0
        cons={n.id:[] for n in topo}
        for n in topo:
            for d in n.deps:
                if d.id in cons: cons[d.id].append(n)
        for n in topo:
            if n.is_fused(): continue
            succs=cons[n.id]
            if len(succs)!=1: continue
            s=succs[0]
            if s.is_fused(): continue
            if n.spiral8_dim==s.spiral8_dim and n.wave_mode==s.wave_mode:
                s.fused_into=n; n.fused_nodes.append(s)
                for d in s.deps:
                    if d.id!=n.id and d not in n.deps: n.deps.append(d)
                n.output_names.extend(s.output_names); fused+=1
        graph._topo=None; return fused

class KuramotoScheduler:
    def __init__(self, lattice, router): self.lattice=lattice; self.router=router
    def schedule(self, graph):
        plan=[]
        for n in graph.topo_sort():
            if not n.is_fused(): self.router.route_node(n); plan.append(n)
        return plan
    def stats(self, plan):
        wc={-1:0,0:0,1:0}; dc={}
        for n in plan: wc[n.wave_mode]=wc.get(n.wave_mode,0)+1; dc[n.spiral8_dim]=dc.get(n.spiral8_dim,0)+1
        return {"total":len(plan),"absorbing":wc[-1],"standing":wc[0],"propagating":wc[1],"dims":dc}

class GraphExecutor:
    def __init__(self, ops, lattice): self.ops=ops; self.lattice=lattice
    def execute(self, graph, plan, context=None):
        if context: graph.tensor_store.update(context)
        for n in plan:
            t0=time.perf_counter(); self._run(n,graph); n.exec_time_ms=(time.perf_counter()-t0)*1000
            for fn in n.fused_nodes:
                t1=time.perf_counter(); self._run(fn,graph); fn.exec_time_ms=(time.perf_counter()-t1)*1000
        return graph.tensor_store
    def _run(self, node, graph):
        if node.op_fn:
            inp={nm:graph.tensor_store.get(nm) for nm in node.input_names}
            r=node.op_fn(inp,graph.tensor_store,**node.kwargs)
            node.result=r
            if node.output_names and r is not None:
                if isinstance(r,dict): graph.tensor_store.update(r)
                elif len(node.output_names)==1: graph.tensor_store[node.output_names[0]]=r
        else:
            node.result={"status":"deferred","pass":node.pass_type.name}
    def report(self, plan):
        return [{"name":n.name,"type":n.pass_type.name,"backend":n.backend.name,
                 "ms":round(n.exec_time_ms,3)} for n in plan]

class TaskCompiler:
    PATTERNS=[
        (["search","find","fetch","lookup"],[PassType.FETCH,PassType.TRANSFORM,PassType.RESPOND]),
        (["run","execute","shell","bash"],[PassType.SHELL,PassType.STORE,PassType.RESPOND]),
        (["code","write","implement","build"],[PassType.CODE,PassType.STORE,PassType.RESPOND]),
        (["read","open","load","recall"],[PassType.RECALL,PassType.TRANSFORM,PassType.RESPOND]),
        (["save","store","persist"],[PassType.RECALL,PassType.TRANSFORM,PassType.STORE]),
        (["browse","visit","navigate","web"],[PassType.BROWSE,PassType.TRANSFORM,PassType.RESPOND]),
        (["analyze","summarize","explain"],[PassType.RECALL,PassType.CODE,PassType.TRANSFORM,PassType.RESPOND]),
    ]
    def __init__(self, lattice): self.lattice=lattice
    def analyze(self, task):
        tl=task.lower()
        for kws,seq in self.PATTERNS:
            if any(k in tl for k in kws): return seq
        return [PassType.RECALL,PassType.TRANSFORM,PassType.RESPOND]
    def compile(self, task):
        seq=self.analyze(task); g=Graph(f"task:{task[:32]}",self.lattice); prev=None
        for i,pt in enumerate(seq):
            n=g.make_node(pt,f"{pt.name}_{i}",deps=[prev] if prev else [])
            n.input_names=[f"step_{i}_in"]; n.output_names=[f"step_{i}_out"]; prev=n
        FusionEngine.fuse(g)
        router=HDGLRouter(self.lattice,BackendRegistry(self.lattice))
        KuramotoScheduler(self.lattice,router).schedule(g)
        return g
