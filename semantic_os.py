"""TRAILBLAZE — Layer 5: Semantic Operating Layer
Capability tokens (lattice=authority), wu-wei 5-strategy codec,
nonlinear context window, memory morphism, lattice-derived scheduler.
"""
import sys,os,json,time,math
from typing import Optional,List,Dict,Any,Tuple
from dataclasses import dataclass,field
from enum import IntEnum
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer0.phi_lattice import PhiLattice,PHI,APA_FLAG_CONSENSUS
from layer3.cognition import CognitionTree,CognitionCell,ERLType

class Capability(IntEnum):
    SHELL_EXEC=1;FILE_WRITE=2;FILE_READ=3;NETWORK_FETCH=4;EPOCH_ADVANCE=5
    BRANCH_CREATE=6;BRANCH_MERGE=7;DB_WRITE=8;AGENT_SPAWN=9;MODEL_LOAD=10

@dataclass
class CapabilityToken:
    capability:Capability; grantee:int; epoch:int; slot:int
    dn_amplitude:float; token_bytes:bytes; expires_at:float

class CapabilityAuthority:
    TTL=300.0
    def __init__(self,tree):
        self.tree=tree; self._tokens:Dict[Tuple,CapabilityToken]={}
        self._defaults={0:list(Capability)}
    def grant(self,cap,branch_id,ttl=None):
        lat=self.tree.lattice; pf=lat.phi_fold; ep=lat.epoch
        slot=lat.slot_for_key(f"cap:{cap.name}:{branch_id}"); dn=lat.slots[slot].dn_amplitude
        tb=pf.hash32(cap.value.to_bytes(4,"little")+branch_id.to_bytes(8,"little")+ep.to_bytes(8,"little")+slot.to_bytes(4,"little"))
        tok=CapabilityToken(capability=cap,grantee=branch_id,epoch=ep,slot=slot,dn_amplitude=dn,token_bytes=tb,expires_at=time.time()+(ttl or self.TTL))
        self._tokens[(cap,branch_id,ep)]=tok; return tok
    def verify(self,cap,branch_id):
        if branch_id==0 and cap in self._defaults.get(0,[]): return True
        lat=self.tree.lattice; ep=lat.epoch; tok=self._tokens.get((cap,branch_id,ep))
        if not tok or time.time()>tok.expires_at: return False
        pf=lat.phi_fold
        exp=pf.hash32(cap.value.to_bytes(4,"little")+branch_id.to_bytes(8,"little")+ep.to_bytes(8,"little")+tok.slot.to_bytes(4,"little"))
        return exp==tok.token_bytes
    def revoke_all(self,bid):
        to_del=[k for k in self._tokens if k[1]==bid]
        for k in to_del: del self._tokens[k]
    def epoch_revoke(self):
        old=self.tree.lattice.epoch-1
        for k in [k for k in self._tokens if k[2]==old]: del self._tokens[k]
    def describe(self):
        lat=self.tree.lattice
        return {"active_tokens":len([t for t in self._tokens.values() if t.epoch==lat.epoch]),"epoch":lat.epoch,"caps":[c.name for c in Capability]}

class WuWeiStrategy(IntEnum):
    DELTA_FOLD=0;PHI_COMPRESS=1;SPIRAL_PACK=2;RESONANCE=3;WUWEI_RAW=4

class WuWeiCodec:
    def __init__(self,lattice): self.lattice=lattice
    def _select(self,data):
        M,L,S=self.lattice.s_u_resonance(); cv=self.lattice.phase_var
        if cv>0.8: return WuWeiStrategy.WUWEI_RAW
        if S>1.5:  return WuWeiStrategy.RESONANCE
        if S>1.0:  return WuWeiStrategy.PHI_COMPRESS
        if abs(math.modf(L)[0])>0.6: return WuWeiStrategy.SPIRAL_PACK
        return WuWeiStrategy.DELTA_FOLD
    def compress(self,data):
        s=self._select(data)
        fns={WuWeiStrategy.WUWEI_RAW:lambda d:d,WuWeiStrategy.PHI_COMPRESS:self._phi_rle,
             WuWeiStrategy.DELTA_FOLD:self._delta_fold,WuWeiStrategy.SPIRAL_PACK:self._spiral_pack,
             WuWeiStrategy.RESONANCE:self._resonance}
        return fns[s](data),s
    def decompress(self,data,s):
        fns={WuWeiStrategy.WUWEI_RAW:lambda d:d,WuWeiStrategy.PHI_COMPRESS:self._phi_rle_d,
             WuWeiStrategy.DELTA_FOLD:self._delta_unfold,WuWeiStrategy.SPIRAL_PACK:self._spiral_unpack,
             WuWeiStrategy.RESONANCE:self._resonance_d}
        return fns[s](data)
    def _phi_rle(self,d):
        th=int(PHI*256)%16; out=bytearray(); i=0
        while i<len(d):
            b=d[i]; r=1
            while i+r<len(d) and d[i+r]==b: r+=1
            if r>th: out.extend([0xFF,b,min(r,255)]); i+=min(r,255)
            else: out.append(b); i+=1
        return bytes(out)
    def _phi_rle_d(self,d):
        out=bytearray(); i=0
        while i<len(d):
            if d[i]==0xFF and i+2<len(d): out.extend([d[i+1]]*d[i+2]); i+=3
            else: out.append(d[i]); i+=1
        return bytes(out)
    def _delta_fold(self,d):
        slots=self.lattice.slots; n=len(slots)
        return bytes((b-int(slots[i%n].value*255.999))&0xFF for i,b in enumerate(d))
    def _delta_unfold(self,d):
        slots=self.lattice.slots; n=len(slots)
        return bytes((b+int(slots[i%n].value*255.999))&0xFF for i,b in enumerate(d))
    def _spiral_pack(self,d):
        from layer0.phi_lattice import SPIRAL8_TABLE
        out=bytearray()
        for i in range(0,len(d),8):
            ch=d[i:i+8]; dim=SPIRAL8_TABLE[i//8%8]["dim"]
            out.extend(((b<<dim)|(b>>(8-dim)))&0xFF for b in ch)
        return bytes(out)
    def _spiral_unpack(self,d):
        from layer0.phi_lattice import SPIRAL8_TABLE
        out=bytearray()
        for i in range(0,len(d),8):
            ch=d[i:i+8]; dim=SPIRAL8_TABLE[i//8%8]["dim"]
            out.extend(((b>>dim)|(b<<(8-dim)))&0xFF for b in ch)
        return bytes(out)
    def _perm(self):
        slots=self.lattice.slots[:256]
        order=sorted(range(256),key=lambda i:-slots[i].dn_amplitude)
        perm=[0]*256; inv=[0]*256
        for rank,idx in enumerate(order): perm[idx]=rank; inv[rank]=idx
        return perm,inv
    def _resonance(self,d): p,_=self._perm(); return bytes(p[b] for b in d)
    def _resonance_d(self,d): _,iv=self._perm(); return bytes(iv[b] for b in d)

class SemanticCompressor:
    def __init__(self,tree,codec): self.tree=tree; self.codec=codec
    def compress_branch(self,bid,max_cells=100):
        hist=self.tree.branch_history(bid,max_depth=500)
        if len(hist)<=max_cells: return {"compressed":0,"total":len(hist),"bytes_saved":0}
        old_cells=hist[max_cells:]; compressed=0; saved=0
        for i,cell in enumerate(old_cells):
            val=cell.value
            raw=json.dumps(val).encode() if isinstance(val,dict) else (val.encode() if isinstance(val,str) else (val if isinstance(val,bytes) else None))
            if raw is None or len(raw)<32: continue
            age=len(hist)-len(old_cells)+i; rel=PHI**(-age*0.1)
            if rel<0.3:
                comp,strat=self.codec.compress(raw)
                if len(comp)<len(raw):
                    cell.value={"__compressed":True,"strategy":int(strat),"data":comp.hex(),"original_len":len(raw)}
                    saved+=len(raw)-len(comp); compressed+=1
        return {"compressed":compressed,"total":len(hist),"bytes_saved":saved}
    def decompress_cell(self,cell):
        v=cell.value
        if not isinstance(v,dict) or not v.get("__compressed"): return v
        raw=self.codec.decompress(bytes.fromhex(v["data"]),WuWeiStrategy(v["strategy"]))
        try: return json.loads(raw.decode())
        except: return raw.decode() if raw else raw

class SemanticContext:
    def __init__(self,tree,ops,branch_id=0,relevance_decay=0.9):
        self.tree=tree; self.ops=ops; self.branch_id=branch_id
        self.relevance_decay=relevance_decay; self._focus=None; self._rcache={}
    def set_focus(self,cid): self._focus=cid; self._rcache.clear()
    def compute_relevance(self,cell):
        if cell.id in self._rcache: return self._rcache[cell.id]
        hist=self.tree.branch_history(self.branch_id); hids=[c.id for c in hist]
        try:
            fp=hids.index(self._focus) if self._focus else 0
            cp=hids.index(cell.id); dist=abs(fp-cp)
        except ValueError: dist=len(hist)
        dn=max(cell.dn_amplitude,0.1); r=dn*(PHI**(-dist*(1.0-self.relevance_decay)))
        self._rcache[cell.id]=r; return r
    def top_k_relevant(self,k=20):
        hist=self.tree.branch_history(self.branch_id,max_depth=500)
        if not hist: return []
        scored=sorted(hist,key=lambda c:-self.compute_relevance(c))
        return scored[:k]
    def context_window_tokens(self,k=20):
        return [{"cell_id":hex(c.id),"domain":c.domain,"epoch":c.epoch,
                 "relevance":round(self.compute_relevance(c),4),
                 "value":(c.value if isinstance(c.value,str) else json.dumps(c.value,default=str))[:200]}
                for c in self.top_k_relevant(k)]
    def describe(self):
        return {"branch_id":self.branch_id,"focus":hex(self._focus) if self._focus else None,
                "relevance_decay":self.relevance_decay,"top10":[hex(c.id) for c in self.top_k_relevant(10)]}

class MemoryMorphism:
    def __init__(self,tree,codec): self.tree=tree; self.codec=codec; self.compressor=SemanticCompressor(tree,codec)
    def consolidate_branch(self,bid,max_depth=50):
        hist=self.tree.branch_history(bid,max_depth=500); nb=len(hist)
        cr=self.compressor.compress_branch(bid,max_cells=max_depth)
        seen={}; deduped=0
        for c in hist:
            k=json.dumps(c.value,default=str,sort_keys=True)[:100]
            if k in seen: c.flags|=0x01; deduped+=1
            else: seen[k]=c.id
        self.tree.ledger.append(ERLType.TOOL_CALL,bid,self.tree.lattice.epoch,{"morphism":"consolidate","nb":nb,"comp":cr["compressed"],"dedup":deduped})
        return {"n_before":nb,"compressed":cr["compressed"],"bytes_saved":cr["bytes_saved"],"deduped":deduped}
    def prune_merged_branches(self):
        to_prune=[bid for bid,b in self.tree._branches.items() if b.merged and bid!=0]
        for bid in to_prune: del self.tree._branches[bid]
        return len(to_prune)
    def extract_semantic_summary(self,bid,n=10):
        hist=self.tree.branch_history(bid,max_depth=n*3)
        top=sorted(hist,key=lambda c:-c.dn_amplitude)[:n]
        parts=[]
        for c in top:
            v=c.value
            parts.append(f"[{c.domain}] {json.dumps(v)[:100] if isinstance(v,dict) else str(v)[:100]}")
        return "\n".join(parts)

class SemanticScheduler:
    def __init__(self,lattice): self.lattice=lattice
    def derive_policy(self):
        M,L,S=self.lattice.s_u_resonance(); cv=self.lattice.phase_var
        cons=bool(self.lattice.slots[0].flags&APA_FLAG_CONSENSUS)
        par=8 if cv<0.2 else 4 if cv<0.5 else 1
        pri="HIGH" if S>1.5 else "NORMAL" if S>0.8 else "LOW"
        to=max(5,int(30*(1.0+math.modf(L)[0])))
        return {"parallelism":par,"priority":pri,"timeout_s":to,"speculative":cons,
                "M_U":round(M,4),"Lambda_U":round(L,4),"S_U":round(S,4),"phase_var":round(cv,4),"consensus":cons}
    def apply_to_graph(self,graph,policy):
        for n in graph.nodes:
            if policy["priority"]=="LOW": n.kwargs["deferred"]=True
            elif policy["priority"]=="HIGH": n.kwargs["priority"]="high"
        return graph
