"""TRAILBLAZE — Layer 1: Tensor Runtime
Epoch-aware tensors, branch-scoped KV cache, quantization, matmul, attention, RMSNorm.
"""
import sys, os, math
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer0.phi_lattice import PhiLattice, Backend, OpClass, BackendRegistry

class DType(IntEnum):
    F32=0; F16=1; BF16=2; Q8_0=3; Q4_K=4; I32=6; I8=7

BLOCK_Q8 = 32

@dataclass
class Tensor:
    data: np.ndarray
    dtype: DType = DType.F32
    backend: Backend = Backend.CPU_NUMPY
    epoch: int = 0
    branch_id: int = 0
    name: str = ""
    scale: Optional[np.ndarray] = None
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def is_valid(self, ep): return self.epoch == ep
    def to_f32(self):
        if self.dtype == DType.F32: return self
        if self.dtype == DType.Q8_0: return _deq8(self)
        return Tensor(self.data.astype(np.float32), DType.F32, self.backend, self.epoch, self.branch_id)
    def __repr__(self): return f"Tensor({self.shape},{self.dtype.name},ep={self.epoch},br={self.branch_id})"

def make_tensor(data, lattice, branch_id=0, name=""):
    return Tensor(np.asarray(data, dtype=np.float32), DType.F32, Backend.CPU_NUMPY, lattice.epoch, branch_id, name)

def quantize_q8(t):
    data = t.data.astype(np.float32).flatten(); orig_len = len(data)
    nb = math.ceil(len(data)/BLOCK_Q8)
    pad = np.zeros(nb*BLOCK_Q8, np.float32); pad[:len(data)] = data
    blocks = pad.reshape(nb, BLOCK_Q8)
    scales = np.max(np.abs(blocks),axis=1,keepdims=True)/127.0
    scales = np.where(scales==0, 1.0, scales)
    q = np.clip(np.round(blocks/scales),-127,127).astype(np.int8)
    qt = Tensor(q.reshape(nb, BLOCK_Q8), DType.Q8_0, t.backend, t.epoch, t.branch_id)
    qt.scale = scales.reshape(-1); qt.name = str(orig_len)
    return qt

def _deq8(t):
    data = t.data.astype(np.float32).flatten(); scales = t.scale
    nb = len(scales); pad = np.zeros(nb*BLOCK_Q8,np.float32); pad[:len(data)]=data
    out = (pad.reshape(nb,BLOCK_Q8)*scales.reshape(-1,1)).flatten()
    orig = int(t.name) if t.name and t.name.isdigit() else len(data)
    return Tensor(out[:orig], DType.F32, t.backend, t.epoch, t.branch_id)

class TensorOps:
    def __init__(self, lattice, registry):
        self.lattice=lattice; self.registry=registry
    def _chk(self, *ts):
        ep = self.lattice.epoch
        for t in ts:
            if not t.is_valid(ep):
                raise ValueError(f"Stale tensor ep={t.epoch} current={ep}")
    def matmul(self, A, B):
        self._chk(A, B)
        return Tensor(np.matmul(A.to_f32().data, B.to_f32().data), DType.F32, Backend.CPU_NUMPY, self.lattice.epoch, A.branch_id)
    def linear(self, x, W, bias=None):
        self._chk(x)
        out = x.to_f32().data @ W.to_f32().data.T
        if bias is not None: out = out + bias.to_f32().data
        return Tensor(out, DType.F32, Backend.CPU_NUMPY, self.lattice.epoch, x.branch_id)
    def softmax(self, x, dim=-1):
        self._chk(x); xf = x.to_f32().data - x.to_f32().data.max(axis=dim,keepdims=True)
        ex = np.exp(xf); return Tensor(ex/ex.sum(axis=dim,keepdims=True), DType.F32, Backend.CPU_NUMPY, self.lattice.epoch, x.branch_id)
    def rms_norm(self, x, w, eps=1e-5):
        self._chk(x); xf = x.to_f32().data
        rms = np.sqrt((xf**2).mean(axis=-1,keepdims=True)+eps)
        return Tensor(xf/rms*w.to_f32().data, DType.F32, Backend.CPU_NUMPY, self.lattice.epoch, x.branch_id)
    def silu(self, x):
        self._chk(x); xf = x.to_f32().data
        return Tensor(xf*(1.0/(1.0+np.exp(-xf))), DType.F32, Backend.CPU_NUMPY, self.lattice.epoch, x.branch_id)
    def gelu(self, x):
        self._chk(x); xf = x.to_f32().data
        return Tensor(0.5*xf*(1+np.tanh(math.sqrt(2/math.pi)*(xf+0.044715*xf**3))), DType.F32, Backend.CPU_NUMPY, self.lattice.epoch, x.branch_id)

@dataclass
class KVCache:
    n_layers: int; n_heads: int; head_dim: int; max_seq: int; branch_id: int; epoch: int
    keys: List[np.ndarray] = field(default_factory=list)
    vals: List[np.ndarray] = field(default_factory=list)
    seq_len: int = 0
    def __post_init__(self):
        if not self.keys:
            self.keys=[np.zeros((self.n_heads,0,self.head_dim),np.float32) for _ in range(self.n_layers)]
            self.vals=[np.zeros((self.n_heads,0,self.head_dim),np.float32) for _ in range(self.n_layers)]
    def is_valid(self, ep): return self.epoch==ep
    def append(self, layer, k, v):
        if k.ndim==4: k=k[0]
        if v.ndim==4: v=v[0]
        self.keys[layer]=np.concatenate([self.keys[layer],k],axis=1)
        self.vals[layer]=np.concatenate([self.vals[layer],v],axis=1)
        if layer==0: self.seq_len+=k.shape[1]
    def fork(self, new_bid):
        c=KVCache(self.n_layers,self.n_heads,self.head_dim,self.max_seq,new_bid,self.epoch)
        c.keys=[k.copy() for k in self.keys]; c.vals=[v.copy() for v in self.vals]; c.seq_len=self.seq_len; return c
    def invalidate(self, ep):
        self.epoch=ep; self.seq_len=0
        self.keys=[np.zeros((self.n_heads,0,self.head_dim),np.float32) for _ in range(self.n_layers)]
        self.vals=[np.zeros((self.n_heads,0,self.head_dim),np.float32) for _ in range(self.n_layers)]
    def reconcile(self, other, lattice):
        m=KVCache(self.n_layers,self.n_heads,self.head_dim,self.max_seq,self.branch_id,lattice.epoch)
        for l in range(self.n_layers):
            if self.keys[l].shape[1]>=other.keys[l].shape[1]:
                m.keys[l]=self.keys[l].copy(); m.vals[l]=self.vals[l].copy()
            else:
                m.keys[l]=other.keys[l].copy(); m.vals[l]=other.vals[l].copy()
        m.seq_len=max(self.seq_len,other.seq_len); return m

def alloc_kvcache(n_layers,n_heads,head_dim,max_seq,branch_id,lattice):
    return KVCache(n_layers,n_heads,head_dim,max_seq,branch_id,lattice.epoch)

class AttentionOp:
    def __init__(self, ops, n_heads, n_kv_heads, head_dim):
        self.ops=ops; self.nh=n_heads; self.nkv=n_kv_heads; self.hd=head_dim
        self.scale=1.0/math.sqrt(head_dim)
    def forward(self, q, k, v, cache, layer_idx, pos):
        ops=self.ops; lat=ops.lattice; B=1; S=q.shape[-2] if q.ndim==3 else 1
        qd=q.to_f32().data.reshape(B,S,self.nh,self.hd).transpose(0,2,1,3)
        kd=k.to_f32().data.reshape(B,S,self.nkv,self.hd).transpose(0,2,1,3)
        vd=v.to_f32().data.reshape(B,S,self.nkv,self.hd).transpose(0,2,1,3)
        if cache is not None:
            if not cache.is_valid(lat.epoch): raise ValueError(f"KV cache stale ep={cache.epoch}")
            cache.append(layer_idx,kd,vd)
            kf=cache.keys[layer_idx]; vf=cache.vals[layer_idx]
        else:
            kf=kd[0]; vf=vd[0]
        if self.nh>self.nkv:
            rep=self.nh//self.nkv; kf=np.repeat(kf,rep,axis=0); vf=np.repeat(vf,rep,axis=0)
        qh=qd[0]; scores=np.matmul(qh,kf.transpose(0,2,1))*self.scale
        total=kf.shape[1]
        if S>1:
            mask=np.triu(np.full((S,total),-1e9),k=total-S+1); scores+=mask[np.newaxis,:,:]
        scores-=scores.max(axis=-1,keepdims=True); attn=np.exp(scores); attn/=attn.sum(axis=-1,keepdims=True)
        ctx=np.matmul(attn,vf).transpose(1,0,2).reshape(B,S,self.nh*self.hd)
        return Tensor(ctx.astype(np.float32),DType.F32,Backend.CPU_NUMPY,lat.epoch,q.branch_id)

@dataclass
class TransformerConfig:
    vocab_size:int=32000; n_layers:int=32; n_heads:int=32; n_kv_heads:int=8
    head_dim:int=128; hidden_dim:int=4096; ffn_dim:int=14336
    rope_base:float=10000.0; max_seq:int=4096; norm_eps:float=1e-5
    use_rms_norm:bool=True; activation:str="silu"; tie_embeddings:bool=True

class TransformerBlock:
    def __init__(self, cfg, layer_idx, ops, lattice):
        self.cfg=cfg; self.li=layer_idx; self.ops=ops; self.lat=lattice
        H=cfg.hidden_dim; D=cfg.head_dim; NH=cfg.n_heads; NK=cfg.n_kv_heads; F=cfg.ffn_dim; ep=lattice.epoch
        rn=lambda r,c: Tensor(np.random.randn(r,c).astype(np.float32)*0.02,epoch=ep,branch_id=0)
        self.wq=rn(NH*D,H); self.wk=rn(NK*D,H); self.wv=rn(NK*D,H); self.wo=rn(H,NH*D)
        self.w1=rn(F,H); self.w2=rn(H,F); self.w3=rn(F,H)
        wn=Tensor(np.ones(H,np.float32),epoch=ep,branch_id=0)
        self.norm1=wn; self.norm2=wn
        self.attn=AttentionOp(ops,NH,NK,D)
    def forward(self, x, cache, pos):
        ops=self.ops; xn=ops.rms_norm(x,self.norm1)
        q=ops.linear(xn,self.wq); k=ops.linear(xn,self.wk); v=ops.linear(xn,self.wv)
        ao=self.attn.forward(q,k,v,cache,self.li,pos)
        ap=ops.linear(ao,self.wo)
        xa=Tensor(x.data+ap.data,DType.F32,Backend.CPU_NUMPY,self.lat.epoch,x.branch_id)
        xn2=ops.rms_norm(xa,self.norm2)
        if self.cfg.activation=="silu":
            gate=ops.silu(ops.linear(xn2,self.w1)); up=ops.linear(xn2,self.w3)
            ff=ops.linear(Tensor(gate.data*up.data,epoch=self.lat.epoch,branch_id=x.branch_id),self.w2)
        else:
            ff=ops.linear(ops.gelu(ops.linear(xn2,self.w1)),self.w2)
        return Tensor(xa.data+ff.data,DType.F32,Backend.CPU_NUMPY,self.lat.epoch,x.branch_id)

class Sampler:
    @staticmethod
    def greedy(logits): return int(np.argmax(logits))
    @staticmethod
    def top_p(logits, p=0.9, temperature=1.0):
        if temperature!=1.0: logits=logits/temperature
        logits-=logits.max(); probs=np.exp(logits); probs/=probs.sum()
        si=np.argsort(-probs); sp=probs[si]; cs=np.cumsum(sp)
        cut=np.searchsorted(cs,p)+1; ti=si[:cut]; tp=sp[:cut]; tp/=tp.sum()
        return int(np.random.choice(ti,p=tp))
    @staticmethod
    def top_k(logits, k=50, temperature=1.0):
        if temperature!=1.0: logits=logits/temperature
        ki=np.argsort(logits)[-k:]; kl=logits[ki]; kl-=kl.max()
        p=np.exp(kl); p/=p.sum(); return int(np.random.choice(ki,p=p))
dequantize_q8 = _deq8
