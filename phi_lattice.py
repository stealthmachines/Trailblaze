"""
TRAILBLAZE — Layer 0: Phi-Lattice Core
Weyl-spaced lattice, Kuramoto coupling, phi_fold hash, PhiStream AEAD,
backend registry. Ported from conscious-128-bit-floor.
"""
import math, os, time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import IntEnum

PHI = 1.6180339887498948
LN_PHI = math.log(PHI)
GAMMA = 0.02
K_COUPLING = 1.0
CONSENSUS_EPS = 1e-6
CONSENSUS_N = 100
APA_FLAG_CONSENSUS = 1 << 4

SPIRAL8_TABLE = [
    {"dim": 1, "wave_mode":  1, "name": "segment"},
    {"dim": 2, "wave_mode":  0, "name": "square"},
    {"dim": 3, "wave_mode": -1, "name": "cube"},
    {"dim": 4, "wave_mode":  1, "name": "tesseract"},
    {"dim": 5, "wave_mode":  0, "name": "5-cube"},
    {"dim": 6, "wave_mode": -1, "name": "6-cube"},
    {"dim": 7, "wave_mode":  1, "name": "7-cube"},
    {"dim": 8, "wave_mode":  0, "name": "8-cube"},
]

@dataclass
class Slot4096:
    value: float = 0.0
    phase: float = 0.0
    freq: float = 1.0
    amp_re: float = 1.0
    amp_im: float = 0.0
    dn_amplitude: float = 0.0
    dimension: int = 1
    wave_mode: int = 0
    flags: int = 0

class PhiFold:
    def __init__(self, lattice):
        self.lattice = lattice
        self._sa = self._sb = None
        self._ep = -1
    def _refresh(self):
        if self._ep == self.lattice.epoch: return
        self._sa = self._sbox(1024); self._sb = self._sbox(2048)
        self._ep = self.lattice.epoch
    def _sbox(self, off):
        s = list(range(256)); slots = self.lattice.slots; n = len(slots)
        for i in range(255, 0, -1):
            j = int(slots[(off + i*3) % n].value * 255.999) % (i+1)
            s[i], s[j] = s[j], s[i]
        return bytes(s)
    def hash32(self, data):
        self._refresh()
        slots = self.lattice.slots; n = len(slots); phi_b = int(PHI*1000)&0xFF
        acc = bytearray(32)
        for i, b in enumerate(data):
            acc[i%32] = (acc[i%32] + b + int(slots[i%n].value*255.999)&0xFF) & 0xFF
        for r in range(12):
            for j in range(32):
                src = (j + r*7) % 32
                sv = (acc[j]+phi_b+acc[src]) & 0xFF
                sv = (sv>>1) | ((sv&1)<<7)
                acc[j] = self._sa[sv]
        return bytes(acc)
    def hash64(self, data):
        self._refresh()
        slots = self.lattice.slots; n = len(slots); phi_b = int(PHI*1000)&0xFF
        lo = bytearray(32); hi = bytearray(32)
        for i, b in enumerate(data):
            sf = int(slots[i%n].value*255.999)&0xFF
            sr = int(slots[(n-1-i%n)].value*255.999)&0xFF
            lo[i%32] = (lo[i%32]+b+sf)&0xFF; hi[i%32] = (hi[i%32]+b+sr)&0xFF
        for r in range(12):
            for j in range(32):
                src = (j+r*7)%32
                sl = (lo[j]+phi_b+lo[src])&0xFF; sh = (hi[j]+phi_b+hi[src])&0xFF
                lo[j] = self._sa[(sl^(hi[j]>>1))&0xFF]; hi[j] = self._sb[(sh^(lo[j]>>1))&0xFF]
        return bytes(lo)+bytes(hi)
    def derive_prk(self, ctx):
        cb = ctx.encode(); return self.hash32(self.hash32(cb)+cb)

class PhiStream:
    def __init__(self, pf):
        self.pf = pf; self._ctr = int.from_bytes(os.urandom(8),"little")
    def _ks(self, prk, ctr, n):
        ks = bytearray(); blk = 0
        while len(ks) < n:
            ks.extend(self.pf.hash32(prk+ctr.to_bytes(8,"little")+blk.to_bytes(4,"little"))); blk+=1
        return bytes(ks[:n])
    def seal(self, pt, ctx="default"):
        prk = self.pf.derive_prk(ctx); ctr = self._ctr; self._ctr += 1
        ks = self._ks(prk, ctr, len(pt)); ct = bytes((p+k)&0xFF for p,k in zip(pt,ks))
        return ctr.to_bytes(8,"little")+self.pf.hash32(ct)+ct
    def unseal(self, env, ctx="default"):
        if len(env)<40: return None
        ctr = int.from_bytes(env[:8],"little"); tag = env[8:40]; ct = env[40:]
        prk = self.pf.derive_prk(ctx); exp = self.pf.hash32(ct)
        if any(a^b for a,b in zip(tag,exp)): return None
        ks = self._ks(prk, ctr, len(ct)); return bytes((c-k)&0xFF for c,k in zip(ct,ks))

class PhiLattice:
    def __init__(self, n_slots=4096, seed=None):
        self.n_slots = n_slots; self.epoch = 0; self.time = 0.0
        self.omega = 1.0; self.phase_var = 1.0; self._csteps = 0
        self.slots: List[Slot4096] = []; self.seed_value = 0
        self._seed(seed)
        self.phi_fold = PhiFold(self); self.phi_stream = PhiStream(self.phi_fold)
        self._prk = self.phi_fold.derive_prk("trailblaze::init")
    def _seed(self, seed):
        if seed is None:
            base = (time.perf_counter_ns()^time.time_ns()^id(self)^int.from_bytes(os.urandom(8),"little"))&(2**64-1)
        else:
            base = seed & (2**64-1)
        self.seed_value = base
        self.slots = []
        for k in range(1, self.n_slots+1):
            val = math.modf(k*PHI)[0]; geo = SPIRAL8_TABLE[(k-1)%8]
            self.slots.append(Slot4096(value=val, phase=2*math.pi*val,
                freq=PHI**(geo["dim"]/8.0), amp_re=math.cos(2*math.pi*val),
                amp_im=math.sin(2*math.pi*val), dimension=geo["dim"], wave_mode=geo["wave_mode"]))
        for _ in range(50): self._step(0.1)
        sb = base.to_bytes(8,"little")
        for i in range(min(64,self.n_slots)):
            self.slots[i].value = math.modf(self.slots[i].value+sb[i%8]/256.0)[0]
        self._dn()
    def _step(self, dt=0.01):
        n = len(self.slots); re=sum(s.amp_re for s in self.slots); im=sum(s.amp_im for s in self.slots)
        R = math.sqrt(re**2+im**2)/n; cv = 1.0-R; self.phase_var = cv
        for s in self.slots:
            c = K_COUPLING*(im*s.amp_re - re*s.amp_im)/n
            ef = s.freq*(1+0.1*s.wave_mode)*(0.1+0.9*cv)*self.omega
            s.phase = (s.phase+(ef+c)*dt)%(2*math.pi)
            s.amp_re = math.cos(s.phase)*math.exp(-GAMMA*dt); s.amp_im = math.sin(s.phase)*math.exp(-GAMMA*dt)
            mag = math.sqrt(s.amp_re**2+s.amp_im**2)
            if mag>1e-9: s.amp_re/=mag; s.amp_im/=mag
        self._csteps = (self._csteps+1) if cv<CONSENSUS_EPS else 0
        if self._csteps>=CONSENSUS_N:
            for s in self.slots: s.flags |= APA_FLAG_CONSENSUS
        self.time += dt
    def _dn(self):
        FIB=[1,1,2,3,5,8,13,21]; PRM=[2,3,5,7,11,13,17,19]
        for s in self.slots:
            n=s.dimension; r=s.value; fn=FIB[n-1]; pn=PRM[n-1]
            om=0.5+0.5*math.sin(math.pi*math.modf(r*n)[0]*PHI)
            try: s.dn_amplitude=math.sqrt(PHI*fn*(2**n)*pn*om)*(r**((n+1)/8.0) if r>0 else 0)
            except: s.dn_amplitude=0.0
    def advance(self, steps=1):
        for _ in range(steps): self._step(0.01)
        rnd=int.from_bytes(os.urandom(2),"little"); ent=((time.perf_counter_ns()&0xFFFF)^rnd).to_bytes(4,"little")
        for i in range(min(32,self.n_slots)):
            self.slots[i].value=math.modf(self.slots[i].value+ent[i%4]/256.0)[0]
        self._dn(); self.epoch+=1; self.phi_fold._ep=-1
        self._prk=self.phi_fold.derive_prk(f"trailblaze::epoch::{self.epoch}")
    def phi_hash_address(self, *parts):
        return int.from_bytes(self.phi_fold.hash64(b"".join(parts))[:16],"little")
    def slot_for_key(self, key):
        h=self.phi_fold.hash32(key.encode()); raw=int.from_bytes(h[:4],"little")
        return int(math.modf(raw*PHI*PHI/(2**32))[0]*self.n_slots)%self.n_slots
    def dn_for_key(self, key): return self.slots[self.slot_for_key(key)].dn_amplitude
    def s_u_resonance(self):
        n=len(self.slots); re=sum(s.amp_re for s in self.slots); im=sum(s.amp_im for s in self.slots)
        M=math.sqrt(re**2+im**2)/n
        if M<1e-9: return (0.0,0.0,0.0)
        L=math.log(M*n)/LN_PHI-1.0/(2*PHI); fl=math.modf(L)[0]
        Om=(1+math.sin(math.pi*fl*PHI))/2
        S=math.sqrt((Om*math.cos(math.pi*L)+1)**2+(Om*math.sin(math.pi*L))**2)
        return (M,L,S)
    def describe(self):
        M,L,S=self.s_u_resonance()
        return {"epoch":self.epoch,"n_slots":self.n_slots,"time":round(self.time,4),
                "phase_var":round(self.phase_var,6),"consensus":bool(self.slots[0].flags&APA_FLAG_CONSENSUS),
                "M_U":round(M,4),"Lambda_U":round(L,4),"S_U":round(S,4),"seed":hex(self.seed_value)}

class Backend(IntEnum):
    CPU_SCALAR=0; CPU_NUMPY=1; ANALOG=2; METAL=3; CUDA=4; SPIRV=5; LLVM_JIT=6

class OpClass(IntEnum):
    MATMUL=0; ATTENTION=1; ELEMENTWISE=2; REDUCTION=3; NTT=4; LATTICE_UPDATE=5; KURAMOTO_STEP=6; PHI_FOLD=7

@dataclass
class BackendDesc:
    type: Backend; available: bool=True; vram_bytes: int=0
    compute_units: int=1; cost_coeffs: dict=field(default_factory=dict)

class BackendRegistry:
    def __init__(self, lattice):
        self.lattice=lattice; self._b: Dict[Backend,BackendDesc]={}
        self.register(BackendDesc(type=Backend.CPU_NUMPY,available=True,compute_units=os.cpu_count() or 1,
            cost_coeffs={OpClass.MATMUL:(1e-9,2.0,1e-4),OpClass.ATTENTION:(2e-9,2.0,1e-4),
                OpClass.ELEMENTWISE:(1e-10,1.0,1e-6),OpClass.REDUCTION:(5e-10,1.0,1e-6),OpClass.PHI_FOLD:(1e-7,1.0,1e-5)}))
        self.register(BackendDesc(type=Backend.ANALOG,available=True,
            cost_coeffs={OpClass.KURAMOTO_STEP:(1e-8,1.0,1e-5),OpClass.LATTICE_UPDATE:(1e-8,1.0,1e-5)}))
    def register(self, d): self._b[d.type]=d
    def select(self, op, n):
        best_b,best_c=Backend.CPU_NUMPY,float("inf")
        for bt,desc in self._b.items():
            if not desc.available or op not in desc.cost_coeffs: continue
            a,k,b=desc.cost_coeffs[op]; c=a*(n**k)+b
            c*=1.0/(1.0+self.lattice.dn_for_key(f"{bt}:{op}")*0.1)
            if c<best_c: best_c,best_b=c,bt
        return best_b
    def describe(self): return {b.name:{"available":d.available,"units":d.compute_units} for b,d in self._b.items()}
