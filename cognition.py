"""TRAILBLAZE — Layer 3: Persistent Cognition Substrate
CognitionCell (u128 phi-addressed, hash-chained), CognitionTree (branch-aware),
ERL v3 ledger (ported from MCP server.js), 4-tier memory, epoch ratchet.
"""
import sys,os,json,time,sqlite3,hashlib
from typing import Optional,List,Dict,Any,Tuple
from dataclasses import dataclass,field
from enum import Enum
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer0.phi_lattice import PhiLattice

class ERLType(str,Enum):
    CELL_COMMIT="CELL_COMMIT"; BRANCH_CREATE="BRANCH_CREATE"; BRANCH_MERGE="BRANCH_MERGE"
    EPOCH_ADVANCE="EPOCH_ADVANCE"; TASK_START="TASK_START"; TASK_COMPLETE="TASK_COMPLETE"
    STATE_SEAL="STATE_SEAL"; AGENT_DELEGATE="AGENT_DELEGATE"; TOOL_CALL="TOOL_CALL"; ERROR="ERROR"
ERLEntryType=ERLType  # alias

@dataclass
class ERLEntry:
    seq:int; type:ERLType; branch_id:int; epoch:int; timestamp:float
    data:dict; parent_hash:str; entry_hash:str=""
    def __post_init__(self):
        if not self.entry_hash: self.entry_hash=self._hash()
    def _hash(self):
        p=json.dumps({"seq":self.seq,"type":self.type,"branch_id":self.branch_id,
            "epoch":self.epoch,"timestamp":self.timestamp,"data":self.data,
            "parent_hash":self.parent_hash},sort_keys=True)
        return hashlib.sha256(p.encode()).hexdigest()
    def verify(self): return self.entry_hash==self._hash()
    def to_dict(self):
        return {"seq":self.seq,"type":self.type,"branch_id":self.branch_id,
            "epoch":self.epoch,"timestamp":self.timestamp,"data":self.data,
            "parent_hash":self.parent_hash,"entry_hash":self.entry_hash}

class ERLLedger:
    GENESIS="0"*64
    def __init__(self, persist_path=None, db_path=None):
        self._entries=[]; self._by_branch={}; self._pp=persist_path; self._db=None
        if db_path: self._init_db(db_path)
        if persist_path and Path(persist_path).exists(): self._load(persist_path)
    def _init_db(self,path):
        self._db=sqlite3.connect(path,check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS erl_entries(seq INTEGER PRIMARY KEY,type TEXT,branch_id INTEGER,epoch INTEGER,timestamp REAL,data TEXT,parent_hash TEXT,entry_hash TEXT)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_b ON erl_entries(branch_id)")
        self._db.commit()
    def append(self,etype,branch_id,epoch,data):
        ph=self._entries[-1].entry_hash if self._entries else self.GENESIS
        e=ERLEntry(seq=len(self._entries),type=etype,branch_id=branch_id,epoch=epoch,
                   timestamp=time.time(),data=data,parent_hash=ph)
        self._entries.append(e); self._by_branch.setdefault(branch_id,[]).append(e)
        if self._db:
            self._db.execute("INSERT INTO erl_entries VALUES(?,?,?,?,?,?,?,?)",
                (e.seq,e.type,e.branch_id,e.epoch,e.timestamp,json.dumps(e.data),e.parent_hash,e.entry_hash))
            self._db.commit()
        if self._pp: self._persist()
        return e
    def verify_chain(self):
        ph=self.GENESIS
        for e in self._entries:
            if not e.verify() or e.parent_hash!=ph: return False,e.seq
            ph=e.entry_hash
        return True,None
    def verify_branch(self,bid):
        for e in self._by_branch.get(bid,[]): 
            if not e.verify(): return False,e.seq
        return True,len(self._by_branch.get(bid,[]))
    def latest(self): return self._entries[-1] if self._entries else None
    def head_hash(self): return self._entries[-1].entry_hash if self._entries else self.GENESIS
    def branch_history(self,bid): return list(self._by_branch.get(bid,[]))
    def entries_since(self,seq): return [e for e in self._entries if e.seq>=seq]
    def _persist(self):
        with open(self._pp,"w") as f: json.dump({"entries":[e.to_dict() for e in self._entries]},f,indent=2)
    def _load(self,path):
        with open(path) as f: data=json.load(f)
        for d in data.get("entries",[]):
            e=ERLEntry(seq=d["seq"],type=ERLType(d["type"]),branch_id=d["branch_id"],
                epoch=d["epoch"],timestamp=d["timestamp"],data=d["data"],
                parent_hash=d["parent_hash"],entry_hash=d["entry_hash"])
            self._entries.append(e); self._by_branch.setdefault(e.branch_id,[]).append(e)
    def summary(self):
        return {"total_entries":len(self._entries),"branches":list(self._by_branch.keys()),
                "head_hash":self.head_hash()[:16]+"...","chain_valid":self.verify_chain()[0]}

@dataclass
class CognitionCell:
    id:int; branch_id:int; parent_id:int; epoch:int; domain:str; value:Any
    lambda_k:float=0.0; sigma:float=0.0; dn_amplitude:float=0.0; flags:int=0
    timestamp:float=field(default_factory=time.time); audit_hash:str=""
    def __post_init__(self):
        if not self.audit_hash: self.audit_hash=self._ah()
    def _ah(self):
        vb=json.dumps(self.value,default=str).encode() if not isinstance(self.value,bytes) else self.value
        return hashlib.sha256(self.parent_id.to_bytes(16,"little")+vb+self.epoch.to_bytes(8,"little")+self.domain.encode()).hexdigest()
    def verify(self): return self.audit_hash==self._ah()
    def to_dict(self):
        v=self.value
        if isinstance(v,bytes): v=v.hex()
        elif not isinstance(v,(str,int,float,dict,list,type(None))): v=repr(v)
        return {"id":hex(self.id),"branch_id":self.branch_id,"parent_id":hex(self.parent_id),
            "epoch":self.epoch,"domain":self.domain,"value":v,"dn_amplitude":self.dn_amplitude,
            "timestamp":self.timestamp,"audit_hash":self.audit_hash}

@dataclass
class BranchHead:
    id:int; tip:int; fork_point:int; parent_branch_id:Optional[int]; epoch_created:int
    kv_cache:Any=None; flow_state:dict=field(default_factory=dict)
    merged:bool=False; merged_into:Optional[int]=None

class CognitionTree:
    MAIN=0
    def __init__(self, lattice, name="default", persist_dir=None):
        self.lattice=lattice; self.name=name
        self._cells:Dict[int,CognitionCell]={}; self._branches:Dict[int,BranchHead]={}
        self._next_bid=1; self.session_memory:Dict[str,Any]={}; self._pd=persist_dir
        ep_path=os.path.join(persist_dir,"erl-ledger.json") if persist_dir else None
        db_path=os.path.join(persist_dir,"cognition.db") if persist_dir else None
        if persist_dir: os.makedirs(persist_dir,exist_ok=True)
        self.ledger=ERLLedger(persist_path=ep_path,db_path=db_path)
        root=self._genesis()
        self._branches[self.MAIN]=BranchHead(id=self.MAIN,tip=root.id,fork_point=root.id,
            parent_branch_id=None,epoch_created=lattice.epoch)
        self.ledger.append(ERLType.BRANCH_CREATE,self.MAIN,lattice.epoch,{"name":"main","root":hex(root.id)})
    def _genesis(self):
        lat=self.lattice
        cid=lat.phi_hash_address(b"TRAILBLAZE::GENESIS",self.name.encode())
        c=CognitionCell(id=cid,branch_id=self.MAIN,parent_id=0,epoch=lat.epoch,
            domain="genesis",value={"trailblaze":True,"name":self.name},dn_amplitude=lat.dn_for_key("genesis"))
        self._cells[cid]=c; return c
    def epoch_advance(self,steps=1):
        old=self.lattice.epoch; self.lattice.advance(steps); new=self.lattice.epoch
        for b in self._branches.values():
            if b.kv_cache is not None: b.kv_cache.invalidate(new)
        self.ledger.append(ERLType.EPOCH_ADVANCE,self.MAIN,new,{"old":old,"new":new,"branches":len(self._branches)})
        return new
    def cell_commit(self,branch_id,value,domain="data"):
        if branch_id not in self._branches: raise KeyError(f"Unknown branch {branch_id}")
        b=self._branches[branch_id]; lat=self.lattice
        vb=value if isinstance(value,bytes) else json.dumps(value,default=str).encode()
        cid=lat.phi_hash_address(b.tip.to_bytes(16,"little"),vb[:64],branch_id.to_bytes(8,"little"),domain.encode())
        c=CognitionCell(id=cid,branch_id=branch_id,parent_id=b.tip,epoch=lat.epoch,
            domain=domain,value=value,dn_amplitude=lat.dn_for_key(f"{branch_id}:{domain}"))
        self._cells[cid]=c; b.tip=cid
        self.ledger.append(ERLType.CELL_COMMIT,branch_id,lat.epoch,{"cell_id":hex(cid),"domain":domain,"vtype":type(value).__name__})
        return c
    def cell_get(self,cid): return self._cells.get(cid)
    def cell_verify(self,cid): c=self._cells.get(cid); return c.verify() if c else False
    def branch_history(self,bid,max_depth=50):
        b=self._branches.get(bid)
        if not b: return []
        cells=[]; cur=b.tip
        for _ in range(max_depth):
            c=self._cells.get(cur)
            if not c or c.parent_id==0:
                if c: cells.append(c)
                break
            cells.append(c); cur=c.parent_id
        return cells
    def branch_create(self,from_bid=0,kv_cache=None):
        src=self._branches.get(from_bid)
        if not src: raise KeyError(f"Unknown branch {from_bid}")
        nid=self._next_bid; self._next_bid+=1
        nkv=None
        if kv_cache is not None: nkv=kv_cache.fork(nid)
        elif src.kv_cache is not None: nkv=src.kv_cache.fork(nid)
        self._branches[nid]=BranchHead(id=nid,tip=src.tip,fork_point=src.tip,
            parent_branch_id=from_bid,epoch_created=self.lattice.epoch,
            kv_cache=nkv,flow_state=dict(src.flow_state))
        self.ledger.append(ERLType.BRANCH_CREATE,nid,self.lattice.epoch,{"from":from_bid,"fork":hex(src.tip)})
        return nid
    def branch_merge(self,src_id,dst_id):
        src=self._branches.get(src_id); dst=self._branches.get(dst_id)
        if not src or not dst or src.merged: return False
        hist=self.branch_history(src_id); fp=src.fork_point
        new_cells=[]
        for c in hist:
            if c.id==fp: break
            new_cells.append(c)
        for c in reversed(new_cells):
            vb=json.dumps(c.value,default=str).encode()
            nid=self.lattice.phi_hash_address(dst.tip.to_bytes(16,"little"),vb[:64],dst_id.to_bytes(8,"little"),c.domain.encode())
            mc=CognitionCell(id=nid,branch_id=dst_id,parent_id=dst.tip,epoch=self.lattice.epoch,
                domain=c.domain,value=c.value,dn_amplitude=c.dn_amplitude)
            self._cells[nid]=mc; dst.tip=nid
        if src.kv_cache and dst.kv_cache: dst.kv_cache=src.kv_cache.reconcile(dst.kv_cache,self.lattice)
        elif src.kv_cache: dst.kv_cache=src.kv_cache
        src.merged=True; src.merged_into=dst_id
        self.ledger.append(ERLType.BRANCH_MERGE,dst_id,self.lattice.epoch,
            {"src":src_id,"dst":dst_id,"cells_merged":len(new_cells)})
        return True
    def branch_tip(self,bid): b=self._branches.get(bid); return b.tip if b else None
    def seal(self,data,domain="state"): return self.lattice.phi_stream.seal(data,ctx=domain)
    def unseal(self,env,domain="state"): return self.lattice.phi_stream.unseal(env,ctx=domain)
    def memory_set(self,k,v): self.session_memory[k]=v
    def memory_get(self,k): return self.session_memory.get(k)
    def notes_write(self,fn,content):
        if not self._pd: return
        nd=os.path.join(self._pd,"notes"); os.makedirs(nd,exist_ok=True)
        open(os.path.join(nd,fn),"w").write(content)
    def notes_read(self,fn):
        if not self._pd: return None
        p=os.path.join(self._pd,"notes",fn)
        return open(p).read() if os.path.exists(p) else None
    def db_exec(self,sql,params=()):
        if not self.ledger._db: return None
        try:
            c=self.ledger._db.execute(sql,params); self.ledger._db.commit(); return c.fetchall()
        except sqlite3.Error as e: return {"error":str(e)}
    def flow_set(self,bid,k,v):
        b=self._branches.get(bid)
        if b: b.flow_state[k]=v
    def flow_get(self,bid,k):
        b=self._branches.get(bid); return b.flow_state.get(k) if b else None
    def record_tool_call(self,bid,tool,args,result):
        self.ledger.append(ERLType.TOOL_CALL,bid,self.lattice.epoch,
            {"tool":tool,"args":args,"rtype":type(result).__name__})
    def describe(self):
        br={}
        for bid,b in self._branches.items():
            br[bid]={"tip":hex(b.tip),"merged":b.merged,"epoch":b.epoch_created,
                     "kv_seq":b.kv_cache.seq_len if b.kv_cache else 0}
        return {"name":self.name,"epoch":self.lattice.epoch,"n_cells":len(self._cells),
                "n_branches":len(self._branches),"branches":br,"ledger":self.ledger.summary(),
                "session_keys":list(self.session_memory.keys())}
