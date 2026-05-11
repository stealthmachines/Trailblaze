"""TRAILBLAZE — Layer 4: Agent Orchestration Fabric
ToolRegistry (23 tools), UnfoldEngine (unfold()→TB_Graph→execute),
FlowStateManager, AgentFabric (recursive delegation), MCPServer (HTTP+SSE),
TrailblazeRuntime (unified entry point). Ported from MCP-Jailbreak-0.4.
"""
import sys,os,json,time,subprocess,threading
from typing import Optional,List,Dict,Any,Callable
from dataclasses import dataclass,field
from pathlib import Path
import http.server
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer0.phi_lattice import PhiLattice,BackendRegistry
from layer2.graph_engine import Graph,Node,PassType,TaskCompiler,GraphExecutor,HDGLRouter,KuramotoScheduler,FusionEngine
from layer3.cognition import CognitionTree,ERLType

@dataclass
class ToolResult:
    success:bool; data:Any; error:Optional[str]=None
    def to_dict(self): return {k:v for k,v in self.__dict__.items() if v is not None}

class Tool:
    def __init__(self,name,desc,fn,schema=None):
        self.name=name; self.description=desc; self.fn=fn; self.schema=schema or {}
    def __call__(self,**kw):
        try: return ToolResult(success=True,data=self.fn(**kw))
        except Exception as e: return ToolResult(success=False,data=None,error=str(e))

class ToolRegistry:
    def __init__(self,tree,router):
        self.tree=tree; self.router=router; self._tools:Dict[str,Tool]={}; self._reg()
    def register(self,t): self._tools[t.name]=t
    def get(self,n): return self._tools.get(n)
    def call(self,name,branch_id=0,**kw):
        t=self._tools.get(name)
        if not t: return ToolResult(success=False,data=None,error=f"Unknown tool: {name}")
        r=t(**kw); self.tree.record_tool_call(branch_id,name,kw,r.data); return r
    def list_tools(self): return [{"name":t.name,"description":t.description} for t in self._tools.values()]
    def _reg(self):
        tree=self.tree
        S=lambda fn,d,**sc: self.register(Tool(fn.__name__ if callable(fn) else fn,d,fn,sc))
        self.register(Tool("shell_exec","Execute shell command",lambda command,timeout=30,**_:self._sh(command,timeout)))
        self.register(Tool("shell_bg","Start background process",lambda command,**_:self._bg(command)))
        self.register(Tool("read_file","Read file",lambda path,**_:Path(path).read_text(errors="replace")))
        self.register(Tool("write_file","Write file",lambda path,content,**_:(Path(path).parent.mkdir(parents=True,exist_ok=True),Path(path).write_text(content),f"wrote {len(content)}B")[-1]))
        self.register(Tool("list_dir","List directory",lambda path=".",**_:[{"name":e.name,"type":"dir" if e.is_dir() else "file","size":e.stat().st_size if e.is_file() else 0} for e in sorted(Path(path).iterdir())]))
        self.register(Tool("file_exists","Check path exists",lambda path,**_:Path(path).exists()))
        self.register(Tool("make_dir","Create directory",lambda path,**_:(Path(path).mkdir(parents=True,exist_ok=True),str(path))[-1]))
        self.register(Tool("delete_file","Delete file",lambda path,**_:(Path(path).unlink(missing_ok=True),f"deleted {path}")[-1]))
        self.register(Tool("memory_set","Session memory set",lambda key,value,**_:(tree.memory_set(key,value),f"set {key}")[-1]))
        self.register(Tool("memory_get","Session memory get",lambda key,**_:tree.memory_get(key)))
        self.register(Tool("memory_list","List memory keys",lambda **_:list(tree.session_memory.keys())))
        self.register(Tool("notes_write","Write durable note",lambda filename,content,**_:(tree.notes_write(filename,content),f"wrote {filename}")[-1]))
        self.register(Tool("notes_read","Read durable note",lambda filename,**_:tree.notes_read(filename)))
        self.register(Tool("db_query","SQL query",lambda sql,**_:tree.db_exec(sql)))
        self.register(Tool("branch_create","Fork branch",lambda from_branch=0,**_:tree.branch_create(from_branch)))
        self.register(Tool("branch_merge","Merge branch",lambda src,dst=0,**_:tree.branch_merge(src,dst)))
        self.register(Tool("cell_commit","Commit to cognition",lambda value,domain="data",branch=0,**_:tree.cell_commit(branch,value,domain).id))
        self.register(Tool("epoch_advance","Ratchet epoch",lambda steps=1,**_:tree.epoch_advance(steps)))
        self.register(Tool("state_describe","Runtime state",lambda **_:tree.describe()))
        self.register(Tool("ledger_verify","Verify ERL chain",lambda **_:tree.ledger.verify_chain()))
        self.register(Tool("ledger_summary","ERL summary",lambda **_:tree.ledger.summary()))
        self.register(Tool("lattice_describe","Lattice state",lambda **_:tree.lattice.describe()))
        self.register(Tool("lattice_advance","Advance lattice",lambda steps=10,**_:(tree.lattice.advance(steps),tree.lattice.describe())[-1]))
    @staticmethod
    def _sh(cmd,timeout=30):
        try:
            p=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout)
            return {"stdout":p.stdout,"stderr":p.stderr,"exit_code":p.returncode}
        except subprocess.TimeoutExpired: return {"error":f"timeout {timeout}s","exit_code":-1}
        except Exception as e: return {"error":str(e),"exit_code":-1}
    @staticmethod
    def _bg(cmd):
        try: p=subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); return {"pid":p.pid}
        except Exception as e: return {"error":str(e)}

class FlowStateManager:
    def __init__(self,tree): self.tree=tree
    def get(self,bid): return {"cwd":self.tree.flow_get(bid,"cwd") or os.getcwd(),"env":self.tree.flow_get(bid,"env") or {},"last_stdout":self.tree.flow_get(bid,"last_stdout"),"active_branch":bid,"task_queue":self.tree.flow_get(bid,"task_queue") or []}
    def update(self,bid,**kw):
        for k,v in kw.items(): self.tree.flow_set(bid,k,v)

@dataclass
class UnfoldResult:
    success:bool; task:str; branch_id:int; pass_sequence:List[str]
    graph_nodes:int; results:Dict[str,Any]; error:Optional[str]=None
    exec_ms:float=0.0; erl_entries:int=0

class UnfoldEngine:
    def __init__(self,tree,tools,lattice):
        self.tree=tree; self.tools=tools; self.lattice=lattice
        self.compiler=TaskCompiler(lattice)
        from layer1.tensor_runtime import TensorOps
        self.ops=TensorOps(lattice,BackendRegistry(lattice))
        self.executor=GraphExecutor(self.ops,lattice)
        self.flow=FlowStateManager(tree)
    def unfold(self,task,branch_id=0,context=None):
        t0=time.perf_counter(); n0=len(self.tree.ledger._entries)
        self.tree.ledger.append(ERLType.TASK_START,branch_id,self.lattice.epoch,{"task":task[:200]})
        try:
            g=self.compiler.compile(task)
            self._bind(g,branch_id)
            reg=BackendRegistry(self.lattice); router=HDGLRouter(self.lattice,reg)
            FusionEngine.fuse(g)
            plan=KuramotoScheduler(self.lattice,router).schedule(g)
            store=dict(context or {}); store["_branch_id"]=branch_id; store["_task"]=task
            results=self.executor.execute(g,plan,context=store)
            rdata={k:v for k,v in results.items() if not k.startswith("_") and isinstance(v,(str,int,float,dict,list))}
            self.tree.cell_commit(branch_id,rdata,"task_result")
            self.flow.update(branch_id,last_task=task,last_result=rdata)
            self.tree.ledger.append(ERLType.TASK_COMPLETE,branch_id,self.lattice.epoch,{"task":task[:200],"success":True,"nodes":len(plan)})
            return UnfoldResult(success=True,task=task,branch_id=branch_id,
                pass_sequence=[n.pass_type.name for n in plan],graph_nodes=len(plan),
                results=rdata,exec_ms=(time.perf_counter()-t0)*1000,
                erl_entries=len(self.tree.ledger._entries)-n0)
        except Exception as e:
            self.tree.ledger.append(ERLType.ERROR,branch_id,self.lattice.epoch,{"task":task[:200],"error":str(e)})
            return UnfoldResult(success=False,task=task,branch_id=branch_id,
                pass_sequence=[],graph_nodes=0,results={},error=str(e),
                exec_ms=(time.perf_counter()-t0)*1000)
    def _bind(self,g,bid):
        tree=self.tools.tree
        for n in g.nodes:
            if n.op_fn: continue
            pt=n.pass_type
            if pt==PassType.SHELL:
                def mks(nd):
                    def fn(inp,store,**kw):
                        r=self.tools.call("shell_exec",bid,command=store.get("_task","echo ok"))
                        store["shell_result"]=r.to_dict(); return r.to_dict()
                    return fn
                n.op_fn=mks(n)
            elif pt==PassType.RECALL:
                def mkr(nd):
                    def fn(inp,store,**kw):
                        v=tree.memory_get(store.get("_task","")[:50].replace(" ","_")); store["recalled"]=v; return v
                    return fn
                n.op_fn=mkr(n)
            elif pt==PassType.STORE:
                def mkst(nd):
                    def fn(inp,store,**kw):
                        k=store.get("_task","")[:50].replace(" ","_"); v=store.get("transform_result") or store.get("recalled")
                        tree.memory_set(k,v); tree.notes_write(f"task_{int(time.time())}.md",f"# {store.get('_task','')}\n{json.dumps(v,default=str)}\n"); return {"stored":k}
                    return fn
                n.op_fn=mkst(n)
            elif pt==PassType.TRANSFORM:
                def mkt(nd):
                    def fn(inp,store,**kw):
                        r={k:v for k,v in store.items() if not k.startswith("_") and isinstance(v,(str,int,float,dict,list))}
                        store["transform_result"]=r; return r
                    return fn
                n.op_fn=mkt(n)
            elif pt in (PassType.RESPOND,PassType.FETCH,PassType.CODE,PassType.BROWSE):
                def mkfb(ptype):
                    def fn(inp,store,**kw):
                        r={k:v for k,v in store.items() if not k.startswith("_") and isinstance(v,(str,int,float,dict,list))}
                        r["pass"]=ptype.name; store["response"]=r; return r
                    return fn
                n.op_fn=mkfb(pt)

@dataclass
class AgentHandle:
    task:str; branch_id:int; result:Optional[UnfoldResult]=None; done:bool=False

class AgentFabric:
    def __init__(self,engine,tree): self.engine=engine; self.tree=tree
    def delegate(self,tasks,from_branch=0):
        handles=[]
        for task in tasks:
            bid=self.tree.branch_create(from_branch)
            self.tree.ledger.append(ERLType.AGENT_DELEGATE,bid,self.tree.lattice.epoch,{"task":task[:200],"from":from_branch})
            r=self.engine.unfold(task,branch_id=bid)
            handles.append(AgentHandle(task=task,branch_id=bid,result=r,done=True))
        for h in handles:
            if h.done and h.result and h.result.success: self.tree.branch_merge(h.branch_id,from_branch)
        return [h.result for h in handles]

class MCPServer:
    def __init__(self,engine,tools,tree,host="0.0.0.0",port=3333):
        self.engine=engine; self.tools=tools; self.tree=tree
        self.host=host; self.port=port; self._srv=None; self._t0=time.time()
    def _make_handler(self):
        engine=self.engine; tools=self.tools; tree=self.tree; srv=self
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def json(self,data,code=200):
                b=json.dumps(data,default=str,indent=2).encode()
                self.send_response(code); self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",len(b)); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(b)
            def do_OPTIONS(self):
                self.send_response(200); self.send_header("Access-Control-Allow-Origin","*")
                self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS"); self.send_header("Access-Control-Allow-Headers","Content-Type"); self.end_headers()
            def do_GET(self):
                p=self.path.split("?")[0]
                if p=="/health": self.json({"status":"ok","version":"trailblaze-0.1","epoch":tree.lattice.epoch,"uptime_s":round(time.time()-srv._t0,1),"n_cells":len(tree._cells),"n_branches":len(tree._branches),"lattice":tree.lattice.describe()})
                elif p=="/tools": self.json({"tools":tools.list_tools()})
                elif p=="/state": self.json(tree.describe())
                elif p=="/ledger": self.json({"summary":tree.ledger.summary(),"latest":tree.ledger.latest().to_dict() if tree.ledger.latest() else None})
                elif p=="/sse":
                    self.send_response(200); self.send_header("Content-Type","text/event-stream")
                    self.send_header("Cache-Control","no-cache"); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
                    for _ in range(5):
                        try:
                            self.wfile.write(f"data: {json.dumps({'type':'heartbeat','epoch':tree.lattice.epoch,'n_cells':len(tree._cells)})}\n\n".encode()); self.wfile.flush(); time.sleep(1)
                        except: break
                else: self.json({"error":"not found"},404)
            def do_POST(self):
                p=self.path; n=int(self.headers.get("Content-Length",0))
                body=self.rfile.read(n) if n else b"{}"
                try: pl=json.loads(body) if body else {}
                except: pl={}
                if p=="/unfold":
                    task=pl.get("task","")
                    if not task: self.json({"error":"task required"},400); return
                    r=engine.unfold(task,branch_id=pl.get("branch_id",0),context=pl.get("context",{}))
                    self.json(r.__dict__)
                elif p.startswith("/tool/"):
                    nm=p[6:].lstrip("/"); bid=pl.pop("branch_id",0)
                    self.json(tools.call(nm,branch_id=bid,**pl).to_dict())
                elif p=="/branch/create": self.json({"branch_id":tree.branch_create(pl.get("from_branch",0))})
                elif p=="/branch/merge": self.json({"merged":tree.branch_merge(pl.get("src"),pl.get("dst",0))})
                elif p=="/epoch/advance": self.json({"epoch":tree.epoch_advance(pl.get("steps",1))})
                else: self.json({"error":"not found"},404)
        return H
    def start(self,blocking=True):
        self._srv=http.server.HTTPServer((self.host,self.port),self._make_handler())
        print(f"[TRAILBLAZE] MCP server http://{self.host}:{self.port}  /health /tools /unfold /state /ledger /sse")
        if blocking: self._srv.serve_forever()
        else:
            t=threading.Thread(target=self._srv.serve_forever,daemon=True); t.start(); return t
    def stop(self):
        if self._srv: self._srv.shutdown()

class TrailblazeRuntime:
    def __init__(self,persist_dir=None,n_lattice_slots=512,seed=None):
        self.lattice=PhiLattice(n_slots=n_lattice_slots,seed=seed)
        self.tree=CognitionTree(self.lattice,name="trailblaze",persist_dir=persist_dir)
        self.registry=BackendRegistry(self.lattice)
        self.router=HDGLRouter(self.lattice,self.registry)
        from layer1.tensor_runtime import TensorOps
        self.ops=TensorOps(self.lattice,self.registry)
        self.tools=ToolRegistry(self.tree,self.router)
        self.engine=UnfoldEngine(self.tree,self.tools,self.lattice)
        self.fabric=AgentFabric(self.engine,self.tree)
        self.server=None
    def unfold(self,task,branch_id=0,**ctx): return self.engine.unfold(task,branch_id=branch_id,context=ctx)
    def delegate(self,tasks,from_branch=0): return self.fabric.delegate(tasks,from_branch=from_branch)
    def serve(self,host="0.0.0.0",port=3333,blocking=True):
        self.server=MCPServer(self.engine,self.tools,self.tree,host,port); return self.server.start(blocking=blocking)
    def describe(self):
        return {"runtime":"TRAILBLAZE","version":"0.1","lattice":self.lattice.describe(),
                "cognition":self.tree.describe(),"tools":len(self.tools._tools),"backends":self.registry.describe()}
