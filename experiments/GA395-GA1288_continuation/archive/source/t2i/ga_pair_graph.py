"""GA600+: jointly model candidate pairs and the atoms they share."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .ga_replica_teacher import trajectory_rows
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT = Path("runs/ga600-graph")


def indexed_graphs(x,d,g,y,origins,names,mu,sd):
    graphs = []
    for n in np.unique(names):
        pair,t = trajectory_rows(n,1)
        for o in np.unique(origins[names==n]):
            idx=np.flatnonzero((names==n)&(origins==o))
            r=pair[t==o]
            np.testing.assert_array_equal(r[:,2],x[idx,0])
            _, endpoints = np.unique(r[:,:2].astype(np.int64),return_inverse=True)
            graphs.append((torch.from_numpy(((features(x[idx],d[idx],g[idx],"level")-mu)/sd).clip(-10,10).astype(np.float32)),
                           torch.from_numpy(endpoints.reshape(-1,2)),torch.from_numpy((y[idx]-x[idx,0]).astype(np.float32)),
                           idx,int(o),int(endpoints.max())+1))
    return graphs


def pack(graphs, device):
    offsets=np.cumsum([0]+[g[5] for g in graphs[:-1]])
    edges=torch.cat([g[1]+n for n,g in zip(offsets,graphs)]).to(device)
    x=torch.cat([g[0] for g in graphs]).to(device)
    y=torch.cat([g[2] for g in graphs]).to(device)
    size=sum(g[5] for g in graphs)
    degree=torch.bincount(edges.ravel(),minlength=size).clamp_min(1).reshape(-1,1)
    return x,edges,degree,y,size


class PairGraph(nn.Module):
    def __init__(self,n,width,rounds,weighted=False):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(n,width),nn.LayerNorm(width),nn.SiLU())
        self.updates=nn.ModuleList(nn.Sequential(nn.Linear(3*width,width),nn.LayerNorm(width),nn.SiLU()) for _ in range(rounds))
        self.route=nn.Linear(width,2) if weighted else None
        self.output=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,x,edges,degree,size):
        q=self.edge(x)
        a,b=edges.T
        for update in self.updates:
            nodes=q.new_zeros((size,q.shape[1]))
            if self.route is None:
                nodes.index_add_(0,a,q)
                nodes.index_add_(0,b,q)
                nodes=nodes/degree
            else:
                left,right=self.route(q).sigmoid().unbind(1)
                nodes.index_add_(0,a,q*left[:,None])
                nodes.index_add_(0,b,q*right[:,None])
                mass=q.new_zeros((size,1))
                mass.index_add_(0,a,left[:,None])
                mass.index_add_(0,b,right[:,None])
                nodes=nodes/mass.clamp_min(.1)
            q=q+update(torch.cat([q,nodes[a],nodes[b]],1))
        return self.output(q).flatten()


def main(capacity=False):
    torch.set_num_threads(3)
    output=Path("runs/ga1133-capacity-graph") if capacity else OUT
    output.mkdir(exist_ok=True)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    if capacity:
        from .ga_competitive_capacity import competition
        x=competition(x,d,o,p,oriented=True)
        v=competition(v,e,vo,vg,oriented=True)
    raw=features(x,d,g,"level")
    mu,sd=raw.mean(0),raw.std(0).clip(.001)
    training=indexed_graphs(x,d,g,y,o,p,mu,sd)
    valid=indexed_graphs(v,e,h,t,vo,vg,mu,sd)
    vb,ve,vd,_,vn=pack(valid,"cpu")
    ix=np.concatenate([z[3] for z in valid])
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    specs=[(r,w,b) for r in (0,1,2,3) for w in (32,72) for b in (1.,4.)]
    if capacity:
        specs=[(rounds,width,4.,weight) for weight in (False,True)
               for rounds in (1,2,3) for width in (48,96)]
    history=[]
    for i in range(len(specs)):
        done={tuple(r["spec"]) for r in history}
        if not history:
            spec=specs[0]
        else:
            best=max(history,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in specs if s not in done),
                     key=lambda s:(sum(a!=b for a,b in zip(s,best["spec"])),
                                   sum(r["spec"][0]==s[0] for r in history)))
        rounds,width,late,*weighted=spec
        torch.manual_seed(600)
        net=PairGraph(raw.shape[1],width,rounds,weighted=bool(weighted and weighted[0])).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.012)
        rng=np.random.default_rng(600)
        start=perf_counter()
        for _ in range(16):
            for batch in np.array_split(rng.permutation(len(training)),32):
                group=[training[j] for j in batch]
                a,edges,degree,target,nnodes=pack(group,"cuda")
                weight=torch.tensor(np.concatenate([np.full(len(z[2]),1. if z[4]==49 else late,dtype=np.float32) for z in group]),device="cuda")
                pred=net(a,edges,degree,nnodes)
                loss=((pred-target).square()*weight).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        seconds=perf_counter()-start
        cpu=deepcopy(net).cpu().eval()
        direct=np.empty_like(t)
        with torch.inference_mode():
            direct[ix]=cpu(vb,ve,vd,vn).numpy()+v[ix,0]
        pred=.65*parent+.35*direct
        scores={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        model_id=f"GA{(1133 if capacity else 600)+i}"
        row=dict(id=model_id,spec=spec,metrics=scores,fit_seconds=seconds)
        history.append(row)
        torch.save(dict(state=cpu.state_dict(),spec=spec,center=mu,scale=sd),output/f"{model_id}.pt")
        np.savez_compressed(output/f"{model_id}.npz",prediction=pred,readout=direct)
        with (output/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(model_id,spec,scores["all"]["residual_correlation"],flush=True)
        if scores["all"]["residual_correlation"]>=.700:
            break
    best=max(history,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (output/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    import sys
    main(capacity="--capacity" in sys.argv)

