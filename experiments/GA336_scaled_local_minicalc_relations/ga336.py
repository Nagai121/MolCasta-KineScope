"""GA336 scales the MiniCalc local-relation representation."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from tseries.data import registry, trajectory, shard, cell
from tseries.features import _mic
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga_local_relations import minicalc
from .ga_force import build
from .ga309 import predict
from .ga_static_offset import VALID, cohort, estimates, extract


DEST=Path("runs/ga336")
K=16


def local(pos,box,pair,atoms,mc):
    endpoints=pair[:,:2].astype(np.intp)
    axes=pair[:,3:6]/pair[:,2,None]
    res=np.asarray(atoms["residue_index"])
    names=np.asarray(atoms["name"])
    elem=np.asarray(atoms["element"])
    cas={r:i for i,(r,n) in enumerate(zip(res,names)) if n=="CA"}
    lookup=np.array([cas.get(r,i) for i,r in enumerate(res)])
    ref=_mic(pos[lookup[endpoints]]-pos[endpoints],box)
    projected=ref-np.sum(axes[:,None]*ref,axis=2,keepdims=True)*axes[:,None]
    short=np.linalg.norm(projected,axis=-1)
    helper=np.eye(3)[np.argmin(np.abs(axes),axis=1)]
    helper-=np.sum(helper*axes,axis=1,keepdims=True)*axes
    second=np.where((short[:,0]>.02)[:,None],projected[:,0],
                    np.where((short[:,1]>.02)[:,None],projected[:,1],helper))
    second/=np.linalg.norm(second,axis=1,keepdims=True)
    basis=np.stack((axes,second,np.cross(axes,second)),axis=2)
    ids=cKDTree(pos).query(pos[endpoints],k=K+1)[1][:,:,1:]
    vec=_mic(pos[ids]-pos[endpoints][:,:,None],box)
    reference=mc.FrameState(np.zeros((vec.size//3,3)),np.repeat(basis,2*K,axis=0),())
    loc=mc.to_local_displacement(reference,vec.reshape(-1,3)).reshape(vec.shape)
    radius=np.linalg.norm(vec,axis=-1)
    scalar=np.stack([radius/6., *(elem[ids]==c for c in ("C","N","O","S")),
                    res[ids]==res[endpoints][:,:,None]],axis=-1)
    return np.concatenate((loc/6.,scalar),axis=3).astype(np.float32)


def pack(task,mc):
    name,r,limit=task
    x,y=build(task)
    cache=DEST/f"{name}-{r}-{limit}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return x,y,z["graph"]
    with np.load(shard(ROOT,name,r,100,8.)) as z:
        mask=z["pair"][:,2]<=6.
        raw=z["pair"][mask][:len(y)]
        times=z["origin"][mask][:len(y)]
    np.testing.assert_array_equal(x[:,0],raw[:,2])
    reg=registry(ROOT,name)
    tid=trajectory(name,r)
    atoms=reg.atoms(tid).to_pydict()
    bounds=np.r_[0,np.flatnonzero(np.diff(times))+1,len(times)]
    features=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        t=int(times[a])
        frame=next(reg.blocks(tid,start=t,stop=t+1))
        features.append(local(frame.positions[0],cell(frame.box[0]),raw[a:b],atoms,mc))
    graph=np.concatenate(features)
    with cache.with_suffix(".tmp").open("wb") as f:
        np.savez(f,graph=graph)
    cache.with_suffix(".tmp").replace(cache)
    return x,y,graph


class Model(nn.Module):
    def __init__(self, static_width, edge_width, mode):
        super().__init__()
        self.mode=mode
        self.edge=nn.Sequential(nn.Linear(edge_width,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU())
        self.angle=nn.Linear(48,1)
        self.static=nn.Sequential(nn.Linear(static_width,96),nn.LayerNorm(96),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(48*4+96,128),nn.LayerNorm(128),nn.SiLU(),
                                nn.Dropout(.08),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,x,g):
        h=self.edge(g)
        if self.mode=="attention":
            m=(h*self.angle(h).softmax(2)).sum(2)
        else:
            m=h.mean(2)
        q=torch.cat((m,h.amax(2)),dim=2).flatten(1)
        return self.head(torch.cat((self.static(x),q),dim=1)).flatten()


def main():
    torch.set_num_threads(4)
    DEST.mkdir(parents=True,exist_ok=True)
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    roster=json.loads(Path("runs/ga318/development.json").read_bytes())["train"]
    with ThreadPoolExecutor(5) as pool:
        work=list(pool.map(lambda p:pack((p,1,2),mc),roster))
    x,y,g=tuple(np.concatenate(v) for v in zip(*work))
    del work
    val=cohort(VALID,3)
    target,views=extract(val)
    q,t,h=tuple(np.concatenate(v) for v in zip(*(pack((n,1,3),mc) for n in VALID)))
    np.testing.assert_allclose(target,t,rtol=0,atol=3.e-6)
    with np.load("runs/ga_native_offset/baselines.npz") as r:
        ref=predict(r,"real_baseline_",views["compact"])
    mu=x.mean(0)
    sd=x.std(0).clip(1.e-3)
    np.savez(DEST/"scaler.npz",center=mu,scale=sd)
    X=torch.from_numpy(np.clip((x-mu)/sd,-10,10)).cuda()
    Y=torch.from_numpy(y-x[:,0]).cuda()
    G=torch.from_numpy(g).cuda()
    Q=torch.from_numpy(np.clip((q-mu)/sd,-10,10)).cuda()
    H=torch.from_numpy(h).cuda()
    results=dict(proteins=roster,rows=len(y),trials={})
    for mode in ("mean","attention"):
        torch.manual_seed(336)
        model=Model(X.shape[1],G.shape[-1],mode).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=nn.HuberLoss(delta=.60)
        start=time.perf_counter()
        for epoch in range(1,31):
            model.train()
            for ix in torch.randperm(len(X),device="cuda").split(512):
                opt.zero_grad(set_to_none=True)
                cost=loss(model(X[ix],G[ix]),Y[ix])
                cost.backward()
                opt.step()
            if epoch%5:
                continue
            model.eval()
            with torch.no_grad():
                pred=model(Q,H).cpu().numpy()+q[:,0]
            name=f"{mode}-{epoch}"
            results["trials"][name]=dict(seconds=time.perf_counter()-start,**estimates(pred,t,ref))
            record(DEST/"development.json",results)
            torch.save(dict(mode=mode,static_width=X.shape[1],edge_width=G.shape[-1],
                            weights={key:w.detach().cpu() for key,w in model.state_dict().items()}),
                       DEST/f"{name}.pt")
            print(name,results["trials"][name],flush=True)


if __name__=="__main__":
    main()
