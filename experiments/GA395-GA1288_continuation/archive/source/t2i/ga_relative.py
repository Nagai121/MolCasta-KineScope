"""GA842+: learn which candidate pairs have stronger persistent movement in each structure."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga842-relative")


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(p)&set(vg)
    hist=targets(x,y,o,p)
    median=hist.mean(0)
    deviation=hist.std(0).clip(.01)
    raw=features(x,d,g,"level")
    dev=features(v,e,h,"level")
    center,scale=raw.mean(0),raw.std(0).clip(.001)
    A=torch.tensor(((raw-center)/scale).clip(-10,10),dtype=torch.float32,device="cuda")
    B=torch.tensor(((dev-center)/scale).clip(-10,10),dtype=torch.float32)
    H=torch.tensor((hist-median)/deviation,dtype=torch.float32,device="cuda")
    # Build reference pair groups by protein and origin, then stratify by current distance.
    group_keys=np.char.add(np.char.add(p.astype(str),"/"),o.astype(str))
    _, ids=np.unique(group_keys,return_inverse=True)
    cut=np.floor(x[:,0]*5).astype(np.int32)
    G=ids*64+cut
    order=np.argsort(G,kind="stable")
    bounds=np.r_[0,np.flatnonzero(np.diff(G[order]))+1,len(G)]
    partners=np.empty((4,len(G)),dtype=np.int64)
    rng=np.random.default_rng(842)
    for a,b in zip(bounds[:-1],bounds[1:]):
        members=order[a:b]
        for row in partners:
            row[members]=rng.permutation(members)
    P=torch.tensor(partners,device="cuda")
    counts=np.bincount(ids)
    weights=torch.tensor(np.where(o==49,1.,4.) / np.sqrt(counts[ids]),device="cuda",dtype=torch.float32)
    weights=weights/weights.mean()
    history=[]
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    options=[(mode,alpha,strength) for mode in ("margin","delta","corr")
             for alpha in (0.,1.) for strength in (.1,.35,.75)]
    for index in range(len(options)):
        tried={tuple(r["spec"]) for r in history}
        if history:
            leader=max(history,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in options if s not in tried),
                     key=lambda s:sum(a!=b for a,b in zip(s,leader["spec"])))
        else:
            spec=options[0]
        mode,alpha,strength=spec
        torch.manual_seed(842)
        net=nn.Sequential(nn.Linear(A.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                          nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,7)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.012)
        for epoch in range(18):
            for i in torch.randperm(len(A),device="cuda").split(1536):
                j=P[epoch%4,i]
                one,two=net(A[i]),net(A[j])
                err=(one-H[i]).square()
                difference=one[:,0]-two[:,0]
                observed=H[i,0]-H[j,0]
                if mode=="delta":
                    relative=(difference-observed).square()
                elif mode=="corr":
                    # A cosine term on differences compares within-origin order and magnitude.
                    similarity=nn.functional.cosine_similarity(difference,observed,dim=0,eps=1e-5)
                    relative=1-similarity
                else:
                    relative=nn.functional.softplus(-difference*observed.sign())*observed.abs().clamp(max=1.)
                loss=((err[:,0]+alpha*err[:,1:].mean(1)+strength*relative)*weights[i]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        cpu=deepcopy(net).cpu().eval()
        with torch.inference_mode():
            direct=cpu(B).numpy()[:,0]*deviation[0]+median[0]+v[:,0]
        pred=.65*parent+.35*direct
        scores={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        mid=f"GA{842+index}"
        torch.save(dict(state=cpu.state_dict(),center=center,scale=scale,output_center=median,
                        output_scale=deviation,spec=spec),OUT/f"{mid}.pt")
        np.savez_compressed(OUT/f"{mid}.npz",prediction=pred,readout=direct)
        row=dict(id=mid,spec=spec,metrics=scores)
        history.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(mid,spec,scores["all"]["residual_correlation"],flush=True)
        if scores["all"]["residual_correlation"]>=.700:
            break
    best=max(history,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()


