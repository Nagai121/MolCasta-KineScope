"""GA393 averages independently trained GA386 readouts before blending."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .explore import record
from .ga373 import assemble
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga393"

def model(n):
    return nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),
                         nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2))

def main():
    D.mkdir(exist_ok=True)
    roster=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    late=json.loads((R/"ga385/lane-0.json").read_bytes())["proteins"]
    sets=[assemble(roster,(49,),R/"ga333",R/"ga355")]
    for origin in (149,249):
        names=[n for n in late if (R/"ga385"/f"{n}-{origin}.npz").exists()]
        sets.append(assemble(names,(origin,),R/"ga365",R/"ga385"))
    x,d,_,y,o,_=tuple(np.concatenate(z) for z in zip(*sets))
    v,e,_,t,vo,vg=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(vg,z["proteins"])
        np.testing.assert_array_equal(vo,z["origin"])
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        base,parent=z["baseline"],z["parent"]
    xx,vv=np.c_[x,d],np.c_[v,e]
    center,scale=xx.mean(0),xx.std(0).clip(.001)
    A=torch.tensor(((xx-center)/scale).clip(-12,12),dtype=torch.float32,device="cuda")
    B=torch.tensor(((vv-center)/scale).clip(-12,12),dtype=torch.float32)
    Y=torch.tensor(y-x[:,0],device="cuda")
    W=torch.tensor(1.+(o!=49),dtype=torch.float32,device="cuda")
    W=W/W.mean()
    seeds=(384,393,394,395,396)
    forecast=[]
    for seed in seeds:
        torch.manual_seed(seed)
        net=model(A.shape[1]).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        for _ in range(40):
            net.train()
            for idx in torch.randperm(len(A),device="cuda").split(768):
                z=net(A[idx])[:,0]
                loss=(W[idx]*nn.functional.smooth_l1_loss(z,Y[idx],beta=.4,reduction="none")).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        cpu=deepcopy(net).cpu().eval()
        with torch.no_grad():
            forecast.append(cpu(B).numpy()[:,0]+v[:,0])
        torch.save(cpu.state_dict(),D/f"seed-{seed}.pt")
        print(seed,flush=True)
    np.savez(D/"scale.npz",center=center,scale=scale)
    combined=np.mean(forecast,axis=0)
    masks={"all":np.ones(len(t),bool),"first":vo==49,"later":vo!=49}
    trials={}
    for w in (.1,.2,.35,.5,.65):
        p=(1-w)*parent+w*combined
        trials[str(w)]={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    record(D/"development.json",dict(best=best,trials=trials,seeds=seeds,rows=len(y)))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
