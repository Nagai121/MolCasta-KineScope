"""GA387 compares pooled, uniform-protein, and group-robust neural supervision."""
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

def main():
    dest=R/"ga387"
    dest.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,_,y,_,groups=assemble(names,(49,),R/"ga333",R/"ga355")
    v,e,_,t,o,proteins=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as f:
        np.testing.assert_array_equal(proteins,f["proteins"])
        np.testing.assert_array_equal(o,f["origin"])
        np.testing.assert_allclose(t,f["target"],rtol=0,atol=3e-6)
        parent,base=f["parent"],f["baseline"]
    a,b=np.c_[x,d],np.c_[v,e]
    c,s=a.mean(0),a.std(0).clip(.001)
    A=torch.tensor(((a-c)/s).clip(-12,12),dtype=torch.float32,device="cuda")
    B=torch.tensor(((b-c)/s).clip(-12,12),dtype=torch.float32)
    Y=torch.tensor(y-x[:,0],device="cuda")
    ids=np.unique(groups,return_inverse=True)[1]
    g=torch.tensor(ids,device="cuda")
    sizes=torch.bincount(g).float()
    total=len(sizes)
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    outputs={}
    for mode,temperature in (("pooled",0.),("uniform",0.),("robust",1.),("robust",3.)):
        torch.manual_seed(384)
        net=nn.Sequential(nn.Linear(a.shape[1],96),nn.LayerNorm(96),nn.SiLU(),
                          nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        assigned=torch.ones(len(Y),device="cuda")
        averaged=torch.zeros(total,device="cuda")
        for epoch in range(1,41):
            net.train()
            sum_loss=torch.zeros(total,device="cuda")
            if mode=="uniform":
                assigned=(len(Y)/(total*sizes))[g]
            elif mode=="robust":
                group_weights=torch.softmax(temperature*averaged,dim=0)
                assigned=(len(Y)*group_weights/sizes)[g].clamp(.15,5.)
                assigned=assigned/assigned.mean()
            for i in torch.randperm(len(Y),device="cuda").split(768):
                p=net(A[i])[:,0]
                err=nn.functional.smooth_l1_loss(p,Y[i],reduction="none",beta=.4)
                sum_loss.scatter_add_(0,g[i],err.detach())
                opt.zero_grad(set_to_none=True)
                (err*assigned[i]).mean().backward()
                opt.step()
            averaged=.7*averaged+.3*sum_loss/sizes
            if epoch not in (10,20,30,40):
                continue
            cpu=deepcopy(net).cpu().eval()
            with torch.no_grad():
                pred=cpu(B).numpy()[:,0]+v[:,0]
            key=f"{mode}-{temperature}-{epoch}"
            torch.save(dict(state=cpu.state_dict(),center=c,scale=s),dest/f"{key}.pt")
            for w in (.1,.2,.35):
                q=(1-w)*parent+w*pred
                outputs[f"{key}/{w}"]={label:estimates(q[m],t[m],base[m]) for label,m in masks.items()}
    best=max(outputs,key=lambda n:outputs[n]["all"]["residual_correlation"])
    record(dest/"development.json",dict(best=best,trials=outputs,train=names,rows=len(y)))
    print(best,outputs[best],flush=True)

if __name__=="__main__":
    main()
