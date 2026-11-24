"""GA389 tests a smooth mixture of distance-conditioned short-response readouts."""
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

class Readout(nn.Module):
    def __init__(self,n,temperature):
        super().__init__()
        self.temperature=temperature
        self.body=nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),
                                nn.Dropout(.12),nn.Linear(96,48),nn.SiLU())
        self.readout=nn.Linear(48,3)
        self.register_buffer("knots",torch.tensor([3.8,4.65,5.65]))
    def forward(self,x,d):
        pred=self.readout(self.body(x))
        weights=(-.5*((d[:,None]-self.knots)/self.temperature).square()).softmax(dim=1)
        return (pred*weights).sum(1)

def main():
    dest=R/"ga389"
    dest.mkdir(exist_ok=True)
    roster=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,g,y,_,_=assemble(roster,(49,),R/"ga333",R/"ga355")
    v,e,h,t,o,n=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    label=torch.tensor(y-x[:,0],device="cuda")
    distance=torch.tensor(x[:,0],device="cuda")
    vd=torch.tensor(v[:,0])
    trials={}
    for kind,a,b in (("structure",np.c_[x,d],np.c_[v,e]),
                     ("relation",np.c_[x[:,:3],d,g],np.c_[v[:,:3],e,h])):
        c,s=a.mean(0),a.std(0).clip(.001)
        aa=torch.tensor(((a-c)/s).clip(-12,12),device="cuda",dtype=torch.float32)
        bb=torch.tensor(((b-c)/s).clip(-12,12),dtype=torch.float32)
        for spread in (.3,.55,.9):
            torch.manual_seed(389)
            model=Readout(a.shape[1],spread).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for ep in range(1,31):
                model.train()
                for ix in torch.randperm(len(aa),device="cuda").split(768):
                    pred=model(aa[ix],distance[ix])
                    loss=nn.functional.smooth_l1_loss(pred,label[ix],beta=.4)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if ep not in (10,20,30):
                    continue
                cpu=deepcopy(model).cpu().eval()
                with torch.no_grad():
                    yhat=cpu(bb,vd).numpy()+v[:,0]
                key=f"{kind}-{spread}-{ep}"
                torch.save(dict(state=cpu.state_dict(),center=c,scale=s,spread=spread),
                           dest/f"{key}.pt")
                for weight in (.1,.2,.35):
                    p=(1-weight)*parent+weight*yhat
                    trials[f"{key}/{weight}"]={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    record(dest/"development.json",dict(best=best,trials=trials,train=roster,rows=len(y)))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
