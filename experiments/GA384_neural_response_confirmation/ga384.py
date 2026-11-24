"""GA384 fits conditional mean and spread of past distance from generated responses."""
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
    dest=R/"ga384"
    dest.mkdir(exist_ok=True)
    train=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,_,y,_,_=assemble(train,(49,),R/"ga333",R/"ga355")
    v,e,_,t,o,n=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(n,z["proteins"])
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        parent,base=z["parent"],z["baseline"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    target=torch.tensor(y-x[:,0],device="cuda")
    result={}
    for kind,width in (("motion",3),("joint",x.shape[1])):
        a,b=np.c_[x[:,:width],d],np.c_[v[:,:width],e]
        center,scale=a.mean(0),a.std(0).clip(.001)
        X=torch.tensor(((a-center)/scale).clip(-12,12),device="cuda",dtype=torch.float32)
        V=torch.tensor(((b-center)/scale).clip(-12,12),dtype=torch.float32)
        for hetero in (False,True):
            torch.manual_seed(384)
            model=nn.Sequential(nn.Linear(a.shape[1],96),nn.LayerNorm(96),nn.SiLU(),
                                nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2)).cuda()
            optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for ep in range(1,31):
                model.train()
                for ix in torch.randperm(len(X),device="cuda").split(768):
                    p=model(X[ix])
                    err=p[:,0]-target[ix]
                    logvar=p[:,1].clamp(-3.,2.)
                    if hetero:
                        loss=(err.square()*torch.exp(-logvar)+logvar).mean()*.5
                    else:
                        loss=nn.functional.smooth_l1_loss(p[:,0],target[ix],beta=.4)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()
                if ep not in (5,10,20,30):
                    continue
                cpu=deepcopy(model).cpu().eval()
                with torch.no_grad():
                    p=cpu(V).numpy()
                pred=p[:,0]+v[:,0]
                conf=np.ones(len(t))
                if hetero:
                    conf=np.exp(-.5*np.clip(p[:,1],-3,2))
                    conf=(conf/np.mean(conf)).clip(.35,1.8)
                for weight in (.1,.2,.35):
                    for gated in ((False,True) if hetero else (False,)):
                        share=weight*conf if gated else weight
                        q=(1-share)*parent+share*pred
                        key=f"{kind}/{hetero}/{ep}/{weight}/{gated}"
                        result[key]={k:estimates(q[m],t[m],base[m]) for k,m in masks.items()}
                torch.save(dict(state=cpu.state_dict(),center=center,scale=scale),
                           dest/f"{kind}-{hetero}-{ep}.pt")
    best=max(result,key=lambda k:result[k]["all"]["residual_correlation"])
    record(dest/"development.json",dict(best=best,trials=result,train=train,rows=len(y),
                                         cuda=torch.cuda.get_device_name()))
    print(best,result[best],flush=True)
if __name__=="__main__":
    main()

