"""GA376 tests smooth kernel regressors for short generated motion."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .explore import record
from .ga373 import assemble
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga376"

def main():
    D.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,g,y,_,_=assemble(names,(49,),R/"ga333",R/"ga355")
    v,e,h,t,o,p=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(p,z["proteins"])
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    trial={}
    label=torch.as_tensor(y-x[:,0],device="cuda")
    for name,u,w in (("motion",np.c_[x[:,:3],d],np.c_[v[:,:3],e]),
                     ("relational",np.c_[x[:,:3],d,g],np.c_[v[:,:3],e,h]),
                     ("structural",np.c_[x,d],np.c_[v,e])):
        center=u.mean(0)
        scale=u.std(0).clip(.01)
        train=torch.as_tensor(((u-center)/scale).clip(-6,6),device="cuda")
        val=torch.as_tensor(((w-center)/scale).clip(-6,6))
        for bandwidth in (.5,1.,2.):
            torch.manual_seed(376)
            width=768
            projection=torch.randn(u.shape[1],width,device="cuda")/(bandwidth*np.sqrt(u.shape[1]))
            phase=torch.rand(width,device="cuda")*(2*np.pi)
            phi=torch.cos(train@projection+phase)*np.sqrt(2/width)
            prediction=torch.cos(val@projection.cpu()+phase.cpu())*np.sqrt(2/width)
            cov=phi.T@phi
            rhs=phi.T@(label-label.mean())
            identity=torch.eye(width,device="cuda")
            for alpha in (.1,1.,10.):
                w=torch.linalg.solve(cov+alpha*identity,rhs)
                p=(prediction@w.cpu()).numpy()+float(label.mean())+v[:,0]
                key=f"{name}-{bandwidth}-{alpha}"
                for weight in (.1,.2,.35):
                    combined=(1-weight)*parent+weight*p
                    trial[f"{key}/{weight}"]={k:estimates(combined[m],t[m],base[m]) for k,m in masks.items()}
                np.savez(D/f"{key}.npz",center=center,scale=scale,projection=projection.cpu().numpy(),
                         phase=phase.cpu().numpy(),weight=w.cpu().numpy(),bias=float(label.mean()))
    first={k:estimates(parent[m],t[m],base[m]) for k,m in masks.items()}
    best=max(trial,key=lambda k:trial[k]["all"]["residual_correlation"])
    record(D/"development.json",dict(best=best,trials=trial,parent=first,train=names,
                                    device=torch.cuda.get_device_name()))
    print(best,trial[best],flush=True)

if __name__=="__main__":
    main()

