"""GA381 learns a regularized local-motion correction with CPU inference."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .explore import record
from .ga377_score import load,minicalc
from .ga379 import moving
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga381"

def main():
    D.mkdir(exist_ok=True)
    names=json.loads((R/"ga377/train/manifest.json").read_bytes())["proteins"]
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    x,d,g,y,_,_=load(names,"train",(49,149),mc,extra=moving)
    v,e,h,t,o,n=load(VALID,"validation",(49,149,249),mc,extra=moving)
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        np.testing.assert_allclose(t,z["target"],atol=3e-6,rtol=0)
        parent,base=z["parent"],z["baseline"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    label=torch.tensor(y-x[:,0],device="cuda")
    trials={}
    for key,a,b in (
        ("distance",np.c_[x[:,:3],d],np.c_[v[:,:3],e]),
        ("direction",np.c_[x[:,:3],d,g],np.c_[v[:,:3],e,h]),
        ("joint",np.c_[x,d,g],np.c_[v,e,h])):
        center=a.mean(0).astype("f")
        scale=a.std(0).clip(.001).astype("f")
        train=torch.tensor(((a-center)/scale).clip(-8,8),dtype=torch.float32,device="cuda")
        valid=torch.tensor(((b-center)/scale).clip(-8,8),dtype=torch.float32)
        torch.manual_seed(381)
        head=nn.Sequential(nn.Linear(a.shape[1],96),nn.LayerNorm(96),nn.SiLU(),
                           nn.Dropout(.15),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1)).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.02)
        for epoch in range(1,31):
            head.train()
            for idx in torch.randperm(len(train),device="cuda").split(512):
                pred=head(train[idx]).flatten()
                loss=nn.functional.smooth_l1_loss(pred,label[idx],beta=.4)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if epoch not in (5,10,20,30):
                continue
            evaluator=deepcopy(head).cpu().eval()
            with torch.no_grad():
                pred=evaluator(valid).flatten().numpy()+v[:,0]
            for w in (.1,.2,.35):
                q=(1-w)*parent+w*pred
                trials[f"{key}/{epoch}/{w}"]={s:estimates(q[m],t[m],base[m]) for s,m in masks.items()}
            torch.save(dict(weights=evaluator.state_dict(),center=center,scale=scale),D/f"{key}-{epoch}.pt")
    best=max(trials,key=lambda s:trials[s]["all"]["residual_correlation"])
    record(D/"development.json",dict(best=best,trials=trials,rows=len(y),device=torch.cuda.get_device_name()))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()

