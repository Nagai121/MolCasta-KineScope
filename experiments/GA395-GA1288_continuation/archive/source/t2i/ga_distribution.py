"""GA773+: estimate the distribution of a pair's past distances from a static origin."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga773-distribution")


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(p)&set(vg)
    label=targets(x,y,o,p,OUT/"training-distribution.npz",distribution=True)
    w=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    yy=torch.tensor(label,device="cuda",dtype=torch.float32)
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    options=[(view,loss,width) for view in ("level","full_response")
             for loss in ("mean","quantiles","mean_and_quantiles","mean_and_order") for width in (64,128)]
    history=[]
    store={}
    for i in range(len(options)):
        done={tuple(r["spec"]) for r in history}
        if not history:
            spec=options[0]
        else:
            leader=max(history,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in options if s not in done),
                     key=lambda s:(sum(a!=b for a,b in zip(s,leader["spec"])),
                                   sum(r["spec"][1]==s[1] for r in history)))
        view,loss,width=spec
        if view not in store:
            a,b=features(x,d,g,view),features(v,e,h,view)
            mu,sd=a.mean(0),a.std(0).clip(.001)
            store[view]=(torch.tensor(((a-mu)/sd).clip(-10,10),device="cuda",dtype=torch.float32),
                         torch.tensor(((b-mu)/sd).clip(-10,10),dtype=torch.float32),mu,sd)
        a,b,mu,sd=store[view]
        torch.manual_seed(773)
        net=nn.Sequential(nn.Linear(a.shape[1],width),nn.LayerNorm(width),nn.SiLU(),
                          nn.Linear(width,width//2),nn.LayerNorm(width//2),nn.SiLU(),
                          nn.Linear(width//2,8)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.012)
        for _ in range(18):
            for ix in torch.randperm(len(a),device="cuda").split(1536):
                pred=net(a[ix])
                err=(pred-yy[ix]).square()
                term=err[:,0]
                if loss=="quantiles":
                    term=err[:,1:].mean(1)
                if loss=="mean_and_quantiles" or loss=="mean_and_order":
                    term=term+err[:,1:].mean(1)
                if loss=="mean_and_order":
                    term=term+.2*torch.relu(pred[:,1:-1]-pred[:,2:]).mean(1)
                opt.zero_grad(set_to_none=True)
                (term*w[ix]).mean().backward()
                opt.step()
        cpu=deepcopy(net).cpu().eval()
        with torch.inference_mode():
            out=cpu(b).numpy()
        quad=out[:,1:].mean(1)
        direct=(quad if loss=="quantiles" else out[:,0])+v[:,0]
        q=.65*parent+.35*direct
        scores={k:estimates(q[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{773+i}"
        torch.save(dict(state=cpu.state_dict(),center=mu,scale=sd,spec=spec),OUT/f"{tag}.pt")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=q,readout=direct,quantiles=out[:,1:])
        row=dict(id=tag,spec=spec,metrics=scores)
        history.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,spec,scores["all"]["residual_correlation"],flush=True)
        if scores["all"]["residual_correlation"]>=.700:
            break
    best=max(history,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
