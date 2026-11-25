"""GA917+: predict a calibrated distribution over persistent displacement bins."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga917-bins")


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    observations=targets(x,y,o,p)
    mean,scale=observations.mean(0),observations.std(0).clip(.01)
    labels=torch.tensor((observations-mean)/scale,dtype=torch.float32,device="cuda")
    w=torch.tensor(np.where(o==49,1.,4.),dtype=torch.float32,device="cuda")
    ins,outs=features(x,d,g,"level"),features(v,e,h,"level")
    c,s=ins.mean(0),ins.std(0).clip(.001)
    a=torch.tensor(((ins-c)/s).clip(-10,10),dtype=torch.float32,device="cuda")
    b=torch.tensor(((outs-c)/s).clip(-10,10),dtype=torch.float32)
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    rows=[]
    options=[(nb,spread,aux) for nb in (9,17,33) for spread in (.15,.35) for aux in (0.,1.)]
    for i in range(len(options)):
        used={tuple(r["spec"]) for r in rows}
        if i==0:
            spec=options[0]
        else:
            best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in options if s not in used),key=lambda s:sum(a!=b for a,b in zip(s,best["spec"])))
        num,spread,aux=spec
        lo,hi=np.quantile((observations[:,0]-mean[0])/scale[0],[.002,.998])
        bins=torch.linspace(float(lo),float(hi),num,device="cuda")
        soft=torch.softmax(-.5*((labels[:,0:1]-bins)/spread).square(),dim=1)
        torch.manual_seed(917)
        model=nn.Sequential(nn.Linear(a.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                            nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,num+6)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.01)
        for _ in range(18):
            for j in torch.randperm(len(labels),device="cuda").split(1536):
                h=model(a[j])
                logp=h[:,:num].log_softmax(1)
                prob=logp.exp()
                cross=-(soft[j]*logp).sum(1)
                expectation=(prob*bins).sum(1)
                mse=(expectation-labels[j,0]).square()
                side=(h[:,num:]-labels[j,1:]).square().mean(1)
                opt.zero_grad(set_to_none=True)
                ((cross+.6*mse+aux*side)*w[j]).mean().backward()
                opt.step()
        net=deepcopy(model).cpu().eval()
        with torch.inference_mode():
            probs=net(b)[:,:num].softmax(1)
            expectation=(probs*bins.cpu()).sum(1).numpy()
        direct=expectation*scale[0]+mean[0]+v[:,0]
        pred=.65*parent+.35*direct
        m={k:estimates(pred[x],t[x],base[x]) for k,x in masks.items()}
        tag=f"GA{917+i}"
        torch.save(dict(state=net.state_dict(),bins=bins.cpu(),center=c,scale=s,
                        label_center=mean,label_scale=scale,spec=spec),OUT/f"{tag}.pt")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=pred,readout=direct)
        row=dict(id=tag,spec=spec,metrics=m)
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,spec,m["all"]["residual_correlation"],flush=True)
        if m["all"]["residual_correlation"]>=.700:
            break
    best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
