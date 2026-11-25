"""GA945+: jointly predict focal history and the component shared across replicas."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga945-shared-replica")


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    label=targets(x,y,o,p)
    with np.load("runs/ga508-replica-teacher/training-teacher.npz") as z:
        np.testing.assert_array_equal(y,z["original"])
        prior=z["prior"][:,1]
        raw_support=z["support"][:,1]
    support=raw_support/(1+raw_support)
    prior=np.where(support>0,prior,label[:,0])
    gap=np.abs(prior-label[:,0])
    all_labels=np.c_[label,prior,gap]
    lm,ls=all_labels.mean(0),all_labels.std(0).clip(.01)
    yy=torch.tensor((all_labels-lm)/ls,device="cuda",dtype=torch.float32)
    strength=torch.tensor(support,device="cuda",dtype=torch.float32)
    weight=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    w_gap=torch.tensor(gap*support,device="cuda",dtype=torch.float32)
    xx,vv=features(x,d,g,"level"),features(v,e,h,"level")
    center,scale=xx.mean(0),xx.std(0).clip(.001)
    a=torch.tensor(((xx-center)/scale).clip(-10,10),device="cuda",dtype=torch.float32)
    b=torch.tensor(((vv-center)/scale).clip(-10,10),dtype=torch.float32)
    options=[(shared,aux,robust) for shared in (.25,1.,2.) for aux in (0.,1.) for robust in (0.,-1.,-2.)]
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    rows=[]
    for i in range(len(options)):
        if not rows:
            spec=options[0]
        else:
            best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            used={tuple(r["spec"]) for r in rows}
            spec=min((s for s in options if s not in used),key=lambda s:sum(u!=v for u,v in zip(s,best["spec"])))
        shared,aux,robust=spec
        torch.manual_seed(688)
        model=nn.Sequential(nn.Linear(a.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                            nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,9)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.012)
        for _ in range(18):
            for j in torch.randperm(len(y),device="cuda").split(1536):
                report=model(a[j])
                err=(report-yy[j]).square()
                residual=err[:,0]
                history=err[:,1:7].mean(1)
                across=err[:,7]+.25*err[:,8]
                w=weight[j]*torch.exp((robust*w_gap[j]).clamp(-2.,2.))
                w=w/w.mean()
                loss=((residual+aux*history+shared*strength[j]*across)*w).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net=deepcopy(model).cpu().eval()
        with torch.inference_mode():
            all_pred=net(b).numpy()*ls+lm
        direct=all_pred[:,0]+v[:,0]
        pred=.65*parent+.35*direct
        results={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{945+i}"
        torch.save(dict(state=net.state_dict(),center=center,scale=scale,target_center=lm,
                        target_scale=ls,spec=spec),OUT/f"{tag}.pt")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=pred,readout=direct)
        row=dict(id=tag,spec=spec,metrics=results)
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,spec,results["all"]["residual_correlation"],flush=True)
        if results["all"]["residual_correlation"]>=.700:
            break
    best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
