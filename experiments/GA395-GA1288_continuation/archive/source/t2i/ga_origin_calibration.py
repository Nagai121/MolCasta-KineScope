"""GA1277+: learn protein-origin offsets and within-origin distance residuals separately."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
import torch
from torch import nn

from .ga_competitive_capacity import competition
from .ga_multiwindow import targets
from .ga_sequential50 import features,pool
from .ga_static_offset import estimates

OUT=Path("runs/ga1277-group-calibration")


def group_data(a,responses,origins,proteins,truth=None):
    combined=np.char.add(np.char.add(proteins.astype(str),"/"),origins.astype(str))
    group_names,groups=np.unique(combined,return_inverse=True)
    group_vectors=[]
    group_target=[]
    for i in range(len(group_names)):
        mask=groups==i
        q=np.c_[a[mask,:14],responses[mask]]
        group_vectors.append(np.r_[q.mean(0),q.std(0),np.log(mask.sum())])
        if truth is not None:
            group_target.append([truth[mask].mean(),np.log(truth[mask].std().clip(.06))])
    return np.asarray(group_vectors),groups,np.asarray(group_target)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    observed=targets(x,y,o,p)
    group,tr,goal=group_data(x,d,o,p,observed[:,0])
    dev,va,_=group_data(v,e,vo,vg)
    mu,sd=group.mean(0),group.std(0).clip(.001)
    estimator=Ridge(alpha=45).fit((group-mu)/sd,goal)
    calibration=estimator.predict((dev-mu)/sd)
    vectors=features(competition(x,d,o,p,oriented=True),d,g,"level")
    query=features(competition(v,e,vo,vg,oriented=True),e,h,"level")
    c,s=vectors.mean(0),vectors.std(0).clip(.001)
    a=torch.tensor(((vectors-c)/s).clip(-10,10),device="cuda",dtype=torch.float32)
    b=torch.tensor(((query-c)/s).clip(-10,10),dtype=torch.float32)
    w=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    options=[(mode,width,seed) for mode in ("raw","center","normalize")
             for width in (64,128) for seed in (688,872)]
    rows=[]
    for i in range(len(options)):
        tried={tuple(row["spec"]) for row in rows}
        if not rows:
            spec=options[0]
        else:
            best=max(rows,key=lambda row:row["metrics"]["all"]["residual_correlation"])
            spec=min((v for v in options if v not in tried),
                     key=lambda v:sum(a!=b for a,b in zip(v,best["spec"])))
        mode,width,seed=spec
        target=observed[:,0].copy()
        if mode!="raw":
            target-=goal[tr,0]
        if mode=="normalize":
            target/=np.exp(goal[tr,1])
        yy=torch.tensor(target,device="cuda",dtype=torch.float32)
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(a.shape[1],width),nn.LayerNorm(width),nn.SiLU(),
                          nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,1)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0019,weight_decay=.012)
        for _ in range(18):
            for j in torch.randperm(len(a),device="cuda").split(1536):
                err=(net(a[j]).flatten()-yy[j]).square()
                opt.zero_grad(set_to_none=True)
                (err*w[j]).mean().backward()
                opt.step()
        model=deepcopy(net).cpu().eval()
        with torch.inference_mode():
            offset=model(b).flatten().numpy()
        if mode=="normalize":
            offset*=np.exp(calibration[va,1])
        if mode!="raw":
            offset+=calibration[va,0]
        direct=v[:,0]+offset
        pred=.65*parent+.35*direct
        result={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{1277+i}"
        row=dict(id=tag,spec=spec,metrics=result)
        rows.append(row)
        torch.save(dict(state=model.state_dict(),center=c,scale=s,spec=spec,
                        group_center=mu,group_scale=sd,group_coef=estimator.coef_,
                        group_intercept=estimator.intercept_),OUT/f"{tag}.pt")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=pred,readout=direct)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,spec,result["all"]["residual_correlation"],flush=True)
    (OUT/"summary.json").write_text(json.dumps(max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"]),
                                              indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
