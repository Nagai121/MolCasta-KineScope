"""GA805+: test the effect of history supervision in held-protein folds."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch
from torch import nn

from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga805-window-grouped")


def main(stability=False):
    out = Path("runs/ga860-reliability-grouped") if stability else OUT
    out.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    label=targets(x,y,o,p)
    if stability:
        with np.load("runs/ga508-replica-teacher/training-teacher.npz") as z:
            np.testing.assert_array_equal(z["original"],y)
            gap=np.abs(z["prior"][:,1]-label[:,0])*z["support"][:,1]/(1+z["support"][:,1])
    train=features(x,d,g,"level")
    val=features(v,e,h,"level")
    splits=list(GroupKFold(4).split(train,y,p))
    refs=np.empty_like(y)
    for tr,va in splits:
        refs[va]=Ridge(alpha=10.).fit(x[tr,:12],y[tr]).predict(x[va,:12])

    def fit(train_idx, query, position, spec):
        group,alpha,seed,*confidence=spec
        a=train[train_idx]
        mu,sd=a.mean(0),a.std(0).clip(.001)
        center,scale=label[train_idx].mean(0),label[train_idx].std(0).clip(.01)
        a=torch.tensor(((a-mu)/sd).clip(-10,10),device="cuda",dtype=torch.float32)
        b=torch.tensor(((query-mu)/sd).clip(-10,10),dtype=torch.float32)
        yy=torch.tensor((label[train_idx]-center)/scale,device="cuda",dtype=torch.float32)
        weight=np.where(o[train_idx]==49,1.,4.)
        if stability:
            weight=weight*np.exp(np.clip(confidence[0]*gap[train_idx],-2.,2.))
        w=torch.tensor(weight,device="cuda",dtype=torch.float32)
        cols={"all":(1,2,3,4,5,6),"short":(1,2,3),"variation":(5,6)}[group]
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(a.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                          nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,7)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.012)
        for _ in range(18):
            for i in torch.randperm(len(a),device="cuda").split(1536):
                err=(net(a[i])-yy[i]).square()
                opt.zero_grad(set_to_none=True)
                local=w[i]
                if stability:
                    local=local/local.mean()
                ((err[:,0]+alpha*err[:,cols].mean(1))*local).mean().backward()
                opt.step()
        cpu=net.cpu().eval()
        with torch.inference_mode():
            q=cpu(b).numpy()[:,0]*scale[0]+center[0]+position
        return q,dict(state=cpu.state_dict(),center=mu,scale=sd,
                      output_center=center,output_scale=scale,spec=spec)

    configs=[(name,a,seed) for seed in (688,773,877)
             for name,a in (("all",0.),("all",1.),("short",1.),("variation",1.))]
    if stability:
        configs=[(name,1.,seed,factor) for seed in (688,773,941)
                 for name in ("all","variation") for factor in (0.,-1.)]
    rows=[]
    for i,spec in enumerate(configs):
        p0=np.empty_like(y)
        for tr,va in splits:
            p0[va],_=fit(tr,train[va],x[va,0],spec)
        fold=[estimates(p0[va],y[va],refs[va])["residual_correlation"] for _,va in splits]
        tag=f"GA{(860 if stability else 805)+i}"
        row=dict(id=tag,spec=spec,folds=fold,cv_mean=float(np.mean(fold)))
        rows.append(row)
        np.savez_compressed(out/f"{tag}-cv.npz",prediction=p0,base=refs,target=y,proteins=p)
        with (out/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
    family={}
    for r in rows:
        name,alpha,_,*confidence=r["spec"]
        family.setdefault((name,alpha,*confidence),[]).append(r["cv_mean"])
    best=max(family,key=lambda k:np.mean(family[k]))
    readout=[]
    for r in rows:
        name,alpha,_,*confidence=r["spec"]
        if (name,alpha,*confidence)!=best:
            continue
        pr,state=fit(np.arange(len(y)),val,v[:,0],r["spec"])
        torch.save(state,out/f"{r['id']}.pt")
        readout.append(pr)
    direct=np.mean(readout,axis=0)
    pred=.65*parent+.35*direct
    development={n:estimates(pred[m],t[m],base[m]) for n,m in
                 (("all",vo>=0),("first",vo==49),("later",vo!=49))}
    result=dict(best_family=best,means={str(k):float(np.mean(a)) for k,a in family.items()},development=development)
    np.savez_compressed(out/"development.npz",prediction=pred,readout=direct)
    (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result,flush=True)


if __name__=="__main__":
    import sys
    main(stability="--stability" in sys.argv)

