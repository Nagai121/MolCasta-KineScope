"""GA725+ pretrains with multiple observed history windows on additional proteins."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_force import build
from .ga_multiwindow import targets
from .ga_replica_teacher import trajectory_rows
from .ga_sequential50 import pool
from .ga_static_offset import estimates

R=Path("runs")
OUT=R/"ga725-multiwindow-pretrain"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,proteins),(v,e,h,t,vo,vg,base,parent)=pool()
    pretrain=json.loads((R/"ga588-transfer/summary.json").read_bytes())["pretrain"]
    assert not set(pretrain)&(set(proteins)|set(vg))
    roster=[]; values=[]; labels=[]; times=[]
    for name in pretrain:
        q,r=build((name,1,4))
        pairs,orig=trajectory_rows(name,1)
        np.testing.assert_array_equal(q[:,0],pairs[:len(r),2])
        roster.extend([name]*len(r))
        values.append(q)
        labels.append(r)
        times.append(orig[:len(r)])
    s,l,z,n=np.concatenate(values),np.concatenate(labels),np.concatenate(times),np.asarray(roster)
    pre=targets(s,l,z,n,OUT/"pretrain-history.npz")
    with np.load(R/"ga688-multiwindow/training-history-targets.npz") as data:
        np.testing.assert_array_equal(data["label"],y)
        fit=data["multi"]
    mean,sd=s.mean(0),s.std(0).clip(.001)
    dm,ds=d.mean(0),d.std(0).clip(.001)
    lm,ls=fit.mean(0),fit.std(0).clip(.01)
    def norm(v,c,s,device):
        return torch.tensor(((v-c)/s).clip(-10,10),dtype=torch.float32,device=device)
    train=norm(x,mean,sd,"cuda"); dev=norm(v,mean,sd,"cpu")
    dtrain=norm(d,dm,ds,"cuda"); ddev=norm(e,dm,ds,"cpu")
    pretrain_x=norm(s,mean,sd,"cuda")
    pretrain_y=norm(pre,lm,ls,"cuda")
    ytrain=norm(fit,lm,ls,"cuda")
    w=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    torch.manual_seed(725)
    backbone=nn.Sequential(nn.Linear(s.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                           nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU()).cuda()
    static=nn.Linear(64,7).cuda()
    opt=torch.optim.AdamW([*backbone.parameters(),*static.parameters()],lr=.002,weight_decay=.01)
    for ep in range(12):
        for batch in torch.randperm(len(s),device="cuda").split(2048):
            error=(static(backbone(pretrain_x[batch]))-pretrain_y[batch]).square()
            loss=(error[:,0]+error[:,1:].mean(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep in (3,7,11):
            print("multiwindow pretraining",ep+1,len(s),flush=True)
    initial=deepcopy(backbone.state_dict())
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    options=[(width,alpha,lr) for width in (64,128) for alpha in (0.,.4,1.) for lr in (.0007,.0017)]
    best=None
    rows=[]
    for i in range(len(options)):
        if best is None:
            spec=(128,1.,.0017)
        else:
            history={tuple(r["spec"]) for r in rows}
            spec=min((s for s in options if s not in history),key=lambda s:sum(a!=b for a,b in zip(s,best["spec"])))
        width,alpha,lr=spec
        torch.manual_seed(726)
        backbone.load_state_dict(initial)
        head=nn.Sequential(nn.Linear(64+d.shape[1],width),nn.LayerNorm(width),nn.SiLU(),
                           nn.Linear(width,7)).cuda()
        optimizer=torch.optim.AdamW([*backbone.parameters(),*head.parameters()],lr=lr,weight_decay=.01)
        for _ in range(18):
            for b in torch.randperm(len(train),device="cuda").split(1536):
                p=head(torch.cat([backbone(train[b]),dtrain[b]],1))
                err=(p-ytrain[b]).square()
                loss=((err[:,0]+alpha*err[:,1:].mean(1))*w[b]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        cpu,ch=deepcopy(backbone).cpu().eval(),head.cpu().eval()
        with torch.inference_mode():
            direct=ch(torch.cat([cpu(dev),ddev],1)).numpy()[:,0]*ls[0]+lm[0]+v[:,0]
        p=.65*parent+.35*direct
        scores={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
        mid=f"GA{725+i}"
        torch.save(dict(encoder=cpu.state_dict(),head=ch.state_dict(),static_center=mean,
                        static_scale=sd,dynamic_center=dm,dynamic_scale=ds,output_center=lm,
                        output_scale=ls,spec=spec),OUT/f"{mid}.pt")
        np.savez_compressed(OUT/f"{mid}.npz",prediction=p,readout=direct)
        row=dict(id=mid,spec=spec,metrics=scores)
        rows.append(row)
        if best is None or scores["all"]["residual_correlation"]>best["metrics"]["all"]["residual_correlation"]:
            best=row
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(mid,spec,scores["all"]["residual_correlation"],flush=True)
        if scores["all"]["residual_correlation"]>=.700:
            break
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
