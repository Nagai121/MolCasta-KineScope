"""GA989+: jointly read the individual simulation branches before pooling."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga333 import WINDOW
from .ga_multiwindow import targets
from .ga_sequential50 import pool
from .ga_static_offset import VALID, estimates

R=Path("runs")
OUT=R/"ga989-branch-fusion"


def raw(names,origins,amber,charmm):
    values=[]
    order=[]
    for name in names:
        for origin in origins:
            cache=f"{name}-{origin}.npz"
            with np.load(R/amber/cache) as a,np.load(R/charmm/cache) as b:
                np.testing.assert_array_equal(a["pair"],b["pair"])
                pair=a["pair"]
                u=np.concatenate([a["tracks"][:2,:len(WINDOW)],b["tracks"][:2,:len(WINDOW)]])
                values.append((u-pair[:,2]).transpose(2,0,1))
                order.extend([(name,origin,float(d)) for d in pair[:,2]])
    return np.concatenate(values),order


def streams():
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    batches=[raw(names,(49,),"ga333","ga355")]
    for lane,a,b in (("ga385/lane-0.json","ga365","ga385"),
                     ("ga392/lane-0.json","ga392/amber","ga392/charmm")):
        names=json.loads((R/lane).read_bytes())["proteins"]
        for origin in (149,249):
            eligible=[n for n in names if all((R/f"{d}/{n}-{origin}.npz").is_file() for d in (a,b))]
            batches.append(raw(eligible,(origin,),a,b))
    train=np.concatenate([a for a,_ in batches])
    order=[entry for _,indices in batches for entry in indices]
    valid,index=raw(VALID,(49,149,249),"ga360/amber","ga360/charmm")
    return train,order,valid,index


class BranchHead(nn.Module):
    def __init__(self,n,t,kind):
        super().__init__()
        self.kind=kind
        self.static=nn.Sequential(nn.Linear(n,112),nn.LayerNorm(112),nn.SiLU(),nn.Linear(112,48),nn.SiLU())
        self.response=nn.Sequential(nn.Linear(t,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.attention=nn.Linear(80,1)
        self.head=nn.Sequential(nn.Linear(48+96,128),nn.LayerNorm(128),nn.SiLU(),
                                nn.Linear(128,64),nn.SiLU(),nn.Linear(64,7))

    def forward(self,x,tracks):
        anchor=self.static(x)
        signal=self.response(tracks).reshape(-1,2,2,32)
        if self.kind=="attention":
            both=anchor[:,None,None].expand(-1,2,2,-1)
            weight=self.attention(torch.cat([signal,both],-1)).softmax(2)
            pooled=(weight*signal).sum(2)
        elif self.kind=="maximum":
            pooled=signal.amax(2)
        else:
            pooled=signal.mean(2)
        first,second=pooled.unbind(1)
        return self.head(torch.cat([anchor,first,second,first-second],1))


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    r,index,s,indices=streams()
    for order,names,origins,coords in ((index,p,o,x),(indices,vg,vo,v)):
        np.testing.assert_array_equal(names,[k[0] for k in order])
        np.testing.assert_array_equal(origins,[k[1] for k in order])
        np.testing.assert_array_equal(coords[:,0],[k[2] for k in order])
    label=targets(x,y,o,p)
    lm,ls=label.mean(0),label.std(0).clip(.01)
    yy=torch.tensor((label-lm)/ls,device="cuda",dtype=torch.float32)
    with np.load(R/"ga508-replica-teacher/training-teacher.npz") as z:
        np.testing.assert_array_equal(y,z["original"])
        gap=np.abs(z["prior"][:,1]-label[:,0])*z["support"][:,1]/(1+z["support"][:,1])
    w=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    uncertainty=torch.tensor(gap,device="cuda",dtype=torch.float32)
    xm,xs=x.mean(0),x.std(0).clip(.001)
    rm,rs=r.mean((0,1)),r.std((0,1)).clip(.001)
    def norm(v,m,s,device):
        return torch.tensor(((v-m)/s).clip(-10,10),device=device,dtype=torch.float32)
    a,b=norm(x,xm,xs,"cuda"),norm(v,xm,xs,"cpu")
    c,f=norm(r,rm,rs,"cuda"),norm(s,rm,rs,"cpu")
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    candidates=[(kind,aux,weight) for kind in ("mean","maximum","attention")
                for aux in (0.,1.) for weight in (0.,-2.)]
    rows=[]
    for i in range(len(candidates)):
        used={tuple(row["spec"]) for row in rows}
        if not rows:
            spec=candidates[0]
        else:
            best=max(rows,key=lambda row:row["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in candidates if s not in used),
                     key=lambda s:sum(a!=b for a,b in zip(s,best["spec"])))
        kind,aux,weight=spec
        torch.manual_seed(688)
        head=BranchHead(a.shape[1],c.shape[2],kind).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.0018,weight_decay=.012)
        for _ in range(18):
            for j in torch.randperm(len(y),device="cuda").split(1536):
                yhat=head(a[j],c[j])
                err=(yhat-yy[j]).square()
                local=w[j]*(uncertainty[j]*weight).clamp(-2.,2.).exp()
                local=local/local.mean()
                opt.zero_grad(set_to_none=True)
                ((err[:,0]+aux*err[:,1:].mean(1))*local).mean().backward()
                opt.step()
        net=deepcopy(head).cpu().eval()
        with torch.inference_mode():
            direct=net(b,f).numpy()[:,0]*ls[0]+lm[0]+v[:,0]
        pred=.65*parent+.35*direct
        metrics={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{989+i}"
        torch.save(dict(state=net.state_dict(),static_mean=xm,static_scale=xs,response_mean=rm,
                        response_scale=rs,output_mean=lm,output_scale=ls,spec=spec),OUT/f"{tag}.pt")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=pred,readout=direct)
        row=dict(id=tag,spec=spec,metrics=metrics)
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as file:
            file.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,spec,metrics["all"]["residual_correlation"],flush=True)
        if metrics["all"]["residual_correlation"]>=.700: break
    best=max(rows,key=lambda row:row["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
