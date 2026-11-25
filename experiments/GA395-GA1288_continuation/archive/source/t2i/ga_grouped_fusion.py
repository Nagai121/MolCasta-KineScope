"""GA621+: protein-grouped selection of static/generated modality fusion."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch
from torch import nn

from .ga_sequential50 import pool
from .ga_static_offset import estimates

RUNS=Path("runs")
OUT=RUNS/"ga621-grouped-fusion"


class Fusion(nn.Module):
    def __init__(self,n,k,style,width=48):
        super().__init__()
        self.style=style
        self.static=nn.Sequential(nn.Linear(n,112),nn.LayerNorm(112),nn.SiLU(),
                                  nn.Linear(112,width),nn.LayerNorm(width),nn.SiLU())
        self.dynamic=nn.Sequential(nn.Linear(k,width),nn.LayerNorm(width),nn.SiLU())
        self.base=nn.Linear(width,1)
        self.mix=nn.Sequential(nn.Linear(2*width,64),nn.SiLU(),nn.Linear(64,1))
        self.gate=nn.Linear(2*width,1)
        self.out=nn.Linear(width,1)

    def forward(self,x,d):
        s=self.static(x)
        q=self.dynamic(d)
        base=self.base(s).flatten()
        if self.style=="uncoupled":
            return base+self.out(q).flatten(),base
        if self.style=="interaction":
            joint=torch.cat([s,q*(1+s.tanh())],1)
        else:
            joint=torch.cat([s,q],1)
        addition=self.mix(joint).flatten()
        if self.style=="gated":
            addition=addition*self.gate(joint).flatten().sigmoid()
        return base+addition,base


def main(extended=False):
    out = RUNS/"ga664-grouped-adaptive" if extended else OUT
    out.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,proteins),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(proteins)&set(vg)
    cv=list(GroupKFold(4).split(x,y,proteins))
    shift=y-x[:,0]
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}

    def run(train,val_x,val_d,spec):
        style,dropped,auxiliary,*tuning=spec
        width, learning, late, epochs=tuning or (48,.0018,3.,14)
        center,scale=x[train].mean(0),x[train].std(0).clip(.001)
        dm,ds=d[train].mean(0),d[train].std(0).clip(.001)
        def tensor(a,mu,sd,device):
            return torch.tensor(((a-mu)/sd).clip(-10,10),device=device,dtype=torch.float32)
        a=tensor(x[train],center,scale,"cuda")
        dyn=tensor(d[train],dm,ds,"cuda")
        target=torch.tensor(shift[train],device="cuda",dtype=torch.float32)
        w=torch.tensor(np.where(o[train]==49,1.,late),device="cuda",dtype=torch.float32)
        torch.manual_seed(621)
        model=Fusion(x.shape[1],d.shape[1],style,width).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=learning,weight_decay=.012)
        for _ in range(epochs):
            for i in torch.randperm(len(a),device="cuda").split(1536):
                values=dyn[i]
                if dropped:
                    values=values*(torch.rand((len(i),1),device="cuda")>=dropped)
                pred,only=model(a[i],values)
                loss=((pred-target[i]).square()*w[i]).mean()
                if auxiliary:
                    loss=loss+.25*((only-target[i]).square()*w[i]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        cpu=deepcopy(model).cpu().eval()
        with torch.inference_mode():
            p,_=cpu(tensor(val_x,center,scale,"cpu"),tensor(val_d,dm,ds,"cpu"))
        return p.numpy()+val_x[:,0],dict(state=cpu.state_dict(),center=center,scale=scale,dyn_mean=dm,dyn_scale=ds,spec=spec)

    specs=[(style,drop,auxiliary) for style in ("uncoupled","joint","gated","interaction")
           for drop,auxiliary in ((0.,False),(.2,True),(.45,True))]
    if extended:
        specs = [(style,0.,False,width,learning,late,epochs)
                 for style in ("interaction","uncoupled","gated")
                 for width,learning,epochs in ((32,.001,24),(64,.001,24),(96,.001,24),(64,.0025,12))
                 for late in (1.,4.)]
    refs=np.empty_like(y)
    for train,val in cv:
        reg=Ridge(alpha=10.).fit(x[train,:12],y[train])
        refs[val]=reg.predict(x[val,:12])
    history=[]
    for i in range(len(specs)):
        done={tuple(v["spec"]) for v in history}
        if i==0:
            spec=specs[0]
        else:
            best=max(history,key=lambda z:z["cv_mean"])
            spec=min((s for s in specs if s not in done),key=lambda s:(
                sum(a!=b for a,b in zip(s,best["spec"])),sum(r["spec"][0]==s[0] for r in history)))
        start=perf_counter()
        pred=np.empty_like(y)
        folds=[]
        for train,val in cv:
            pred[val],_=run(train,x[val],d[val],spec)
            folds.append(estimates(pred[val],y[val],refs[val])["residual_correlation"])
        model_id=f"GA{(664 if extended else 621)+i}"
        row=dict(id=model_id,spec=spec,cv_mean=float(np.mean(folds)),folds=folds,
                 cv_pooled=estimates(pred,y,refs),seconds=perf_counter()-start)
        history.append(row)
        np.savez_compressed(out/f"{model_id}-cv.npz",prediction=pred,target=y,baseline=refs,proteins=proteins)
        with (out/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
    best=max(history,key=lambda z:z["cv_mean"])
    direct,state=run(np.arange(len(x)),v,e,best["spec"])
    p=.65*parent+.35*direct
    best["development"]={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
    torch.save(state,out/f"{best['id']}.pt")
    np.savez_compressed(out/"development.npz",prediction=p,readout=direct)
    (out/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
    print(best,flush=True)


if __name__=="__main__":
    import sys
    main(extended="--extended" in sys.argv)

