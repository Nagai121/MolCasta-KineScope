"""GA704+: initialize from GA594 and fine-tune on multiple history targets."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .ga_sequential50 import pool
from .ga_static_offset import estimates

RUNS=Path("runs")
OUT=RUNS/"ga704-transfer-multiwindow"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(p)&set(vg)
    with np.load(RUNS/"ga688-multiwindow/training-history-targets.npz") as z:
        np.testing.assert_array_equal(y,z["label"])
        target=z["multi"]
    source=torch.load(RUNS/"ga588-transfer/GA594.pt",map_location="cpu",weights_only=False)
    mu,sd=target.mean(0),target.std(0).clip(.01)
    predict_mean=torch.tensor(mu,device="cuda")
    predict_scale=torch.tensor(sd,device="cuda")
    def norm(values,c,s,device):
        return torch.tensor(((values-c)/s).clip(-10,10),device=device,dtype=torch.float32)
    a=norm(x,source["mu"],source["scale"],"cuda")
    b=norm(v,source["mu"],source["scale"],"cpu")
    u=norm(d,source["dynamic_mean"],source["dynamic_scale"],"cuda")
    q=norm(e,source["dynamic_mean"],source["dynamic_scale"],"cpu")
    labels=torch.tensor(target,device="cuda",dtype=torch.float32)
    later=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    width=source["width"]
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    specs=[(epoch,alpha,which) for epoch in (4,12,24) for alpha in (0.,.2,.6,1.)
           for which in ("all","short") if alpha or which=="all"]
    trials=[]
    for i in range(len(specs)):
        if not trials:
            spec=(4,0.,"all")
        else:
            tried={tuple(r["spec"]) for r in trials}
            best=max(trials,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in specs if s not in tried),
                     key=lambda s:(sum(a!=b for a,b in zip(s,best["spec"])),
                                   sum(r["spec"][1]==s[1] for r in trials)))
        epochs,alpha,which=spec
        torch.manual_seed(704)
        enc=nn.Sequential(nn.Linear(x.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                          nn.Linear(128,48),nn.LayerNorm(48),nn.SiLU())
        head=nn.Sequential(nn.Linear(48+d.shape[1],width),nn.LayerNorm(width),
                           nn.SiLU(),nn.Linear(width,7))
        enc.load_state_dict(source["encoder"])
        with torch.no_grad():
            for layer in ("0","1"):
                for param in ("weight","bias"):
                    getattr(head[int(layer)],param).copy_(source["head"][f"{layer}.{param}"])
            head[3].weight[0].copy_(source["head"]["3.weight"][0])
            head[3].bias[0].copy_(source["head"]["3.bias"][0])
        enc,head=enc.cuda(),head.cuda()
        optim=torch.optim.AdamW([*enc.parameters(),*head.parameters()],lr=.00065,weight_decay=.012)
        start=perf_counter()
        for _ in range(epochs):
            for idx in torch.randperm(len(a),device="cuda").split(1536):
                value=head(torch.cat([enc(a[idx]),u[idx]],1))
                truth=labels[idx]
                main=((value[:,0]-truth[:,0])/predict_scale[0]).square()
                limit=7 if which=="all" else 4
                sub=(value[:,1:limit]-(truth[:,1:limit]-predict_mean[1:limit])/predict_scale[1:limit]).square().mean(1)
                loss=((main+alpha*sub)*later[idx]).mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
        torch.cuda.synchronize()
        fit=perf_counter()-start
        ce,ch=deepcopy(enc).cpu().eval(),deepcopy(head).cpu().eval()
        with torch.inference_mode():
            direct=ch(torch.cat([ce(b),q],1)).numpy()[:,0]+v[:,0]
        pred=.65*parent+.35*direct
        scores={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        mid=f"GA{704+i}"
        torch.save(dict(encoder=ce.state_dict(),head=ch.state_dict(),mu=source["mu"],scale=source["scale"],
                        dynamic_mean=source["dynamic_mean"],dynamic_scale=source["dynamic_scale"],
                        width=width,blend=.35),OUT/f"{mid}.pt")
        np.savez_compressed(OUT/f"{mid}.npz",prediction=pred,readout=direct)
        row=dict(id=mid,spec=spec,metrics=scores,fit_seconds=fit)
        trials.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(mid,spec,scores["all"]["residual_correlation"],flush=True)
        if scores["all"]["residual_correlation"]>=.700:
            break
    best=max(trials,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

