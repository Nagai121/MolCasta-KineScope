"""GC14: recover the deployment-relevant history summary from static geometry."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tseries.hb_single_structure import OUT
from tseries.hb_validation10 import panel10
from .frontends import data, design, infer
from .explore import record, score
from .gc_history import past, stats
from .gc_assist import matched


def main():
    torch.set_num_threads(3)
    dest=Path("runs/gc")
    training=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    tr,a,base_t=data(training)
    va,b,base_v=data(panel10())
    with ThreadPoolExecutor(3) as pool:
        ht=np.concatenate(list(pool.map(past,tr)))
        hv=np.concatenate(list(pool.map(past,va)))
    dt,dv=[np.concatenate([p["pair"][:,2] for p in parts]) for parts in (tr,va)]
    yt=stats(ht,dt,5)
    yv=stats(hv,dv,5)
    packed=lambda s:np.column_stack((s["mean"],s["occupancy"])).astype(np.float32)
    result={}
    for family in ("distance","existing_static","combined"):
        x=design(a[family],base_t)
        q=design(b[family],base_v)
        center=torch.from_numpy(x.mean(0))
        scale=torch.from_numpy(np.maximum(x.std(0),.001))
        x=((torch.from_numpy(x)-center)/scale).clamp(-10,10).cuda()
        q=((torch.from_numpy(q)-center)/scale).clamp(-10,10)
        y=torch.from_numpy(packed(yt)).cuda()
        torch.manual_seed(141)
        model=nn.Sequential(nn.Linear(x.shape[1],48),nn.SiLU(),nn.Linear(48,2)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.002)
        for _ in range(6):
            for i in torch.randperm(len(x),device=x.device).split(4096):
                p=model(x[i])
                loss=F.smooth_l1_loss(p[:,0],y[i,0])+F.binary_cross_entropy_with_logits(p[:,1],y[i,1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.cpu().eval()
        checkpoint=dest/f"GC14-{family}.pt"
        torch.save(dict(state=model.state_dict(),center=center,scale=scale,width=q.shape[1]),checkpoint)
        with torch.inference_mode():
            p=model(q).numpy().copy()
        p[:,1]=1/(1+np.exp(-p[:,1]))
        predicted=design(p,base_v)
        readout=dest/"GC11-500ps-mean_and_occupancy.pt"
        p1=infer(predicted,readout)
        comparison=matched(va,base_v,{"frozen_t1":base_v,"inferred_summary":p1})
        result[family]=dict(mean_rmse=float(np.sqrt(np.mean((p[:,0]-yv["mean"][:,0])**2))),
                            occupancy_rmse=float(np.sqrt(np.mean((p[:,1]-yv["occupancy"][:,0])**2))),
                            metrics=score(va,p1),conditional=comparison["results"]["inferred_summary"],
                            checkpoint=checkpoint.name)
    observed=infer(design(packed(yv),base_v),dest/"GC11-500ps-mean_and_occupancy.pt")
    record(dest/"GC14.json",dict(baseline=score(va,base_v),observed=score(va,observed),results=result))
    print("Completed GC14",flush=True)


if __name__=="__main__":
    main()
