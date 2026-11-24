"""GA329: CUDA neural structural priors with frozen static-input features."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .explore import record
from .ga309 import predict
from .ga_force import build
from .ga_static_offset import VALID, cohort, estimates, extract


class Prior(nn.Module):
    def __init__(self, features, width, dropout):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(features,width),nn.LayerNorm(width),nn.SiLU())
        self.blocks=nn.ModuleList(
            nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.SiLU(),
                          nn.Dropout(dropout),nn.Linear(width,width)) for _ in range(3)
        )
        self.output=nn.Linear(width,1)

    def forward(self,x):
        x=self.input(x)
        for block in self.blocks:
            x=x+block(x)
        return self.output(x).squeeze(1)


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_float32_matmul_precision("high")
    dest=Path("runs/ga329")
    dest.mkdir(parents=True,exist_ok=True)
    roster=json.loads(Path("runs/ga318/development.json").read_bytes())["train"]
    tasks=[(n,r,4) for n in roster for r in (1,2,3)]
    with ThreadPoolExecutor(4) as pool:
        items=list(pool.map(build,tasks))
    x,y=[np.concatenate(a) for a in zip(*items)]
    del items
    t,v=extract(cohort(VALID,3))
    val=np.concatenate([build((n,1,3))[0] for n in VALID])
    a=np.mean(x,axis=0)
    b=np.std(x,axis=0).clip(1.e-3)
    np.savez(dest/"standardize.npz",center=a,scale=b)
    tx=torch.from_numpy(x).cuda()
    ty=torch.from_numpy(y-x[:,0]).cuda()
    del x,y
    center=torch.as_tensor(a,device="cuda")
    scale=torch.as_tensor(b,device="cuda")
    xv=((torch.as_tensor(val,device="cuda")-center)/scale).clamp(-10.,10.)
    tx.sub_(center).div_(scale).clamp_(-10.,10.)
    with np.load("runs/ga_native_offset/baselines.npz") as z:
        base=predict(z,"real_baseline_",v["compact"])
    ledger=dict(proteins=roster,rows=len(tx),device=torch.cuda.get_device_name(),trials={})
    for width,dropout,beta,lr in ((128,.08,.50,.001),(256,.10,.35,.0008),(256,.05,1.,.0008)):
        torch.manual_seed(329)
        model=Prior(tx.shape[1],width,dropout).cuda()
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.005)
        loss=nn.HuberLoss(delta=beta)
        tag=f"{width}-{dropout}-{beta}"
        ledger["trials"][tag]={}
        started=time.perf_counter()
        for ep in range(1,26):
            model.train()
            for idx in torch.randperm(len(tx),device="cuda").split(4096):
                optim.zero_grad(set_to_none=True)
                error=loss(model(tx[idx]),ty[idx])
                error.backward()
                optim.step()
            if ep in (5,10,15,20,25):
                model.eval()
                with torch.no_grad():
                    p=model(xv).cpu().numpy()+val[:,0]
                stats=estimates(p,t,base)
                ledger["trials"][tag][str(ep)]=dict(seconds=time.perf_counter()-started,**stats)
                torch.save(dict(width=width,dropout=dropout,features=tx.shape[1],
                                weights={k:w.detach().cpu() for k,w in model.state_dict().items()}),
                           dest/f"{tag}-{ep}.pt")
                record(dest/"development.json",ledger)
                print(tag,ep,stats,flush=True)


if __name__=="__main__":
    main()
