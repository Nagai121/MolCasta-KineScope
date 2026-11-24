"""GA386 compares later-origin supervision with matched first-origin controls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .explore import record
from .ga373 import assemble
from .ga_static_offset import VALID, estimates

R=Path("runs")

def main(expanded_pool=False):
    dest=R/("ga392/readout" if expanded_pool else "ga386")
    dest.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    lanes=[json.loads((R/f"ga385/lane-{i}.json").read_bytes()) for i in (0,1)]
    extra=tuple(lanes[0]["proteins"])
    assert not set(extra)&set(VALID)
    x,d,_,y,o,p=assemble(names,(49,),R/"ga333",R/"ga355")
    batches=[(extra,R/"ga365",R/"ga385")]
    if expanded_pool:
        state=json.loads((R/"ga392/lane-0.json").read_bytes())
        more=tuple(state["proteins"])
        assert not set(extra)&set(more)
        assert not set(more)&set(VALID)
        batches.append((more,R/"ga392/amber",R/"ga392/charmm"))
    extra_batches=[]
    for roster,amber,charmm in batches:
        for origin in (149,249):
            found=[n for n in roster if (charmm/f"{n}-{origin}.npz").exists()
                   and (amber/f"{n}-{origin}.npz").exists()]
            if found:
                a,b,_,t,u,g=assemble(found,(origin,),amber,charmm)
                extra_batches.append((a,b,t,u,g))
    if not extra_batches:
        raise RuntimeError("No matched later training data")
    a,b,h,orig,g=tuple(np.concatenate(z) for z in zip(*extra_batches))
    v,e,_,t,vo,vg=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(vg,z["proteins"])
        np.testing.assert_array_equal(vo,z["origin"])
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        parent,base=z["parent"],z["baseline"]
    V=np.c_[v,e]
    anchored={}
    masks={"all":np.ones(len(t),bool),"first":vo==49,"later":vo!=49}
    specs={"control":(x,d,y,o),"expanded":(np.concatenate((x,a)),np.concatenate((d,b)),
                 np.concatenate((y,h)),np.concatenate((o,orig)))}
    for kind,(q,rep,truth,origins) in specs.items():
        inp=np.c_[q,rep]
        c,s=inp.mean(0),inp.std(0).clip(.001)
        A=torch.tensor(((inp-c)/s).clip(-12,12),device="cuda",dtype=torch.float32)
        B=torch.tensor(((V-c)/s).clip(-12,12),dtype=torch.float32)
        target=torch.tensor(truth-q[:,0],device="cuda")
        for later in ((0.,) if kind=="control" else (0.,1.,3.)):
            weights=torch.tensor(1.+later*(origins!=49),device="cuda",dtype=torch.float32)
            weight=weights/weights.mean()
            torch.manual_seed(384)
            model=nn.Sequential(nn.Linear(inp.shape[1],96),nn.LayerNorm(96),nn.SiLU(),
                    nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2)).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for ep in range(1,41):
                model.train()
                for idx in torch.randperm(len(A),device="cuda").split(768):
                    yhat=model(A[idx])[:,0]
                    loss=(nn.functional.smooth_l1_loss(yhat,target[idx],reduction="none",beta=.4)*weight[idx]).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if ep not in (10,20,30,40):
                    continue
                cpu=deepcopy(model).cpu().eval()
                with torch.no_grad():
                    score=cpu(B).numpy()[:,0]+v[:,0]
                key=f"{kind}-{later}-{ep}"
                for share in (.1,.2,.35):
                    yhat=(1-share)*parent+share*score
                    anchored[f"{key}/{share}"]={k:estimates(yhat[m],t[m],base[m]) for k,m in masks.items()}
                torch.save(dict(state=cpu.state_dict(),center=c,scale=s),dest/f"{key}.pt")
    best=max(anchored,key=lambda k:anchored[k]["all"]["residual_correlation"])
    record(dest/"development.json",dict(best=best,trials=anchored,training_first=len(y),
                                         training_later=len(h),later_proteins=sorted(set(g))))
    print(best,anchored[best],flush=True)

if __name__=="__main__":
    main()

