"""Confirm the frozen GA384 candidate on the two previously reported panels."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from tseries.data import TEST
from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_native_offset import HOLD
from .ga_static_offset import estimates

R=Path("runs")
D=R/"ga384"

def main(blind=False):
    info=json.loads((D/"development.json").read_bytes())
    kind,hetero,epoch,weight,gated=info["best"].split("/")
    if gated!="False":
        raise RuntimeError("This confirmation requires the frozen fixed-weight model")
    state=torch.load(D/f"{kind}-{hetero}-{epoch}.pt",map_location="cpu",weights_only=False)
    c,s=state["center"],state["scale"]
    head=nn.Sequential(nn.Linear(len(c),96),nn.LayerNorm(96),nn.SiLU(),
                       nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2))
    head.load_state_dict(state["state"])
    head.eval()
    reserve=tuple(json.loads((R/"ga321/reserve.json").read_bytes())["proteins"])
    panels=(
        ("previous",TEST+HOLD,R/"ga312",R/"ga321"),
        ("reserve",reserve,R/"ga321/reserve/amber",R/"ga321/reserve"))
    if blind:
        plan=json.loads((R/"ga351/reserved.json").read_bytes())
        roster=tuple(p["id"] for p in plan["proteins"])
        assert not set(roster)&(set(TEST+HOLD+reserve)|set(info["train"]))
        panels=(("blind-ga351",roster,R/"ga351/blind/amber",R/"ga351/blind/charmm"),)
    for name,roster,a,b in panels:
        data=components(roster,a,b,("generated-6",),"all-9")
        parent,_=forecasts(data,("generated-6",.25,.2))
        t,base,grp,*_=data
        x,d,y=inputs(roster,a,b)
        np.testing.assert_allclose(t,y,rtol=0,atol=3e-6)
        feature=np.c_[x[:,:3] if kind=="motion" else x,d]
        with torch.no_grad():
            p=head(torch.tensor(((feature-c)/s).clip(-12,12),dtype=torch.float32)).numpy()[:,0]+x[:,0]
        w=float(weight)
        q=(1-w)*parent+w*p
        out=dict(pairs=len(t),rule=info["best"],metrics=estimates(q,t,base),
                 parent=estimates(parent,t,base))
        record(D/f"{name}.json",out)
        np.savez(D/f"{name}.npz",predicted=q,target=t,static=base,proteins=grp)
        print(name,out,flush=True)

if __name__=="__main__":
    main()

