"""Confirm the validation-selected GA386 candidate on two held-out panels."""
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
D=R/"ga386"

def main():
    info=json.loads((D/"development.json").read_bytes())
    model,weight=info["best"].split("/")
    state=torch.load(D/f"{model}.pt",map_location="cpu",weights_only=False)
    c,s=state["center"],state["scale"]
    head=nn.Sequential(nn.Linear(len(c),96),nn.LayerNorm(96),nn.SiLU(),
                       nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2))
    head.load_state_dict(state["state"])
    head.eval()
    reserve=tuple(json.loads((R/"ga321/reserve.json").read_bytes())["proteins"])
    panels=(
        ("previous",TEST+HOLD,R/"ga312",R/"ga321"),
        ("reserve",reserve,R/"ga321/reserve/amber",R/"ga321/reserve"))
    for name,roster,a,b in panels:
        data=components(roster,a,b,("generated-6",),"all-9")
        parent,_=forecasts(data,("generated-6",.25,.2))
        t,base,grp,*_=data
        x,d,y=inputs(roster,a,b)
        np.testing.assert_allclose(t,y,rtol=0,atol=3e-6)
        feature=np.c_[x,d]
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


