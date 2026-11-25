"""Lock one transfer model on development before reading GA351 results."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga348 import components, forecasts
from .ga373 import assemble
from .ga_static_offset import estimates

RUNS = Path("runs")


def main():
    stages=[RUNS/"ga588-transfer",RUNS/"ga616-transfer-expanded"]
    candidates=[(p,json.loads((p/"summary.json").read_bytes())["best"]) for p in stages]
    path, row=max(candidates,key=lambda item:item[1]["metrics"]["all"]["residual_correlation"])
    model=row["id"]
    (path/"ga351-lock.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    roster=tuple(r["id"] for r in json.loads((RUNS/"ga351/reserved.json").read_bytes())["proteins"])
    amber,charmm=[RUNS/f"ga351/blind/{f}" for f in ("amber","charmm")]
    state=torch.load(path/f"{model}.pt",map_location="cpu",weights_only=False)
    data=components(roster,amber,charmm,("generated-6",),"all-9")
    parent,_=forecasts(data,("generated-6",.25,.2))
    x,d,_,y,_,p=assemble(roster,(49,),amber,charmm)
    target,base,groups,*_=data
    np.testing.assert_array_equal(groups,p)
    np.testing.assert_allclose(target,y,rtol=0,atol=3e-6)
    with np.load(RUNS/"ga384/blind-ga351.npz") as old:
        np.testing.assert_array_equal(groups,old["proteins"])
        np.testing.assert_allclose(base,old["static"],rtol=0,atol=3e-6)
        np.testing.assert_allclose(target,old["target"],rtol=0,atol=3e-6)
    encoder=nn.Sequential(nn.Linear(x.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                          nn.Linear(128,48),nn.LayerNorm(48),nn.SiLU()).eval()
    width=state["width"]
    head=nn.Sequential(nn.Linear(48+d.shape[1],width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,1)).eval()
    encoder.load_state_dict(state["encoder"])
    head.load_state_dict(state["head"])
    xnorm=torch.tensor(((x-state["mu"])/state["scale"]).clip(-10,10),dtype=torch.float32)
    dnorm=torch.tensor(((d-state["dynamic_mean"])/state["dynamic_scale"]).clip(-10,10),dtype=torch.float32)
    with torch.inference_mode():
        direct=head(torch.cat([encoder(xnorm),dnorm],1)).numpy().ravel()+x[:,0]
    w=state["blend"]
    pred=(1-w)*parent+w*direct
    result=dict(id=model,pairs=len(pred),proteins=list(roster),
                metrics=estimates(pred,target,base),parent=estimates(parent,target,base))
    (path/"ga351-confirmation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(path/"ga351-confirmation.npz",prediction=pred,parent=parent,target=target,
                        baseline=base,proteins=groups)
    print(result,flush=True)


if __name__=="__main__":
    main()
