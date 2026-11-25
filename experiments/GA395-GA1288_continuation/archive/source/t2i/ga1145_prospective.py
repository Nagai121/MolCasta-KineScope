"""Lock the best GA1086 family member before acquiring an independent panel."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga351_generate import main as generate
from .ga348 import components, forecasts
from .ga373 import assemble
from .ga351 import main as reserve
from .ga351_acquire import main as acquire
from .ga_competitive_capacity import competition

OUT=Path("runs/ga1145-prospective")
from .ga_sequential50 import features
from .ga_static_offset import estimates

RUNS=Path("runs")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--lock-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not (OUT/"reserved.json").exists():
        reserve(OUT/"reserved.json","GA-1145/")
    training=RUNS/"ga1077-reliability-width"
    selected=json.loads((training/"summary.json").read_bytes())
    roster=tuple(row["id"] for row in json.loads((OUT/"reserved.json").read_bytes())["proteins"])
    learning=json.loads((RUNS/"ga445-494/summary.json").read_bytes())
    assert not set(roster)&(set(learning["train_proteins"])|set(learning["validation_proteins"]))
    checkpoint=training/f"{selected['id']}.pt"
    lock=dict(id=selected["id"],validation=selected["metrics"]["all"]["residual_correlation"],
              reference="frozen static Ridge",parents="GA348 fixed 65 percent",
              origin=49,replica=1,response="Amber 8 and CHARMM 2, 100 ps; readout consumes the first 2 of both at <=25 ps",
              proteins=list(roster),goal=.700,checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest())
    path=OUT/"model-lock.json"
    if path.exists():
        assert json.loads(path.read_bytes())==lock,"Model or protocol changed after registration"
    else:
        path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    if args.lock_only:
        print(lock,flush=True)
        return
    if not (OUT/"acquisition.json").exists():
        acquire(OUT)
    acquisition=json.loads((OUT/"acquisition.json").read_bytes())["results"]
    if any(len(acquisition.get(n,{}).get("frames",{}))!=3 for n in roster):
        raise RuntimeError("One or more reserved proteins are not ready")
    torch.set_num_threads(3)
    from .frontends import data as frontends
    print("Preparing fixed ROD candidate data",flush=True)
    frontends(roster)
    for field in ("amber","charmm"):
        print("Starting",field,flush=True)
        generate(OUT,field,resolve_terminal_histidine=True)
    amber,charmm=[OUT/"blind"/k for k in ("amber","charmm")]
    data=components(roster,amber,charmm,("generated-6",),"all-9")
    parent,_=forecasts(data,("generated-6",.25,.2))
    y,base,groups,*_=data
    x,d,g,target,_,p=assemble(roster,(49,),amber,charmm)
    np.testing.assert_array_equal(groups,p)
    np.testing.assert_allclose(y,target,rtol=0,atol=3e-6)
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    view,mixture,*_=state["spec"]
    x=competition(x,d,np.full(len(x),49),p,oriented=view=="angles")
    a=features(x,d,g,"level")
    width=128
    net=nn.Sequential(nn.Linear(a.shape[1],width),nn.LayerNorm(width),nn.SiLU(),
                      nn.Linear(width,width//2),nn.LayerNorm(width//2),nn.SiLU(),
                      nn.Linear(width//2,mixture*3+6)).eval()
    net.load_state_dict(state["state"])
    with torch.inference_mode():
        z=torch.tensor(((a-state["static_center"])/state["static_scale"]).clip(-10,10),dtype=torch.float32)
        out=net(z)
        expected=(out[:,:mixture].softmax(1)*out[:,mixture:2*mixture]).sum(1).numpy()
        direct=expected*state["target_scale"][0]+state["target_center"][0]+x[:,0]
    prediction=.65*parent+.35*direct
    result=dict(id=lock["id"],n_pairs=len(y),proteins=list(roster),
                metrics=estimates(prediction,y,base),parent=estimates(parent,y,base),
                by_protein={n:estimates(prediction[groups==n],y[groups==n],base[groups==n])
                            for n in roster if (groups==n).sum()>1})
    np.savez_compressed(OUT/"confirmation.npz",prediction=prediction,parent=parent,target=y,baseline=base,proteins=groups)
    (OUT/"confirmation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result,flush=True)


if __name__=="__main__":
    main()


