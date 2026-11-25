"""GA1246: matched historical comparison for already selected mixture checkpoints."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tseries.data import TEST
from .ga_competitive_capacity import competition
from .ga348 import components,forecasts
from .ga373 import assemble
from .ga_native_offset import HOLD
from .ga_sequential50 import features
from .ga_static_offset import estimates

R=Path("runs")
OUT=R/"ga1246-historical-comparison"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    ids={"GA1039":"ga1037-oriented-capacity","GA1086":"ga1077-reliability-width"}
    results={}
    panels={
        "historical":(TEST+HOLD,R/"ga312",R/"ga321"),
        "reserve":(tuple(json.loads((R/"ga321/reserve.json").read_bytes())["proteins"]),
                   R/"ga321/reserve/amber",R/"ga321/reserve"),
    }
    for group,(roster,amber,charmm) in panels.items():
        data=components(roster,amber,charmm,("generated-6",),"all-9")
        parent,_=forecasts(data,("generated-6",.25,.2))
        target,base,proteins,*_=data
        x,d,z,y,o,p=assemble(roster,(49,),amber,charmm)
        np.testing.assert_array_equal(proteins,p)
        np.testing.assert_allclose(y,target,rtol=0,atol=3e-6)
        x=competition(x,d,o,p,oriented=True)
        a=features(x,d,z,"level")
        results[group]={"GA348":estimates(parent,y,base)}
        predictions={"parent":parent,"target":y,"base":base,"proteins":proteins}
        for tag,folder in ids.items():
            state=torch.load(R/folder/f"{tag}.pt",map_location="cpu",weights_only=False)
            _,count,*_=state["spec"]
            model=nn.Sequential(nn.Linear(a.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                                nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,count*3+6))
            model.load_state_dict(state["state"])
            model.eval()
            q=np.clip((a-state["static_center"])/state["static_scale"],-10,10)
            with torch.inference_mode():
                out=model(torch.tensor(q,dtype=torch.float32))
                expected=(out[:,:count].softmax(1)*out[:,count:2*count]).sum(1).numpy()
            readout=x[:,0]+state["target_center"][0]+state["target_scale"][0]*expected
            pred=.65*parent+.35*readout
            predictions[tag]=pred
            results[group][tag]=estimates(pred,y,base)
        previous=R/"ga386"/("previous.json" if group=="historical" else "reserve.json")
        if previous.exists():
            recorded=json.loads(previous.read_bytes())
            results[group]["GA386_record"]=recorded.get("metrics")
        results[group]["pairs"]=len(y)
        np.savez_compressed(OUT/f"{group}.npz",**predictions)
        print(group,results[group],flush=True)
    (OUT/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
