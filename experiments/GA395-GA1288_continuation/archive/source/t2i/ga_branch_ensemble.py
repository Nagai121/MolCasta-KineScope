"""GA1259+: integrate the nonlinear readout over pairs of simulated Amber branches."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga333 import WINDOW
from .ga_competitive_capacity import competition
from .ga_sequential50 import features,pool
from .ga_static_offset import VALID,estimates

R=Path("runs")
OUT=R/"ga1259-branch-ensemble"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    _,(v,e,h,t,vo,vg,base,parent)=pool()
    source=R/"ga1077-reliability-width"
    state=torch.load(source/"GA1086.pt",map_location="cpu",weights_only=False)
    width=len(state["static_center"])
    count=state["spec"][1]
    net=nn.Sequential(nn.Linear(width,128),nn.LayerNorm(128),nn.SiLU(),
                      nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,count*3+6)).eval()
    net.load_state_dict(state["state"])
    amber=[]
    charmm=[]
    for name in VALID:
        for origin in (49,149,249):
            f=f"{name}-{origin}.npz"
            with np.load(R/"ga360/amber"/f) as a,np.load(R/"ga360/charmm"/f) as b:
                np.testing.assert_array_equal(a["pair"],b["pair"])
                ref=a["pair"][:,2]
                mask=(vg==name)&(vo==origin)
                np.testing.assert_array_equal(ref,v[mask,0])
                amber.append(a["tracks"][:8,:len(WINDOW)]-ref)
                charmm.append(b["tracks"][:2,:len(WINDOW)]-ref)
    amber,charmm=np.concatenate(amber,axis=2),np.concatenate(charmm,axis=2)
    groups=list(combinations(range(8),2))

    def summarize(a,b):
        x,y=a.mean(0).T,b.mean(0).T
        return np.c_[x,y,x-y,a.std((0,1)),b.std((0,1)),a.mean(1).std(0),b.mean(1).std(0)]

    def predict(d):
        oriented=competition(v,d,vo,vg,oriented=True)
        x=features(oriented,d,h,"level")
        x=((x-state["static_center"])/state["static_scale"]).clip(-10,10).astype(np.float32)
        with torch.inference_mode():
            out=net(torch.from_numpy(x))
            expectation=(out[:,:count].softmax(1)*out[:,count:2*count]).sum(1).numpy()
        return expectation*state["target_scale"][0]+state["target_center"][0]+v[:,0]

    np.testing.assert_allclose(summarize(amber[:2],charmm),e,rtol=0,atol=3e-6)
    members=np.stack([predict(summarize(amber[list(g)],charmm)) for g in groups])
    pooled=members.mean(0)
    anchors=members[0]
    with np.load(source/"GA1086.npz") as z:
        np.testing.assert_allclose(.65*parent+.35*anchors,z["prediction"],rtol=0,atol=3e-6)
    runs={
        "fixed_control":anchors,
        "readout_average":pooled,
        "readout_median":np.median(members,axis=0),
        "all_branches_one_readout":predict(summarize(amber,charmm)),
        "half_anchor":.5*(pooled+anchors),
        "soft_uncertainty":anchors+(pooled-anchors)*(.1/(.1+members.std(0))),
    }
    result={}
    for i,(name,readout) in enumerate(runs.items()):
        model=f"GA{1259+i}"
        prediction=.65*parent+.35*readout
        scores={k:estimates(prediction[m],t[m],base[m]) for k,m in
                (("all",vo>=0),("first",vo==49),("later",vo!=49))}
        result[model]=dict(method=name,metrics=scores,amber_group_count=len(groups))
        np.savez_compressed(OUT/f"{model}.npz",prediction=prediction,readout=readout)
        print(model,name,scores["all"]["residual_correlation"],flush=True)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
