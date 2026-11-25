"""GA1247+: learn adapters on proteins absent from the frozen GA386 readout."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from xgboost import XGBRegressor

from .ga_competitive_capacity import competition
from .ga_sequential50 import features,pool
from .ga_static_offset import estimates

R=Path("runs")
OUT=R/"ga1247-frozen-adapter"


def load_anchor():
    folder=R/"ga386"
    key,share=json.loads((folder/"development.json").read_bytes())["best"].split("/")
    state=torch.load(folder/f"{key}.pt",map_location="cpu",weights_only=False)
    width=len(state["center"])
    model=nn.Sequential(nn.Linear(width,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),
                        nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2)).eval()
    model.load_state_dict(state["state"])
    return model,state,float(share)


def readout(model,state,x,d):
    a=np.c_[x,d]
    a=np.clip((a-state["center"])/state["scale"],-12,12).astype(np.float32)
    with torch.inference_mode():
        return model(torch.from_numpy(a)).numpy()[:,0]+x[:,0]


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    _,(v,e,h,t,vo,vg,base,parent)=pool()
    with np.load(R/"ga1029-later-expansion/train-extra.npz") as z:
        x,d,g,y,o,p=[z[k] for k in ("x","d","g","y","origins","proteins")]
    trained=set(json.loads((R/"ga355/development.json").read_bytes())["successful"])
    trained.update(json.loads((R/"ga385/lane-0.json").read_bytes())["proteins"])
    assert not set(p)&(trained|set(vg))
    locked,state,share=load_anchor()
    pred=readout(locked,state,x,d)
    val=readout(locked,state,v,e)
    vintage=(1-share)*parent+share*val
    rows=[]
    train_views={
        "raw":np.c_[features(x,d,g,"level"),pred-x[:,0]],
        "angles":np.c_[features(competition(x,d,o,p,oriented=True),d,g,"level"),pred-x[:,0]],
    }
    val_views={
        "raw":np.c_[features(v,e,h,"level"),val-v[:,0]],
        "angles":np.c_[features(competition(v,e,vo,vg,oriented=True),e,h,"level"),val-v[:,0]],
    }
    splits=list(GroupKFold(4).split(x,y,p))
    residual=y-pred
    for i,(view,depth) in enumerate((k,d) for k in train_views for d in (2,4,6)):
        a,b=train_views[view],val_views[view]
        def fitted(indices):
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=300,max_depth=depth,
                               learning_rate=.031,max_bin=96,n_jobs=3,subsample=.88,reg_lambda=25)
            model.fit(a[indices],residual[indices])
            assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
            model.set_params(device="cpu")
            return model
        inf=np.empty_like(y)
        for tr,va in splits:
            inf[va]=fitted(tr).predict(a[va])
        weight=min((0.,.25,.5,.75,1.),key=lambda w:np.mean((residual-w*inf)**2))
        model=fitted(np.arange(len(y)))
        changed=vintage+share*weight*model.predict(b)
        masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
        metrics={k:estimates(changed[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{1247+i}"
        model.save_model(OUT/f"{tag}.ubj")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=changed,anchor=vintage)
        row=dict(id=tag,view=view,depth=depth,blend=weight,metrics=metrics,
                 training_cv_rmse=float(np.sqrt(np.mean((residual-weight*inf)**2))))
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
    summary=dict(best=min(rows,key=lambda r:r["training_cv_rmse"]),train_proteins=sorted(set(p)),
                 train_pairs=len(y),frozen_share=share,baseline=estimates(vintage,t,base),trials=rows)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
