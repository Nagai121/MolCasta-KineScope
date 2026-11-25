"""Static origin input with independently generated solvent-MD features."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga_static_offset import TRAIN_GROUPS,VALID,cohort,extract,estimates,ridge

DEST=Path("runs/ga_native_offset")
CACHE=Path("runs/ga_native240")
HOLD=("4ftf_A","3msw_A","3wx4_A","4wlr_B")


def native(rows):
    fields=[]
    for r in rows:
        with np.load(CACHE/f"{r['protein']}-{r['origin']}.npz") as z:
            if not np.array_equal(z["pair"],r["pair"]):
                raise RuntimeError("Native ROD candidates changed")
            h=z["tracks"]
        d=r["pair"][:,2]
        delta=h-d
        fields.append(np.c_[delta.mean((0,1)),delta.mean(1).T,delta[:,:5].mean((0,1)),
                            delta[:,-5:].mean((0,1)),delta.std((0,1)),
                            delta.min((0,1)),delta.max((0,1)),(h<=3.5).mean((0,1))])
    return np.concatenate(fields).astype(np.float32)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    train=[r for group in TRAIN_GROUPS for r in cohort(group,1)]
    valid=cohort(VALID,3)
    test=cohort(TEST,3)+cohort(HOLD[:2],3)+cohort(HOLD[2:],2)
    assert not set(r["protein"] for r in train)&set(r["protein"] for r in valid+test)
    yt,xt=extract(train)
    yv,xv=extract(valid)
    ye,xe=extract(test)
    reference=ridge(xt["compact"],yt)
    rt,rv,re=[reference.predict(x["compact"]) for x in (xt,xv,xe)]
    a,b,c=[native(x) for x in (train,valid,test)]
    static=tuple(x["compact"] for x in (xt,xv,xe))
    proj=ridge(static[0],a)
    transforms={}
    for label,fitted in (("real_baseline",reference),("native_baseline",proj)):
        prep,head=fitted.steps[0][1],fitted.steps[1][1]
        transforms.update({f"{label}_center":prep.mean_,f"{label}_scale":prep.scale_,
                           f"{label}_weight":head.coef_,f"{label}_bias":head.intercept_})
    np.savez(DEST/"baselines.npz",**transforms)
    ra,rb,rc=[x-proj.predict(y) for x,y in zip((a,b,c),static)]
    oof=np.empty(len(yt),np.float32)
    groups=np.concatenate([np.full(len(r["y"]),r["protein"]) for r in train])
    for protein in np.unique(groups):
        member=groups==protein
        oof[member]=ridge(static[0][~member],yt[~member]).predict(static[0][member])
    choices={}
    packets={"mean":tuple(x[:,:1] for x in (ra,rb,rc)),
             "raw":(a,b,c),"residual":(ra,rb,rc),
             "hybrid":tuple(np.c_[x,y,z] for x,y,z in zip(static,(a,b,c),(ra,rb,rc))),
             "rich":tuple(np.c_[x["structural"],r] for x,r in zip((xt,xv,xe),(ra,rb,rc)))}
    ledger=dict(trials={},test_proteins=TEST+HOLD,criterion=.5)
    for k,(t,v,h) in packets.items():
        for method in ("ridge","tree2","tree4"):
            key=f"{k}-{method}"
            if method=="ridge":
                model=ridge(t,yt-oof)
                transformer,head=model.steps[0][1],model.steps[1][1]
                np.savez(DEST/f"{key}.npz",center=transformer.mean_,scale=transformer.scale_,
                         weight=head.coef_,bias=head.intercept_)
            else:
                model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=250,
                                   learning_rate=.045,max_depth=int(method[-1]),
                                   subsample=.85,colsample_bytree=.83,random_state=113,n_jobs=4)
                model.fit(t,yt-oof)
                model.set_params(device="cpu")
                model.save_model(DEST/f"{key}.ubj")
            val=model.predict(v)+rv
            choices[key]=(model,h)
            ledger["trials"][key]=dict(validation=estimates(val,yv,rv))
            print(key,ledger["trials"][key],flush=True)
    # Direct-shape controls select amplitude on the same validation rows.
    for i,scale in enumerate((.25,.5,.75,1.)):
        key=f"native-offset-{i}"
        choices[key]=(None,scale)
        ledger["trials"][key]=dict(validation=estimates(rv+scale*rb[:,0],yv,rv),scale=scale)
    winner=max(choices,key=lambda k:ledger["trials"][k]["validation"]["residual_correlation"])
    predictor,features=choices[winner]
    pred=re+(features*rc[:,0] if predictor is None else predictor.predict(features))
    ledger["selected"]=winner
    ledger["test_metrics"]=estimates(pred,ye,re)
    np.savez(DEST/"evaluation.npz",predicted=pred,target=ye,static=re,
             proteins=np.concatenate([np.full(len(r["y"]),r["protein"]) for r in test]),
             origin=np.concatenate([np.full(len(r["y"]),r["origin"]) for r in test]),
             pair=np.concatenate([r["pair"][:,:2] for r in test]))
    record(DEST/"results.json",ledger)
    print(winner,ledger["test_metrics"],flush=True)


if __name__=="__main__":
    main()
