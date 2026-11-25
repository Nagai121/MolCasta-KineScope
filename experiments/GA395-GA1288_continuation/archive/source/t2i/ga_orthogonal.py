"""GA647+: remove static-predictable components from generated pair responses."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga647-orthogonal")


def main():
    OUT.mkdir(exist_ok=True)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    raw=features(x,d,g,"level")[:,x.shape[1]:]
    val=features(v,e,h,"level")[:,x.shape[1]:]
    folds=list(GroupKFold(4).split(x,y,p))
    y0=y-x[:,0]
    center,scale=x.mean(0),x.std(0).clip(.001)
    a=((x-center)/scale).clip(-8,8)
    b=((v-center)/scale).clip(-8,8)
    rng=np.random.default_rng(647)
    proj=rng.standard_normal((a.shape[1],128),dtype=np.float32) / np.sqrt(a.shape[1])
    phase=rng.uniform(0,2*np.pi,size=128).astype(np.float32)
    representations={
        "raw":(a,b),
        "wide":(np.c_[a,np.cos(a@proj+phase)],np.c_[b,np.cos(b@proj+phase)]),
        "compact":(a[:,:12],b[:,:12]),
    }
    rows=[]
    for tag,(z,w) in representations.items():
        forecast=np.empty_like(raw)
        static=np.empty_like(y)
        for train,valid in folds:
            remap=Ridge(alpha=30.).fit(z[train],raw[train])
            forecast[valid]=remap.predict(z[valid])
            nuisance=Ridge(alpha=30.).fit(z[train],y0[train])
            static[valid]=nuisance.predict(z[valid])
        remap=Ridge(alpha=30.).fit(z,raw)
        nuisance=Ridge(alpha=30.).fit(z,y0)
        valres=val-remap.predict(w)
        offset=nuisance.predict(w)
        np.savez_compressed(OUT/f"projection-{tag}.npz",normalize_center=center,normalize_scale=scale,
                            random_projection=proj,phase=phase,dyn_coef=remap.coef_,
                            dyn_intercept=remap.intercept_,target_coef=nuisance.coef_,
                            target_intercept=nuisance.intercept_)
        for strength,depth in ((.5,4),(1.,4),(.5,7),(1.,7)):
            aa=np.c_[x,raw-strength*forecast]
            bb=np.c_[v,val-strength*(val-valres)]
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=450,
                                 max_depth=depth,max_bin=96,learning_rate=.038,n_jobs=4,
                                 subsample=.86,colsample_bytree=.9,reg_lambda=12,random_state=647)
            tic=perf_counter()
            model.fit(aa,y0-static,sample_weight=np.where(o==49,1.,4.))
            assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
            model.set_params(device="cpu")
            direct=model.predict(bb)+offset+v[:,0]
            pred=.65*parent+.35*direct
            metrics={k:estimates(pred[m],t[m],base[m]) for k,m in
                     (("all",vo>=0),("first",vo==49),("later",vo!=49))}
            mid=f"GA{647+len(rows)}"
            model.save_model(OUT/f"{mid}.ubj")
            np.savez_compressed(OUT/f"{mid}.npz",prediction=pred,readout=direct)
            row=dict(id=mid,projection=tag,strength=strength,depth=depth,metrics=metrics,seconds=perf_counter()-tic)
            rows.append(row)
            with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
            print(mid,tag,strength,depth,metrics["all"]["residual_correlation"],flush=True)
            if metrics["all"]["residual_correlation"]>=.700:
                (OUT/"summary.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
                return
    best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

