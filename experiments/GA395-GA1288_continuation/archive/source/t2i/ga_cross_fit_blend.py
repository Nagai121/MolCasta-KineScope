"""Fit and audit a second-stage blend on out-of-protein predictions."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import GroupKFold

from .ga_sequential50 import pool
from .ga_static_offset import estimates

R=Path("runs")
OUT=R/"ga1128-cross-fit-blend"
FAMILIES=("ga805-window-grouped","ga860-reliability-grouped","ga1104-grouped-capacity")


def main(nonlinear=False):
    out=Path("runs/ga1182-nonlinear-blend") if nonlinear else OUT
    out.mkdir(exist_ok=True)
    (x,d,_,y,o,names),(v,e,_,t,vo,_,base,parent)=pool()
    origin_delta=y-x[:,0]
    train,valid=[],[]
    members=[]
    for family in FAMILIES:
        folder=R/family
        picks=sorted(p.stem for p in folder.glob("GA*.pt"))
        assert len(picks)==3, (family,picks)
        forecasts=[]
        for tag in picks:
            with np.load(folder/f"{tag}-cv.npz") as z:
                np.testing.assert_array_equal(z["proteins"],names)
                np.testing.assert_array_equal(z["target"],y)
                forecasts.append(z["prediction"])
        train.append(np.mean(forecasts,axis=0)-x[:,0])
        with np.load(folder/"development.npz") as z:
            valid.append((z["prediction"]-.65*parent)/.35-v[:,0])
        members.append(picks)
    a,b=np.column_stack(train),np.column_stack(valid)
    contexts=np.column_stack((x[:,0],d[:,:6].mean(1),d[:,6:12].mean(1),d[:,-4:]))
    dev_context=np.column_stack((v[:,0],e[:,:6].mean(1),e[:,6:12].mean(1),e[:,-4:]))
    cv=list(GroupKFold(4).split(a,y,names))
    baseline=np.empty_like(y)
    for tr,va in cv:
        baseline[va]=Ridge(alpha=10).fit(x[tr,:12],y[tr]).predict(x[va,:12])

    def fit(kind, tr, q, cq):
        matrix,target=a[tr],origin_delta[tr]
        c=contexts[tr]
        if kind=="equal":
            return q.mean(1),{"kind":kind}
        if kind=="nnls":
            weight=nnls(matrix,target)[0]
            return q@weight,{"weight":weight.tolist()}
        if kind=="ridge":
            fit=Ridge(alpha=100).fit(matrix,target)
            return fit.predict(q),{"weight":fit.coef_.tolist(),"bias":fit.intercept_.item()}
        if kind=="huber":
            fit=HuberRegressor(alpha=15,max_iter=180).fit(matrix,target)
            return fit.predict(q),{"weight":fit.coef_.tolist(),"bias":float(fit.intercept_)}
        if kind=="contextual":
            def expand(p,c):
                return np.column_stack((p,(p[:,:,None]*c[:,None,:]).reshape(len(p),-1)))
            mu,s=c.mean(0),c.std(0).clip(.01)
            model=Ridge(alpha=350).fit(expand(matrix,((c-mu)/s).clip(-4,4)),target)
            out=model.predict(expand(q,((cq-mu)/s).clip(-4,4)))
            return out,{"weight":model.coef_.tolist(),"bias":model.intercept_.item(),
                        "context_mean":mu.tolist(),"context_scale":s.tolist()}
        if kind.startswith("xgb"):
            from xgboost import XGBRegressor
            _,depth,feature_set=kind.split("-")
            depth=int(depth)
            source=np.c_[matrix,c]
            query=np.c_[q,cq]
            if feature_set=="physical":
                # Corresponding static and dynamic terms are supplied with the
                # candidate distributions, using no other proteins at inference.
                source=np.c_[source,extra_train[tr]]
                query=np.c_[query,extra_query]
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=240,
                               max_depth=depth,learning_rate=.036,max_bin=96,n_jobs=3,
                               reg_lambda=25,subsample=.88)
            model.fit(source,target)
            assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
            model.set_params(device="cpu")
            return model.predict(query),model
        raise ValueError(kind)

    extra_train=np.c_[x[:,:14],x[:,-34:],d]
    extra_development=np.c_[v[:,:14],v[:,-34:],e]
    extra_query=None
    records=[]
    methods=("equal","nnls","ridge","huber","contextual")
    if nonlinear:
        methods=tuple(f"xgb-{depth}-{view}" for view in ("summary","physical") for depth in (2,4,6))
    for i,kind in enumerate(methods):
        cross=np.empty_like(y)
        for tr,va in cv:
            extra_query=extra_train[va]
            cross[va]=fit(kind,tr,a[va],contexts[va])[0]+x[va,0]
        folds=[estimates(cross[va],y[va],baseline[va])["residual_correlation"] for _,va in cv]
        tag=f"GA{(1182 if nonlinear else 1128)+i}"
        row={"id":tag,"kind":kind,"folds":folds,"cv_mean":float(np.mean(folds))}
        records.append(row)
        np.savez_compressed(out/f"{tag}-cv.npz",prediction=cross,target=y,baseline=baseline,proteins=names)
        print(row,flush=True)
    winner=max(records,key=lambda r:r["cv_mean"])
    extra_query=extra_development
    extra,weights=fit(winner["kind"],np.arange(len(y)),b,dev_context)
    if nonlinear:
        weights.save_model(out/f"{winner['id']}.ubj")
        weights={"model":f"{winner['id']}.ubj"}
    pred=.65*parent+.35*(extra+v[:,0])
    scores={key:estimates(pred[m],t[m],base[m]) for key,m in
            (("all",vo>=0),("first",vo==49),("later",vo!=49))}
    summary=dict(winner=winner,trials=records,development=scores,weights=weights,
                 families=FAMILIES,members=members)
    np.savez_compressed(out/"development.npz",prediction=pred,readout=extra+v[:,0])
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(scores,flush=True)


if __name__=="__main__":
    import sys
    main(nonlinear="--nonlinear" in sys.argv)

