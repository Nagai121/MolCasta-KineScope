"""Evaluate GA355 readouts with a locked validation-only selection rule."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_native_offset import HOLD
from .ga_static_offset import VALID

R = Path("runs")
DEST = R/"ga355"


def evaluate(names, amber, charmm, models):
    ref = components(names, amber, charmm, ("generated-6",), "all-9")
    target, base, *_, scores = ref
    x, d, y = inputs(names, amber, charmm)
    np.testing.assert_allclose(y,target,rtol=0,atol=3e-6)
    for name in models:
        head = XGBRegressor(device="cpu")
        head.load_model(DEST/f"{name}.ubj")
        view=np.c_[x[:, :3] if name.startswith("generated") else x,d]
        scores[f"scaled-{name}"]=head.predict(view)+x[:,0]-base
    return ref


def main():
    train = json.loads((DEST/"development.json").read_bytes())
    if len(train["trials"]) != 6:
        raise RuntimeError("GA355 training has not completed")
    models = tuple(train["trials"])
    ranks = []
    dev = evaluate(VALID,R/"ga310",R/"ga313",models)
    _,baseline=forecasts(dev,("generated-6",.25,.2))
    for name,w in product(models,(.10,.20,.35,.50)):
        rule=(f"scaled-{name}",.25,w)
        _,metrics=forecasts(dev,rule)
        ranks.append(dict(rule=rule,metrics=metrics))
    ranks.sort(key=lambda row: -row["metrics"]["residual_correlation"])
    best=ranks[0]
    delta=best["metrics"]["residual_correlation"]-baseline["residual_correlation"]
    record(DEST/"fusion.json",dict(baseline=baseline,best=best,trials=ranks,delta=delta))
    print(best,delta,flush=True)
    if delta<.003:
        return
    reserve=tuple(json.loads((R/"ga321/reserve.json").read_bytes())["proteins"])
    for tag,n,a,c in (
        ("previous",TEST+HOLD,R/"ga312",R/"ga321"),
        ("reserve",reserve,R/"ga321/reserve/amber",R/"ga321/reserve")):
        data=evaluate(n,a,c,(best["rule"][0].removeprefix("scaled-"),))
        y,b,g,*_=data
        p,m=forecasts(data,best["rule"])
        _,old=forecasts(data,("generated-6",.25,.2))
        record(DEST/f"{tag}.json",dict(rule=best["rule"],pairs=len(y),metrics=m,parent=old))
        np.savez(DEST/f"{tag}.npz",predicted=p,target=y,static=b,proteins=g)
        print(tag,m,flush=True)


if __name__=="__main__":
    main()
