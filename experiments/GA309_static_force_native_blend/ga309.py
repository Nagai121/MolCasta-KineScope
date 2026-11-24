"""GA309: validation-only blend of independently trained static and dynamic models."""
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga_force import build
from .ga_native_offset import HOLD, native
from tseries.data import TEST
from .ga_static_offset import VALID, cohort, estimates, extract


def predict(z, prefix, x):
    return (x-z[prefix+"center"])/z[prefix+"scale"] @ z[prefix+"weight"].T + z[prefix+"bias"]


def main():
    valid = cohort(VALID, 3)
    dest = Path("runs/ga309")
    y, x = extract(valid)
    def load(name):
        path = Path("runs/ga_native_offset") / name
        return np.load(path)
    with load("baselines.npz") as z, load("residual-ridge.npz") as m:
        s = x["compact"]
        base = predict(z,"real_baseline_",s)
        d = predict(m,"",native(valid)-predict(z,"native_baseline_",s))
    a = np.concatenate([build((p,1,3))[0] for p in VALID])
    model = XGBRegressor(device="cpu")
    model.load_model("runs/ga_force/all-8.ubj")
    f = model.predict(a)+a[:,0]-base
    result={}
    for w in np.linspace(-1.,1.,81):
        result[str(w)] = estimates(base+d+w*f,y,base)
    ranked=sorted(result,key=lambda k:result[k]["residual_correlation"],reverse=True)
    record(dest/"development.json",dict(trials=result,best=ranked[:5]))
    print(ranked[0],result[ranked[0]],flush=True)
    locked = float(ranked[0])
    evaluable = cohort(TEST, 3) + cohort(HOLD[:2], 3) + cohort(HOLD[2:], 2)
    target, views = extract(evaluable)
    structural = views["compact"]
    with load("baselines.npz") as z, load("residual-ridge.npz") as m:
        baseline = predict(z, "real_baseline_", structural)
        residual = predict(m, "", native(evaluable)-predict(z, "native_baseline_", structural))
    force = np.concatenate([build((p,1,3 if p in TEST+HOLD[:2] else 2))[0] for p in TEST+HOLD])
    extra = model.predict(force)+force[:,0]-baseline
    combined = baseline+residual+locked*extra
    report = dict(weight=locked, pairs=len(target), test=estimates(combined,target,baseline),
                  ga305=estimates(baseline+residual,target,baseline), proteins=TEST+HOLD)
    record(dest/"confirmation.json", report)
    np.savez(dest/"evaluation.npz", predicted=combined, target=target, static=baseline)
    print(report, flush=True)


if __name__ == "__main__":
    main()
