"""GA354 averages nonlinear readouts over pairs of generated branches."""
from itertools import combinations
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import load
from .ga348 import components, forecasts
from .ga333 import WINDOW
from .ga_static_offset import VALID, estimates


ROOT = Path("runs")


def bag(a, c, x, model):
    a, c = a[:, :len(WINDOW)], c[:, :len(WINDOW)]
    outputs = []
    for first in combinations(range(len(a)), 2):
        u = a[list(first)]
        for second in combinations(range(len(c)), 2):
            v = c[list(second)]
            am, bm = u.mean(0).T, v.mean(0).T
            d = np.c_[am, bm, am-bm, u.std((0, 1)), v.std((0, 1)),
                      u.mean(1).std(0), v.mean(1).std(0)]
            outputs.append(model.predict(np.c_[x, d]))
    return np.mean(outputs, axis=0)


def main():
    from .ga_force import build
    dest = ROOT/"ga354"
    dest.mkdir(exist_ok=True)
    amber, charmm = ROOT/"ga310", ROOT/"ga313"
    a, y, baseline, _, _, _ = load(VALID, amber)
    c, *_ = load(VALID, charmm)
    x = np.concatenate([build((n, 1, 1))[0][:, :3] for n in VALID])
    model = XGBRegressor(device="cpu")
    model.load_model(ROOT/"ga344/generated-6.ubj")
    ref = components(VALID, amber, charmm, ("generated-6",), "all-9")
    parent, parent_metrics = forecasts(ref, ("generated-6", .25, .2))
    prediction = parent+.2*(bag(a, c, x, model)+x[:,0]-baseline-ref[-1]["generated-6"])
    out = dict(parent=parent_metrics, bagged=estimates(prediction,y,baseline), pairs=len(y),
               amber_combinations=28, charmm_combinations=1)
    record(dest/"development.json",out)
    print(out,flush=True)


if __name__ == "__main__":
    main()

