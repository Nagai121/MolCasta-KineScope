"""Measure seed averaging against same-origin observations (development only)."""
from pathlib import Path
from itertools import combinations

import numpy as np

from .explore import record
from .ga309 import predict
from .ga_force import build
from xgboost import XGBRegressor
from .ga_static_offset import VALID, cohort, estimates, extract


def main():
    dest = Path("runs/ga310")
    rows = [r for r in cohort(VALID,1) if (dest/f"{r['protein']}-{r['origin']}.npz").exists()]
    y,x = extract(rows)
    s=x["compact"]
    with np.load("runs/ga_native_offset/baselines.npz") as m:
        base = predict(m,"real_baseline_",s)
        proj = predict(m,"native_baseline_",s)[:,0]
    branches = []
    for r in rows:
        with np.load(dest/f"{r['protein']}-{r['origin']}.npz") as z:
            np.testing.assert_array_equal(r["pair"], z["pair"])
            branches.append(z["tracks"].mean(1)-r["pair"][:,2][None,:])
    fields=np.concatenate(branches,axis=1)
    force = []
    for r in rows:
        a, t = build((r["protein"], 1, 1))
        np.testing.assert_allclose(r["pair"][:, 2], a[:,0], rtol=0, atol=1.e-6)
        force.append(a)
    force = np.concatenate(force)
    head = XGBRegressor(device="cpu")
    head.load_model("runs/ga_force/all-8.ubj")
    static = head.predict(force) + force[:,0] - base
    out={}
    for count in (1,2,3,4,6,8):
        metrics=[estimates(base+.4*(fields[list(ids)].mean(0)-proj),y,base)
                 for ids in combinations(range(len(fields)),count)]
        values=np.array([m["residual_correlation"] for m in metrics])
        hybrid=np.array([
            estimates(base+.4*(fields[list(ids)].mean(0)-proj)+.525*static,y,base)["residual_correlation"]
            for ids in combinations(range(len(fields)),count)
        ])
        out[str(count)] = dict(hybrid_mean=float(hybrid.mean()), combinations=len(values), mean=float(values.mean()),
                               minimum=float(values.min()), maximum=float(values.max()))
    report=dict(proteins=[r["protein"] for r in rows], pairs=len(y), results=out)
    record(dest/"development.json", report)
    print(report,flush=True)


if __name__ == "__main__":
    main()
