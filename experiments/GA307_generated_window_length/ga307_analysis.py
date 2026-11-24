"""Compare matched simulation windows without fitting on the assessed protein."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga309 import predict
from .ga307 import TIMES
from .ga_static_offset import VALID, cohort, estimates, extract, ridge


def main():
    dest = Path("runs/ga307/2fs")
    rows = [r for r in cohort(VALID, 3) if (dest/f"{r['protein']}-{r['origin']}.npz").exists()]
    if not rows:
        raise RuntimeError("No completed runs")
    y, x = extract(rows)
    s = x["compact"]
    source = Path("runs/ga_native_offset/baselines.npz")
    with np.load(source) as a:
        fixed_projection = predict(a,"native_baseline_",s)[:,0]
        base = ((s-a["real_baseline_center"])/a["real_baseline_scale"] @
                a["real_baseline_weight"] + a["real_baseline_bias"])
    group = np.concatenate([np.repeat(r["protein"], len(r["y"])) for r in rows])
    fields = {}
    for window in (10, 25, 50, 100, 200, 300, 500):
        for mode in ("mean", "late", "trapezoid"):
            blocks = []
            for r in rows:
                path = dest/f"{r['protein']}-{r['origin']}.npz"
                with np.load(path) as a:
                    np.testing.assert_array_equal(a["pair"], r["pair"])
                    h = a["tracks"]
                grid = np.array(TIMES)
                take = grid <= window
                if mode == "late":
                    take &= grid > window/2
                a = h[:, take].mean(0)
                if mode == "trapezoid":
                    value = np.trapezoid(a, grid[take], axis=0)/(grid[take][-1]-grid[take][0])
                else:
                    value = a.mean(0)
                blocks.append(value-r["pair"][:, 2])
            field = np.concatenate(blocks)
            field_metrics = {"fixed_100ps_projection": estimates(base+field-fixed_projection,y,base)}
            prediction = np.empty_like(field)
            for g in np.unique(group):
                select = group == g
                if np.all(select):
                    break
                p = ridge(s[~select],field[~select])
                prediction[select] = field[select]-p.predict(s[select])
            if len(np.unique(group)) >= 2:
                field_metrics["development_lopo_projection"] = estimates(base+prediction,y,base)
            fields[f"{window}/{mode}"] = field_metrics
    record(dest/"development.json", dict(pairs=len(y), proteins=np.unique(group).tolist(), probes=fields))
    print(sorted(fields.items(), key=lambda item: -item[1]["fixed_100ps_projection"]["residual_correlation"])[:8], flush=True)


if __name__ == "__main__":
    main()
