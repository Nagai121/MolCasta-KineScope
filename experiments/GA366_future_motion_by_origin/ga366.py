"""GA366 compares true future motion by origin for information diagnosis only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .explore import record
from .ga315 import observed
from .ga_static_offset import VALID, TRAIN_GROUPS, cohort, estimates, extract, ridge

R=Path("runs")

def main():
    train=[r for group in TRAIN_GROUPS for r in cohort(group,1)]
    validation=cohort(VALID,3)
    _,x=extract(train)
    y,q=extract(validation)
    with ThreadPoolExecutor(4) as pool:
        a=np.concatenate(list(pool.map(observed,train)))
        b=np.concatenate(list(pool.map(observed,validation)))
    drift=b-ridge(x["compact"],a).predict(q["compact"])
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(z["target"],y,rtol=0,atol=3e-6)
        base,parent,origin=z["baseline"],z["parent"],z["origin"]
    out={}
    for key,m in (("first",origin==49),("later",origin!=49)):
        metrics={"ga348":estimates(parent[m],y[m],base[m])}
        for i,window in enumerate(("100ps","0-200ps","0-500ps","0-1000ps","600-1000ps")):
            metrics[window]=estimates(base[m]+drift[m,i],y[m],base[m])
        out[key]=dict(pairs=int(m.sum()),metrics=metrics)
    dest=R/"ga366"
    dest.mkdir(exist_ok=True)
    record(dest/"diagnostics.json",dict(scope="observed future unavailable during inference",results=out))
    print({n:{k:m["residual_correlation"] for k,m in v["metrics"].items()} for n,v in out.items()},flush=True)

if __name__=="__main__":
    main()
