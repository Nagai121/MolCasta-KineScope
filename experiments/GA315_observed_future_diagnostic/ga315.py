"""Information diagnostic only: observed future trajectory versus past target."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tseries.data import registry, trajectory, cell
from tseries.features import _mic
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga_static_offset import VALID, TRAIN_GROUPS, cohort, estimates, extract, ridge


def observed(row):
    name = row["protein"]
    origin = row["origin"]
    reg = registry(ROOT,name)
    frames = next(reg.blocks(trajectory(name,1),start=origin,stop=origin+11,chunk_frames=11))
    if len(frames.time)!=11 or abs(frames.time[-1]-frames.time[0]-1000.) > .01:
        raise RuntimeError("Unexpected diagnostic sampling grid")
    i,j = row["pair"][:,:2].astype(int).T
    h = np.stack([np.linalg.norm(_mic(p[j]-p[i],cell(c)),axis=1)
                  for p,c in zip(frames.positions,frames.box)])
    return np.c_[h[1],h[:3].mean(0),h[:6].mean(0),h.mean(0),h[6:].mean(0)]


def main():
    train = [r for group in TRAIN_GROUPS for r in cohort(group,1)]
    valid = cohort(VALID,3)
    y,x=extract(train)
    t,z=extract(valid)
    s,sv=x["compact"],z["compact"]
    ref=ridge(s,y).predict(sv)
    with ThreadPoolExecutor(4) as pool:
        a=np.concatenate(list(pool.map(observed,train)))
        b=np.concatenate(list(pool.map(observed,valid)))
    p=ridge(s,a)
    dz=b-p.predict(sv)
    out={}
    for i,k in enumerate(("frame100ps","0-200ps","0-500ps","0-1000ps","600-1000ps")):
        out[k]=estimates(ref+dz[:,i],t,ref)
    dest=Path("runs/ga315")
    out=dict(purpose="observed future diagnostic only, excluded from predictors",
             proteins=VALID,pairs=len(t),results=out)
    record(dest/"observed_diagnostic.json",out)
    print(out,flush=True)


if __name__=="__main__":
    main()
