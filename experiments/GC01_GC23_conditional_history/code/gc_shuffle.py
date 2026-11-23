"""GC18: keep the history distribution but scramble candidate identity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tseries.hb_validation10 import panel10
from .frontends import data, design, infer
from .explore import record, score
from .gc_history import past, stats


def main():
    torch.set_num_threads(3)
    dest=Path("runs/gc")
    rows,_,base=data(panel10())
    with ThreadPoolExecutor(3) as pool:
        h=np.concatenate(list(pool.map(past,rows)))
    d=np.concatenate([p["pair"][:,2] for p in rows])
    protein=np.concatenate([np.full(len(p["y"]),i//3,dtype=int) for i,p in enumerate(rows)])
    distance=np.floor((d-3.5)/.15).astype(int).clip(0,17)
    odds=np.floor((design(d[:,None],base)[:,0]+15)/.3).astype(int).clip(0,100)
    key=protein*18*101+distance*101+odds
    order=np.argsort(key,kind="stable")
    groups=np.split(order,np.flatnonzero(np.diff(key[order]))+1)
    distinct=[g for g in groups if len(g)>1]
    rng=np.random.default_rng(181)
    checkpoint=dest/"GC10-500ps-all.pt"
    def evaluate(values):
        p=infer(design(stats(values,d,5)["all"],base),checkpoint)
        return score(rows,p)
    out=[]
    for _ in range(10):
        perm=np.arange(len(h))
        for g in distinct:
            g=rng.permutation(g)
            perm[g]=np.roll(g,1)
        swapped=h[perm]-d[perm,None]+d[:,None]
        out.append(evaluate(swapped))
    record(dest/"GC18.json",dict(baseline=score(rows,base),clean=evaluate(h),shuffled=out,
                                 shuffled_candidates=sum(map(len,distinct)),
                                 distance_width=.15,logit_width=.3))
    print("Completed GC18",flush=True)


if __name__=="__main__":
    main()
