"""GC22–23: temporal order ablation and per-statistic identity substitution."""
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
    parts,_,base=data(panel10())
    with ThreadPoolExecutor(3) as pool:
        history=np.concatenate(list(pool.map(past,parts)))[:,-5:]
    distance=np.concatenate([p["pair"][:,2] for p in parts])
    checkpoint=dest/"GC10-500ps-all.pt"
    def evaluate(x):
        return score(parts,infer(design(x,base),checkpoint))
    clean=stats(history,distance,5)["all"]
    rng=np.random.default_rng(227)
    ordered=[]
    for _ in range(5):
        order=np.argsort(rng.random(history.shape),axis=1)
        shuffled=np.take_along_axis(history,order,axis=1)
        ordered.append(evaluate(stats(shuffled,distance,5)["all"]))
    record(dest/"GC22.json",dict(unchanged=evaluate(clean),
                                reversed=evaluate(stats(history[:,::-1],distance,5)["all"]),
                                randomized=ordered))
    movements=clean[:,[3,4,5]]
    rows={}
    for name,cols in (("mean",[0]),("minimum",[1]),("occupancy",[2]),("temporal",[3,4,5])):
        ablated=clean.copy()
        ablated[:,cols]=0.
        rows[name]=evaluate(ablated)
    rows["all"]=evaluate(clean)
    rows["none"]=evaluate(np.zeros_like(clean))
    record(dest/"GC23.json",dict(rows=rows))
    print("Completed GC22 and GC23",flush=True)


if __name__=="__main__":
    main()
