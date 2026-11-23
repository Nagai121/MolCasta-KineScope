"""GC12–13: sensitivity of the frozen measured-history correction."""
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
        history=np.concatenate(list(pool.map(past,parts)))
    distance=np.concatenate([p["pair"][:,2] for p in parts])
    checkpoint=dest/"GC10-500ps-all.pt"
    clean=score(parts,infer(design(stats(history,distance,5)["all"],base),checkpoint))
    rows={}
    rng=np.random.default_rng(490)
    for kind in ("framewise","pairwise"):
        for scale in (.1,.2,.4,.6):
            metrics=[]
            for _ in range(5):
                shape=history.shape if kind=="framewise" else (len(history),1)
                distorted=history+scale*rng.standard_normal(shape)
                features=stats(distorted,distance,5)["all"]
                metrics.append(score(parts,infer(design(features,base),checkpoint)))
            rows[f"{kind}-{scale}"] = metrics
    record(dest/"GC12.json",dict(base=score(parts,base),clean=clean,rows=rows))
    rows={}
    for retain in (.0,.25,.5,.75,1.):
        shrunk=distance[:,None]+retain*(history-distance[:,None])
        x=stats(shrunk,distance,5)["all"]
        rows[str(retain)]=score(parts,infer(design(x,base),checkpoint))
    record(dest/"GC13.json",dict(base=score(parts,base),clean=clean,rows=rows))
    print("Completed GC12 and GC13",flush=True)


if __name__=="__main__":
    main()
