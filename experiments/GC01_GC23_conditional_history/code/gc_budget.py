"""GC20–21: incomplete and low-precision observed histories."""
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
        h=np.concatenate(list(pool.map(past,parts)))
    d=np.concatenate([p["pair"][:,2] for p in parts])
    checkpoint=dest/"GC10-500ps-all.pt"
    def probabilities(history):
        return infer(design(stats(history,d,5)["all"],base),checkpoint)
    full=probabilities(h)
    rng=np.random.default_rng(2001)
    masks={}
    for cov in (.1,.25,.5,.75,.9):
        measures=[]
        for _ in range(5):
            has=rng.random(len(d))<cov
            measures.append(score(parts,np.where(has,full,base)))
        masks[str(cov)]=measures
    record(dest/"GC20.json",dict(baseline=score(parts,base),full=score(parts,full),rows=masks))
    precision={}
    for q in (.05,.1,.2,.4,.8):
        nearest=np.round(h/q)*q
        # Threshold labels are recorded separately from the coarse distances.
        x=stats(nearest,d,5)["all"]
        preserved=x.copy()
        preserved[:,2]=(h[:,-5:]<=3.5).mean(1)
        precision[str(q)]=dict(quantized=score(parts,probabilities(nearest)),
                                true_occupancy=score(parts,infer(design(preserved,base),checkpoint)))
    record(dest/"GC21.json",dict(baseline=score(parts,base),full=score(parts,full),rows=precision))
    print("Completed GC20 and GC21",flush=True)


if __name__=="__main__":
    main()
