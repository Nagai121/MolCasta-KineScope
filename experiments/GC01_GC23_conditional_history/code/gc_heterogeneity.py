"""GC16–17: locate and quantify observed-history gains."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tseries.hb_validation10 import panel10
from .audit import paired_ap
from .frontends import data, design, infer
from .explore import record, score
from .gc_history import past, stats
from .gc_temporal import subset


def main():
    torch.set_num_threads(3)
    dest=Path("runs/gc")
    parts,_,base=data(panel10())
    with ThreadPoolExecutor(3) as pool:
        h=np.concatenate(list(pool.map(past,parts)))
    d=np.concatenate([p["pair"][:,2] for p in parts])
    x=stats(h,d,5)["all"]
    p=infer(design(x,base),dest/"GC10-500ps-all.pt")
    rows={}
    for a,b in ((3.5,4.),(4.,4.5),(4.5,5.),(5.,6.01)):
        keep=(d>a)&(d<=b)
        group=subset(parts,keep)
        rows[f"{a}-{b}"]=dict(base=score(group,base[keep]),corrected=score(group,p[keep]))
    record(dest/"GC16.json",dict(rows=rows))
    y=np.concatenate([v["y"] for v in parts])
    delta={}
    for name in panel10():
        keep=np.concatenate([np.full(len(v["y"]),v["protein"]==name) for v in parts])
        group=subset(parts,keep)
        a,b=score(group,base[keep]),score(group,p[keep])
        delta[name]=dict(base=a,corrected=b,delta_auprc=b["auprc"]-a["auprc"])
    ci=paired_ap(parts,p,{"frozen_t1":base})
    record(dest/"GC17.json",dict(proteins=delta,delta_ci=ci,positive_rows=int(y.sum())))
    print("Completed GC16 and GC17",flush=True)


if __name__=="__main__":
    main()
