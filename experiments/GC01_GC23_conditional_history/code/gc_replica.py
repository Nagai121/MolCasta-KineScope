"""GC19: substitute history of the same pair from another replica."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib

import numpy as np
import torch

from tseries.data import registry, trajectory
from tseries.hb_single_structure import ROOT
from tseries.hb_validation10 import panel10
from .frontends import data, design, infer
from .explore import record, score
from .gc_history import past, stats
from .gc_temporal import subset


def keys(p):
    a,b=p["pair"][:,:2].astype(np.uint64).T
    if max(a.max(),b.max())>=2**20:
        raise ValueError("Pair IDs out of range")
    return (p["origin"].astype(np.uint64)<<40)|(a<<20)|b


def topology_signature(p):
    reg=registry(ROOT,p["protein"])
    atoms=reg.atoms(trajectory(p["protein"],p["replica"]))
    identity=atoms.select(["name","residue_index","residue_name","element"]).to_pylist()
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def main():
    torch.set_num_threads(3)
    dest=Path("runs/gc")
    parts,_,base=data(panel10())
    with ThreadPoolExecutor(3) as pool:
        values=list(pool.map(past,parts))
    by={(p["protein"],p["replica"]):i for i,p in enumerate(parts)}
    observed,substitute=[],[]
    checkpoint=dest/"GC10-500ps-all.pt"
    for i,p in enumerate(parts):
        j=by[(p["protein"],p["replica"]%3+1)]
        q=parts[j]
        if topology_signature(p)!=topology_signature(q):
            raise ValueError("Replica topology differs")
        target=keys(p)
        source=keys(q)
        order=np.argsort(source)
        at=np.searchsorted(source[order],target).clip(0,len(source)-1)
        pick=order[at]
        keep=source[pick]==target
        d=p["pair"][:,2]
        other=q["pair"][pick,2]
        matched=values[j][pick]-other[:,None]+d[:,None]
        observed.append(keep)
        substitute.append(matched)
    mask=np.concatenate(observed)
    d=np.concatenate([p["pair"][:,2] for p in parts])[mask]
    groups=subset(parts,mask)
    def evaluate(values):
        p=infer(design(stats(values,d,5)["all"],base[mask]),checkpoint)
        return score(groups,p)
    record(dest/"GC19.json",dict(coverage=float(mask.mean()),baseline=score(groups,base[mask]),
                                  observed=evaluate(np.concatenate(values)[mask]),
                                  substitute=evaluate(np.concatenate(substitute)[mask])))
    print("Completed GC19",flush=True)


if __name__=="__main__":
    main()
