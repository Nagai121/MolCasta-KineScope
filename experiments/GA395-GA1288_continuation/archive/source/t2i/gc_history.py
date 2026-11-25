"""GC10–11: observed-history window and carrier ablations."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from tseries.data import cell, registry, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import ROOT, OUT
from tseries.hb_validation10 import panel10
from .frontends import data, design, fit, infer
from .explore import record, score


def past(part):
    reg=registry(ROOT,part["protein"])
    tid=trajectory(part["protein"],part["replica"])
    chunks=[]
    for a,b in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][a])
        frames=next(reg.blocks(tid,start=t-10,stop=t+1,chunk_frames=11))
        if len(frames.time)!=11 or not np.allclose(frames.time[-1]-frames.time, np.arange(10,-1,-1)*100,
                                                   rtol=0,atol=.01):
            raise ValueError("Expected 100 ps history grid")
        pairs=part["pair"][a:b]
        i,j=pairs[:,:2].astype(np.intp).T
        chunks.append(np.column_stack([np.linalg.norm(_mic(
            p[j]-p[i],cell(box)),axis=1) for p,box in zip(frames.positions[:-1],frames.box[:-1])]))
    return np.concatenate(chunks)


def stats(values,distance,window):
    v=values[:,-window:]
    mean=distance-v.mean(1)
    closest=distance-v.min(1)
    occupancy=(v<=3.5).mean(1)
    return dict(mean=mean[:,None], closest=closest[:,None], occupancy=occupancy[:,None],
                mean_and_occupancy=np.column_stack((mean,occupancy)),
                motion=np.column_stack((distance-v[:,-1],v[:,0]-v[:,-1],v.std(1))),
                all=np.column_stack((mean,closest,occupancy,distance-v[:,-1],v[:,0]-v[:,-1],v.std(1))))


def main():
    torch.set_num_threads(3)
    dest=Path("runs/gc")
    names=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    tr,_,pr=data(names)
    va,_,pv=data(panel10())
    with ThreadPoolExecutor(3) as pool:
        ht=np.concatenate(list(pool.map(past,tr)))
        hv=np.concatenate(list(pool.map(past,va)))
    dt,dv=[np.concatenate([p["pair"][:,2] for p in rows]) for rows in (tr,va)]
    labels=np.concatenate([p["y"] for p in tr])
    baseline=score(va,pv)
    for trial, windows, carriers in ((10,(1,2,3,5,10),("all",)),
                                      (11,(5,10),("mean","closest","occupancy","mean_and_occupancy","motion"))):
        rows={}
        for w in windows:
            xt,xv=stats(ht,dt,w),stats(hv,dv,w)
            for c in carriers:
                key=f"{w*100}ps-{c}"
                path=dest/f"GC{trial}-{key}.pt"
                elapsed,peak=fit(design(xt[c],pr),labels,path)
                prob=infer(design(xv[c],pv),path)
                rows[key]=dict(metrics=score(va,prob),fit_seconds=elapsed,cuda_peak_bytes=peak,
                               checkpoint=path.name)
        record(dest/f"GC{trial}.json",dict(baseline=baseline,rows=rows))
        print(f"Completed GC{trial}",flush=True)


if __name__=="__main__":
    main()
