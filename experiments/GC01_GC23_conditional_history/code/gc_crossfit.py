"""GC15: train the frozen correction on cross-fitted surrogate histories."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from tseries.hb_single_structure import OUT
from tseries.hb_validation10 import panel10
from .frontends import data, design, fit, infer
from .explore import record, score
from .gc_history import past, stats
from .gc_temporal import readout


def history_features(parts):
    with ThreadPoolExecutor(3) as pool:
        h=np.concatenate(list(pool.map(past,parts)))
    d=np.concatenate([p["pair"][:,2] for p in parts])
    values=stats(h,d,5)
    return np.column_stack((values["mean"],values["occupancy"]))


def main():
    torch.set_num_threads(3)
    names=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    tr,a,bt=data(names)
    va,b,bv=data(panel10())
    teacher=history_features(tr)
    fold=np.concatenate([np.full(len(p["y"]),names.index(p["protein"])%3) for p in tr])
    y=np.concatenate([p["y"] for p in tr])
    dest=Path("runs/gc")
    rows={}
    for family in ("distance","existing_static","combined"):
        x=design(a[family],bt).astype(float)
        v=design(b[family],bv).astype(float)
        outputs=np.zeros_like(teacher)
        estimates=[]
        for col in range(teacher.shape[1]):
            val=[]
            for i in range(3):
                keep=fold!=i
                outputs[~keep,col]=readout(x[keep],x[~keep],teacher[keep,col])
                val.append(readout(x[keep],v,teacher[keep,col]))
            estimates.append(np.mean(val,axis=0))
        estimates=np.column_stack(estimates)
        outputs[:,1]=outputs[:,1].clip(0,1)
        estimates[:,1]=estimates[:,1].clip(0,1)
        ckpt=dest/f"GC15-{family}.pt"
        seconds,peak=fit(design(outputs,bt),y,ckpt)
        prob=infer(design(estimates,bv),ckpt)
        rows[family]=dict(metrics=score(va,prob),fit_seconds=seconds,cuda_peak_bytes=peak,
                          checkpoint=ckpt.name)
    record(dest/"GC15.json",dict(baseline=score(va,bv),rows=rows))
    print("Completed GC15",flush=True)


if __name__=="__main__":
    main()
