"""GC03–05: threshold crossings, temporal direction, and static readout."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import torch

from tseries.data import cell, registry, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import ROOT, OUT
from tseries.hb_validation10 import panel10
from .frontends import data
from .explore import record
from .gc_assist import matched

STEPS = (-5, -2, -1, 1, 2, 5)


def offsets(part):
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    rows = []
    for a,b in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        t = int(part["origin"][a])
        frames = next(reg.blocks(tid, start=t-5, stop=t+6, chunk_frames=11))
        steps = np.array(STEPS)+5
        if not np.allclose(frames.time[steps]-frames.time[5], np.array(STEPS)*100, atol=.01, rtol=0):
            raise ValueError("Unexpected trajectory times")
        pairs = part["pair"][a:b]
        left,right = pairs[:, :2].astype(np.intp).T
        values = [np.linalg.norm(_mic(frames.positions[i, right]-frames.positions[i, left],
                                     cell(frames.box[i])), axis=1) for i in steps]
        rows.append(np.column_stack(values))
    return np.concatenate(rows)


def subset(parts, keep):
    out = []
    start = 0
    for p in parts:
        end = start+len(p["y"])
        select = keep[start:end]
        out.append(dict(protein=p["protein"], pair=p["pair"][select], y=p["y"][select]))
        start = end
    return out


def readout(a,b,y):
    center = a.mean(0)
    scale = np.maximum(a.std(0), .001)
    a,b = [(x-center)/scale for x in (a,b)]
    a,b = [np.column_stack((np.ones(len(x)),x,x*x)) for x in (a,b)]
    z = torch.from_numpy(a).cuda()
    target = torch.from_numpy(y.astype(float)).cuda()
    penalty = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)*len(a)*.001
    weights = torch.linalg.solve(z.T@z+penalty, z.T@target).cpu().numpy()
    return b@weights


def main():
    from pathlib import Path
    destination = Path("runs/gc")
    destination.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    va,xv,base = data(panel10())
    with ThreadPoolExecutor(3) as pool:
        future = np.concatenate(list(pool.map(offsets,va)))
    distance = np.concatenate([p["pair"][:,2] for p in va])
    ahead = future[:,3]
    near = ahead <= 3.5
    scores = dict(frozen_t1=base, observed_100ps=distance-ahead, crossing=near.astype(float))
    all_rows = matched(va,base,scores)
    noncrossing = matched(subset(va,~near),base[~near],
                          {k:v[~near] for k,v in scores.items() if k != "crossing"})
    record(destination/"GC03.json", dict(all=all_rows, noncrossing=noncrossing,
           excluded_rows=int(near.sum())))
    scores = {"frozen_t1":base}
    for i,t in enumerate(STEPS):
        scores[f"offset_{t*100}_ps"] = distance-future[:,i]
    record(destination/"GC04.json",matched(va,base,scores))
    print("Completed GC03 and GC04",flush=True)
    names = tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    tr,xt,_ = data(names)
    with ThreadPoolExecutor(3) as pool:
        train = np.concatenate(list(pool.map(offsets,tr)))
    train_distance=np.concatenate([p["pair"][:,2] for p in tr])
    y=train_distance-train[:,3]
    scores={"frozen_t1":base,"observed_100ps":distance-ahead}
    for name in xt:
        scores[name]=readout(xt[name],xv[name],y)
    record(destination/"GC05.json",matched(va,base,scores))
    print("Completed GC05",flush=True)


if __name__=="__main__":
    main()
