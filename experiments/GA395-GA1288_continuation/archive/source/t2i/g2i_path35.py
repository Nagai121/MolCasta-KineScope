"""Thirty-five frozen-readout experiments on static approach paths."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from tseries.data import cell, registry, trajectory
from tseries.features import _mic
from tseries.hb_heavy_dynamics import neighbor_pairs
from tseries.hb_single_structure import OUT, ROOT
from tseries.hb_validation10 import panel10
from .explore import record, score
from .frontends import data, design, infer
from .gc_assist import matched
from .g2i_rank import fit, pairs

FAMILIES = ("distance", "straight", "detour", "joint_route", "anisotropy", "availability", "hybrid")
WEIGHTS = (0., .025, .075, .15, .30)


def paths(position, box, pair):
    a,b=pair[:,:2].astype(np.intp).T
    ends=np.stack((a,b))
    axis=pair[:,3:6]/pair[:,2,None]
    reference=np.eye(3)[np.abs(axis).argmin(axis=1)]
    cross=np.cross(axis,reference)
    cross/=np.linalg.norm(cross,axis=1)[:,None]
    normal=np.cross(axis,cross)
    ang=np.arange(8)*np.pi/4
    lateral=(np.cos(ang)[:,None,None]*cross+np.sin(ang)[:,None,None]*normal)
    left,right=neighbor_pairs(position,box,5.5).T
    source=np.r_[left,right]
    target=np.r_[right,left]
    order=np.argsort(source,kind="stable")
    ends_ptr=np.searchsorted(source[order],np.arange(len(position)+1))
    offsets=np.arange(28)
    start=ends_ptr[ends]
    slots=start[...,None]+offsets
    mask=slots<ends_ptr[ends+1][...,None]
    if len(order):
        neighbors=target[order[slots.clip(0,len(order)-1)]]
    else:
        neighbors=np.zeros(slots.shape,dtype=int)
    dv=_mic(position[neighbors]-position[ends][...,None,:],box)
    d0=np.linalg.norm(dv,axis=-1)
    valid=mask&(d0>2.05)&(neighbors!=ends[::-1][...,None])
    overlap=np.maximum(2.75-d0,0.)**2
    forward=np.stack((axis,-axis))
    def cost(displace):
        v=np.linalg.norm(dv-displace[...,None,:],axis=-1)
        return ((np.maximum(2.75-v,0.)**2-overlap)*valid).sum(-1)
    straight=cost(.65*forward)
    arc=[]
    for direction in lateral:
        waypoint=cost(.32*forward+.40*direction)
        endpoint=cost(.65*forward+.22*direction)
        arc.append(.6*waypoint+endpoint)
    arc=np.asarray(arc)
    direct=np.column_stack((straight.T,straight.sum(0),straight.max(0)))
    free=np.column_stack((arc.min(0).T,arc.mean(0).T,arc.max(0).T))
    total=arc.sum(1)
    opposite=(arc[:,0]+np.roll(arc[:,1],4,axis=0))
    joint=np.column_stack((total.min(0),opposite.min(0),total.mean(0),opposite.mean(0),
                           total.min(0)-straight.sum(0)))
    anis=np.column_stack((arc.std(0).T,(arc.max(0)-arc.min(0)).T,
                          (arc<straight).mean(0).T))
    favorable=np.exp(-np.maximum(total.min(0),0.))*np.exp(-(pair[:,2]-3.5))
    availability=[]
    for endpoint in ends:
        volume=np.bincount(endpoint,weights=favorable)[endpoint]
        availability.extend((favorable/(.05+volume),volume-favorable))
    return np.column_stack((direct,free,joint,anis,*availability)).astype(np.float32)


def build(part):
    reg=registry(ROOT,part["protein"])
    tid=trajectory(part["protein"],part["replica"])
    chunks=[]
    for a,b in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        frame=int(part["origin"][a])
        block=next(reg.blocks(tid,start=frame,stop=frame+1,chunk_frames=1))
        chunks.append(paths(block.positions[0],cell(block.box[0]),part["pair"][a:b]))
    return np.concatenate(chunks)


def views(p,old):
    d=old["distance"][:,:1]
    spans=(None,slice(0,4),slice(4,10),slice(10,15),slice(15,21),slice(21,25),slice(None))
    result={}
    for family,sl in zip(FAMILIES,spans):
        if sl is None:
            result[family]=d
        elif family=="hybrid":
            result[family]=np.column_stack((d,p,old["combined"]))
        else:
            result[family]=np.column_stack((d,p[:,sl]))
    return result


def main():
    torch.set_num_threads(3)
    dst=Path("runs/g2i/path35")
    dst.mkdir(parents=True,exist_ok=True)
    ids=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    tr,a,bt=data(ids)
    va,b,bv=data(panel10())
    started=time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        x=np.concatenate(list(pool.map(build,tr)))
        v=np.concatenate(list(pool.map(build,va)))
    elapsed=time.perf_counter()-started
    print(f"Prepared {len(x)+len(v)} rows in {elapsed:.2f}s",flush=True)
    x,v=views(x,a),views(v,b)
    y=np.concatenate([p["y"] for p in tr])
    positive,negative=pairs(tr,design(a["distance"],bt))
    report=dict(baseline=score(va,bv),training_proteins=ids,validation_proteins=panel10(),
                feature_seconds=elapsed,matched_train_pairs=len(positive),trials={})
    scores={"frozen_t1":bv}
    for family in FAMILIES:
        train=design(x[family],bt)
        validation=design(v[family],bv)
        for weight in WEIGHTS:
            name=f"{family}-{weight}"
            path=dst/f"{name}.pt"
            started=time.perf_counter()
            fit(train,y,positive,negative,weight,path)
            training_seconds=time.perf_counter()-started
            started=time.perf_counter()
            p=infer(validation,path)
            seconds=time.perf_counter()-started
            scores[name]=p
            row=dict(metrics=score(va,p),train_seconds=training_seconds,cpu_seconds=seconds,
                     checkpoint=path.name)
            report["trials"][name]=row
            record(dst/"results.json",report)
            print(name,row["metrics"]["auprc"],flush=True)
    report["conditional"]=matched(va,bv,scores,same_frame=True)
    record(dst/"results.json",report)


if __name__=="__main__":
    main()
