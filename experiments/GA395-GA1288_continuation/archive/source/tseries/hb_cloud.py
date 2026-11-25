"""Rigid-motion-invariant local atom density around a hydrogen-bond option."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np
from scipy.spatial import cKDTree

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_single_structure import ROOT, explore, load
from .search_events import EXTRA_TRAINING


def cloud(pos,box,pair,types):
    a,b=pair[:,:2].astype(np.int64).T
    delta=_mic(pos[b]-pos[a],box)
    midpoint=pos[a]+.5*delta
    axis=delta/pair[:,2,None]
    if box.ndim==1:
        edge=np.nextafter(box.astype(np.float64),0)
        positions=np.minimum(np.remainder(pos.astype(np.float64),box),edge)
        queries=np.minimum(np.remainder(midpoint.astype(np.float64),box),edge)
        radius=6.25
        tree=cKDTree(positions,boxsize=box)
    else:
        transform=np.linalg.inv(box.astype(np.float64))
        positions=np.minimum(np.remainder(pos@transform,1),np.nextafter(1.,0.))
        queries=np.minimum(np.remainder(midpoint@transform,1),np.nextafter(1.,0.))
        radius=6.25/np.linalg.svd(box,compute_uv=False)[-1]
        tree=cKDTree(positions,boxsize=1.)
    lists=tree.query_ball_point(queries,radius)
    lengths=np.asarray([len(x) for x in lists])
    row=np.repeat(np.arange(len(pair),dtype=np.int64),lengths)
    site=np.concatenate(lists).astype(np.int64)
    relative=_mic(pos[site]-midpoint[row],box)
    norm2=np.einsum("ij,ij->i",relative,relative)
    axial=np.einsum("ij,ij->i",relative,axis[row])
    radial=np.sqrt(np.maximum(0,norm2-axial**2))
    keep=(site!=a[row])&(site!=b[row])&(norm2<=6.25**2)&(radial<=5)
    row,site,norm2,axial,radial=(x[keep] for x in (row,site,norm2,axial,radial))
    element=np.zeros(len(pos),dtype=np.int64)
    for number,category in ((7,1),(8,2),(16,3),(15,3),(34,3)):
        element[types==number]=category
    type_at=element[site]
    radial_bin=np.searchsorted([1.2,2.2,3.4],radial,side="right")
    axis_bin=np.searchsorted([-.60,-.2,.2,.60],axial/pair[row,2],side="right")
    lane=row*80+axis_bin*16+radial_bin*4+type_at
    weight=np.exp(-norm2/22)
    hist=np.bincount(lane,weights=weight,minlength=len(pair)*80).reshape(-1,5,4,4)
    inner=hist[:,1:4,:2,:].sum((1,2))
    edges=hist[:,[0,4],:2,:].sum((1,2))
    return np.column_stack((inner,edges,hist.reshape(-1,80))).astype(np.float32)


def add_cloud(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-cloud"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["cloud"]=z["features"]
                return
    types=topology(reg.atoms(tid).to_pydict())[3]
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        b=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(cloud(b.positions[0],cell(b.box[0]),part["pair"][start:end],types))
    part["cloud"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["cloud"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    valid=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_cloud,part,stride)
              for parts,stride in ((train,25),(extra,50),(valid,100)) for part in parts]
        for job in jobs:
            job.result()
    variants=(
        ("xgb",118,"hb-origin-cloud",False,4,1.0),
        ("xgb",118,"hb-origin-cloud",True,6,1.0),
        ("xgb",166,"hb-origin-cloud",False,5,1.0),
        ("xgb",166,"hb-origin-cloud",True,6,1.0),
        ("xgb-soft-tune-0.15-0.25-320-local",118,"hb-origin-cloud",True,5,1.0),
        ("xgb-soft-tune-0.25-0.4-240-local",166,"hb-origin-cloud",False,4,1.0),
    )
    explore("cloud",variants,841,train+extra,valid)


if __name__=="__main__":
    main()
