"""Estimate heavy-atom directional compliance from one origin frame."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic, _pd
from .hb_local_geometry import project, unit
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


def stiffness(position, box, pair, topo):
    if box.ndim==1:
        edge=np.nextafter(box.astype(np.float64),0)
        periodic=np.minimum(np.remainder(position.astype(np.float64),box),edge)
        contact=cKDTree(periodic,boxsize=box).query_pairs(4.5,output_type="ndarray")
        i,j=contact.T
    else:
        i,j=np.where(np.triu(_pd(position,position,box)<=4.5,1))
    offset=_mic(position[j]-position[i],box)
    distance=np.linalg.norm(offset,axis=1)
    directions=unit(offset)
    residue=topo[2][1]
    nearby=abs(residue[i]-residue[j])<=1
    covalent=nearby & (residue[i]>=0) & (residue[j]>=0) & (distance<1.95)
    local_prior=np.where(covalent,20.,np.exp(-((distance-2.6)/1.2)**2))
    spring=directions[:,:,None]*directions[:,None,:]*local_prior[:,None,None]
    diagonal=np.broadcast_to(.4*np.eye(3),(len(position),3,3)).copy()
    np.add.at(diagonal,i,spring)
    np.add.at(diagonal,j,spring)
    compliance=np.linalg.inv(diagonal)
    pressure=np.zeros((len(position),3))
    force=(~covalent*np.maximum(3.1-distance,0)**2)[:,None]*directions
    np.add.at(pressure,i,-force)
    np.add.at(pressure,j,force)
    relaxed=np.einsum("nij,nj->ni",compliance,pressure)
    a,b=pair[:,:2].astype(np.intp).T
    axis=unit(_mic(position[b]-position[a],box))
    along_a=np.einsum("ni,nij,nj->n",axis,compliance[a],axis)
    along_b=np.einsum("ni,nij,nj->n",axis,compliance[b],axis)
    trace=np.trace(compliance,axis1=1,axis2=2)
    tangent_a=(trace[a]-along_a)*.5
    tangent_b=(trace[b]-along_b)*.5
    projected=project(relaxed[b]-relaxed[a],axis)
    return np.column_stack((along_a,along_b,tangent_a,tangent_b,along_a+along_b,
                            along_a/(tangent_a+.01),along_b/(tangent_b+.01),
                            projected)).astype(np.float32)


def add_stiffness(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-atom-stiffness"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as arrays:
            if np.array_equal(arrays["pair"],part["pair"]) and np.array_equal(arrays["origin"],part["origin"]):
                part["stiffness"]=arrays["features"]
                return
    topo=topology(reg.atoms(tid).to_pydict())
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        block=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(stiffness(block.positions[0],cell(block.box[0]),part["pair"][start:end],topo))
    part["stiffness"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["stiffness"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    panel=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    basic=load(TRAIN,25,geometry=True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    extra=load(panel,100,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_stiffness,part,stride) for parts,stride in
              ((basic,25),(middle,50),(extra,100),(valid,100)) for part in parts]
        for job in jobs:
            job.result()
    options=(("xgb",118,"hb-origin-stiffness",False,6,1.),
             ("xgb",166,"hb-origin-stiffness",False,5,1.),
             ("xgb",166,"hb-origin-stiffness",True,5,1.))
    explore("stiffness",options,1291,basic+middle+extra,valid)


if __name__=="__main__":
    main()
