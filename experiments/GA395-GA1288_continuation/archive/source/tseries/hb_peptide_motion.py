"""Test small peptide-plane rotations around consecutive Cα anchors."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic, _pd
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


ROTATIONS=np.radians((-26.,-13.,13.,26.))


def peptide_motion(position,box,pair,topo):
    maps,names,_,_=topo
    states=np.repeat(position[:,None,:],len(ROTATIONS)+1,axis=1)
    permitted=np.zeros(states.shape[:2],dtype=bool)
    permitted[:,0]=True
    atoms=np.arange(len(position))
    for idx,(left,right) in enumerate(zip(maps[:-1],maps[1:])):
        if names[idx+1]=="PRO" or "OXT" in left:
            continue
        bond=_mic(position[right["N"]]-position[left["C"]],box)
        if not 1.0<=np.linalg.norm(bond)<=1.8:
            continue
        moving=np.array([left["C"],left["O"],right["N"]],dtype=np.intp)
        start=position[left["CA"]]
        axis=_mic(position[right["CA"]]-start,box)
        norm=np.linalg.norm(axis)
        if norm<1e-6:
            continue
        axis=axis/norm
        offsets=_mic(position[moving]-start,box)
        projection=(offsets@axis)[:,None]*axis
        radial=offsets-projection
        tangential=np.cross(axis,radial)
        fixed=atoms[~np.isin(atoms,moving)]
        existing=_pd(start+offsets,position[fixed],box)
        max_motion=2*np.sin(max(abs(ROTATIONS))/2)*np.linalg.norm(radial,axis=1).max()
        nearby=existing.min(axis=0)<2.25+max_motion
        boundary=np.minimum(2.25,existing[:,nearby]-.12)
        for index,angle in enumerate(ROTATIONS,1):
            rotated=start+projection+np.cos(angle)*radial+np.sin(angle)*tangential
            if np.any(_pd(rotated,position[fixed[nearby]],box)<boundary):
                continue
            states[moving,index]=rotated
            permitted[moving,index]=True
    d,a=pair[:,:2].astype(np.intp).T
    fields=[]
    for mobile,static in ((d,a),(a,d)):
        length=np.linalg.norm(_mic(position[static,None,:]-states[mobile],box),axis=2)
        diff=length-length[:,:1]
        fields.extend((-np.where(permitted[mobile],diff,0).min(axis=1),
                       np.where(permitted[mobile],diff,0).max(axis=1),
                       permitted[mobile,1:].sum(axis=1),
                       (permitted[mobile]&(length<=3.5)).sum(axis=1)))
    return np.column_stack(fields).astype(np.float32)


def add_peptide_motion(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-peptide-motion"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.exists():
        with np.load(cache) as arrays:
            if np.array_equal(arrays["pair"],part["pair"]) and np.array_equal(arrays["origin"],part["origin"]):
                part["peptide"]=arrays["features"]
                return
    topo=topology(reg.atoms(tid).to_pydict())
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        snapshot=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(peptide_motion(snapshot.positions[0],cell(snapshot.box[0]),part["pair"][start:end],topo))
    part["peptide"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["peptide"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    added=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    train=load(TRAIN,25,geometry=True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    late=load(added,100,geometry=True)
    validation=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_peptide_motion,part,stride) for parts,stride in
              ((train,25),(middle,50),(late,100),(validation,100)) for part in parts]
        for job in jobs:
            job.result()
    configs=(("xgb",118,"hb-origin-peptide",False,6,1.),
             ("xgb",166,"hb-origin-peptide",False,5,1.))
    explore("peptide_motion",configs,1411,train+middle+late,validation)


if __name__=="__main__":
    main()
