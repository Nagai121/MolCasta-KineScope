"""Origin-frame geometry of hydrogen-bond partners already within 3.5 Å."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic, _pd
from .hb_local_geometry import project, unit
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


def occupied(position, box, pair, topo):
    maps, _, ctx, _ = topo
    donors, donor_anchors, acceptors, acceptor_anchors, cross = (ctx[i] for i in (5,8,9,12,13))
    d, a = np.where((_pd(position[donors],position[acceptors],box)<=3.5) & cross)
    i,j = donors[d],acceptors[a]
    toward = unit(_mic(position[i]-position[donor_anchors[d]],box))
    previous=np.full(len(position),-1,dtype=np.int32)
    for prior,current in zip(maps[:-1],maps[1:]):
        previous[current["N"]]=prior["C"]
    predecessor=_mic(position[i]-position[np.maximum(previous[i],0)],box)
    bonded=(previous[i]>=0)&(np.linalg.norm(predecessor,axis=1)<1.8)
    toward=unit(toward+unit(predecessor)*bonded[:,None])
    delta=_mic(position[j]-position[i],box)
    ha=delta-toward
    away=unit(_mic(position[j]-position[acceptor_anchors[a]],box))
    quality=(np.exp(-((np.linalg.norm(ha,axis=1)-1.9)/.7)**2)
             * np.maximum(project(toward,unit(ha)),0)**2
             * (.15+.85*np.maximum(project(away,-unit(ha)),0)))
    ai,aj=pair[:,:2].astype(np.intp).T
    axis=unit(_mic(position[aj]-position[ai],box))
    columns=[]
    for atoms,target,sign in ((i,ai,1),(j,aj,-1)):
        count=np.bincount(atoms,minlength=len(position))
        mass=np.bincount(atoms,weights=quality,minlength=len(position))
        strongest=np.zeros(len(position))
        np.maximum.at(strongest,atoms,quality)
        aggregate=np.zeros_like(position,dtype=np.float64)
        np.add.at(aggregate,atoms,sign*unit(delta)*quality[:,None])
        alignment=project(sign*axis,aggregate[target])/np.maximum(mass[target],1e-6)
        columns.extend((count[target],strongest[target],mass[target],alignment))
    return np.column_stack(columns).astype(np.float32)


def add_occupied(part,stride):
    protein, replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-occupied"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as arrays:
            if np.array_equal(arrays["pair"],part["pair"]) and np.array_equal(arrays["origin"],part["origin"]):
                part["occupied"]=arrays["features"]
                return
    topo=topology(reg.atoms(tid).to_pydict())
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        block=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(occupied(block.positions[0],cell(block.box[0]),part["pair"][start:end],topo))
    part["occupied"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["occupied"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    added=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    basic=load(TRAIN,25,geometry=True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    extra=load(added,100,geometry=True)
    validation=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_occupied,part,stride) for parts,stride in
              ((basic,25),(middle,50),(extra,100),(validation,100)) for part in parts]
        for job in jobs:
            job.result()
    choices=(("xgb",118,"hb-origin-occupied",False,6,1.0),
             ("xgb",166,"hb-origin-occupied",False,5,1.0),
             ("xgb",166,"hb-origin-occupied",True,5,1.0))
    explore("occupied",choices,1261,basic+middle+extra,validation)


if __name__=="__main__":
    main()
