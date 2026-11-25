"""Score whether a candidate can approach under small local side-chain rotations."""
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


BACKBONE={"N","CA","C","O","OXT"}
ANGLES=np.radians((-35.,-18.,18.,35.))


def connected(adjacency, start, removed):
    seen={start}
    queue=[start]
    while queue:
        current=queue.pop()
        for other in adjacency[current]:
            if (current,other) in removed or other in seen:
                continue
            seen.add(other)
            queue.append(other)
    return seen


def conformations(position,box,topo):
    states=[[xyz] for xyz in position]
    index=np.arange(len(position))
    for atoms in topo[0]:
        names=list(atoms)
        ids=np.array([atoms[name] for name in names])
        root=names.index("CA")
        local=position[ids[root]]+_mic(position[ids]-position[ids[root]],box)
        lengths=np.linalg.norm(local[:,None]-local[None,:],axis=-1)
        bonds=(lengths>.55)&(lengths<1.95)
        adjacency=[np.flatnonzero(row).tolist() for row in bonds]
        candidates=np.argwhere(np.triu(bonds,1))
        for a,b in candidates:
            if names[a] in BACKBONE and names[b] in BACKBONE:
                continue
            removed={(a,b),(b,a)}
            retained=connected(adjacency,root,removed)
            if (a in retained)==(b in retained):
                continue
            fixed_axis,mobile_axis=(a,b) if a in retained else (b,a)
            mobile=np.array(sorted(connected(adjacency,mobile_axis,removed)-{mobile_axis}))
            if not len(mobile) or any(names[k] in BACKBONE for k in mobile):
                continue
            ligand=ids[mobile]
            fixed=index[~np.isin(index,ligand)]
            axis=(local[mobile_axis]-local[fixed_axis])/lengths[mobile_axis,fixed_axis]
            anchor=local[mobile_axis]
            delta=local[mobile]-anchor
            parallel=(delta@axis)[:,None]*axis
            perpendicular=delta-parallel
            tangent=np.cross(axis,perpendicular)
            before=_pd(local[mobile],position[fixed],box)
            crosslink=(before<2.12)&(fixed[None,:]!=ids[mobile_axis])
            if np.any(crosslink):
                continue
            max_sweep=2*np.sin(np.max(abs(ANGLES))/2)*np.linalg.norm(perpendicular,axis=1).max()
            useful=before.min(axis=0)<2.15+max_sweep
            points=position[fixed[useful]]
            boundary=np.minimum(2.15,before[:,useful]-.12)
            for angle in ANGLES:
                move=anchor+parallel+np.cos(angle)*perpendicular+np.sin(angle)*tangent
                if len(points) and np.any(_pd(move,points,box)<boundary):
                    continue
                for atom,xyz in zip(ligand,move):
                    states[atom].append(xyz)
    return states


def torsional(position,box,pair,topo):
    states=conformations(position,box,topo)
    limit=max(map(len,states))
    moves=np.repeat(position[:,None,:],limit,axis=1)
    count=np.array([len(options) for options in states])
    for idx,options in enumerate(states):
        moves[idx,:len(options)]=options
    a,b=pair[:,:2].astype(np.intp).T
    columns=[]
    for source,destination in ((a,b),(b,a)):
        length=np.linalg.norm(_mic(moves[source]-position[destination,None,:],box),axis=-1)
        change=length-length[:,:1]
        columns.extend((-change.min(axis=1),change.max(axis=1),np.log1p(count[source]-1),
                        (change<-.25).sum(axis=1)/np.maximum(count[source]-1,1)))
    return np.column_stack(columns).astype(np.float32)


def add_torsional(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-torsional"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path=directory/f"{stride}-{version}.npz"
    if path.is_file():
        with np.load(path) as arrays:
            if np.array_equal(arrays["pair"],part["pair"]) and np.array_equal(arrays["origin"],part["origin"]):
                part["torsional"]=arrays["features"]
                return
    topo=topology(reg.atoms(tid).to_pydict())
    result=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        origin=int(part["origin"][start])
        snapshot=next(reg.blocks(tid,start=origin,stop=origin+1))
        result.append(torsional(snapshot.positions[0],cell(snapshot.box[0]),part["pair"][start:end],topo))
    part["torsional"]=np.concatenate(result)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["torsional"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,path)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    panel=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    basic=load(TRAIN,25,geometry=True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    late=load(panel,100,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_torsional,part,stride) for parts,stride in
              ((basic,25),(middle,50),(late,100),(valid,100)) for part in parts]
        for job in jobs:
            job.result()
    configs=(("xgb",118,"hb-origin-torsional",False,6,1.),
             ("xgb",166,"hb-origin-torsional",True,5,1.))
    explore("torsional",configs,1331,basic+middle+late,valid)


if __name__=="__main__":
    main()
