"""Origin-frame charge context with a fixed library built only from training topology."""
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree

from .data import TRAIN,VALIDATION,cell,registry,trajectory
from .features import _mic
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING,dump


def keys(atoms):
    residue=np.asarray(atoms["residue_index"])
    end=np.where(residue==residue.min(),"first",np.where(residue==residue.max(),"last","internal"))
    return [f"{rn}/{atom}/{end}" for rn,atom,end in zip(atoms["residue_name"],atoms["name"],end)]


def make_library():
    lookup=defaultdict(list)
    verified={}
    for name in TRAIN+EXTRA_TRAINING+ADDITIONAL_TRAINING:
        reg=registry(ROOT,name)
        tid=trajectory(name,1)
        atoms=reg.atoms(tid).to_pydict()
        source=np.asarray(atoms["source_atom"],dtype=np.int64)
        file=next((ROOT/name/"sources"/"ATLAS").rglob(f"{name}_R1.tpr"))
        universe=mda.Universe(str(file))
        tpr=universe.atoms
        if not np.array_equal(tpr.names[source],atoms["name"]):
            verified[name]=False
            continue
        hydrogen=(tpr.masses>=.5)&(tpr.masses<2.5)
        q=np.asarray(tpr.charges).copy()
        a,b=universe.bonds.indices.T
        np.add.at(q,a[~hydrogen[a]&hydrogen[b]],q[b[~hydrogen[a]&hydrogen[b]]])
        np.add.at(q,b[hydrogen[a]&~hydrogen[b]],q[a[hydrogen[a]&~hydrogen[b]]])
        for k,value in zip(keys(atoms),q[source]):
            lookup[k].append(float(value))
        verified[name]=True
    if not lookup:
        raise ValueError("No training topology was matched")
    library={k:float(np.median(v)) for k,v in sorted(lookup.items())}
    dump(OUT/"charge_library.json",dict(library=library,verified_training_topologies=verified))
    return library


def features(pos,box,pair,charge):
    endpoint,inverse=np.unique(pair[:,:2].astype(np.int64),return_inverse=True)
    a,b=inverse.reshape(-1,2).T
    if box.ndim==1:
        edge=np.nextafter(box.astype(np.float64),0)
        work=np.minimum(np.remainder(pos.astype(np.float64),box),edge)
        tree=cKDTree(work,boxsize=box); radius=9.0
    else:
        work=np.minimum(np.remainder(pos.astype(np.float64)@np.linalg.inv(box.astype(np.float64)),1),
                        np.nextafter(1.,0.))
        tree=cKDTree(work,boxsize=1)
        radius=9/np.linalg.svd(box,compute_uv=False)[-1]
    neighborhoods=tree.query_ball_point(work[endpoint],radius)
    row=np.repeat(np.arange(len(endpoint),dtype=np.int64),[len(v) for v in neighborhoods])
    index=np.concatenate(neighborhoods).astype(np.int64)
    d=_mic(pos[index]-pos[endpoint[row]],box)
    r2=np.einsum("ij,ij->i",d,d)
    keep=(r2>1e-6)&(r2<=81)&(index!=endpoint[row])
    row,index,d,r2=(v[keep] for v in (row,index,d,r2))
    r=np.sqrt(r2)
    last=len(endpoint)
    electro=(charge[index]*np.exp(-r/4.0))/np.maximum(.8,r)
    potential=np.bincount(row,weights=electro,minlength=last)
    nearby_net=np.bincount(row,weights=charge[index]*(r<=4),minlength=last)
    field=np.column_stack([np.bincount(row,weights=electro*d[:,i]/np.maximum(r2,.8),minlength=last)
                           for i in range(3)])
    ion_counts=np.bincount(row,weights=(r<=6)*(abs(charge[index])>.3),minlength=last)
    unit=_mic(pos[endpoint[b]]-pos[endpoint[a]],box)/pair[:,2,None]
    fa=np.einsum("ij,ij->i",field[a],unit)
    fb=np.einsum("ij,ij->i",field[b],-unit)
    c1,c2=charge[endpoint[a]],charge[endpoint[b]]
    out=np.column_stack((c1,c2,potential[a],potential[b],fa,fb,nearby_net[a],nearby_net[b],
                         c1*c2,c1*potential[a],c2*potential[b],fa+fb,fa-fb,
                         ion_counts[a],ion_counts[b],potential[a]-potential[b])).astype(np.float32)
    assert np.isfinite(out).all()
    return out


def add_charge(part,stride,library=None):
    artifact=OUT/"charge_library.json"
    checksum=hashlib.sha256(Path(__file__).read_bytes()+artifact.read_bytes()).hexdigest()[:12]
    if library is None: library=json.loads(artifact.read_text())["library"]
    reg=registry(ROOT,part["protein"])
    tid=trajectory(part["protein"],part["replica"])
    path=ROOT/part["protein"]/"trajectories"/tid/"hb-charge"
    path.mkdir(exist_ok=True)
    cache=path/f"{stride}-{checksum}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["charge"]=z["features"]
                return
    atoms=reg.atoms(tid).to_pydict()
    charges=np.array([library.get(k,library.get(k.rpartition("/")[0]+"/internal",0.)) for k in keys(atoms)],
                     dtype=np.float32)
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        block=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(features(block.positions[0],cell(block.box[0]),part["pair"][start:end],charges))
    part["charge"]=np.concatenate(rows)
    tmp=NamedTemporaryFile(dir=path,delete=False)
    try:
        with tmp:
            np.savez(tmp,features=part["charge"],pair=part["pair"],origin=part["origin"])
        os.replace(tmp.name,cache)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main():
    library=make_library()
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    valid=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_charge,p,stride,library)
              for rows,stride in ((train,25),(extra,50),(valid,100)) for p in rows]
        for job in jobs: job.result()
    configs=(
        ("xgb",118,"hb-origin-charge",False,4,1.),
        ("xgb",118,"hb-origin-charge",True,6,1.),
        ("xgb",166,"hb-origin-charge",True,5,1.),
        ("xgb-soft-tune-0.15-0.25-320-local",118,"hb-origin-charge",False,5,1.),
        ("xgb-soft-tune-0.25-0.4-240-local",166,"hb-origin-charge",True,5,1.),
    )
    explore("charge",configs,861,train+extra,valid)


if __name__=="__main__":
    main()
