"""Approximate directional solvent exposure from one heavy-atom frame."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np
from scipy.spatial import cKDTree

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_single_structure import ROOT, explore, load
from .search_events import EXTRA_TRAINING


def exposure(pos,box,pairs,types,samples=48):
    # Approximate van der Waals radii (Å), expanded by a solvent probe.
    radius=np.full(len(pos),1.70,dtype=np.float32)
    for element,value in ((7,1.55),(8,1.52),(15,1.80),(16,1.80),(34,1.90)):
        radius[types==element]=value
    radius+=1.4
    endpoint, inverse=np.unique(pairs[:,:2].astype(np.int64),return_inverse=True)
    a,b=inverse.reshape(-1,2).T
    steps=np.arange(samples,dtype=np.float32)
    z=1-2*(steps+.5)/samples
    angles=steps*np.pi*(3-np.sqrt(5))
    radius_xy=np.sqrt(1-z*z)
    points=np.column_stack((np.cos(angles)*radius_xy,np.sin(angles)*radius_xy,z)).astype(np.float32)
    if box.ndim==1:
        upper=np.nextafter(box.astype(np.float64),0)
        wrapped=np.minimum(np.remainder(pos.astype(np.float64),box),upper)
        tree=cKDTree(wrapped,boxsize=box)
        lists=tree.query_ball_point(wrapped[endpoint],radius.max()*2)
    else:
        inv=np.linalg.inv(box.astype(np.float64))
        xyz=np.minimum(np.remainder(pos.astype(np.float64)@inv,1),np.nextafter(1.0,0))
        tree=cKDTree(xyz,boxsize=1)
        lists=tree.query_ball_point(xyz[endpoint],radius.max()*2/np.linalg.svd(box,compute_uv=False)[-1])
    visible=[]
    for t in range(0,len(endpoint),96):
        atoms=endpoint[t:t+96]
        neighbors=lists[t:t+96]
        count=max(len(row) for row in neighbors)
        chosen=np.full((len(atoms),count),-1,dtype=np.int32)
        for row,site in enumerate(neighbors):
            chosen[row,:len(site)]=site
        safe=np.maximum(chosen,0)
        delta=_mic(pos[safe]-pos[atoms,None],box)
        r2=np.einsum("ijk,ijk->ij",delta,delta)
        central=radius[atoms,None]
        outer=radius[safe]
        boundary=(central**2+r2-outer**2)/(2*central)
        possible=(chosen>=0)&(chosen!=atoms[:,None])&(r2<(central+outer)**2)
        blocked=(np.einsum("ndj,kj->ndk",delta,points)>boundary[:,:,None])&possible[:,:,None]
        visible.append(~blocked.any(1))
    visible=np.concatenate(visible)
    fraction=visible.mean(1)
    moment=visible.astype(np.float32)@points/samples
    ab=_mic(pos[endpoint[b]]-pos[endpoint[a]],box)/pairs[:,2,None]
    projection_a=np.einsum("ij,ij->i",moment[a],ab)
    projection_b=np.einsum("ij,ij->i",moment[b],-ab)
    orientation=ab@points.T
    side_a=(orientation>=.50)
    side_b=(orientation<=-.50)
    open_a=(visible[a]&side_a).sum(1)/np.maximum(1,side_a.sum(1))
    open_b=(visible[b]&side_b).sum(1)/np.maximum(1,side_b.sum(1))
    output=np.column_stack((fraction[a],fraction[b],open_a,open_b,projection_a,projection_b,
                            fraction[a]*fraction[b],open_a*open_b,
                            np.abs(fraction[a]-fraction[b]),np.minimum(open_a,open_b),
                            fraction[a]*(pairs[:,2]-3.5),fraction[b]*(pairs[:,2]-3.5))).astype(np.float32)
    assert np.isfinite(output).all()
    return output


def add_exposure(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    path=ROOT/protein/"trajectories"/tid/"hb-exposure"
    path.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=path/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["exposure"]=z["features"]
                return
    types=topology(reg.atoms(tid).to_pydict())[3]
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        b=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(exposure(b.positions[0],cell(b.box[0]),part["pair"][start:end],types))
    part["exposure"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=path,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["exposure"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    from concurrent.futures import ThreadPoolExecutor
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    valid=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        work=[pool.submit(add_exposure,part,stride)
              for parts,stride in ((train,25),(extra,50),(valid,100)) for part in parts]
        for job in work: job.result()
    configs=(
        ("xgb",118,"hb-origin-exposure",False,4,1.0),
        ("xgb",118,"hb-origin-exposure",True,6,1.0),
        ("xgb",166,"hb-origin-exposure",True,6,1.0),
        ("xgb-soft-tune-0.15-0.25-320-local",118,"hb-origin-exposure",True,4,1.0),
        ("xgb-soft-tune-0.25-0.4-240-local",166,"hb-origin-exposure",True,5,1.0),
    )
    explore("exposure",configs,741,train+extra,valid)


if __name__=="__main__":
    main()
