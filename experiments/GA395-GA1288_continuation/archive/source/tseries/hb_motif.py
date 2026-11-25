"""Secondary-structure and charged-neighbor features from one structure."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_single_structure import OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


def motif_features(pos, box, pairs, topo):
    maps, names, _, _ = topo
    n=len(maps)
    ordinals=np.full(len(pos),-1,dtype=np.int32)
    for i, m in enumerate(maps): ordinals[list(m.values())]=i
    a,b=pairs[:,:2].astype(np.int64).T
    i,j=ordinals[a],ordinals[b]
    assert np.all(i>=0) and np.all(j>=0)
    nitrogen=np.array([m["N"] for m in maps])
    oxygen=np.array([m["O"] for m in maps])
    alpha=np.array([m["CA"] for m in maps])
    d=np.linalg.norm(_mic(pos[nitrogen,None]-pos[oxygen][None,:],box),axis=2)
    def neighbor(x,y):
        valid=(x>=0)&(x<n)&(y>=0)&(y<n)&(abs(x-y)>1)
        return np.where(valid,d[x.clip(0,n-1),y.clip(0,n-1)],15)
    parallel_before,parallel_after=neighbor(i-1,j-1),neighbor(i+1,j+1)
    anti_before,anti_after=neighbor(i-1,j+1),neighbor(i+1,j-1)
    reverse=neighbor(j,i)
    antiparallel=np.minimum(neighbor(j-1,i+1),neighbor(j+1,i-1))
    helix_i=np.minimum(neighbor(i,i-4),neighbor(i+4,i))
    helix_j=np.minimum(neighbor(j,j-4),neighbor(j+4,j))
    ids=np.arange(n)
    tangent=_mic(pos[alpha[(ids+1).clip(0,n-1)]]-pos[alpha[(ids-1).clip(0,n-1)]],box)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6)
    alignment=np.einsum("ij,ij->i",tangent[i],tangent[j])
    axis=_mic(pos[b]-pos[a],box)/pairs[:,2,None]
    signed={"ARG":1.0,"LYS":1.0,"ASP":-1.0,"GLU":-1.0}
    sites={"ARG":"CZ","LYS":"NZ","ASP":"CG","GLU":"CD"}
    charge=np.array([signed.get(r,0) for r in names],dtype=np.float32)
    charge_ids=np.array([m.get(sites.get(r,"CA"),m["CA"]) for m,r in zip(maps,names)])
    sources=pos[charge_ids]
    diff=_mic(pos[:,None]-sources[None,:],box)
    r2=np.einsum("ijk,ijk->ij",diff,diff)
    weights=np.where((r2<=144)&(r2>1),charge[None,:]/np.maximum(r2,1),0)
    weights[np.arange(len(pos))[ordinals>=0],ordinals[ordinals>=0]]=0
    field=np.einsum("ik,ikl->il",weights/np.sqrt(np.maximum(r2,1)),diff)
    scalar=weights.sum(1)
    field_a=np.einsum("ij,ij->i",field[a],axis)
    field_b=np.einsum("ij,ij->i",field[b],axis)
    out=np.column_stack((parallel_before,parallel_after,anti_before,anti_after,
                         reverse,antiparallel,helix_i,helix_j,
                         alignment,np.minimum(parallel_before,parallel_after),
                         np.minimum(anti_before,anti_after), scalar[a],scalar[b],
                         field_a,field_b,field_a-field_b,charge[i],charge[j])).astype(np.float32)
    assert np.isfinite(out).all()
    return out


def add_motif(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    path=ROOT/protein/"trajectories"/tid/"hb-motif"
    path.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=path/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["motif"]=z["features"]
                return
    topo=topology(reg.atoms(tid).to_pydict())
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        b=next(reg.blocks(tid,start=t,stop=t+1))
        rows.append(motif_features(b.positions[0],cell(b.box[0]),part["pair"][start:end],topo))
    part["motif"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=path,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["motif"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    valid=load(VALIDATION,100,geometry=True)
    for parts,stride in ((train,25),(extra,50),(valid,100)):
        for part in parts:
            add_motif(part,stride)
    configs=(
        ("xgb",118,"hb-origin-motif",False,4,1.0),
        ("xgb",118,"hb-origin-motif",True,6,1.0),
        ("xgb",166,"hb-origin-motif",True,6,1.0),
        ("xgb",166,"hb-origin-motif",False,4,1.0),
        ("xgb-soft-tune-0.15-0.25-320-local",118,"hb-origin-motif",True,4,1.0),
        ("xgb-soft-tune-0.25-0.4-240-local",166,"hb-origin-motif",True,5,1.0),
    )
    explore("motif",configs,721,train+extra,valid)


if __name__=="__main__":
    main()
