"""Use explicit hydrogens from the same frame as the heavy-atom ROD input."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import MDAnalysis as mda
import numpy as np
from minitopol.atlas import check_atlas

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_local_geometry import unit
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


@lru_cache(maxsize=24)
def source_plans(protein):
    reg=registry(ROOT,protein)
    return {row["replica"]:row for row in check_atlas(reg.workspace,protein,"analysis")}


def hydrogen_indices(raw_topology,plan,source_atoms):
    source_to_tpr=np.asarray(plan["source_to_tpr"],dtype=np.intp)
    tpr=mda.Universe(str(plan["tpr"]))
    if len(source_to_tpr)!=len(raw_topology.atoms):
        raise ValueError("MiniTopol did not map every atom in the ATLAS source")
    mass=tpr.atoms.masses
    hydrogen=(mass>=.5)&(mass<2.5)
    inverse=np.full(len(mass),-1,dtype=np.intp)
    inverse[source_to_tpr]=np.arange(len(source_to_tpr))
    heavy_index=np.full(len(mass),-1,dtype=np.intp)
    heavy_index[source_to_tpr[source_atoms]]=np.arange(len(source_atoms))
    by_atom=[[] for _ in source_atoms]
    for left,right in tpr.bonds.indices:
        if hydrogen[left] and not hydrogen[right]:
            left,right=right,left
        donor=heavy_index[left]
        explicit=inverse[right]
        if donor>=0 and hydrogen[right] and explicit>=0:
            by_atom[donor].append(explicit)
    if not any(by_atom):
        raise ValueError("No explicit donor hydrogens in the MiniTopol ATLAS source")
    width=max(map(len,by_atom))
    index=np.full((len(source_atoms),width),-1,dtype=np.intp)
    for row,items in zip(index,by_atom):
        row[:len(items)]=items
    return index


def geometric_neighborhood(raw_topology,source_atoms):
    atoms=raw_topology.atoms
    hydrogens=np.flatnonzero(np.isin(atoms.elements,("H","D","T")))
    if not len(hydrogens):
        raise ValueError("The ATLAS source contains no explicit hydrogen atoms")
    by_residue={}
    for rank,source in enumerate(source_atoms):
        by_residue.setdefault(atoms.resindices[source],[]).append(rank)
    eligible=[by_residue.get(residue,()) for residue in atoms.resindices[hydrogens]]
    width=max(map(len,eligible))
    nearest=np.full((len(hydrogens),width),-1,dtype=np.intp)
    for row,items in zip(nearest,eligible):
        row[:len(items)]=items
    return hydrogens,nearest


def geometric_hydrogens(heavy,raw,box,neighborhood):
    hydrogens,near=neighborhood
    delta=_mic(heavy[np.maximum(near,0)]-raw[hydrogens,None,:],box)
    distance=np.where(near>=0,np.linalg.norm(delta,axis=2),np.inf)
    order=np.argsort(distance,axis=1)
    row=np.arange(len(hydrogens))
    closest=distance[row,order[:,0]]
    next_best=distance[row,order[:,1]] if distance.shape[1]>1 else np.full(len(row),np.inf)
    certain=(closest>=.65)&(closest<=1.3)&(next_best>closest+.15)
    by_atom=[[] for _ in heavy]
    for h,atom in zip(hydrogens[certain],near[row[certain],order[certain,0]]):
        by_atom[atom].append(h)
    index=np.full((len(heavy),max(1,max(map(len,by_atom)))),-1,dtype=np.intp)
    for row,items in zip(index,by_atom):
        row[:len(items)]=items
    return index


def explicit_geometry(heavy, raw, box, pair, acceptor_anchor, attached):
    a,b=pair[:,:2].astype(np.intp).T
    index=attached[a]
    available=index>=0
    h=raw[np.maximum(index,0)]
    dh=_mic(h-heavy[a,None,:],box)
    ha=_mic(heavy[b,None,:]-h,box)
    distance=np.linalg.norm(ha,axis=-1)
    donor=np.sum(dh*ha,axis=-1)/np.maximum(np.linalg.norm(dh,axis=-1)*distance,1e-8)
    acceptor=unit(_mic(heavy[b]-heavy[acceptor_anchor[b]],box))
    direction=-ha/np.maximum(distance[:,:,None],1e-8)
    acceptor=np.einsum("ni,nki->nk",acceptor,direction)
    assessment=np.exp(-((distance-1.9)/.8)**2)*np.maximum(donor,0)**2*(.2+.8*np.maximum(acceptor,0))
    best=np.argmax(np.where(available,assessment,-1.),axis=1)
    row=np.arange(len(pair))
    count=available.sum(axis=1)
    order=np.sort(np.where(available,distance,12.),axis=1)
    second=order[:,1] if order.shape[1]>1 else np.full(len(pair),12.)
    return np.column_stack((order[:,0], np.where(count,donor[row,best],0),
                            np.where(count,acceptor[row,best],0), np.where(count,assessment[row,best],0),
                            count, second,
                            np.where(count,distance[row,best],12.),
                            np.sum(available & (donor>.75) & (distance<2.8),axis=1))).astype(np.float32)


@contextmanager
def open_explicit(reg,protein,replica):
    tid=trajectory(protein,replica)
    plan=source_plans(protein)[f"R{replica}"]
    if plan.get("error"):
        raise ValueError(f"MiniTopol ATLAS source verification failed for {tid}: {plan['error']}")
    atoms=reg.atoms(tid).to_pydict()
    source_atoms=np.asarray(atoms["source_atom"],dtype=np.intp)
    topo=topology(atoms)
    anchor=np.arange(len(source_atoms))
    anchor[topo[2][9]]=topo[2][12]
    raw_topology=mda.Universe(str(plan["topology"]),str(plan["xtc"]))
    with raw_topology.trajectory as reader:
        if not np.array_equal(raw_topology.atoms.names[source_atoms],atoms["name"]):
            raise ValueError(f"Source atom order differs from MiniTopol for {tid}")
        use_tpr=not plan.get("bond_error") and plan.get("source_to_tpr") is not None
        bonded=hydrogen_indices(raw_topology,plan,source_atoms) if use_tpr else None
        neighborhood=None if use_tpr else geometric_neighborhood(raw_topology,source_atoms)

        def read(origin,snapshot,pair):
            step=reader[origin]
            if not np.isclose(step.time,snapshot.time[0],rtol=0,atol=.01):
                raise ValueError(f"Explicit hydrogens and heavy atoms have different timestamps for {tid}")
            if not np.allclose(step.positions[source_atoms],snapshot.positions[0],rtol=0,atol=.002):
                raise ValueError(f"Explicit hydrogens and heavy atoms have different coordinates for {tid}")
            box=cell(snapshot.box[0])
            attached=bonded if use_tpr else geometric_hydrogens(snapshot.positions[0],step.positions,box,neighborhood)
            return explicit_geometry(snapshot.positions[0],step.positions,box,pair,anchor,attached)

        yield read


def add_explicit(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-explicit-hydrogen"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as arrays:
            if np.array_equal(arrays["pair"],part["pair"]) and np.array_equal(arrays["origin"],part["origin"]):
                part["explicit"]=arrays["features"]
                return
    with open_explicit(reg,protein,replica) as read:
        rows=[]
        for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
            origin=int(part["origin"][start])
            snapshot=next(reg.blocks(tid,start=origin,stop=origin+1))
            rows.append(read(origin,snapshot,part["pair"][start:end]))
    part["explicit"]=np.concatenate(rows)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,features=part["explicit"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,cache)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    panel=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    basic=load(TRAIN,25,geometry=True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    late=load(panel,100,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(add_explicit,part,stride) for parts,stride in
              ((basic,25),(middle,50),(late,100),(valid,100)) for part in parts]
        for job in jobs:
            job.result()
    configs=(("xgb",118,"hb-origin-explicit",False,6,1.),
             ("xgb",166,"hb-origin-explicit",False,5,1.),
             ("xgb",166,"hb-origin-explicit",True,5,1.))
    explore("explicit_hydrogen",configs,1351,basic+middle+late,valid)


if __name__=="__main__":
    main()
