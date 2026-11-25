"""Approximate flexibility from a single conformation's C-alpha network."""
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_single_structure import OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


def network_features(positions, box, pair, maps):
    ca_ids = np.array([m["CA"] for m in maps])
    residue = np.full(len(positions), -1, dtype=np.int32)
    for i, atoms in enumerate(maps):
        residue[list(atoms.values())] = i
    a, b = pair[:, :2].astype(np.int32).T
    left, right = residue[a],residue[b]
    if not (np.all(left >= 0) and np.all(right >= 0)):
        raise ValueError("Missing C-alpha assignment")
    ca = positions[ca_ids]
    delta = _mic(ca[:, None]-ca[None, :],box)
    length2 = np.einsum("ijk,ijk->ij",delta,delta)
    weights = np.where((length2 > 1e-5) & (length2 <= 100),16/np.maximum(length2,1),0)
    degree = weights.sum(1)
    stiffness = np.diag(degree+.10)-weights
    covariance = solve(stiffness,np.eye(len(ca)),assume_a="pos",check_finite=False)
    left_var, right_var = covariance.diagonal()[left], covariance.diagonal()[right]
    relative = np.maximum(left_var+right_var-2*covariance[left,right],0)
    direction = _mic(positions[b]-positions[a],box) / pair[:, 2:3]
    vector = delta/np.sqrt(np.maximum(length2,1e-6))[..., None]
    tensor = np.einsum("ij,ijk,ijl->ikl",weights,vector,vector)
    stiffness_a = np.einsum("ki,kij,kj->k",direction,tensor[left],direction)
    stiffness_b = np.einsum("ki,kij,kj->k",direction,tensor[right],direction)
    score = np.column_stack((
        relative, left_var, right_var, covariance[left,right],
        stiffness_a, stiffness_b, degree[left], degree[right],
    )).astype(np.float32)
    assert np.isfinite(score).all()
    return score


def add_flexibility(part,stride):
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-origin-flexibility"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.exists():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["flexibility"]=z["features"]
                return
    maps=topology(reg.atoms(tid).to_pydict())[0]
    out=[]
    with threadpool_limits(limits=1,user_api="blas"):
        for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
            origin=int(part["origin"][start])
            block=next(reg.blocks(tid,start=origin,stop=origin+1))
            out.append(network_features(block.positions[0],cell(block.box[0]),part["pair"][start:end],maps))
    features=np.concatenate(out)
    tmp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with tmp:
            np.savez(tmp,features=features,pair=part["pair"],origin=part["origin"])
        os.replace(tmp.name,cache)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    part["flexibility"]=features


def main():
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    valid=load(VALIDATION,100,geometry=True)
    for parts,stride in ((train,25),(extra,50),(valid,100)):
        for part in parts:
            add_flexibility(part,stride)
    configurations=(
        ("xgb",118,"hb-origin-flexibility",False,4,1.0),
        ("xgb",166,"hb-origin-flexibility",True,5,1.0),
        ("xgb",118,"hb-origin-flexibility",True,6,1.0),
        ("xgb-soft-local",118,"hb-origin-flexibility",False,4,1.0),
        ("xgb-soft-wide-local",166,"hb-origin-flexibility",True,5,1.0),
        ("xgb-soft-tune-0.15-0.25-320-local",142,"hb-origin-flexibility",True,5,1.0),
        ("xgb-soft-tune-0.35-0.25-400-local",118,"hb-origin-flexibility",True,4,1.0),
        ("xgb",166,"hb-origin-flexibility",True,6,.75),
    )
    explore("flexibility",configurations,621,train+extra,valid)
