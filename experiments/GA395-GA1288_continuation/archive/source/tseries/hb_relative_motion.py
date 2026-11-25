"""Predict projected pair motion with exact endpoint supervision at 1 ns."""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn

from .data import TRAIN, VALIDATION, cell, registry, trajectory
from .features import _mic
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_occupied import add_occupied
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump, view


def training_motion(part,stride):
    if "future_distance" not in part:
        raise ValueError("Motion supervision requires a training-only shard")
    protein,replica=part["protein"],part["replica"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-relative-motion"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path=directory/f"{stride}-{version}.npz"
    if path.is_file():
        with np.load(path) as data:
            if np.array_equal(data["pair"],part["pair"]) and np.array_equal(data["origin"],part["origin"]):
                part["relative_motion"]=data["targets"]
                return
    rows=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        block=next(reg.blocks(tid,start=t,stop=t+11,chunk_frames=11))
        if not np.isclose(block.time[-1]-block.time[0],1000,rtol=0,atol=.01):
            raise ValueError("Motion supervision requires coordinates exactly 1 ns apart")
        pair=part["pair"][start:end]
        a,b=pair[:,:2].astype(np.intp).T
        displacement=_mic(block.positions[-1,b]-block.positions[-1,a],cell(block.box[-1]))
        if not np.array_equal(np.linalg.norm(displacement,axis=-1)<=3.5,part["y"][start:end]):
            raise ValueError("Motion supervision and exact endpoint labels differ")
        direction=pair[:,3:6]/pair[:,2:3]
        radial=np.einsum("ni,ni->n",direction,displacement)
        lateral=np.linalg.norm(displacement-radial[:,None]*direction,axis=1)
        rows.append(np.column_stack((np.clip(radial-pair[:,2],-6,8),np.log1p(np.minimum(lateral,12)))))
    part["relative_motion"]=np.concatenate(rows).astype(np.float32)
    temp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with temp:
            np.savez(temp,targets=part["relative_motion"],pair=part["pair"],origin=part["origin"])
        os.replace(temp.name,path)
    finally:
        Path(temp.name).unlink(missing_ok=True)


class RelativeMotion(nn.Module):
    def __init__(self,features,hidden):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(features,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                   nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Linear(hidden,5))

    def forward(self,x,distance,prior,kinetic_weight):
        raw=self.network(x)
        mean=raw[:,1:3]
        scale=nn.functional.softplus(raw[:,3:5])+.08
        lateral=torch.expm1(mean[:,1].clamp(0,2.2))
        radius=(3.5**2-lateral.square()).clamp_min(1e-6).sqrt()
        projected=distance+mean[:,0]
        lower=(-radius-projected)/scale[:,0]
        upper=(radius-projected)/scale[:,0]
        chance=.5*(torch.erf(upper/np.sqrt(2.))-torch.erf(lower/np.sqrt(2.)))
        classifier=torch.sigmoid(.75*prior+raw[:,0])
        return (1-kinetic_weight)*classifier+kinetic_weight*chance,mean,scale


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).astype(np.float32,copy=False)


def fit(name,folder,train,config,kinetic_weight,aux_weight):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(27182)
    features=np.concatenate([view(p,config) for p in train])
    center=features.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(features.std(0,dtype=np.float64),1e-6).astype(np.float32)
    x=torch.as_tensor((features-center)/scale,device="cuda")
    del features
    distance=torch.as_tensor(np.concatenate([p["pair"][:,2] for p in train]),device="cuda")
    prior=torch.as_tensor(logit(np.concatenate([p["old"] for p in train])),device="cuda")
    y=torch.as_tensor(np.concatenate([p["y"] for p in train]),device="cuda",dtype=torch.float32)
    motion=torch.as_tensor(np.concatenate([p["relative_motion"] for p in train]),device="cuda")
    model=RelativeMotion(x.shape[1],config[4]).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.025)
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(22):
        for index in torch.randperm(len(x),device="cuda").split(8192):
            optim.zero_grad(set_to_none=True)
            p,mean,sigma=model(x[index],distance[index],prior[index],kinetic_weight)
            classification=nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y[index])
            dynamics=(.5*((motion[index]-mean)/sigma).square()+sigma.log()).mean()
            (classification+aux_weight*dynamics).backward()
            optim.step()
    torch.cuda.synchronize()
    path=folder/f"{name}.pt"
    torch.save(dict(model=model.cpu().state_dict(),center=torch.from_numpy(center),scale=torch.from_numpy(scale),
                    hidden=config[4],kinetic_weight=kinetic_weight),path)
    return dict(checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cuda="cuda:0",
                updater="AdamW",fit_seconds=time.perf_counter()-start,train_rows=len(x),
                train_positives=int(y.sum().item()),cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                kinetic_weight=kinetic_weight,auxiliary_weight=aux_weight)


def predict_motion(parts,folder,record):
    state=torch.load(folder/record["fit"]["checkpoint"],map_location="cpu",weights_only=True)
    center,scale=state["center"],state["scale"]
    model=RelativeMotion(len(center),state["hidden"]).eval()
    model.load_state_dict(state["model"])
    rows=[]
    with torch.inference_mode():
        for part in parts:
            x=(torch.from_numpy(view(part,record["config"]))-center)/scale
            distance=torch.from_numpy(part["pair"][:,2])
            prior=torch.from_numpy(logit(part["old"]))
            p,_,_=model(x,distance,prior,state["kinetic_weight"])
            rows.append(p.numpy())
    return rows


def main():
    folder=OUT/"relative_motion"
    folder.mkdir(exist_ok=True)
    added=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    basic=load(TRAIN,25,targets=True,geometry=True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,targets=True,geometry=True)
    late=load(added,100,targets=True,geometry=True)
    valid=load(VALIDATION,100,geometry=True)

    def prepare(part,stride,training):
        add_cloud(part,stride)
        add_directional(part,stride)
        add_occupied(part,stride)
        if training:
            training_motion(part,stride)

    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(prepare,part,stride,training) for parts,stride,training in
              ((basic,25,True),(middle,50,True),(late,100,True),(valid,100,False)) for part in parts]
        for job in jobs:
            job.result()
    result={}
    for index,(width,rep,kinetic,aux) in enumerate(((118,"directional",.08,.15),
                                                   (118,"stacked-directional",.18,.12),
                                                   (166,"stacked-cloud",0.,.2)),1371):
        name=f"T1I{index}"
        config=("hb-relative-motion",width,"hb-origin-"+rep,False,128,1.)
        trained=fit(name,folder,basic+middle+late,config,kinetic,aux)
        row=dict(config=config,fit=trained)
        row["validation"]=validation_report(valid,predict_motion(valid,folder,row))
        result[name]=row
        dump(folder/"validation.json",result)
        print(name,row["validation"]["candidate"]["auprc"],key(row),flush=True)
    best=max(result,key=lambda n:key(result[n]))
    dump(folder/"selection.json",dict(name=best,**result[best]))


if __name__=="__main__":
    main()
