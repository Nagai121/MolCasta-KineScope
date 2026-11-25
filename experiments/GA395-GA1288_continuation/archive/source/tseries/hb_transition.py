"""Learn from multiple future labels while inferring only from the origin frame."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
from torch import nn

from .data import HISTORICAL, TEST, TRAIN, VALIDATION, cell, registry, trajectory
from .features import _mic
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump, view

LAGS = (1, 3, 5, 10)


def auxiliary_labels(part, stride, training=TRAIN+EXTRA_TRAINING):
    protein, replica = part["protein"],part["replica"]
    if protein not in training or protein in VALIDATION+TEST+HISTORICAL:
        raise ValueError("Auxiliary targets are restricted to training proteins")
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    directory=ROOT/protein/"trajectories"/tid/"hb-auxiliary-targets"
    directory.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=directory/f"{stride}-{version}.npz"
    if cache.exists():
        with np.load(cache) as f:
            if np.array_equal(f["pair"],part["pair"]) and np.array_equal(f["origin"],part["origin"]):
                part["aux"]=f["aux"]
                return
    result=[]
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        t=int(part["origin"][start])
        b=next(reg.blocks(tid,start=t,stop=t+11,chunk_frames=11))
        if not np.allclose(b.time[list(LAGS)]-b.time[0],100*np.asarray(LAGS),rtol=0,atol=.01):
            raise ValueError(f"Unexpected time grid in {tid}")
        a,c=part["pair"][start:end,:2].astype(np.int64).T
        d=np.column_stack([np.linalg.norm(_mic(b.positions[h,c]-b.positions[h,a],cell(b.box[h])),axis=1)
                           for h in LAGS])
        if not np.array_equal(d[:,-1]<=3.5,part["y"][start:end]):
            raise ValueError(f"Auxiliary label mismatch in {tid}")
        result.append(d)
    aux=np.concatenate(result).astype(np.float32)
    tmp=NamedTemporaryFile(dir=directory,delete=False)
    try:
        with tmp:
            np.savez(tmp,pair=part["pair"],origin=part["origin"],aux=aux)
        os.replace(tmp.name,cache)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    part["aux"]=aux


class Transition(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                   nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                   nn.Linear(hidden,5))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    def forward(self,x,prior):
        raw=self.network(x)
        return torch.cat((raw[:,:3],raw[:,3:4]+prior[:,None],raw[:,4:]),dim=1)


def to_logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def train_network(name,folder,parts,cfg,aux_weight,tail_weight):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.manual_seed(27182)
    x=np.concatenate([view(p,cfg) for p in parts])
    distances=np.concatenate([p["aux"] for p in parts])
    labels=distances<=3.5
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(1e-6,x.std(0,dtype=np.float64)).astype(np.float32)
    static=torch.as_tensor((x-center)/scale,device="cuda")
    target=torch.as_tensor(labels,device="cuda",dtype=torch.float32)
    delta=torch.as_tensor(np.clip(distances[:,-1]-3.5,-1.5,3),device="cuda")
    raw_prior=np.concatenate([p["old"] for p in parts])
    prior=torch.as_tensor(to_logit(raw_prior),device="cuda")
    hidden=cfg[4]
    model=Transition(x.shape[1],hidden).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.025)
    positives=torch.as_tensor(np.flatnonzero(labels[:,-1]),device="cuda")
    negatives=torch.as_tensor(np.flatnonzero(~labels[:,-1]),device="cuda")
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(20):
        for ids in torch.randperm(len(x),device="cuda").split(8192):
            optimizer.zero_grad(set_to_none=True)
            p=model(static[ids],prior[ids])
            loss=nn.functional.binary_cross_entropy_with_logits(p[:,3],target[ids,3])
            loss += aux_weight*nn.functional.binary_cross_entropy_with_logits(p[:,:3],target[ids,:3])
            loss += .04*nn.functional.smooth_l1_loss(p[:,4],delta[ids])
            if tail_weight:
                pos=positives[torch.randint(len(positives),(96,),device="cuda")]
                neg=negatives[torch.randint(len(negatives),(384,),device="cuda")]
                low=model(static[pos],prior[pos])[:,3].sort().values[13:36]
                high=model(static[neg],prior[neg])[:,3].topk(40).values
                loss += tail_weight*nn.functional.softplus(high[:,None]-low).mean()
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    checkpoint=folder/f"{name}.pt"
    torch.save(dict(model=model.cpu().state_dict(),center=torch.from_numpy(center),
                     scale=torch.from_numpy(scale),hidden=hidden),checkpoint)
    return dict(cuda="cuda:0",updater="AdamW",checkpoint=checkpoint.name,
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                fit_seconds=time.perf_counter()-start,train_rows=len(labels),
                train_positives=int(labels[:,-1].sum()),cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                auxiliary_weight=aux_weight,tail_weight=tail_weight)


def transition_predict(parts,folder,record):
    torch.set_num_threads(4)
    state=torch.load(folder/record["fit"]["checkpoint"],map_location="cpu",weights_only=True)
    center,scale=state["center"],state["scale"]
    model=Transition(len(center),state["hidden"]).eval()
    model.load_state_dict(state["model"])
    result=[]
    with torch.inference_mode():
        for part in parts:
            x=torch.from_numpy(view(part,record["config"]))
            prior=torch.from_numpy(to_logit(part["old"]))
            result.append(torch.sigmoid(model((x-center)/scale,prior)[:,3]).numpy())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--volume",action="store_true")
    args=parser.parse_args()
    folder=OUT/("transition_volume" if args.volume else "transition")
    folder.mkdir(exist_ok=True)
    train=load(TRAIN,25,geometry=True)
    extra=load(EXTRA_TRAINING,50,geometry=True)
    validation=load(VALIDATION,100,geometry=True)
    configs=((118,"environment",96,.2,0),(166,"competition",128,.25,0),
             (118,"local",96,.30,0),(166,"local",96,.15,.02),
             (118,"local",128,.35,.04),(118,"environment",96,0,.025))
    allowed=TRAIN+EXTRA_TRAINING
    late=[]
    if args.volume:
        from .hb_cloud import add_cloud
        from .hb_directional import add_directional
        from .hb_occupied import add_occupied

        files=(OUT/"diverse_training_panel.json",OUT/"volume_training_panel.json")
        added=tuple(protein for path in files for protein in json.loads(path.read_text(encoding="utf-8"))["proteins"])
        allowed+=ADDITIONAL_TRAINING+added
        extra+=load(ADDITIONAL_TRAINING,50,geometry=True)
        late=load(added,100,geometry=True)
        configs=((118,"directional",128,.30,0),(118,"stacked-directional",128,.20,.012),
                 (166,"stacked-occupied",128,.28,.025))
        for parts,stride in ((train,25),(extra,50),(late,100),(validation,100)):
            for part in parts:
                add_cloud(part,stride)
                add_directional(part,stride)
                add_occupied(part,stride)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(auxiliary_labels,p,stride,allowed) for parts,stride in ((train,25),(extra,50),(late,100))
              for p in parts]
        for job in jobs:
            job.result()
    extra+=late
    log={}
    for i,(width,rep,hidden,aux,tail) in enumerate(configs,1341 if args.volume else 681):
        name=f"T1I{i}"
        cfg=("hb-transition",width,f"hb-origin-{rep}",False,hidden,1.0)
        record=dict(config=cfg,fit=train_network(name,folder,train+extra,cfg,aux,tail))
        record["validation"]=validation_report(validation,transition_predict(validation,folder,record))
        log[name]=record
        dump(folder/"validation.json",log)
        print(name,record["validation"]["candidate"]["auprc"],key(record),flush=True)
    best=max(log,key=lambda n:key(log[n]))
    dump(folder/"selection.json",dict(name=best,**log[best]))


if __name__=="__main__":
    main()
