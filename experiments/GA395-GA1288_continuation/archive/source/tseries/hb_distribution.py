"""Predict a distribution of 1 ns distances from origin-only features."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from .data import TRAIN, VALIDATION
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump, view


class DistanceDistribution(nn.Module):
    def __init__(self,width,hidden,components):
        super().__init__()
        self.components=components
        self.encoder=nn.Sequential(nn.Linear(width,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                   nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                   nn.Linear(hidden,components*3))
    def forward(self, x, origin):
        weight, mean, width = self.encoder(x).chunk(3,dim=1)
        weight=weight.log_softmax(dim=1)
        mean=origin[:,None]+4*torch.tanh(mean)
        width=.08+nn.functional.softplus(width)
        return weight,mean,width


def probability(weight,mean,width):
    return (weight.exp()*torch.special.ndtr((3.5-mean)/width)).sum(dim=1)


def predict_distribution(parts,folder,row):
    torch.set_num_threads(4)
    state=torch.load(folder/row["fit"]["checkpoint"],map_location="cpu",weights_only=True)
    mean,scale=state["center"],state["scale"]
    model=DistanceDistribution(len(mean),state["hidden"],state["count"]).eval()
    model.load_state_dict(state["model"])
    result=[]
    with torch.inference_mode():
        for p in parts:
            x=torch.from_numpy(view(p,row["config"]))
            distance=torch.from_numpy(p["pair"][:,2])
            estimate=probability(*model((x-mean)/scale,distance)).numpy()
            result.append((1-state["prior"])*estimate+state["prior"]*p["old"])
    return result


def fit_distribution(name,folder,parts,cfg,count,coefficient,prior):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(27182)
    x=np.concatenate([view(p,cfg) for p in parts])
    distance=np.concatenate([p["pair"][:,2] for p in parts])
    y=np.concatenate([p["future_distance"] for p in parts])
    labels=y<=3.5
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(1e-6,x.std(0,dtype=np.float64)).astype(np.float32)
    features=torch.as_tensor((x-center)/scale,device="cuda")
    origin=torch.as_tensor(distance,device="cuda")
    outcome=torch.as_tensor(np.clip(y,1,16),device="cuda")
    target=torch.as_tensor(labels,device="cuda",dtype=torch.float32)
    previous=torch.as_tensor(np.concatenate([p["old"] for p in parts]),device="cuda")
    hidden=cfg[4]
    model=DistanceDistribution(x.shape[1],hidden,count).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.025)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(19):
        for ids in torch.randperm(len(y),device="cuda").split(8192):
            opt.zero_grad(set_to_none=True)
            w,m,s=model(features[ids],origin[ids])
            log_density=(w-.5*((outcome[ids,None]-m)/s)**2-s.log()).logsumexp(1)
            p=(1-prior)*probability(w,m,s)+prior*previous[ids]
            loss=nn.functional.binary_cross_entropy(p.clamp(1e-7,1-1e-7),target[ids])
            near=torch.where(origin[ids]<=4.2,1.0,.6)
            loss-=coefficient*(near*log_density).mean()
            loss.backward()
            opt.step()
    torch.cuda.synchronize()
    path=folder/f"{name}.pt"
    torch.save(dict(model=model.cpu().state_dict(),center=torch.from_numpy(center),
                    scale=torch.from_numpy(scale),hidden=hidden,count=count,prior=prior),path)
    return dict(checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cuda="cuda:0",updater="AdamW",fit_seconds=time.perf_counter()-start,
                train_rows=len(y),train_positives=int(labels.sum()),components=count,
                likelihood_weight=coefficient,prior_weight=prior)


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--volume",action="store_true")
    modes.add_argument("--expanded10",action="store_true")
    parser.add_argument("--explicit",action="store_true")
    args=parser.parse_args()
    if args.explicit and not args.expanded10:
        parser.error("--explicit requires --expanded10")
    folder=OUT/("ap10_distribution_explicit" if args.explicit else "ap10_distribution_expanded" if args.expanded10 else
                "distribution_volume" if args.volume else "distribution")
    folder.mkdir(exist_ok=True)
    basic=load(TRAIN,25,True,True)
    middle=load(EXTRA_TRAINING,50,True,True)
    train=basic+middle
    validation=VALIDATION
    if args.expanded10:
        from .hb_validation10 import panel10
        validation=panel10()
    valid=load(validation,100,geometry=True)
    choices=((118,"environment",96,2,.12,.35),(118,"local",96,3,.15,.50),
             (166,"local",128,3,.06,.40),(142,"competition",96,2,.20,.60),
             (118,"local",96,2,.06,.70))
    if args.volume or args.expanded10:
        from .hb_directional import add_directional
        from .hb_scaling import take_groups

        panels=(OUT/"diverse_training_panel.json",OUT/"volume_training_panel.json")
        if args.expanded10:
            panels+=tuple(OUT/f"{name}.json" for name in (
                "volume_96_training_panel", "volume_128_training_panel", "ap10_volume_144_training_panel"))
        extra=tuple(p for panel in panels for p in json.loads(panel.read_text(encoding="utf-8"))["proteins"])
        if set(TRAIN+EXTRA_TRAINING+ADDITIONAL_TRAINING+extra)&set(validation):
            raise ValueError("Training and validation proteins overlap")
        groups=((basic,25),(middle+load(ADDITIONAL_TRAINING,50,True,True),50),
                (load(extra,100,True,True),100))
        for parts,stride in (*groups,(valid,100)):
            for part in parts:
                add_directional(part,stride)
        if args.explicit:
            from concurrent.futures import ThreadPoolExecutor
            from .hb_cloud import add_cloud
            from .hb_explicit_context import add_explicit_context
            from .hb_explicit_hydrogen import add_explicit
            from .hb_occupied import add_occupied

            def prepare(part, stride):
                add_cloud(part,stride)
                add_occupied(part,stride)
                add_explicit(part,stride)
                add_explicit_context(part)

            with ThreadPoolExecutor(3) as executor:
                jobs = [executor.submit(prepare,part,stride)
                        for parts,stride in (*groups,(valid,100)) for part in parts]
                for job in jobs:
                    job.result()
        train=take_groups([part for parts,_ in groups for part in parts],
                          1_400_000 if args.expanded10 else 720_000,27182)
        choices=(choices[3],(118,"directional",128,3,.08,.25),
                 (118,"directional",128,2,.12,.1))
    if args.expanded10:
        choices=((118,"directional",128,2,.12,.10),(166,"directional",160,3,.06,.10))
        print("AP_DISTANCE_DISTRIBUTION_DATA",len({p["protein"] for p in train}),
              sum(len(p["y"]) for p in train),len(validation),flush=True)
    if args.explicit:
        choices=((118,"stacked-explicit",128,2,.10,.10),
                 (118,"stacked-explicit_context",128,3,.06,.10))
    log={}
    first=1821 if args.explicit else 1721 if args.expanded10 else 1191 if args.volume else 801
    for nr,(width,rep,hidden,components,loss,prior) in enumerate(choices,first):
        name=f"T1I{nr}"
        cfg=("hb-distribution",width,f"hb-origin-{rep}",False,hidden,1.)
        row=dict(config=cfg,fit=fit_distribution(name,folder,train,cfg,components,loss,prior))
        row["validation"]=validation_report(valid,predict_distribution(valid,folder,row))
        log[name]=row
        dump(folder/"validation.json",log)
        print(name,row["validation"]["candidate"]["auprc"],key(row),flush=True)
    best=max(log,key=lambda n:key(log[n]))
    dump(folder/"selection.json",dict(name=best,**log[best]))


if __name__=="__main__":
    main()
