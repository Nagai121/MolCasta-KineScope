"""CUDA training with a differentiable review-budget objective at recall 80%."""
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from .data import VALIDATION
from .hb_directional import add_directional
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump, view


class ReviewModel(nn.Module):
    def __init__(self,n,hidden):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(n,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self,x,base_logit):
        return base_logit + self.body(x).squeeze(1)


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).astype(np.float32)


def predict_review(parts,folder,row):
    torch.set_num_threads(4)
    state=torch.load(folder/row["fit"]["checkpoint"],map_location="cpu",weights_only=True)
    center,scale=state["center"],state["scale"]
    model=ReviewModel(len(center),state["hidden"]).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        return [model((torch.from_numpy(view(p,row["config"]))-center)/scale,
                       torch.from_numpy(logit(p["old"]))).sigmoid().numpy() for p in parts]


def train_review(name,folder,train,cfg,temperature,emphasis,prior_strength):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(27182)
    feature=np.concatenate([view(p,cfg) for p in train])
    y=np.concatenate([p["y"] for p in train])
    prior=logit(np.concatenate([p["old"] for p in train]))
    center=feature.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(1e-6,feature.std(0,dtype=np.float64)).astype(np.float32)
    xx=torch.as_tensor((feature-center)/scale,device="cuda")
    yy=torch.as_tensor(y,device="cuda",dtype=torch.float32)
    pp=torch.as_tensor(prior,device="cuda")
    pos=torch.as_tensor(np.flatnonzero(y),device="cuda")
    neg=torch.as_tensor(np.flatnonzero(~y),device="cuda")
    hidden=cfg[4]
    model=ReviewModel(feature.shape[1],hidden).cuda()
    threshold=torch.nn.Parameter(torch.tensor(float(np.quantile(prior[y],.2)),device="cuda"))
    dual=torch.tensor(.5,device="cuda")
    opt=torch.optim.AdamW([{"params":model.parameters(),"weight_decay":.035},
                           {"params":[threshold],"weight_decay":0.}],lr=.0007)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(18):
        for rows in torch.randperm(len(y),device="cuda").split(8192):
            opt.zero_grad(set_to_none=True)
            rank=model(xx[rows],pp[rows])
            positive=pos[torch.randint(len(pos),(768,),device="cuda")]
            negative=neg[torch.randint(len(neg),(2048,),device="cuda")]
            positive_score=model(xx[positive],pp[positive])
            negative_score=model(xx[negative],pp[negative])
            recall=((positive_score-threshold)/temperature).sigmoid().mean()
            false_review=((negative_score-threshold)/temperature).sigmoid().mean()
            violation=.80-recall
            loss=nn.functional.binary_cross_entropy_with_logits(rank,yy[rows])
            loss+=emphasis*(false_review+dual.detach()*violation)
            loss+=prior_strength*(rank-pp[rows]).square().mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                dual.add_(.018*violation.detach()).clamp_(0.,15.)
    torch.cuda.synchronize()
    checkpoint=folder/f"{name}.pt"
    torch.save(dict(model=model.cpu().state_dict(),center=torch.from_numpy(center),
                    scale=torch.from_numpy(scale),hidden=hidden),checkpoint)
    return dict(checkpoint=checkpoint.name,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                cuda="cuda:0",updater="AdamW",fit_seconds=time.perf_counter()-start,
                train_rows=len(y),train_positives=int(y.sum()),temperature=temperature,
                budget_emphasis=emphasis,prior_strength=prior_strength,
                trained_threshold=float(threshold.detach().cpu()),dual=float(dual.cpu()))


def main():
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("--volume",action="store_true")
    args=parser.parse_args()
    folder=OUT/("constraint_volume" if args.volume else "constraint")
    folder.mkdir(exist_ok=True)
    panel=json.loads((OUT/"diverse_training_panel.json").read_text())["proteins"]
    if args.volume:
        panel+=json.loads((OUT/"volume_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    former=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    recent=load(panel,100,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    for parts,stride in ((former,50),(recent,100),(valid,100)):
        for part in parts:
            add_directional(part,stride)
    variants=(
        ("environment",118,64,.3,.2,.002),
        ("local",118,64,.25,.12,.004),
        ("directional",118,96,.40,.2,.003),
        ("directional",142,64,.3,.35,.008),
        ("directional",166,96,.45,.1,.012),
        ("environment",118,96,.65,.08,.015),
    )
    if args.volume:
        from .hb_cloud import add_cloud
        from .hb_occupied import add_occupied

        for parts,stride in ((former,50),(recent,100),(valid,100)):
            for part in parts:
                add_cloud(part,stride)
                add_occupied(part,stride)
        variants=(("stacked-directional",118,128,.35,.15,.012),
                  ("stacked-occupied",166,128,.45,.2,.02),
                  ("directional",118,96,.55,.1,.022))
    result={}
    for number,(rep,width,hidden,temp,focus,prior) in enumerate(variants,1381 if args.volume else 1021):
        name=f"T1I{number}"
        cfg=("hb-constraint",width,"hb-origin-"+rep,False,hidden,1.)
        row=dict(config=cfg,fit=train_review(name,folder,former+recent,cfg,temp,focus,prior))
        row["validation"]=validation_report(valid,predict_review(valid,folder,row))
        result[name]=row
        dump(folder/"validation.json",result)
        print(name,row["validation"]["candidate"]["auprc"],key(row),flush=True)
    best=max(result,key=lambda n:key(result[n]))
    dump(folder/"selection.json",dict(name=best,**result[best]))


if __name__=="__main__":
    main()
