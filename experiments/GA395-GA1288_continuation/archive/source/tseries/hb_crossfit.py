"""Cross-fitted static ensembles with calibration only on training proteins."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .data import TRAIN, VALIDATION
from .hb_directional import add_directional
from .hb_exposure import add_exposure
from .hb_motif import add_motif
from .hb_single_structure import OUT, key, load, validation_report
from .search_events import EXTRA_TRAINING, cpu_predict, dump, fit, view


def add_union(parts,stride):
    def one(part):
        add_directional(part,stride)
        add_motif(part,stride)
        add_exposure(part,stride)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(one,parts))


def odds(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def cross_predict(parts,folder,record):
    path=folder/record["fit"]["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record["fit"]["sha256"]:
        raise ValueError("Crossfit manifest changed")
    descriptor=json.loads(path.read_text())
    matrices=[view(p,descriptor["config"]) for p in parts]
    aggregate=[np.zeros(len(p["pair"]),dtype=np.float64) for p in parts]
    for model in descriptor["folds"]:
        file=folder/model["checkpoint"]
        if hashlib.sha256(file.read_bytes()).hexdigest()!=model["sha256"]:
            raise ValueError("Crossfit component changed")
        predict=cpu_predict(file,descriptor["config"])
        for dst,x in zip(aggregate,matrices):
            dst+=predict(x)/len(descriptor["folds"])
    scale,prior_weight,bias=descriptor["calibration"]
    return [1/(1+np.exp(-np.clip(scale*odds(p)+prior_weight*odds(row["old"])+bias,-40,40)))
            for row,p in zip(parts,aggregate)]


def crossfit(tag,folder,train,config,inherited):
    partitions=TRAIN+EXTRA_TRAINING
    models=[]
    calibration=[]
    total=0.
    for fold in range(4):
        held=set(partitions[fold::4])
        fit_parts=[p for p in train if p["protein"] not in held]
        held_parts=[p for p in train if p["protein"] in held]
        record=fit(f"{tag}_fold{fold}",folder,fit_parts,config)
        total+=record["fit_seconds"]
        predictor=cpu_predict(folder/record["checkpoint"],config)
        for p in held_parts:
            if not inherited or p["protein"] in EXTRA_TRAINING:
                calibration.append((np.column_stack((odds(predictor(view(p,config))),odds(p["old"]))),
                                    p["y"]))
        models.append(record)
    x=torch.as_tensor(np.concatenate([p[0] for p in calibration]),device="cuda",dtype=torch.float32)
    y=torch.as_tensor(np.concatenate([p[1] for p in calibration]),device="cuda",dtype=torch.float32)
    parameters=torch.nn.Parameter(torch.tensor([1.,0.,0.],device="cuda"))
    active=torch.tensor(float(inherited),device="cuda")
    optimizer=torch.optim.Adam([parameters],lr=.018)
    for _ in range(260):
        optimizer.zero_grad()
        raw=x[:,0]*parameters[0]+active*x[:,1]*parameters[1]+parameters[2]
        penalty=.005*(parameters[0]-1).square()+.004*parameters[1].square()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(raw,y)+penalty
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    p=parameters.detach().cpu().tolist()
    if not inherited: p[1]=0.
    manifest=folder/f"{tag}_ensemble.json"
    dump(manifest,dict(config=config,folds=models,calibration=p))
    return dict(checkpoint=manifest.name,sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                fit_seconds=total,cuda="cuda:0",updater="grow_gpu_hist",folds=4,
                train_rows=sum(len(p["y"]) for p in train))


def main():
    folder=OUT/"crossfit"
    folder.mkdir(exist_ok=True)
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    valid=load(VALIDATION,100,geometry=True)
    for rows,stride in ((train,25),(extra,50),(valid,100)):
        add_union(rows,stride)
    options=(
        (("xgb",118,"hb-origin-union",False,5,1.0),False),
        (("xgb",166,"hb-origin-union",False,6,1.0),True),
        (("xgb-soft-tune-0.15-0.25-320-local",118,"hb-origin-union",False,5,1.0),False),
        (("xgb",118,"hb-origin-environment",False,6,1.0),True),
    )
    log={}
    for number,(cfg,prior) in enumerate(options,781):
        name=f"T1I{number}"
        row=dict(config=("hb-crossfit",*cfg[1:]),fit=crossfit(name,folder,train+extra,cfg,prior))
        row["validation"]=validation_report(valid,cross_predict(valid,folder,row))
        log[name]=row
        dump(folder/"validation.json",log)
        print(name,row["validation"]["candidate"]["auprc"],key(row),flush=True)
    winner=max(log,key=lambda n:key(log[n]))
    dump(folder/"selection.json",dict(name=winner,**log[winner]))


if __name__=="__main__":
    main()
