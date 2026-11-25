"""Train ensemble calibration on proteins reserved from all component fits."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from .data import HISTORICAL, TEST, TRAIN, VALIDATION
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_occupied import add_occupied
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, key, load, predict, validation_report
from .search_events import EXTRA_TRAINING, dump


def members(model):
    selection=json.loads((OUT/"selections"/model/"selection.json").read_text(encoding="utf-8"))
    weights={name:weight for name,weight in selection["weights"].items() if weight}
    records={}
    for path in sorted(OUT.glob("*/validation.json")):
        if not path.with_name("selection.json").is_file():
            continue
        rows=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows,dict):
            continue
        for name in set(weights)&rows.keys():
            row=rows[name]
            if not isinstance(row,dict) or "fit" not in row or "config" not in row:
                continue
            if name in records:
                raise ValueError(f"Duplicate component record for {name}")
            records[name]=(path.parent,row)
    if records.keys()!=weights.keys():
        raise ValueError("Missing ensemble components")
    return [(weights[name],records[name][0],records[name][1]) for name in weights]


def assemble(parts, components, scorer=predict):
    all_scores=[scorer(parts,folder,row) for _,folder,row in components]
    return [np.column_stack(scores).astype(np.float32) for scores in zip(*all_scores)]


def distance_context(part):
    return np.clip((part["pair"][:,2]-4.0)/2,-.25,1).astype(np.float32)


def role_context(part):
    base=part["base"]
    return np.column_stack((distance_context(part),base[:,3]-.5,base[:,5]-.5,
                            (base[:,3]>.5)*(base[:,5]>.5)-.5)).astype(np.float32)


def evaluate(scores,distance,record,role=None):
    weights=np.asarray(record["weights"],dtype=np.float32)
    if "gate" in record:
        logits=np.log(weights)[None,:]+role@np.asarray(record["gate"],dtype=np.float32)
        local_weights=np.exp(logits-logits.max(axis=1,keepdims=True))
        local_weights/=local_weights.sum(axis=1,keepdims=True)
        mixed=(scores*local_weights).sum(axis=1)
    else:
        mixed=scores@weights
    p=np.clip(mixed,1e-6,1-1e-6)
    margin=np.log(p/(1-p))*(record["slope"]+record["distance_slope"]*distance)
    margin+=record["offset"]+record["distance_offset"]*distance
    return 1/(1+np.exp(-np.clip(margin,-40,40)))


def read_calibration(folder,record):
    payload=(folder/record["fit"]["checkpoint"]).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=record["fit"]["sha256"]:
        raise ValueError("Calibration checkpoint changed")
    return json.loads(payload)


def component_representations(folder,record):
    representations={record["config"][2]}
    if record["config"][0]=="hb-calibration":
        for item in read_calibration(folder,record)["components"]:
            representations.update(component_representations(OUT/item["folder"],item["record"]))
    return representations


def predict_calibrated(parts,folder,record,scorer=predict):
    manifest=read_calibration(folder,record)
    components=[(None, OUT/row["folder"], row["record"]) for row in manifest["components"]]
    for _,component_folder,component in components:
        checkpoint=component_folder/component["fit"]["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=component["fit"]["sha256"]:
            raise ValueError(f"Calibration component changed: {checkpoint}")
    scores=assemble(parts,components,scorer)
    parameters=manifest["parameters"]
    return [evaluate(s,distance_context(p),parameters,role_context(p) if "gate" in parameters else None)
            for s,p in zip(scores,parts)]


def fit(scores,distances,labels,initial,protein_weights,option,roles=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Calibration requires CUDA")
    device="cuda"
    p=torch.as_tensor(scores,device=device)
    d=torch.as_tensor(distances,device=device)
    y=torch.as_tensor(labels,device=device,dtype=torch.float32)
    row_weight=torch.as_tensor(protein_weights,device=device)
    origin=torch.as_tensor(initial,device=device,dtype=torch.float32).log()
    adjustment=torch.nn.Parameter(torch.zeros_like(origin))
    intercept=torch.nn.Parameter(torch.zeros(2,device=device))
    log_slope=torch.nn.Parameter(torch.zeros(2,device=device))
    use_gate=option in ("role","role-tail")
    train_roles=torch.as_tensor(roles,device=device) if use_gate else None
    gate=(torch.nn.Parameter(torch.zeros((train_roles.shape[1],len(initial)),device=device))
          if use_gate else None)
    parameters=(adjustment,intercept,log_slope,gate) if use_gate else (adjustment,intercept,log_slope)
    optimizer=torch.optim.Adam(parameters,lr=.025)
    shape=option=="distance" or use_gate
    do_weights=option!="temperature"
    cutoff=float(np.quantile((scores@np.asarray(initial,dtype=np.float32))[labels],.20))
    if option=="role-tail":
        threshold=torch.logit(torch.tensor(cutoff,device=device).clamp(1e-5,1-1e-5))
        recall_reference=torch.tensor(.80,device=device)
    start=time.perf_counter()
    for _ in range(320):
        optimizer.zero_grad(set_to_none=True)
        coefficients=origin+do_weights*adjustment
        if use_gate:
            weights=(coefficients+train_roles@gate).softmax(1)
            mixed=(p*weights).sum(1)
        else:
            mixed=p@coefficients.softmax(0)
        margin=torch.logit(mixed,eps=1e-6)*(log_slope[0].exp()+shape*.4*log_slope[1].tanh()*d)
        margin+=intercept[0]+shape*intercept[1]*d
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(margin,y,reduction="none")*row_weight).mean()
        loss+=.012*adjustment.square().mean()+.007*(intercept[1].square()+log_slope[1].square())
        if use_gate:
            loss+=.025*gate.square().mean()
        if option=="role-tail":
            review=((margin-threshold)/.22).sigmoid()
            recall=(review*y).sum()/y.sum()
            false_review=(review*(1-y)).sum()/(1-y).sum()
            loss+=.12*(false_review+3*(recall_reference-recall).clamp_min(0).square())
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    parameters=dict(weights=(origin+do_weights*adjustment).softmax(0).detach().cpu().tolist(),
                    slope=float(log_slope[0].exp().item()),distance_slope=float(shape*.4*log_slope[1].tanh().item()),
                    offset=float(intercept[0].item()),distance_offset=float(shape*intercept[1].item()))
    if use_gate:
        parameters["gate"]=gate.detach().cpu().tolist()
    return parameters,time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("model",choices=("T1I2003","T1I2005"))
    args=parser.parse_args()
    gated=args.model=="T1I2005"
    panel_name="ensemble_gate_panel.json" if gated else "ensemble_calibration_panel.json"
    panel=json.loads((OUT/panel_name).read_text(encoding="utf-8"))["proteins"]
    if gated:
        previous=json.loads((OUT/"ensemble_calibration_panel.json").read_text(encoding="utf-8"))["proteins"]
        if set(panel)&set(previous):
            raise ValueError("Conditional ensemble training overlaps existing calibration proteins")
    for name in ("diverse_training_panel","volume_training_panel","volume_96_training_panel","volume_128_training_panel"):
        path=OUT/(name+".json")
        if not path.is_file():
            continue
        used=json.loads(path.read_text(encoding="utf-8"))["proteins"]
        if set(panel)&set(used):
            raise ValueError("Calibration proteins overlap a component training panel")
    if set(panel)&set(TRAIN+EXTRA_TRAINING+ADDITIONAL_TRAINING+VALIDATION+TEST+HISTORICAL):
        raise ValueError("Calibration proteins overlap a frozen evaluation or training panel")
    components=members(args.model)
    for _,folder,row in components:
        checkpoint=folder/row["fit"]["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=row["fit"]["sha256"]:
            raise ValueError(f"Ensemble component changed: {checkpoint}")
    calibration=load(panel,100,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    for part in calibration+valid:
        add_cloud(part,100)
        add_directional(part,100)
        add_occupied(part,100)
    cal=assemble(calibration,components)
    val=assemble(valid,components)
    initial=[weight for weight,_,_ in components]
    before=[s@np.asarray(initial,dtype=np.float32) for s in val]
    folder=OUT/("ensemble_gate" if gated else "ensemble_calibration")
    folder.mkdir(exist_ok=True)
    totals={name:sum(len(p["y"]) for p in calibration if p["protein"]==name) for name in panel}
    size=sum(totals.values())
    per_part=[np.full(len(p["y"]),size/(len(totals)*totals[p["protein"]]),dtype=np.float32) for p in calibration]
    result={}
    options=("temperature","role","role-tail") if gated else ("temperature","weights","distance")
    role_data=np.concatenate([role_context(part) for part in calibration]) if gated else None
    for index,option in enumerate(options,1451 if gated else 1321):
        name=f"T1I{index}"
        params,seconds=fit(np.concatenate(cal),np.concatenate([distance_context(p) for p in calibration]),
                           np.concatenate([p["y"] for p in calibration]),initial,np.concatenate(per_part),option,role_data)
        manifest=dict(parameters=params,source_model=args.model,calibration=panel,
                      components=[dict(folder=p.name,record=r) for _,p,r in components])
        path=folder/f"{name}.json"
        dump(path,manifest)
        row=dict(config=("hb-calibration",118,"hb-origin-calibration",False,0,1.0),
                 fit=dict(checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          cuda="cuda:0",updater="Adam",fit_seconds=seconds,train_rows=size,option=option),
                 validation=validation_report(valid,[evaluate(s,distance_context(p),params,role_context(p))
                                                     for s,p in zip(val,valid)]))
        result[name]=row
        print(name,key(row),flush=True)
    dump(folder/"validation.json",result)
    best=max(result,key=lambda name:key(result[name]))
    dump(folder/"selection.json",dict(name=best,parent=validation_report(valid,before),**result[best]))


if __name__=="__main__":
    main()
