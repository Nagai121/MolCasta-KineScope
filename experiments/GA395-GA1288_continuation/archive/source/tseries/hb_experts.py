"""Distance- and chemistry-specific origin-frame classifiers."""
import hashlib
import json
import time

import numpy as np
import torch
import xgboost as xgb

from .data import TRAIN, VALIDATION
from .hb_single_structure import OUT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump, view


def groups(part, mode):
    x=part["base"]
    if mode=="chemistry":
        return (2*(x[:,3]==1)+(x[:,5]==1)).astype(np.int32)
    limits=np.array([4.0,4.5,5.0]) if mode=="distance" else np.array([4.0,4.75])
    return np.searchsorted(limits,x[:,0],side="left")


def logits(parts, models, cfg, mode):
    for part in parts:
        group=groups(part,mode)
        x=view(part,cfg)
        raw=np.empty(len(group),dtype=np.float32)
        for index,model in enumerate(models):
            keep=group==index
            if keep.any():
                raw[keep]=model.inplace_predict(x[keep],predict_type="margin")
        yield raw,group


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def predict_experts(parts,folder,record):
    info=record["fit"]
    path=folder/info["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info["sha256"]:
        raise ValueError("Modified expert manifest")
    manifest=json.loads(path.read_text())
    models=[]
    for file,sha in manifest["models"]:
        path=folder/file
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError(f"Modified checkpoint {file}")
        model=xgb.Booster()
        model.load_model(path)
        model.set_param(dict(device="cpu",nthread=4))
        models.append(model)
    weight=np.asarray(manifest["weights"])
    bias=np.asarray(manifest["bias"])
    result=[]
    for (raw,group),part in zip(logits(parts,models,record["config"],manifest["grouping"]),parts):
        adjusted=weight[0]*raw+weight[1]*logit(part["old"])+bias[group]
        result.append(1/(1+np.exp(-np.clip(adjusted,-40,40))))
    return result


def fit_experts(name,folder,train,calibration,cfg,mode,aux,rounds=180):
    started=time.perf_counter()
    x=np.concatenate([view(p,cfg) for p in train])
    y=np.concatenate([p["y"] for p in train])
    classes=np.concatenate([groups(p,mode) for p in train])
    actual="cuda:0"
    updater="grow_gpu_hist"
    count=3 if mode=="coarse" else 4
    models=[]
    paths=[]
    labels=y
    if aux:
        gap=np.concatenate([p["future_distance"] for p in train])-3.5
        labels=.82*y+.18/(1+np.exp(np.clip(gap/.25,-30,30)))
    for index in range(count):
        keep=classes==index
        if len(np.unique(y[keep]))!=2:
            raise ValueError(f"Training group {index} requires both classes")
        data=xgb.QuantileDMatrix(x[keep],label=labels[keep],nthread=4)
        model=xgb.Booster(dict(device="cuda:0",tree_method="hist",objective="binary:logistic",
                                eta=.05,max_depth=cfg[4],seed=27182,colsample_bytree=.9,
                                subsample=.9,min_child_weight=10,nthread=4),[data])
        model.update(data,0)
        info=json.loads(model.save_config())["learner"]
        actual=info["generic_param"]["device"]
        updater=info["gradient_booster"]["gbtree_train_param"]["updater_seq"]
        if actual!="cuda:0" or "gpu_hist" not in updater:
            raise RuntimeError("Expert training did not use the CUDA histogram updater")
        for i in range(1,rounds):
            model.update(data,i)
        path=folder/f"{name}_group{index}.ubj"
        model.save_model(path)
        paths.append((path.name,hashlib.sha256(path.read_bytes()).hexdigest()))
        cpu=xgb.Booster(model_file=path)
        cpu.set_param(dict(device="cpu",nthread=4))
        models.append(cpu)
        del model, data
        if rounds >= 400:
            print(name,"CUDA expert",index+1,"/",count,flush=True)
    pairs=list(logits(calibration,models,cfg,mode))
    feature=np.column_stack((np.concatenate([p[0] for p in pairs]),
                             logit(np.concatenate([p["old"] for p in calibration]))))
    xx=torch.as_tensor(feature,device="cuda",dtype=torch.float32)
    sections=torch.as_tensor(np.concatenate([p[1] for p in pairs]),device="cuda",dtype=torch.long)
    target=torch.as_tensor(np.concatenate([p["y"] for p in calibration]),device="cuda",dtype=torch.float32)
    raw=torch.nn.Parameter(torch.zeros(2,device="cuda"))
    bias=torch.nn.Parameter(torch.zeros(count,device="cuda"))
    optimizer=torch.optim.Adam((raw,bias),lr=.02)
    for _ in range(250):
        optimizer.zero_grad()
        weights=raw.softmax(dim=0)
        penalty=.003*bias.square().mean()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(xx@weights+bias[sections],target)+penalty
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    path=folder/f"{name}_experts.json"
    proteins=tuple(dict.fromkeys(part["protein"] for part in calibration))
    dump(path,dict(models=paths,grouping=mode,weights=raw.softmax(dim=0).detach().cpu().tolist(),
                   bias=bias.detach().cpu().tolist(),calibration_proteins=proteins,
                   calibration_protein=proteins[0] if len(proteins)==1 else None))
    return dict(checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cuda=actual,updater=updater,fit_seconds=time.perf_counter()-started,
                train_rows=len(y),train_positives=int(y.sum()),grouping=mode,soft_target=aux,rounds=rounds)


def main():
    import argparse
    from .hb_single_structure import ADDITIONAL_TRAINING

    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--volume",action="store_true")
    modes.add_argument("--expanded10",action="store_true")
    parser.add_argument("--long",action="store_true")
    args=parser.parse_args()
    if args.long and not args.expanded10:
        parser.error("--long requires --expanded10")
    folder=OUT/("ap10_experts_long" if args.long else "ap10_experts_expanded" if args.expanded10
                else "experts_volume" if args.volume else "experts")
    folder.mkdir(exist_ok=True)
    controls=(("distance","local",118,4,False),("distance","environment",118,5,True),
              ("coarse","local",142,5,True),("coarse","competition",142,4,False),
              ("chemistry","environment",118,6,False),("chemistry","local",166,5,True))
    if args.expanded10:
        from concurrent.futures import ThreadPoolExecutor
        from .hb_cloud import add_cloud
        from .hb_directional import add_directional
        from .hb_explicit_context import add_explicit_context
        from .hb_explicit_hydrogen import add_explicit
        from .hb_occupied import add_occupied
        from .hb_validation10 import panel10

        validation_names=panel10()
        calibration_names=tuple(json.loads((OUT/"ap10_calibration_panel.json").read_text())["proteins"])
        panels=("diverse_training_panel","volume_training_panel","volume_96_training_panel",
                "volume_128_training_panel","ap10_volume_144_training_panel")
        extra=tuple(p for name in panels for p in json.loads((OUT/f"{name}.json").read_text())["proteins"])
        middle=EXTRA_TRAINING+ADDITIONAL_TRAINING
        if set(TRAIN+middle+extra)&set(validation_names+calibration_names):
            raise ValueError("Training overlaps validation or calibration")
        if set(validation_names)&set(calibration_names):
            raise ValueError("Calibration overlaps validation")
        batches=[(load(names,stride,geometry=True),stride) for names,stride in
                 ((TRAIN,25),(middle,50),(extra,100))]
        calibration=load(calibration_names,100,geometry=True)
        valid=load(validation_names,100,geometry=True)

        def prepare(part, stride):
            add_cloud(part,stride)
            add_directional(part,stride)
            add_occupied(part,stride)
            add_explicit(part,stride)
            add_explicit_context(part)

        with ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(prepare,p,stride) for parts,stride in
                  (*batches,(calibration,100),(valid,100)) for p in parts]
            for job in jobs:
                job.result()
        train=[p for parts,_ in batches for p in parts]
        controls=(("chemistry","stacked-explicit",166,7,False),
                  ("coarse","stacked-explicit_context",166,7,False))
        if args.long:
            controls=controls[:1]
    else:
        train=load(TRAIN,25,True,True)+load(tuple(p for p in EXTRA_TRAINING if p!="2w0g_A"),50,True,True)
        calibration=load(("2w0g_A",),50,geometry=True)
        valid=load(VALIDATION,100,geometry=True)
    if args.volume:
        panels=(OUT/"diverse_training_panel.json",OUT/"volume_training_panel.json")
        extra=tuple(p for panel in panels for p in json.loads(panel.read_text(encoding="utf-8"))["proteins"])
        train+=load(ADDITIONAL_TRAINING,50,True,True)+load(extra,100,True,True)
        controls=controls[1::2]
    result={}
    first=1871 if args.long else 1831 if args.expanded10 else 1181 if args.volume else 701
    for index,(group,kind,width,depth,aux) in enumerate(controls,first):
        tag=f"T1I{index}"
        cfg=("hb-experts",width,"hb-origin-"+kind,False,depth,1.0)
        r=dict(config=cfg,fit=fit_experts(tag,folder,train,calibration,cfg,group,aux,
                                        rounds=600 if args.long else 420 if args.expanded10 else 180))
        r["validation"]=validation_report(valid,predict_experts(valid,folder,r))
        result[tag]=r
        dump(folder/"validation.json",result)
        print(tag,r["validation"]["candidate"]["auprc"],key(r),flush=True)
    winner=max(result,key=lambda k:key(result[k]))
    dump(folder/"selection.json",dict(name=winner,**result[winner]))


if __name__=="__main__":
    main()
