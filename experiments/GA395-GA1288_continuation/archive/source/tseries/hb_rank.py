"""Train and calibrate origin-only GPU rankers for hydrogen-bond candidates."""
import hashlib
import json
import time

import numpy as np
import torch
import xgboost as xgb

from .hb_single_structure import OUT, key, load, validation_report
from .data import TRAIN, VALIDATION
from .search_events import EXTRA_TRAINING, dump, view


def rank_predict(parts, folder, record):
    path = folder/record["fit"]["calibration"]
    downloaded = path.read_bytes()
    if hashlib.sha256(downloaded).hexdigest() != record["fit"]["calibration_sha256"]:
        raise ValueError("Ranking calibration changed")
    metadata = json.loads(downloaded)
    model = xgb.Booster()
    model.load_model(folder/record["fit"]["checkpoint"])
    model.set_param({"device":"cpu", "nthread":4})
    output = []
    for part in parts:
        logits = model.inplace_predict(view(part,record["config"]),predict_type="margin")
        baseline = np.clip(part["old"],1e-6,1-1e-6)
        x = np.column_stack((logits,np.log(baseline/(1-baseline))))
        calibrated = x@metadata["weights"]+metadata["bias"]
        output.append(1/(1+np.exp(-np.clip(calibrated,-40,40))))
    return output


def learn(name, folder, train, calibration, config, method, blend, rounds=160, objective="rank:pairwise"):
    start = time.perf_counter()
    if not torch.cuda.is_available() or not xgb.build_info()["USE_CUDA"]:
        raise RuntimeError("Training requires CUDA")
    data = np.concatenate([view(p,config) for p in train])
    labels = np.concatenate([p["y"] for p in train])
    group = np.concatenate([np.diff(np.r_[0,p["ends"]]) for p in train])
    matrix_type = xgb.QuantileDMatrix if rounds > 160 else xgb.DMatrix
    matrix = matrix_type(data,label=labels,group=group,nthread=4)
    settings = dict(tree_method="hist",device="cuda:0",objective=objective,
                    eta=.05,max_depth=config[4],lambdarank_pair_method=method,
                    lambdarank_num_pair_per_sample=16 if method=="topk" else 8,seed=27182,nthread=4)
    model = xgb.Booster(settings,[matrix])
    model.update(matrix,0)
    actual = json.loads(model.save_config())["learner"]
    device = actual["generic_param"]["device"]
    updater = actual["gradient_booster"]["gbtree_train_param"]["updater_seq"]
    if device!="cuda:0" or "gpu_hist" not in updater:
        raise RuntimeError(f"Ranking used {device}, {updater}")
    for i in range(1,rounds):
        model.update(matrix,i)
    checkpoint = folder/f"{name}.ubj"
    model.save_model(checkpoint)
    model.set_param({"device":"cpu", "nthread":4})
    features = []
    for part in calibration:
        raw = model.inplace_predict(view(part,config),predict_type="margin")
        previous = np.clip(part["old"],1e-6,1-1e-6)
        features.append(np.column_stack((raw,np.log(previous/(1-previous)))))
    x = torch.as_tensor(np.concatenate(features),device="cuda",dtype=torch.float32)
    target = torch.as_tensor(np.concatenate([q["y"] for q in calibration]),device="cuda",dtype=torch.float32)
    weights = torch.nn.Parameter(torch.tensor([1.,0.],device="cuda"))
    intercept = torch.nn.Parameter(torch.tensor(-2.,device="cuda"))
    optimizer = torch.optim.Adam((weights,intercept),lr=.025)
    active = torch.tensor([1.,float(blend)],device="cuda")
    for _ in range(280):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x@(active*weights)+intercept,target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    calibration_path=folder/f"{name}_calibration.json"
    proteins=tuple(dict.fromkeys(part["protein"] for part in calibration))
    dump(calibration_path,dict(weights=(active*weights).detach().cpu().tolist(),
                               bias=float(intercept.detach().cpu()),proteins=proteins,
                               protein=proteins[0] if len(proteins)==1 else None))
    return dict(cuda=device,updater=updater,fit_seconds=time.perf_counter()-start,
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                checkpoint=checkpoint.name,calibration=calibration_path.name,
                calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                method=method,objective=objective,rounds=rounds,train_rows=len(labels),train_positives=int(labels.sum()))


def main():
    import argparse
    from .hb_directional import add_directional
    from .hb_single_structure import ADDITIONAL_TRAINING

    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--volume",action="store_true")
    modes.add_argument("--expanded10",action="store_true")
    args=parser.parse_args()
    folder=OUT/("ap10_rank_expanded" if args.expanded10 else "ranking_volume" if args.volume else "ranking")
    folder.mkdir(parents=True,exist_ok=True)
    basic=load(TRAIN,25,geometry=True)
    learning_middle=tuple(p for p in EXTRA_TRAINING if args.expanded10 or p!="2w0g_A")
    middle=load(learning_middle,50,geometry=True)
    train=basic+middle
    calibration_names=("2w0g_A",)
    validation_names=VALIDATION
    if args.expanded10:
        from .hb_validation10 import panel10

        calibration_names=tuple(json.loads((OUT/"ap10_calibration_panel.json").read_text())["proteins"])
        validation_names=panel10()
    calibration_stride=100 if args.expanded10 else 50
    calibration=load(calibration_names,calibration_stride,geometry=True)
    valid=load(validation_names,100,geometry=True)
    configurations=(
        ("environment",118,3,"mean",False),("environment",118,4,"topk",True),
        ("local",142,3,"mean",True),("local",166,4,"topk",False),
        ("local",118,5,"mean",False),("competition",142,5,"topk",True),
    )
    if args.volume or args.expanded10:
        panels=(OUT/"diverse_training_panel.json",OUT/"volume_training_panel.json")
        if args.expanded10:
            panels+=tuple(OUT/f"{name}.json" for name in (
                "volume_96_training_panel","volume_128_training_panel","ap10_volume_144_training_panel"))
        extra=tuple(p for panel in panels for p in json.loads(panel.read_text(encoding="utf-8"))["proteins"])
        if set(TRAIN+learning_middle+ADDITIONAL_TRAINING+extra)&set(validation_names+calibration_names):
            raise ValueError("Training overlaps validation or calibration")
        if set(validation_names)&set(calibration_names):
            raise ValueError("Validation overlaps calibration")
        middle+=load(ADDITIONAL_TRAINING,50,geometry=True)
        late=load(extra,100,geometry=True)
        for parts,stride in ((basic,25),(middle,50),(late,100),(valid,100),(calibration,calibration_stride)):
            for part in parts:
                add_directional(part,stride)
        train=basic+middle+late
        configurations=(("environment",118,4,"topk",True),
                        ("directional",118,5,"mean",True),
                        ("directional",166,5,"topk",True))
    if args.expanded10:
        from .hb_cloud import add_cloud
        from .hb_explicit_hydrogen import add_explicit
        from .hb_occupied import add_occupied

        for parts,stride in ((basic,25),(middle,50),(late,100),(valid,100),(calibration,calibration_stride)):
            for part in parts:
                add_cloud(part,stride)
                add_occupied(part,stride)
                add_explicit(part,stride)
        configurations=(("stacked-explicit",166,8,"topk",True),
                        ("stacked-explicit",166,8,"mean",False))
        print("AP_RANK_DATA",len({part["protein"] for part in train}),len(validation_names),flush=True)
    log={}
    first=1771 if args.expanded10 else 1201 if args.volume else 611
    for index,(representation,width,depth,method,blend) in enumerate(configurations,first):
        name=f"T1I{index}"
        config=("xgb-rank",width,f"hb-origin-{representation}",False,depth,1.0)
        record=dict(config=config,fit=learn(name,folder,train,calibration,config,method,blend,
                                          rounds=300 if args.expanded10 else 160))
        record["validation"]=validation_report(valid,rank_predict(valid,folder,record))
        log[name]=record
        dump(folder/"validation.json",log)
        print(name,record["validation"]["candidate"]["auprc"],key(record),flush=True)
    best=max(log,key=lambda n:key(log[n]))
    dump(folder/"selection.json",dict(name=best,**log[best]))


if __name__=="__main__":
    main()
