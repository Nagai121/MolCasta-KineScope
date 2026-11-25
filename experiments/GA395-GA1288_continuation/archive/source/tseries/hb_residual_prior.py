"""Learn a residual over a strong static prior using held-protein predictions."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .data import HISTORICAL, TEST, TRAIN, VALIDATION
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_occupied import add_occupied
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, key, load, validation_report
from .hb_transition import auxiliary_labels, train_network, transition_predict
from .search_events import EXTRA_TRAINING, cpu_predict, dump, fit, view


PRIOR_CONFIG=("xgb",118,"hb-origin-stacked-directional",False,6,1.)


def prior_parts(parts,folder,record):
    checkpoint=folder/record["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=record["sha256"]:
        raise ValueError("Static prior changed")
    runner=cpu_predict(checkpoint,PRIOR_CONFIG)
    return [dict(part,old=runner(view(part,PRIOR_CONFIG)).astype(np.float32,copy=False)) for part in parts]


def predict_residual_prior(parts,folder,record):
    prior=record["fit"]["prior"]
    prepared=prior_parts(parts,OUT/prior["folder"],prior["fit"])
    return transition_predict(prepared,folder,record)


def main():
    folder=OUT/"residual_prior"
    folder.mkdir(exist_ok=True)
    panel=tuple(json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"])
    middle=EXTRA_TRAINING+ADDITIONAL_TRAINING
    names=TRAIN+middle+panel
    if set(names)&set(VALIDATION+TEST+HISTORICAL):
        raise ValueError("Residual-prior training panel overlaps evaluation proteins")
    groups=((load(TRAIN,25,geometry=True),25,True),
            (load(middle,50,geometry=True),50,True),
            (load(panel,100,geometry=True),100,True),
            (load(VALIDATION,100,geometry=True),100,False))

    def prepare(part,stride,training):
        add_cloud(part,stride)
        add_directional(part,stride)
        add_occupied(part,stride)
        if training:
            auxiliary_labels(part,stride,names)

    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(prepare,part,stride,training) for parts,stride,training in groups for part in parts]
        for job in jobs:
            job.result()
    train=[part for parts,_,training in groups if training for part in parts]
    valid=groups[-1][0]
    ordered=sorted(names,key=lambda p:hashlib.sha256(p.encode()).digest())
    sets=[]
    oof=[]
    for fold in range(4):
        hidden=set(ordered[fold::4])
        fit_parts=[part for part in train if part["protein"] not in hidden]
        hidden_parts=[part for part in train if part["protein"] in hidden]
        model=fit(f"T1I{1494+fold}",folder,fit_parts,PRIOR_CONFIG)
        oof.extend(prior_parts(hidden_parts,folder,model))
        sets.append(dict(fit=model,held_proteins=sorted(hidden),fit_proteins=sorted(set(names)-hidden)))
    if sum(len(p["y"]) for p in oof)!=sum(len(p["y"]) for p in train):
        raise ValueError("Cross-protein static priors did not cover every row")
    dump(folder/"out_of_fold.json",sets)
    record=json.loads((OUT/"stacked/validation.json").read_text(encoding="utf-8"))["T1I1271"]
    if tuple(record["config"])!=PRIOR_CONFIG:
        raise ValueError("The frozen full-panel prior does not match the fold models")
    prior=dict(folder="stacked",fit=record["fit"])
    prepared=prior_parts(valid,OUT/prior["folder"],prior["fit"])
    rows={}
    for idx,(width,rep,aux,tail) in enumerate(((118,"stacked-directional",.15,0.),
                                              (118,"stacked-directional",.20,.025),
                                              (166,"stacked-occupied",.28,.04)),1491):
        name=f"T1I{idx}"
        cfg=("hb-residual-prior",width,"hb-origin-"+rep,False,128,1.)
        trained=train_network(name,folder,oof,cfg,aux,tail)
        trained["prior"]=prior
        row=dict(config=cfg,fit=trained)
        row["validation"]=validation_report(valid,transition_predict(prepared,folder,row))
        rows[name]=row
        dump(folder/"validation.json",rows)
        print(name,row["validation"]["candidate"]["auprc"],key(row),flush=True)
    best=max(rows,key=lambda name:key(rows[name]))
    dump(folder/"selection.json",dict(name=best,**rows[best]))


if __name__=="__main__":
    main()
