"""Compare training diversity at a fixed example budget and fixed model."""
import json
import argparse

import numpy as np

from .data import TRAIN, VALIDATION
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, explore, load
from .search_events import EXTRA_TRAINING, dump


def take_groups(parts, budget, seed):
    totals={}
    for p in parts:
        totals[p["protein"]]=totals.get(p["protein"],0)+len(p["y"])
    capacity=min(budget,sum(totals.values()))
    low,high=0,max(totals.values())
    while low<high:
        pivot=(low+high+1)//2
        if sum(min(c,pivot) for c in totals.values())<=capacity:
            low=pivot
        else:
            high=pivot-1
    quotas={g:min(n,low) for g,n in totals.items()}
    remaining=capacity-sum(quotas.values())
    for g in sorted(quotas):
        if remaining and quotas[g]<totals[g]:
            quotas[g]+=1
            remaining-=1
    rng=np.random.default_rng(seed)
    selected=[]
    for name, quota in quotas.items():
        same=[part for part in parts if part["protein"]==name]
        offsets=np.r_[0,np.cumsum([len(p["y"]) for p in same])]
        global_rows=np.sort(rng.permutation(totals[name])[:quota])
        for part,start,end in zip(same,offsets[:-1],offsets[1:]):
            indices=global_rows[(global_rows>=start)&(global_rows<end)]-start
            if not len(indices):
                continue
            selected.append({field:(value[indices] if isinstance(value,np.ndarray)
                                     and len(value)==len(part["y"]) else value)
                             for field,value in part.items()})
    assert sum(len(part["y"]) for part in selected)==capacity
    return selected


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--volume-doses",action="store_true")
    mode.add_argument("--compare-96",action="store_true")
    mode.add_argument("--compare-128",action="store_true")
    mode.add_argument("--ablate-96",action="store_true")
    mode.add_argument("--replicate",type=int,choices=(32,64,96))
    args=parser.parse_args()
    first=load(TRAIN,25)+load(EXTRA_TRAINING,50)
    second=first+load(ADDITIONAL_TRAINING,50)
    panel=json.loads((OUT/"diverse_training_panel.json").read_text())["proteins"]
    third=second+load(panel,100)
    volume_panel=OUT/"volume_training_panel.json"
    cohorts=[(8,first),(16,second),(32,third)]
    if volume_panel.is_file():
        added=json.loads(volume_panel.read_text(encoding="utf-8"))["proteins"]
        cohorts.append((64,third+load(added,100)))
    valid=load(VALIDATION,100)
    config=(("xgb",118,"hb-origin-environment",False,6,1.0),)
    if args.compare_96 or args.compare_128 or args.ablate_96 or args.replicate==96:
        panel_96=OUT/"volume_96_training_panel.json"
        if cohorts[-1][0]!=64 or not panel_96.is_file():
            parser.error("The 96-protein runs require both cached expansion panels")
        panel=json.loads(panel_96.read_text(encoding="utf-8"))
        added=load(panel["proteins"],100)
        train=cohorts[-1][1]+added
        cohorts.append((96,train))
    if args.ablate_96:
        classes=sorted({row["structure_class"] for row in panel["metadata"].values()})
        studies={}
        for index,kind in enumerate(classes,1241):
            group=[part for part in added if panel["metadata"][part["protein"]]["structure_class"]==kind]
            subset=take_groups(cohorts[-2][1]+group,800000,27182)
            stage=f"scaling_72_class_{index}"
            explore(stage,config,index,subset,valid)
            studies[stage]=dict(structure_class=kind,proteins=sorted({p["protein"] for p in group}),
                                train_rows=sum(len(p["y"]) for p in subset))
        subset=take_groups(train,800000,27182)
        stage="scaling_96_800000"
        explore(stage,config,1251,subset,valid)
        studies[stage]=dict(proteins=sorted({p["protein"] for p in added}),
                            train_rows=sum(len(p["y"]) for p in subset))
        dump(OUT/"class_expansion_comparison.json",studies)
        return
    if args.compare_96:
        subsets=take_groups(train,400000,27182)
        explore("scaling_96",config,1141,subsets,valid)
        dump(OUT/"fixed_budget_scaling_96.json",dict(rows=sum(len(p["y"]) for p in subsets),
                                                      proteins=sorted({p["protein"] for p in subsets})))
        return
    if args.compare_128:
        path=OUT/"volume_128_training_panel.json"
        panel=json.loads(path.read_text(encoding="utf-8"))["proteins"]
        train=cohorts[-1][1]+load(panel,100)
        for index,budget in enumerate((400000,800000),1401):
            subset=take_groups(train,budget,27182)
            explore(f"scaling_128_{budget}",config,index,subset,valid)
        return
    if args.replicate:
        if args.replicate==64 and cohorts[-1][0]!=64:
            parser.error("This comparison requires the cached 64-protein panel")
        train=dict(cohorts)[args.replicate]
        start={32:1161,64:1151,96:1231}[args.replicate]
        for index,seed in enumerate((14159,26535,89793),start):
            subsets=take_groups(train,400000,seed)
            explore(f"scaling_{args.replicate}_seed_{seed}",config,index,subsets,valid)
        return
    if args.volume_doses:
        if cohorts[-1][0]!=64:
            parser.error("This comparison requires the cached 64-protein panel")
        train=cohorts[-1][1]
        for index,budget in enumerate((600000,800000,1000000),1121):
            subsets=take_groups(train,budget,27182)
            explore(f"scaling_64_{budget}",config,index,subsets,valid)
            print("TRAINING_DOSE",budget,sum(len(p["y"]) for p in subsets),flush=True)
        return
    record={}
    for index,(size,parts) in enumerate(cohorts,921):
        subsets=take_groups(parts,400000,27182)
        actual=sum(len(p["y"]) for p in subsets)
        name=f"scaling_{size}"
        explore(name,config,index,subsets,valid)
        record[str(size)]=dict(rows=actual,proteins=sorted({p["protein"] for p in subsets}))
        print("TRAINING_SIZE",size,actual,flush=True)
    dump(OUT/"fixed_budget_scaling.json",record)


if __name__=="__main__":
    main()
