"""Expand the static training panel with the same held-out test."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .audit_split import similarity
from .data import PANELS, TRAIN, VALIDATION, acquire
from .hb_directional import add_directional
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING, dump


def choose(existing, path, per_class=8):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["proteins"]
    with Path("atlas-info.tsv").open(encoding="utf-8") as handle:
        rows={r["PDB"]:r for r in csv.DictReader(handle,delimiter="\t")}
    reference=existing+PANELS["validation"]+PANELS["test"]+PANELS["historical"]
    used={rows[p]["UniProt"] for p in reference}
    record={}
    classes=(("Mainly Alpha",),("Mainly Beta",),("Alpha Beta",),
             ("Special","Few Secondary Structures","-"))
    quotas=(per_class,)*len(classes) if isinstance(per_class,int) else per_class
    for group,quota in zip(classes,quotas,strict=True):
        if quota == 0:
            continue
        candidates=[r for r in rows.values() if r["non_redundant_protein"]=="True"
                    and r["CATH_class"].strip('"') in group and 70<=int(r["length"])<=150
                    and r["UniProt"] not in used and r["UniProt"] not in ("","-")
                    and not any(t in r["protein_name"].lower() for t in ("toxin","effector"))
                    and "virus" not in r["organism"].lower()]
        candidates.sort(key=lambda r:(abs(int(r["length"])-106),r["PDB"]))
        count=0
        architectures=set()
        for r in candidates:
            if r["UniProt"] in used or r["PDB"] in reference or r["PDB"] in record:
                continue
            arch=r["CATH_architecture"]
            if arch in architectures and count<2 and len(group)==1:
                continue
            scores={q:similarity(r["sequence"],rows[q]["sequence"])["identity_over_shorter"]
                    for q in reference+tuple(record)}
            if max(scores.values())>.30:
                continue
            record[r["PDB"]]=dict(uniprot=r["UniProt"],length=int(r["length"]),
                        structure_class=r["CATH_class"],architecture=arch,
                        closest=max(scores,key=scores.get),max_identity=max(scores.values()))
            count+=1
            used.add(r["UniProt"])
            architectures.add(arch)
            if count==quota:
                break
        if count!=quota:
            raise ValueError(f"Only {count} independent training chains in {group}")
    dump(path,dict(proteins=list(record),metadata=record,validation=PANELS["validation"],test=PANELS["test"]))
    return list(record)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--size",type=int,choices=(64,96,128),default=64)
    args=parser.parse_args()
    former=tuple(json.loads((OUT/"diverse_training_panel.json").read_text())["proteins"])
    existing=TRAIN+EXTRA_TRAINING+ADDITIONAL_TRAINING+former
    panel=OUT/"volume_training_panel.json"
    previous=()
    if args.size>=96:
        previous=tuple(json.loads(panel.read_text(encoding="utf-8"))["proteins"])
        panel=OUT/"volume_96_training_panel.json"
    if args.size==128:
        previous+=tuple(json.loads(panel.read_text(encoding="utf-8"))["proteins"])
        panel=OUT/"volume_128_training_panel.json"
    cal_paths=(OUT/"ensemble_calibration_panel.json",OUT/"ensemble_gate_panel.json",OUT/"ap10_calibration_panel.json")
    reserved=tuple(p for path in cal_paths if path.is_file()
                   for p in json.loads(path.read_text(encoding="utf-8"))["proteins"])
    validation_path=OUT/"validation10_added.json"
    validation_reserved=tuple(json.loads(validation_path.read_text(encoding="utf-8"))["proteins"]) if validation_path.is_file() else ()
    added=choose(existing+previous+reserved+validation_reserved,panel)
    print("VOLUME_TRAIN_ONLY",added,flush=True)
    if args.prepare:
        def get(p):
            acquire(ROOT,p)
            print("VOLUME_ACQUIRED",p,flush=True)
        with ThreadPoolExecutor(5) as pool:
            list(pool.map(get,added))
    basic=load(TRAIN,25,True,True)
    middle=load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,True,True)
    current=load(former,100,True,True)
    expanded=load((*previous,*added),100,True,True)
    valid=load(VALIDATION,100,geometry=True)
    for parts,stride in ((basic,25),(middle,50),(current,100),(expanded,100),(valid,100)):
        with ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(add_directional,p,stride) for p in parts]
            for job in jobs: job.result()
    print("VOLUME_DATA",sum(len(p["y"]) for p in expanded),flush=True)
    configs=(
        ("xgb",118,"hb-origin-environment",False,6,1.0),
        ("xgb",118,"hb-origin-directional",False,6,1.0),
        ("xgb",166,"hb-origin-directional",False,5,1.0),
        ("xgb",166,"hb-origin-environment",False,5,1.0),
        ("xgb-soft-tune-0.15-0.25-320-local",118,"hb-origin-directional",False,5,1.0),
    )
    stage,start={64:("volume",981),96:("volume_96",1131),128:("volume_128",1391)}[args.size]
    explore(stage,configs,start,basic+middle+current+expanded,valid)


if __name__=="__main__":
    main()
