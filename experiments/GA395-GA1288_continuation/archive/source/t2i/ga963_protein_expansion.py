"""GA963+: add new proteins to the matched two-field dynamic training set."""
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import openmm

from .ga333 import WINDOW
from .ga365 import parts
from .ga373 import assemble
from .ga_multiwindow import targets
from .ga_native240 import simulate
from .ga_openmm_pilot import build

R=Path("runs")
OUT=R/"ga963-protein-expansion"


def main(later=False, scale=False):
    out=Path("runs/ga1029-later-expansion") if later else OUT
    if scale:
        out=R/"ga1289-scale-146"
    out.mkdir(exist_ok=True)
    frames=(49,149,249) if scale else ((149,249) if later else (49,))
    prior=json.loads((R/"ga588-transfer/summary.json").read_bytes())
    learned=json.loads((R/"ga445-494/summary.json").read_bytes())
    banned=set(learned["train_proteins"])|set(learned["validation_proteins"])
    for panel in ("ga351","ga841-fresh-panel","ga1145-prospective"):
        banned.update(row["id"] for row in json.loads((R/panel/"reserved.json").read_bytes())["proteins"])
    pool=[n for n in prior["pretrain"] if n not in banned]
    names=sorted(pool,key=lambda name:sha256(("GA963/"+name).encode()).digest())[:24]
    if scale:
        previous=set(json.loads((R/"ga1029-later-expansion/plan.json").read_bytes())["proteins"])
        candidates=[n for n in pool if n not in previous]
        names=sorted(candidates,key=lambda n:sha256(("GA1289/"+n).encode()).digest())[:48]
        assert len(names)==48
    else:
        assert len(names)==24
    plan={"proteins":names,"replica":1,"origins":list(frames),"branches_per_field":2,"ps":25,
          "purpose":"additional proteins for generated-response training"}
    lock=out/"plan.json"
    if lock.exists():
        assert json.loads(lock.read_bytes())==plan
    else:
        lock.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    for label,step,iterations,fn in (
        ("amber",4,25,build),
        ("charmm",2,60,partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml")))):
        if scale and label=="charmm":
            from .ga_terminal_histidine import NeutralTerminalHistidine
            fn=partial(fn,forcefield_class=NeutralTerminalHistidine)
        folder=out/label
        folder.mkdir(exist_ok=True)
        print("generating",label,len(names),flush=True)
        for name in names:
            argv=dict(destination=folder,samples=WINDOW,branches=2,setup_fn=fn,
                      parts_fn=partial(parts,origins=frames))
            try:
                simulate((name,),len(frames),step_fs=step,minimize_iterations=iterations,**argv)
            except openmm.OpenMMException as exc:
                if "NaN" not in str(exc):
                    raise
                (folder/f"{name}-recovery.json").write_text(
                    json.dumps({"step_fs":step//2,"minimize":100}),encoding="utf-8")
                simulate((name,),len(frames),step_fs=step//2,minimize_iterations=100,**argv)
    x,d,g,y,origins,proteins=assemble(names,frames,out/"amber",out/"charmm")
    label=targets(x,y,origins,proteins,out/"extra-history.npz")
    store=dict(x=x,d=d,g=g,y=y,origins=origins,proteins=proteins,label=label)
    if later or scale:
        prior_folder=R/"ga1029-later-expansion" if scale else OUT
        with np.load(prior_folder/"train-extra.npz") as first:
            store={key:np.concatenate([first[key],value]) for key,value in store.items()}
    np.savez_compressed(out/"train-extra.npz",**store)
    print("assembled expanded training pool",len(names),len(y),flush=True)
    from .ga_mixture import main as train
    train(stability=True,new_proteins=not (later or scale),late_expansion=later,scale_146=scale)


if __name__=="__main__":
    import sys
    main(later="--later" in sys.argv,scale="--scale" in sys.argv)


