"""GC06–07: measure measured-history headroom with a frozen T1I2007."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from tseries.data import TEST
from tseries.hb_single_structure import OUT
from tseries.hb_validation10 import panel10
from .frontends import data, design, fit, infer
from .explore import record, score
from .gc_temporal import offsets


def measured_history(parts, front):
    with ThreadPoolExecutor(3) as pool:
        history = np.concatenate(list(pool.map(offsets, parts)))[:, :3]
    d = np.concatenate([p["pair"][:, 2] for p in parts])
    delta = d[:, None] - history
    return {"distance":d[:, None], "past_100ps":delta[:, 2:],
            "past_500ps":np.column_stack((delta, delta.mean(1), (history <= 3.5).mean(1))),
            "static_and_past":np.column_stack((front["combined"], delta,
                                               (history <= 3.5).mean(1)))}


def main():
    torch.set_num_threads(3)
    dest = Path("runs/gc")
    names = tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    valid = panel10()
    assert not set(names)&set(valid+TEST)
    train, ft, pt = data(names)
    dev, fv, pv = data(valid)
    xt,xv = measured_history(train,ft),measured_history(dev,fv)
    y = np.concatenate([p["y"] for p in train])
    results = dict(train=names, validation=valid, baseline=score(dev,pv), rows={})
    for name in xt:
        ckpt=dest/f"GC06-{name}.pt"
        elapsed, peak = fit(design(xt[name],pt),y,ckpt)
        p = infer(design(xv[name],pv),ckpt)
        results["rows"][name] = dict(metrics=score(dev,p),fit_seconds=elapsed,cuda_peak_bytes=peak,
                                    checkpoint=ckpt.name)
    record(dest/"GC06.json",results)
    print("Completed GC06",flush=True)
    possible = [n for n in xt if n!="distance"]
    winner=max(possible,key=lambda n:results["rows"][n]["metrics"]["auprc"])
    if results["rows"][winner]["metrics"]["auprc"] <= results["baseline"]["auprc"]:
        record(dest/"GC07.json",dict(selected=None, reason="Validation AUPRC gate not met"))
        return
    test, ff, pp = data(TEST)
    x = measured_history(test,ff)
    p = infer(design(x[winner],pp),dest/f"GC06-{winner}.pt")
    record(dest/"GC07.json",dict(selected=winner,proteins=TEST,baseline=score(test,pp),
                                  corrected=score(test,p)))
    print("Completed GC07",flush=True)


if __name__ == "__main__":
    main()
