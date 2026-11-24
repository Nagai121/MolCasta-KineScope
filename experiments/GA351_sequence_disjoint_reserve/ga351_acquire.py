"""Acquire the already locked GA351 confirmation data using MiniTopol."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from tseries.data import acquire, registry, trajectory
from tseries.hb_single_structure import ROOT
from .explore import record

def fetch(name):
    try:
        acquire(ROOT,name)
        reg=registry(ROOT,name)
        frames={str(r):reg.manifest(trajectory(name,r))["frames"] for r in (1,2,3)}
        return name,dict(frames=frames)
    except Exception as exc:
        return name,dict(error=f"{type(exc).__name__}: {exc}")

def main():
    dest=Path("runs/ga351")
    panel=json.loads((dest/"reserved.json").read_bytes())
    names=[p["id"] for p in panel["proteins"]]
    results={}
    with ThreadPoolExecutor(2) as jobs:
        for name,result in jobs.map(fetch,names):
            results[name]=result
            record(dest/"acquisition.json",dict(panel=names,results=results))
            print(name,result,flush=True)

if __name__=="__main__":
    main()
