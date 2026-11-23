"""GC08–09: stricter matching and historical-state ablations."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tseries.hb_validation10 import panel10
from .frontends import data
from .explore import record
from .gc_assist import matched
from .gc_temporal import offsets


def main():
    torch.set_num_threads(3)
    dest = Path("runs/gc")
    parts, x, p = data(panel10())
    with ThreadPoolExecutor(3) as pool:
        d = np.concatenate(list(pool.map(offsets, parts)))
    origin = np.concatenate([q["pair"][:, 2] for q in parts])
    past = d[:, :3]
    recent = origin - past[:, -1]
    older = origin - past[:, :2].mean(1)
    minimum = origin - past.min(1)
    hist = origin - past.mean(1)
    resident = (past <= 3.5).mean(1)
    future = origin - d[:, 3]
    scores = dict(frozen_t1=p, past_100ps=recent, past_older_mean=older, past_mean=hist,
                  past_min=minimum, past_occupancy=resident, future_100ps=future,
                  static_local_closing=-x["local_response"][:, 1],
                  static_network_closing=-x["network_response"][:, 4])
    record(dest/"GC08.json", matched(parts,p,scores,same_frame=True))
    keep = (past > 3.5).all(1)
    from .gc_temporal import subset
    chosen = {n:v[keep] for n,v in scores.items() if n!="past_occupancy"}
    record(dest/"GC09.json", dict(excluded=int((~keep).sum()),
                                  result=matched(subset(parts,keep),p[keep],chosen)))
    print("Completed GC08 and GC09",flush=True)


if __name__ == "__main__":
    main()
