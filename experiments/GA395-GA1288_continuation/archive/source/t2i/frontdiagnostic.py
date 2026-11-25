"""Measure which static front ends predict an actual next-100-ps distance change."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import torch

from tseries.data import cell, registry, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import OUT, ROOT
from tseries.hb_validation10 import panel10
from .explore import record
from .frontends import DEST, VARIANTS, data


def short_target(part):
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    result = []
    for a, b in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        origin = int(part["origin"][a])
        frame = next(reg.blocks(tid, start=origin, stop=origin+2, chunk_frames=2))
        if not np.isclose(frame.time[-1]-frame.time[0], 100., rtol=0., atol=.01):
            raise ValueError("Expected a 100 ps forward interval")
        pair = part["pair"][a:b]
        left, right = pair[:, :2].astype(np.intp).T
        p = frame.positions[-1]
        d_next = np.linalg.norm(_mic(p[right]-p[left], cell(frame.box[-1])), axis=1)
        result.append(d_next - pair[:, 2])
    return np.concatenate(result)


def main():
    train = tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    tr, xt, _ = data(train)
    va, xv, _ = data(panel10())
    with ThreadPoolExecutor(3) as pool:
        yt = np.concatenate(list(pool.map(short_target, tr)))
        yv = np.concatenate(list(pool.map(short_target, va)))
    result = {"horizon_ps":100, "train_rows":len(yt), "validation_rows": len(yv),
              "zero_change_rmse": float(np.sqrt(np.mean(yv**2))), "trials": {}}
    for name in VARIANTS:
        a, b = xt[name].astype(float), xv[name].astype(float)
        center, scale = a.mean(0), np.maximum(a.std(0), .001)
        a, b = [(x-center)/scale for x in (a, b)]
        a, b = [np.column_stack((np.ones(len(x)), x, x*x)) for x in (a, b)]
        design = torch.from_numpy(a).cuda()
        response = torch.from_numpy(yt.astype(float)).cuda()
        ridge = torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
        weights = torch.linalg.solve(design.T@design + ridge*len(a)*.001,
                                     design.T@response).cpu().numpy()
        predicted = b@weights
        row = {"rmse": float(np.sqrt(np.mean((yv-predicted)**2))),
               "bias": float(np.mean(predicted-yv)), "correlation": float(np.corrcoef(predicted, yv)[0, 1])}
        result["trials"][name] = row
        print(name, json.dumps(row), flush=True)
    record(DEST/"short_response.json", result)


if __name__ == "__main__":
    main()
