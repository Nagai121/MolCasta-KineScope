"""Static-origin front ends with one frozen T1I2007 residual readout."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logit, ndtr
import torch
from torch import nn
from torch.nn import functional as F

from tseries.data import registry, trajectory, cell, TEST, HISTORICAL
from tseries.features import _mic
from tseries.hb_ap_replay import replay
from tseries.hb_heavy_dynamics import neighbor_pairs
from tseries.hb_preprocessing_mechanics import prepare
from tseries.hb_single_structure import OUT, ROOT, load
from tseries.hb_validation10 import panel10
from .explore import STRIDE, score, record

DEST = Path("runs/t2i_front")
VARIANTS = ("distance", "existing_static", "local_response", "network_response",
            "occupancy", "competitors", "combined")


def response(position, box, pair):
    """Solve a damped heavy-atom contact network's local pressure response."""
    left, right = neighbor_pairs(position, box, 5.).T
    delta = _mic(position[right] - position[left], box)
    r = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(r[:, None], .001)
    nonbonded = r > 2.05
    load = (np.maximum(3.2-r, 0.) - .14*np.exp(-((r-3.9)/.65)**2)) * nonbonded
    force = np.zeros_like(position)
    np.add.at(force, left, -load[:, None]*direction)
    np.add.at(force, right, load[:, None]*direction)
    spring = np.where(nonbonded, .35*np.exp(-.7*(r-3.)), 2.)
    outer = spring[:, None, None] * direction[:, :, None] * direction[:, None, :]
    stiffness = np.broadcast_to(np.eye(3)*.8, (len(position), 3, 3)).copy()
    np.add.at(stiffness, left, outer)
    np.add.at(stiffness, right, outer)
    inverse = np.linalg.inv(stiffness)
    local = np.einsum("nij,nj->ni", inverse, force)
    disp = local
    snapshots = [local]
    for step in range(8):
        rhs = force.copy()
        np.add.at(rhs, left, np.einsum("nij,nj->ni", outer, disp[right]))
        np.add.at(rhs, right, np.einsum("nij,nj->ni", outer, disp[left]))
        disp = .35*disp + .65*np.einsum("nij,nj->ni", inverse, rhs)
        if step in (1, 3, 7):
            snapshots.append(disp)
    a, b = pair[:, :2].astype(np.intp).T
    v = pair[:, 3:6]
    axis = v / pair[:, 2, None]
    changed = [np.linalg.norm(v + u[b]-u[a], axis=1) - pair[:, 2] for u in snapshots]
    along = np.einsum("ni,nij,nj->n", axis, inverse[a]+inverse[b], axis)
    cross = np.trace(inverse[a]+inverse[b], axis1=1, axis2=2)-along
    return np.column_stack((*changed, along, cross)).astype(np.float32)


def build(part):
    prepare(part, STRIDE)
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    rows = {name: [] for name in VARIANTS}
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        t = int(part["origin"][start])
        block = next(reg.blocks(tid, start=t, stop=t+1, chunk_frames=1))
        pair = part["pair"][start:end]
        d = pair[:, 2]
        m = response(block.positions[0], cell(block.box[0]), pair)
        q = part["explicit"][start:end, 3].clip(0., 1.)
        # Reduced scales, with no conversion to physical time.
        occ = np.column_stack([ndtr((3.5-d-m[:, i]*f)/np.sqrt(.02+f*m[:, 4]*.2))
                               for i in (0, 3) for f in (.25, .5, 1., 2.)])
        competitive = []
        for endpoint in (0, 1):
            groups = pair[:, endpoint].astype(np.intp)
            affinity = (.15+.85*q)*occ[:, 5]
            total = np.bincount(groups, weights=affinity)[groups]
            strongest = np.zeros(total.size if not len(groups) else groups.max()+1)
            np.maximum.at(strongest, groups, affinity)
            competitive.extend((affinity/(.1+total), total-affinity,
                                affinity-strongest[groups]))
        static = np.column_stack((part["explicit_context"][start:end, :8],
                                  part["compliance"][start:end, :5]))
        for name, x in zip(VARIANTS, (d[:, None], static, m[:, [0, 4, 5]], m,
                                     occ, np.column_stack(competitive),
                                     np.column_stack((m, occ, *competitive)))):
            rows[name].append(np.column_stack((d, x)).astype(np.float32))
    return {key: np.concatenate(items) for key, items in rows.items()}


def data(names):
    started = time.perf_counter()
    parts = load(names, STRIDE, geometry=True)
    recipe = OUT / "selections/T1I2007/ap_recipe.json"
    signature = hashlib.sha256(Path(__file__).read_bytes()+recipe.read_bytes()).hexdigest()
    cache_key = hashlib.sha256((" ".join(names)+signature).encode()).hexdigest()[:20]
    path = DEST / f"cache-{cache_key}.npz"
    check = hashlib.sha256()
    for part in parts:
        for key in ("pair", "origin"):
            check.update(part[key].tobytes())
    candidates = check.hexdigest()
    if path.exists():
        with np.load(path) as cache:
            if str(cache["candidates"]) != candidates:
                raise RuntimeError("Candidate alignment changed")
            return parts, {n: cache[n] for n in VARIANTS}, cache["base"]
    chunks = [build(part) for part in parts]
    x = {n: np.concatenate([c[n] for c in chunks]) for n in VARIANTS}
    _, p = replay(parts, json.loads(recipe.read_bytes()))
    base = np.concatenate(p)
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez(f, candidates=candidates, base=base, **x)
    path.with_suffix(".tmp").replace(path)
    print(f"Prepared {len(base)} candidates in {time.perf_counter()-started:.2f}s", flush=True)
    return parts, x, base


def net(width):
    return nn.Sequential(nn.Linear(width, 24), nn.SiLU(), nn.Linear(24, 1))


def design(features, anchor):
    return np.column_stack((logit(anchor.clip(1e-6, 1-1e-6)), features)).astype(np.float32)


def fit(x, y, checkpoint):
    torch.manual_seed(41)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    center = torch.from_numpy(x.mean(0))
    scale = torch.from_numpy(np.maximum(x.std(0), .001))
    values = ((torch.from_numpy(x)-center)/scale).clamp(-10, 10).cuda()
    offset = torch.from_numpy(x[:, 0].copy()).cuda()
    target = torch.from_numpy(y.astype(np.float32)).cuda()
    model = net(values.shape[1]).cuda()
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(5):
        for i in torch.randperm(len(x), device="cuda").split(4096):
            correction = .35*model(values[i]).flatten().tanh()
            loss = F.binary_cross_entropy_with_logits(offset[i]+correction, target[i])
            loss = loss + .01*correction.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    assert model[-1].weight.grad.is_cuda
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter()-started
    torch.save(dict(state=model.cpu().state_dict(), center=center, scale=scale), checkpoint)
    return fit_seconds, torch.cuda.max_memory_allocated()


def infer(x, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = net(x.shape[1]).eval()
    model.load_state_dict(saved["state"])
    values = torch.from_numpy(x)
    with torch.inference_mode():
        standardized = ((values-saved["center"])/saved["scale"]).clamp(-10, 10)
        shift = torch.cat([model(b).flatten().tanh() for b in standardized.split(16384)])
        return (values[:, 0]+.35*shift).sigmoid().numpy()


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    DEST.mkdir(parents=True, exist_ok=True)
    train = tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    valid = panel10()
    assert not set(train) & set(valid+TEST+HISTORICAL)
    train_parts, xt, anchor_t = data(train)
    valid_parts, xv, anchor_v = data(valid)
    y = np.concatenate([part["y"] for part in train_parts])
    baseline = score(valid_parts, anchor_v)
    results = {"train": train, "validation": valid, "baseline": baseline, "trials": {}}
    for name in VARIANTS:
        checkpoint = DEST/f"{name}.pt"
        seconds, peak = fit(design(xt[name], anchor_t), y, checkpoint)
        started = time.perf_counter()
        p = infer(design(xv[name], anchor_v), checkpoint)
        row = dict(metrics=score(valid_parts, p), train_seconds=seconds,
                   cpu_seconds=time.perf_counter()-started, cuda_peak_bytes=peak,
                   checkpoint=checkpoint.name)
        results["trials"][name] = row
        record(DEST/"validation.json", results)
        print(name, json.dumps(row), flush=True)
    best = max(VARIANTS, key=lambda n: results["trials"][n]["metrics"]["auprc"])
    if results["trials"][best]["metrics"]["auprc"] > baseline["auprc"]:
        test_parts, xx, anchor = data(TEST)
        p = infer(design(xx[best], anchor), DEST/f"{best}.pt")
        record(DEST/"test.json", dict(name=best, baseline=score(test_parts, anchor),
                                     corrected=score(test_parts, p), proteins=TEST))


if __name__ == "__main__":
    main()
