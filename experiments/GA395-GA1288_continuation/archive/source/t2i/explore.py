"""T2I static-origin exploration on the archived hydrogen-bond ROD panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit, logit

from tseries.data import HISTORICAL, TEST, cell, registry, shard, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import OUT, ROOT, load
from tseries.hb_preprocessing_search import prepare, competition
from tseries.hb_validation10 import panel10
from tseries.search_events import count_to_recall, metrics

DEST = Path("runs/t2i")
STRIDE = 100
FAMILIES = ("geometry", "competition", "interaction", "hydrogen", "cloud",
            "corridor", "matching", "radial", "closure", "all")
STRUCTURES = ("residual", "experts", "additive", "gated", "lowrank")
OBJECTIVES = ("binary", "mean", "residual", "occupancy", "joint", "rank", "tail",
              "distribution", "rank-joint", "calibrated")


def record(path, item):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(item, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def history(part):
    """Observed backwards 1 ns, labels for training only."""
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    blocks = []
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        origin = int(part["origin"][start])
        frame = next(reg.blocks(tid, start=origin-10, stop=origin+1, chunk_frames=11))
        if len(frame.time) != 11 or abs(frame.time[-1] - frame.time[0] - 1000) > .01:
            raise RuntimeError("Invalid history time grid")
        a, b = part["pair"][start:end, :2].astype(np.intp).T
        dist = np.stack([np.linalg.norm(_mic(p[b] - p[a], cell(box)), axis=1)
                         for p, box in zip(frame.positions, frame.box)])
        blocks.append(np.column_stack((dist.mean(0), (dist <= 3.5).mean(0),
                                      dist.std(0), np.quantile(dist, .1, axis=0),
                                      dist[0] - dist[-1])))
    return np.concatenate(blocks).astype(np.float32)


def prepare_one(part, training):
    started = time.perf_counter()
    prepare(part, STRIDE)
    pair = part["pair"]
    x = part["base"]
    d = pair[:, 2]
    prep = {
        "geometry": x[:, :118],
        "competition": x[:, 118:142],
        "interaction": x[:, 142:166],
        "hydrogen": part["explicit_context"],
        "cloud": np.column_stack((part["directional"], part["occupied"])),
        "corridor": part["corridor"][:, :20],
        "matching": competition(part),
        "radial": np.exp(-((d[:, None] - np.linspace(3.5, 6, 8)) / .25)**2),
        "closure": np.stack([expit((3.5-d)/scale) * (.5+.5*part["explicit"][:, 3])
                             for scale in (.15, .25, .4, .65, 1., 1.6, 2.5, 4.)], axis=1),
    }
    ext = {
        "geometry": np.column_stack((x[:, 6:8], x[:, 12:14], x[:, 39:41], x[:, 56:58])),
        "competition": prep["competition"][:, :8],
        "interaction": prep["interaction"][:, :8],
        "hydrogen": prep["hydrogen"][:, :8],
        "cloud": prep["cloud"][:, :8],
        "corridor": prep["corridor"][:, :8],
        "matching": prep["matching"][:, :8],
        "radial": prep["radial"],
        "closure": prep["closure"],
        "all": np.column_stack([values[:, 0] for values in list(prep.values())[:8]]),
    }
    prep["all"] = np.column_stack(list(prep.values())[1:])
    reference = np.column_stack((np.ones(len(d)), d, d*d, x[:, 6:8], x[:, 12:14],
                                 x[:, 39:41], pair[:, 6:8])).astype(np.float32)
    result = dict(pair=pair, base=x, extras=ext, blocks=prep, fixed=reference,
                  y=part["y"], protein=part["protein"], replica=part["replica"],
                  origin=part["origin"], feature_seconds=time.perf_counter()-started)
    if training:
        result["history"] = history(part)
    return result


def matrix(parts, family):
    output = []
    for p in parts:
        candidate = np.column_stack((p["pair"], p["extras"][family])).astype(np.float32)
        assert candidate.shape[1] == 16
        features = p["blocks"]["geometry"]
        if family != "geometry":
            features = np.column_stack((features, p["blocks"][family]))
        output.append(np.column_stack((candidate[:, [2, 6, 7]], features, candidate[:, 8:])))
    result = np.concatenate(output).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError(f"Nonfinite inputs for {family}")
    return result


class Model(nn.Module):
    def __init__(self, width, architecture, auxiliary):
        super().__init__()
        self.architecture = architecture
        self.input = nn.Linear(width, 96)
        self.linear = nn.Linear(width, 1)
        self.hidden = nn.Linear(96, 96)
        self.gate = nn.Linear(width, 3 if architecture == "experts" else 96)
        self.project = nn.Linear(width, 12)
        self.lift = nn.Linear(12, 96)
        self.out = nn.Linear(96, 5 if architecture == "experts" else 3)
        self.aux = nn.Linear(96, 8)
        self.link = nn.Linear(8, 1)
        self.auxiliary = auxiliary

    def forward(self, x):
        h = F.silu(self.input(x))
        if self.architecture == "residual":
            h = h + F.silu(self.hidden(h))
        elif self.architecture == "gated":
            h = F.silu(self.hidden(h)) * self.gate(x).sigmoid()
        elif self.architecture == "lowrank":
            h = F.silu(self.lift(self.project(x).square())) + h
        z = self.out(h)
        if self.architecture == "experts":
            probs = (z[:, :3].sigmoid() * self.gate(x).softmax(1)).sum(1)
            pred = torch.logit(probs.clamp(.000001, .999999))
        elif self.architecture == "additive":
            pred = self.linear(x).flatten() + z[:, 0]
        else:
            pred = z[:, 0]
        aux = self.aux(h)
        if self.auxiliary:
            pred = pred + self.link(aux).flatten()
        return pred, aux


def gaussian_stats(parts, train_names):
    # Fit nuisance projections out of the fold for every training protein.
    matrices = {name: np.concatenate([p["fixed"] for p in parts if p["protein"] == name])
                for name in train_names}
    targets = {name: np.concatenate([p["history"][:, :2] for p in parts if p["protein"] == name])
               for name in train_names}
    summaries = {n: (m.astype("d").T @ m, m.astype("d").T @ targets[n])
                 for n, m in matrices.items()}
    cov = sum(a for a, _ in summaries.values())
    rhs = sum(b for _, b in summaries.values())
    out = []
    for p in parts:
        a, b = summaries[p["protein"]]
        typical = p["fixed"] @ np.linalg.solve(cov-a+np.eye(len(cov))*.01, rhs-b)
        values = p["history"]
        delta = values[:, :2] - typical
        out.append(np.column_stack((values[:, 0]-p["pair"][:, 2], values[:, 1],
                                    delta, values[:, 2:],
                                    values[:, 1] > 0)))
    return np.concatenate(out).astype(np.float32)


def loss_for(objective, logits, aux, y, teacher):
    loss = F.binary_cross_entropy_with_logits(logits, y)
    if "rank" in objective:
        pos, neg = logits[y > .5], logits[y <= .5]
        n = min(len(pos), len(neg))
        if n:
            loss = loss + .16 * F.softplus(neg[:n] - pos[:n]).mean()
    columns = {"mean": (0,), "residual": (2,), "occupancy": (1, 3),
               "joint": (0, 2, 3), "tail": (4, 5, 6),
               "distribution": (0, 1, 2, 3, 4, 5, 6),
               "rank-joint": (0, 2, 3)}
    if objective in columns:
        cols = columns[objective]
        loss = loss + .2 * F.smooth_l1_loss(aux[:, cols], teacher[:, cols])
    if objective == "distribution":
        loss = loss + .05 * F.binary_cross_entropy_with_logits(aux[:, 7], teacher[:, 7])
    if objective == "calibrated":
        loss = loss + .6 * (logits.sigmoid() - y).square().mean()
    return loss


def estimate(index, family, architecture, objective, xt, xv, y, teacher, epochs):
    torch.manual_seed(20000 + index)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    mean = xt.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(xt.std(0, dtype=np.float64).astype(np.float32), .001)
    def normalize(x):
        return torch.from_numpy(np.clip((x-mean)/scale, -15, 15))
    x = normalize(xt).cuda()
    labels = torch.from_numpy(y.astype(np.float32)).cuda()
    aux = torch.from_numpy(teacher).cuda()
    use_aux = objective not in ("binary", "rank", "calibrated")
    model = Model(x.shape[1], architecture, use_aux).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.003)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=x.device).split(4096):
            l, a = model(x[idx])
            loss = loss_for(objective, l, a, labels[idx], aux[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    assert next(model.parameters()).grad.is_cuda
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    model.cpu().eval()
    name = f"T2I{index:02}"
    checkpoint = DEST / f"{name}.pt"
    torch.save(dict(model=model.state_dict(), family=family, architecture=architecture,
                    objective=objective, auxiliary=use_aux,
                    mean=torch.from_numpy(mean), scale=torch.from_numpy(scale)), checkpoint)
    started = time.perf_counter()
    prediction = predict(checkpoint, xv)
    return prediction, dict(name=name, family=family, architecture=architecture,
                            objective=objective, fit_seconds=fit_seconds,
                            cpu_seconds=time.perf_counter()-started, cuda_peak_bytes=peak,
                            checkpoint=checkpoint.name, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            epochs=epochs, seed=20000+index)


def predict(checkpoint, x):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    normalized = ((torch.from_numpy(x) - saved["mean"]) / saved["scale"]).clamp(-15, 15)
    model = Model(x.shape[1], saved["architecture"], saved["auxiliary"])
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(b)[0].sigmoid().numpy() for b in normalized.split(16384)])


def score(parts, p):
    y = np.concatenate([q["y"] for q in parts])
    pooled = metrics(y, p)
    pooled.update(rows=len(y), positives=int(y.sum()), r80=count_to_recall(y, p, np.ones(len(y), bool)))
    return pooled


def cohort(names, training):
    base = load(names, STRIDE, geometry=False)
    with ThreadPoolExecutor(3) as pool:
        return list(pool.map(lambda p: prepare_one(p, training), base))


def roster(rows):
    scores = {family: max((r["validation"]["auprc"] for r in rows if r["family"] == family),
                          default=0.) for family in FAMILIES}
    architecture = {name: max((r["validation"]["auprc"] for r in rows if r["architecture"] == name),
                              default=0.) for name in STRUCTURES}
    used = {(r["family"], r["objective"]) for r in rows}
    objective = len(rows) // len(FAMILIES)
    remaining = [f for f in FAMILIES if (f, OBJECTIVES[objective]) not in used]
    family = (remaining[0] if objective == 0 else max(remaining, key=scores.get))
    if objective < len(STRUCTURES):
        structure = STRUCTURES[(FAMILIES.index(family)+objective) % len(STRUCTURES)]
    else:
        prior = [r for r in rows if r["family"] == family]
        structure = max(STRUCTURES, key=lambda a: max(
            (r["validation"]["auprc"] for r in prior if r["architecture"] == a),
            default=architecture[a]))
    return family, structure, OBJECTIVES[objective]


def peak_rss():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ("peak", "working", "qppeak", "qp", "qnpeak",
                                              "qn", "page", "pagepeak")]
    mem = Memory()
    mem.cb = ctypes.sizeof(mem)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    get = ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)
    if get(current(), ctypes.byref(mem), mem.cb):
        return mem.peak
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training is required")
    DEST.mkdir(parents=True, exist_ok=True)
    valid_names = panel10()
    train_names = tuple(json.loads((OUT / "volume_training_panel.json").read_text())["proteins"])[:24]
    assert len(train_names) == 24 and len(set(train_names)) == 24
    assert not set(train_names) & set(valid_names + TEST + HISTORICAL)
    run = dict(train=train_names, validation=valid_names, test=TEST, stride=STRIDE,
               horizon_ns=1, origin_first_frame=49, epochs=args.epochs,
               cuda=torch.version.cuda, accelerator=torch.cuda.get_device_name(),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest = DEST / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(run)):
        raise RuntimeError("Run manifest mismatch")
    record(manifest, run)
    print(json.dumps(run), flush=True)
    train, valid = cohort(train_names, True), cohort(valid_names, False)
    print(f"Prepared {sum(len(p['y']) for p in train)} train and {sum(len(p['y']) for p in valid)} validation candidates", flush=True)
    y = np.concatenate([p["y"] for p in train])
    teacher = gaussian_stats(train, train_names)
    path = DEST / "results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    for row in rows:
        if hashlib.sha256((DEST / row["checkpoint"]).read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise RuntimeError("Previously recorded checkpoint changed")
    for i in range(len(rows), args.limit):
        family, architecture, objective = roster(rows)
        p, row = estimate(i+1, family, architecture, objective, matrix(train, family),
                          matrix(valid, family), y, teacher, args.epochs)
        row["validation"] = score(valid, p)
        row["process_peak_rss_bytes"] = peak_rss()
        rows.append(row)
        record(path, rows)
        print(json.dumps(row), flush=True)
    if len(rows) == 100:
        top = max(rows, key=lambda r: (r["validation"]["auprc"], -r["validation"]["logloss"]))
        test = cohort(TEST, False)
        start = time.perf_counter()
        p = predict(DEST / top["checkpoint"], matrix(test, top["family"]))
        record(DEST / "test.json", dict(name=top["name"], metrics=score(test, p),
                                       score_seconds=time.perf_counter()-start,
                                       proteins=TEST, selection="validation AUPRC then logloss"))
    record(DEST / "preprocessing.json", {
        "sum_parallel_work_seconds": {key: sum(p["feature_seconds"] for p in parts)
                                      for key, parts in (("train", train), ("validation", valid))},
        "training_rows": len(y), "training_positives": int(y.sum()),
    })


if __name__ == "__main__":
    main()
