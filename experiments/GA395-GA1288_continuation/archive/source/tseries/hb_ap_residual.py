"""Calibrate and rerank a frozen selector with a differentiable AP surrogate."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_occupied import add_occupied
from .hb_single_structure import OUT, load, predict, validation_report
from .hb_validation10 import panel10
from .search_events import dump, view

FEATURES = ("xgb", 118, "hb-origin-environment", False, 6, 1.)
PRIOR = "T1I1301"


def prior_scores(parts, prior):
    if prior == "T1I2006":
        from .hb_ensemble_calibration import assemble, members
        components = members(prior)
        return [sum(weight * matrix[:, i] for i, (weight, _, _) in enumerate(components))
                for matrix in assemble(parts, components)]
    folder = OUT / ("ap10_portfolio_deep" if prior == "T1I1813" else "stacked_volume_96")
    rows = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    return predict(parts, folder, rows[prior])


def network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.LayerNorm(32), nn.SiLU(), nn.Linear(32, 1))


def residue_context(part, probabilities):
    from .data import registry, trajectory
    from .hb_single_structure import ROOT

    reg = registry(ROOT, part["protein"])
    atoms = reg.atoms(trajectory(part["protein"], part["replica"])).to_pydict()
    _, residue = np.unique(atoms["residue_index"], return_inverse=True)
    size = int(residue.max()) + 1
    rows = []
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        a, b = residue[part["pair"][start:end, :2].astype(np.intp)].T
        p = probabilities[start:end]
        key = a * size + b
        reverse = b * size + a
        total = np.bincount(key, p, minlength=size * size)
        count = np.bincount(key, minlength=size * size)
        best = np.zeros(size * size)
        np.maximum.at(best, key, p)
        outgoing = np.bincount(a, p, minlength=size)
        incoming = np.bincount(b, p, minlength=size)
        grouped = total[key]
        rows.append(np.column_stack((
            grouped - p, np.log1p(count[key] - 1), best[key] - p, p / (grouped + .01),
            outgoing[a] - grouped, incoming[b] - grouped, total[reverse], best[reverse],
        )))
    return np.concatenate(rows).astype(np.float32)


def matrix(parts, frame_context=False, prior=PRIOR, relational=False):
    matrices = []
    for part, probabilities in zip(parts, prior_scores(parts, prior)):
        p = np.clip(probabilities, 1e-6, 1 - 1e-6)
        fields = [view(part, FEATURES)]
        if frame_context:
            from .hb_origin_context import context
            fields.append(context(dict(part, old=p)))
        if relational:
            fields.append(residue_context(part, p))
        fields.append(np.log(p / (1 - p)))
        matrices.append(np.column_stack(fields).astype(np.float32))
    return matrices


def cpu_scores(parts, folder, record):
    path = folder / record["fit"]["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["fit"]["sha256"]:
        raise ValueError(f"Checkpoint changed: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return score_matrices(matrix(parts, record["fit"].get("frame_context", False),
                                 record["fit"]["prior"], record["fit"].get("relational", False)), state)


def score_matrices(matrices, state):
    model = network(len(state["center"])).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        values = [torch.from_numpy(x) for x in matrices]
        return [torch.sigmoid(x[:, -1] + model((x-state["center"])/state["scale"]).squeeze(1)).numpy()
                for x in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--incumbent", action="store_true")
    parser.add_argument("--relational", action="store_true")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    if args.relational and not args.incumbent:
        parser.error("--relational requires --incumbent")
    if args.deep and (args.incumbent or args.relational or args.context):
        parser.error("--deep is a separate residual configuration")
    panel = panel10()
    calibration_files = ("ap10_volume_176_training_panel",) if args.deep else (
        ("ap10_calibration_panel",) if args.incumbent else ("ensemble_calibration_panel", "ensemble_gate_panel"))
    calibration = tuple(p for stem in calibration_files
                        for p in json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))["proteins"])
    prior_training = tuple(p for stem in ("diverse_training_panel", "volume_training_panel", "volume_96_training_panel")
                           for p in json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))["proteins"])
    from .data import HISTORICAL, TEST, TRAIN
    from .hb_single_structure import ADDITIONAL_TRAINING
    from .search_events import EXTRA_TRAINING
    if args.incumbent or args.deep:
        prior_training += tuple(p for stem in ("volume_128_training_panel", "ensemble_calibration_panel",
                                               "ensemble_gate_panel")
                                for p in json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))["proteins"])
    if args.deep:
        prior_training += tuple(p for stem in ("ap10_volume_144_training_panel", "ap10_calibration_panel")
                                for p in json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))["proteins"])

    if set(calibration) & set(panel + prior_training + TRAIN + EXTRA_TRAINING + ADDITIONAL_TRAINING + TEST + HISTORICAL):
        raise ValueError("Residual training overlaps prior fitting or evaluation")
    train, valid = load(calibration, 100, geometry=True), load(panel, 100, geometry=True)
    for part in train + valid:
        add_cloud(part, 100)
        add_directional(part, 100)
        add_occupied(part, 100)
    if args.deep:
        from .hb_explicit_context import add_explicit_context
        from .hb_explicit_hydrogen import add_explicit

        for part in train + valid:
            add_explicit(part, 100)
            add_explicit_context(part)
    use_context = args.context or args.incumbent or args.deep
    use_relations = args.relational or args.deep
    prior = "T1I1813" if args.deep else "T1I2006" if args.incumbent else PRIOR
    x = np.concatenate(matrix(train, use_context, prior, use_relations))
    valid_x = matrix(valid, use_context, prior, use_relations)
    y = np.concatenate([part["y"] for part in train])
    center = torch.from_numpy(x.mean(0, dtype=np.float64).astype(np.float32))
    scale = torch.from_numpy(np.maximum(x.std(0, dtype=np.float64), 1e-6).astype(np.float32))
    tensor = torch.as_tensor(x, device="cuda")
    features = (tensor-center.cuda())/scale.cuda()
    target = torch.as_tensor(y, device="cuda", dtype=torch.float32)
    pos = torch.as_tensor(np.flatnonzero(y), device="cuda")
    neg = torch.as_tensor(np.flatnonzero(~y), device="cuda")
    factor = len(neg)/len(pos)*128/1024
    folder = OUT / ("ap10_residual_deep" if args.deep else
                    "ap10_residual_relational" if args.relational else "ap10_residual_incumbent" if args.incumbent else
                    "ap10_residual_context" if args.context else "ap10_residual")
    folder.mkdir(exist_ok=True)
    results = {}
    strengths = (0., .03) if args.deep else (0., .05, .15) if args.incumbent else (0., .15, .4)
    start = 1841 if args.deep else 1621 if args.relational else 1591 if args.incumbent else 1571 if args.context else 1561
    penalty = .25 if args.deep else .05
    for i, emphasis in enumerate(strengths, start):
        torch.manual_seed(27182)
        model = network(features.shape[1]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
        started = time.perf_counter()
        for _ in range(10):
            for ids in torch.randperm(len(y), device="cuda").split(8192):
                optimizer.zero_grad(set_to_none=True)
                residual = model(features[ids]).squeeze(1)
                logits = tensor[ids,-1] + residual
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target[ids]) + penalty * residual.square().mean()
                if emphasis:
                    pi = pos[torch.randint(len(pos), (128,), device="cuda")]
                    ni = neg[torch.randint(len(neg), (1024,), device="cuda")]
                    ps = tensor[pi,-1] + model(features[pi]).squeeze(1)
                    ns = tensor[ni,-1] + model(features[ni]).squeeze(1)
                    tp = ((ps[None,:]-ps[:,None])/.30).sigmoid().sum(1) + .5
                    fp = ((ns[None,:]-ps[:,None])/.30).sigmoid().sum(1)
                    loss -= emphasis * (tp/(tp + factor*fp)).mean()
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        state = dict(model=model.cpu().state_dict(), center=center, scale=scale)
        name = f"T1I{i}"
        path = folder / f"{name}.pt"
        torch.save(state, path)
        report = validation_report(valid, score_matrices(valid_x, state))
        results[name] = dict(fit=dict(checkpoint=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      cuda="cuda:0", updater="AdamW", train_rows=len(y), fit_seconds=seconds,
                                      prior=prior, ap_weight=emphasis, training_proteins=calibration,
                                      frame_context=use_context, relational=use_relations), validation=report)
        dump(folder / "validation.json", results)
        print(name, report["candidate"], flush=True)


if __name__ == "__main__":
    main()
