"""Screen the 24 non-all-atom trajectory proposals with bounded CPU approximations."""
import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, log_loss

from .data import HISTORICAL, TEST, TRAIN, cell, registry, topology, trajectory
from .features import _mic
from .hb_ap_replay import replay
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_explicit_context import add_explicit_context
from .hb_explicit_hydrogen import add_explicit
from .hb_occupied import add_occupied
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load, predict, validation_report
from .hb_validation10 import panel10
from .search_events import EXTRA_TRAINING, dump

STAGE = "ap10_trajectory_24"
KINETIC = {
    2: "heavy endpoint propagation with virtual donor-hydrogen direction",
    3: "residue motion plus local endpoint rotation for side chains",
    4: "candidate-local damped endpoint dynamics with anchored environment",
    5: "residue motion with local angular relaxation",
    6: "small correlated rotations about backbone anchors",
    7: "correlated neighboring peptide rotations",
    8: "side-chain conformation samples from the existing collision-filtered generator",
    9: "loop-like deformation with a zero displacement envelope at both ends",
    10: "rigid movement of connected short backbone segments",
    11: "superposed low-frequency elastic graph modes",
    12: "candidate-oriented elastic mode selection",
    13: "packing-dependent anisotropic displacement",
    14: "slow elastic deformation with fast local rotation",
    15: "residue contact graph updated during displacement",
    16: "paired approach and retreat probes",
    17: "release after a contact-directed displacement",
    18: "shared-donor alternative acceptor relaxation",
    19: "adaptive allocation to ambiguous trial paths",
    20: "shortlist coarse paths followed by endpoint rotation",
    21: "learned terminal-distance distribution and interpolated paths",
    22: "learned additive correction to coarse path distances",
    23: "learned coarse distance-state transition probabilities",
    24: "empirical neighbor transport of terminal displacement",
    25: "prediction-loss-trained latent displacement paths",
}

T = np.linspace(.125, 1., 8, dtype=np.float32)


def summary(track, d):
    track = np.asarray(track, dtype=np.float32).reshape(len(d), -1, 8)
    delta = track - d[:, None, None]
    return np.column_stack((delta.mean((1, 2)), delta.std((1, 2)), delta.min((1, 2)),
                            delta.max((1, 2)), (track <= 3.5).mean((1, 2)),
                            (track <= 4.).mean((1, 2)), delta[:, :, -1].mean(1),
                            (np.diff(track, axis=2) < 0).mean((1, 2)))).astype(np.float32)


def physical(position, box, pair, topo, rng):
    res = topo[0]
    a, b = pair[:, :2].astype(np.intp).T
    owner = np.full(len(position), -1)
    for i, atoms in enumerate(res):
        owner[list(atoms.values())] = i
    if np.any(owner[a] < 0) or np.any(owner[b] < 0):
        raise ValueError("Missing residue anchor")
    ia, ib = owner[a], owner[b]
    ca = position[[atoms["CA"] for atoms in res]]
    rel = _mic(position - ca[np.maximum(owner, 0)], box)
    delta = _mic(position[b] - position[a], box)
    d = pair[:, 2]
    axis = delta / d[:, None]
    cab = _mic(ca[:, None] - ca[None, :], box)
    identity = np.eye(len(ca), dtype=bool)
    contacts = (np.linalg.norm(cab, axis=-1) < 9.) & ~identity
    degree = contacts.sum(1)
    graph = contacts.astype(np.float32)
    v = rng.normal(0., .38, (len(ca), 3)).astype(np.float32)
    w = rng.normal(0., .30, (len(ca), 3)).astype(np.float32)
    smooth = (graph @ w + w) / (1 + degree[:, None])
    v = (graph @ v + 3. * v) / (3. + degree[:, None])
    displ = v[ib] - v[ia]
    rotation = np.cross(w[ib], rel[b]) - np.cross(w[ia], rel[a])
    correlated = np.cross(smooth[ib], rel[b]) - np.cross(smooth[ia], rel[a])
    neighbor_axis = _mic(ca[np.minimum(np.arange(len(ca)) + 1, len(ca) - 1)] -
                         ca[np.maximum(np.arange(len(ca)) - 1, 0)], box)
    neighbor_axis /= np.maximum(np.linalg.norm(neighbor_axis, axis=1, keepdims=True), 1e-6)
    twist = (w * neighbor_axis).sum(1, keepdims=True) * neighbor_axis
    hinge = np.cross(twist[ib], rel[b]) - np.cross(twist[ia], rel[a])
    amplitude = .65 / np.sqrt(1. + degree[ia] + degree[ib])
    sign = np.array([-1., 1.], dtype=np.float32)

    def paths(shift, ramp=T):
        vector = delta[:, None, None, :] + shift[:, None, None, :] * sign[None, :, None, None] * ramp[None, None, :, None]
        return np.linalg.norm(vector, axis=-1)

    def scalar(step, ramp=T):
        return d[:, None, None] + step[:, None, None] * sign[None, :, None] * ramp[None, None]

    n_anchor = np.arange(len(ca))
    virtual = _mic(position[a] - ca[ia], box)
    virtual /= np.maximum(np.linalg.norm(virtual, axis=1, keepdims=True), 1e-5)
    virtual_direction = (axis * virtual).sum(1)
    backbone = set(i for atoms in res for k,i in atoms.items() if k in ("N", "CA", "C", "O", "OXT"))
    rotate_side = (np.cross(w[ib], rel[b]) * ~np.isin(b, list(backbone))[:, None]
                   - np.cross(w[ia], rel[a]) * ~np.isin(a, list(backbone))[:, None])
    loop = np.sin(np.pi * (n_anchor % 7) / 6.)
    segment = n_anchor // 5
    group_v = v[np.minimum(segment * 5 + 2, len(v) - 1)]

    eigenvalues, modes = np.linalg.eigh(np.diag(degree.astype(np.float32)) - graph)
    modes = modes[:, eigenvalues > .0001][:, :5]
    if not modes.shape[1]:
        modes = np.zeros((len(ca), 1), dtype=np.float32)
    modal = modes @ rng.normal(0., .35, (modes.shape[1], 3))
    selected = .45 * (modes[ib] - modes[ia])
    magnitude = np.max(np.abs(selected), axis=1)
    crowd = np.bincount(a, minlength=len(position))[a]
    competition = np.zeros(len(position))
    np.minimum.at(competition, a, d - 6.)
    other = np.maximum(d - (competition[a] + 6.), 0.)

    updated = []
    shift = np.zeros_like(v)
    velocity = v.copy()
    for step in range(8):
        moved = cab + shift[:, None] - shift[None, :]
        g = ((np.linalg.norm(moved, axis=-1) < 9.) & ~identity).astype(np.float32)
        velocity = .88 * velocity + .18 * ((g @ shift) / np.maximum(g.sum(1)[:, None], 1.) - shift)
        shift += .22 * velocity
        change = shift[ib] - shift[ia]
        updated.append(np.linalg.norm(delta[:, None] + change[:, None] * sign[None, :, None], axis=-1))

    track = {
        2: paths(displ + .14 * virtual * virtual_direction[:, None]),
        3: paths(displ + rotate_side),
        4: scalar(amplitude, (1. - np.exp(-3. * T))),
        5: paths(displ + correlated, (1. - np.exp(-4. * T)) * np.exp(-T)),
        6: paths(hinge, np.sin(np.pi * T / 2.)),
        7: paths(correlated, np.sin(np.pi * T)),
        9: paths(v[ib] * loop[ib, None] - v[ia] * loop[ia, None] + .2 * correlated),
        10: paths(group_v[ib] - group_v[ia]),
        11: paths(modal[ib] - modal[ia], np.sin(np.pi * T / 2.)),
        12: scalar(magnitude, np.sin(np.pi * T / 2.)),
        13: paths(rotation + axis * amplitude[:, None]),
        14: paths(modal[ib] - modal[ia]) + scalar((rotation * axis).sum(1), .3 * np.sin(4. * np.pi * T)) - d[:, None, None],
        15: np.stack(updated, axis=-1),
        16: scalar(.35 / (1. + .08 * (degree[ia] + degree[ib]))),
        17: scalar(amplitude * 2., np.where(T < .5, 2. * T, np.exp(-3. * (T - .5)))),
        18: scalar(.6 * amplitude * (np.log1p(crowd) + other), np.sin(np.pi * T)),
        19: np.concatenate((scalar(amplitude), scalar(amplitude * np.where(abs(d - 4.) < .3, 1.8, 1.)) ), axis=1),
        20: paths(displ + np.where(abs(d - 4.)[:, None] < .45, rotation, 0.), T),
    }
    return {i: summary(values, d) for i, values in track.items()}


def features(part, stride):
    from .hb_torsional import add_torsional

    add_cloud(part, stride)
    add_directional(part, stride)
    add_occupied(part, stride)
    add_torsional(part, stride)
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    topo = topology(reg.atoms(tid).to_pydict())
    out = {i: [] for i in range(2, 21) if i != 8}
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        if start == end:
            continue
        origin = int(part["origin"][start])
        block = next(reg.blocks(tid, start=origin, stop=origin + 1))
        blocks = physical(block.positions[0], cell(block.box[0]), part["pair"][start:end], topo,
                          np.random.default_rng(origin + 27182))
        for i, values in blocks.items():
            out[i].append(values)
    for i, rows in out.items():
        part[f"motion{i}"] = np.concatenate(rows)
    part["motion8"] = part["torsional"]


def generator_input(part):
    return np.column_stack((part["pair"][:, 2], part["base"][:, :8],
                            part["directional"][:, :8], part["motion11"])).astype(np.float32)


def generator_network(width):
    from torch import nn

    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 14))


def train_generator(parts, folder):
    import torch
    from torch.nn import functional as F

    torch.manual_seed(27182)
    x = np.concatenate([generator_input(p) for p in parts])
    distance = x[:, 0].copy()
    y = np.concatenate([p["future_distance"] for p in parts]).astype(np.float32)
    coarse = np.concatenate([p["motion11"][:, 6] for p in parts])
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    x_gpu, d_gpu, y_gpu, c_gpu = (torch.as_tensor(a, device="cuda") for a in ((x-center)/scale, distance, y, coarse))
    model = generator_network(x.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(9):
        for ix in torch.randperm(len(y), device="cuda").split(4096):
            opt.zero_grad()
            z = model(x_gpu[ix])
            shift = (y_gpu[ix] - d_gpu[ix]).clamp(-4, 4)
            scale_pred = F.softplus(z[:, 1]) + .08
            nll = .5 * ((shift - z[:, 0]) / scale_pred).square() + scale_pred.log()
            state = torch.bucketize(y_gpu[ix], y_gpu.new_tensor([3.5, 4.75]))
            latent = d_gpu[ix, None] + 2. * z[:, 5:11].tanh()
            logit = (3.5 - latent).amax(1) / .30 + z[:, 0]
            loss = (nll.mean() + F.smooth_l1_loss(z[:, 2]+c_gpu[ix], shift)
                    + F.cross_entropy(z[:, 11:14], state)
                    + F.binary_cross_entropy_with_logits(logit, (y_gpu[ix] <= 3.5).float()))
            loss.backward()
            opt.step()
    torch.cuda.synchronize()
    checkpoint = folder / "generator.pt"
    torch.save(dict(model=model.cpu().state_dict(), center=torch.from_numpy(center),
                    scale=torch.from_numpy(scale)), checkpoint)
    ix = np.random.default_rng(27182).choice(len(y), min(16000, len(y)), replace=False)
    np.savez_compressed(folder / "neighbors.npz", x=((x-center)/scale)[ix, :9], shift=(y-distance)[ix])
    return model.eval(), center, scale, checkpoint


def learned(part, model, center, scale, tree, library):
    import torch

    x = generator_input(part)
    d = x[:, 0]
    norm = (x-center)/scale
    with torch.inference_mode():
        z = model(torch.from_numpy(norm)).numpy()
    spreads = np.logaddexp(0., z[:, 1]) + .08
    shifts = z[:, :1] + spreads[:, None] * np.array([-1., -.35, .35, 1.])
    correction = z[:, 2] + part["motion11"][:, 6]
    probabilities = torch.softmax(torch.from_numpy(z[:, 11:14]), dim=1).numpy()
    state_mean = probabilities @ np.array([3.25, 4.1, 5.6])
    _, near = tree.query(norm[:, :9], k=4)
    options = {
        21: shifts,
        22: correction[:, None] + spreads[:, None] * np.array([-.5, .5]),
        23: (state_mean-d)[:, None] * np.array([.6, 1.4]),
        24: library[near],
        25: 2. * np.tanh(z[:, 5:11]),
    }
    for i, displacement in options.items():
        track = d[:, None, None] + displacement[:, :, None] * T
        values = summary(track, d)
        if i == 23:
            values[:, :3] = probabilities
        part[f"motion{i}"] = values


def band_comparison(valid, p, y, bands, delta, guard, baseline_ap):
    best, chosen, best_ap = p, [0., 0., 0.], baseline_ap
    tried = []
    for weights in itertools.product((0., .03125, .0625, .125), repeat=3):
        q = p + np.take(weights, bands) * delta
        ap, loss = float(average_precision_score(y,q)), float(log_loss(y,q))
        tried.append(dict(weights=weights, auprc=ap, logloss=loss))
        if ap > best_ap and loss <= guard:
            best, chosen, best_ap = q, weights, ap
    breaks = np.cumsum([len(part["y"]) for part in valid])[:-1]
    return dict(weights=chosen, metrics=validation_report(valid,np.split(best,breaks)),trials=tried,
                best_nonzero=max((item for item in tried if any(item["weights"])),key=lambda item:item["auprc"]))


def compare_bands(panel):
    import torch

    folder = OUT / STAGE
    original_ledger = json.loads((folder / "comparison_2_25.json").read_bytes())
    if tuple(original_ledger["validation"]) != panel:
        raise ValueError("Validation panel changed")
    checkpoint = folder / "generator.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original_ledger["generator_sha256"]:
        raise ValueError("Generator checkpoint changed")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = generator_network(len(state["center"]))
    model.load_state_dict(state["model"])
    model.eval()
    with np.load(folder / "neighbors.npz") as library:
        tree, shifts = cKDTree(library["x"]), library["shift"]
    valid = load(panel, 100, geometry=True)
    started = time.perf_counter()
    for part in valid:
        part_start = time.perf_counter()
        features(part, 100)
        learned(part, model, state["center"].numpy(), state["scale"].numpy(), tree, shifts)
        part["motion_wall_seconds"] = time.perf_counter() - part_start
        add_explicit(part, 100)
        add_explicit_context(part)
    elapsed = time.perf_counter()-started
    motion_seconds = sum(part["motion_wall_seconds"] for part in valid)
    frozen = json.loads((OUT / "selections/T1I2007/ap_recipe.json").read_bytes())
    _, original = replay(valid, frozen)
    baseline = validation_report(valid, original)
    p = np.concatenate(original)
    y = np.concatenate([part["y"] for part in valid])
    bands = np.digitize(np.concatenate([part["pair"][:, 2] for part in valid]), [4., 4.75])
    guard = baseline["candidate"]["logloss"]
    result = dict(validation=panel, baseline=baseline["candidate"],
                  validation_cpu_preparation_seconds=elapsed, validation_motion_seconds=motion_seconds,
                  models={})
    for i in [0, *range(2, 26)]:
        stage = folder / f"idea_{i:02d}"
        row = json.loads((stage / "selection.json").read_bytes())
        if hashlib.sha256((stage / row["fit"]["checkpoint"]).read_bytes()).hexdigest() != row["fit"]["sha256"]:
            raise ValueError(f"Selector checkpoint changed for idea {i}")
        delta = np.concatenate(predict(valid, stage, row)).astype(np.float64) - p
        result["models"][i] = band_comparison(valid,p,y,bands,delta,guard,baseline["candidate"]["auprc"])
        print("BANDS", i, result["models"][i]["weights"], result["models"][i]["best_nonzero"], flush=True)
        dump(folder / "bands.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=2)
    parser.add_argument("--last", type=int, default=25)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.first <= args.last <= 25:
        parser.error("Choose an inclusive range from 2 to 25")
    panel = panel10()
    if args.compare:
        compare_bands(panel)
        return
    extra = tuple(json.loads((OUT / "diverse_training_panel.json").read_bytes())["proteins"])
    generator_names = extra[:8]
    middle = EXTRA_TRAINING + ADDITIONAL_TRAINING
    selector_names = TRAIN + middle + extra[8:]
    if set(generator_names) & set(selector_names) or set(generator_names + selector_names) & set(panel+TEST+HISTORICAL):
        raise ValueError("Overlapping protein partitions")
    groups = ((load(TRAIN,25,geometry=True),25), (load(middle,50,geometry=True),50),
              (load(extra[8:],100,geometry=True),100))
    auxiliary = load(generator_names,100,targets=True,geometry=True)
    valid = load(panel,100,geometry=True)
    folder = OUT / STAGE
    folder.mkdir(exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(3) as pool:
        jobs = [pool.submit(features, p, stride) for parts,stride in (*groups,(auxiliary,100),(valid,100)) for p in parts]
        for job in jobs:
            job.result()
    seconds = time.perf_counter() - started
    print("CPU preprocessing ready", seconds, flush=True)
    model, center, scale, checkpoint = train_generator(auxiliary, folder)
    with np.load(folder / "neighbors.npz") as data:
        tree, library = cKDTree(data["x"]), data["shift"]
    train = [p for parts,_ in groups for p in parts]
    for p in train + valid:
        learned(p, model, center, scale, tree, library)
    for p in valid:
        add_explicit(p,100)
        add_explicit_context(p)
    frozen = json.loads((OUT/"selections/T1I2007/ap_recipe.json").read_bytes())
    _, baseline = replay(valid, frozen)
    p = np.concatenate(baseline)
    y = np.concatenate([q["y"] for q in valid])
    baseline_metrics = validation_report(valid, baseline)["candidate"]
    if not np.isclose(baseline_metrics["auprc"], frozen["best"]["auprc"], atol=1e-12, rtol=0):
        raise ValueError("Frozen T1I2007 score changed")
    stage_config = folder / f"comparison_{args.first}_{args.last}.json"
    ledger = dict(training=selector_names, generator_training=generator_names, validation=panel,
                  cpu_feature_wall_seconds=seconds, baseline=baseline_metrics,
                  generator_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), models={})
    indices = ([0] if args.first == 2 else []) + list(range(args.first,args.last+1))
    for i in indices:
        stage = f"{STAGE}/idea_{i:02d}"
        kind = f"motion{i}" if i else "directional"
        config = ("xgb", 118, f"hb-origin-stacked-{kind}", False, 6, 1.)
        number = 1900+i
        explore(stage, (config,), number, train, valid)
        path = OUT / stage
        row = json.loads((path/"selection.json").read_bytes())
        q = np.concatenate(predict(valid,path,row)).astype(np.float64)
        mixes = []
        for weight in (0., .03125, .0625, .125, .25):
            scores = p + weight * (q-p)
            mixes.append(dict(weight=weight, auprc=float(average_precision_score(y,scores)),
                              logloss=float(log_loss(y,scores))))
        ledger["models"][i] = dict(description=KINETIC.get(i,"matched static control"), record=row, mixtures=mixes)
        dump(stage_config, ledger)
        print("IDEA",i,"AP",row["validation"]["candidate"]["auprc"],"mixtures",mixes,flush=True)


if __name__ == "__main__":
    main()
