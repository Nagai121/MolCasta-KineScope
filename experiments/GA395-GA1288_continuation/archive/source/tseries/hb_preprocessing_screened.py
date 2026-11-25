"""Integrate protein heavy atoms with an effective screened charge interaction."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import HISTORICAL, TEST, TRAIN, cell, registry, trajectory
from .features import _mic
from .hb_explicit_hydrogen import source_plans
from .hb_heavy_dynamics import bonded_terms, neighbor_pairs
from .hb_preprocessing_search import prepare as prepare_static
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .hb_trajectory_screen import summary
from .hb_validation10 import panel10
from .search_events import EXTRA_TRAINING


def charge_map(atoms, plan):
    import MDAnalysis as mda

    if plan.get("bond_error") or plan.get("source_to_tpr") is None:
        elements = np.asarray(atoms["element"])
        fallback = np.select([elements == "O", elements == "N", elements == "C"],
                             [-.30, -.10, .10])
        return fallback.astype(np.float64), False
    mapped = np.asarray(plan["source_to_tpr"])[np.asarray(atoms["source_atom"], dtype=np.intp)]
    tpr = mda.Universe(str(plan["tpr"]))
    charges = tpr.atoms.charges.astype(np.float64)
    mass = tpr.atoms.masses
    bonds = tpr.bonds.indices
    left, right = np.concatenate((bonds, bonds[:, ::-1])).T
    attached = (mass[left] < 2.5) & (mass[right] >= 2.5)
    folded = charges + np.bincount(right[attached], weights=charges[left[attached]], minlength=len(charges))
    return folded[mapped], True


def paths(position, box, pair, atoms, charges, rng, coupling):
    edges, rest, k, mass, _ = bonded_terms(position, box, atoms)
    n = len(position)
    excluded = edges[:, 0]*n+edges[:, 1]
    a, b = pair[:, :2].astype(np.intp).T
    contact = neighbor_pairs(position, box, 8.0)
    soft = contact[~np.isin(contact[:, 0]*n+contact[:, 1], excluded)]
    left, right = np.concatenate((edges, soft)).T
    cp = charges[soft[:, 0]]*charges[soft[:, 1]]
    initial = np.linalg.norm(_mic(position[soft[:, 1]]-position[soft[:, 0]], box), axis=1)
    core = np.minimum(initial, 2.8)
    tracks = []
    for _ in range(3):
        x = position.astype(np.float64, copy=True)
        v = rng.normal(size=x.shape) * np.sqrt(.59/mass[:, None])
        snapshots = []
        for i in range(32):
            delta = _mic(x[right]-x[left], box)
            length = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
            d = length[len(rest):]
            safe = np.maximum(d, .8)
            screened = np.exp(-safe/3.)
            attr = -coupling * cp * screened * (1./safe**2 + 1./(3.*safe))
            taper = np.clip((8.-d), 0., 1.)
            overlap = np.minimum(d-core, 0.)
            radial = np.r_[k*(length[:len(rest)]-rest), attr*taper+32.*overlap]
            reaction = delta*(radial/length)[:, None]
            force = np.column_stack([np.bincount(left, z, n)-np.bincount(right, z, n) for z in reaction.T])
            v = .95*v + .04*(force-1.2*(x-position))/mass[:, None]
            v += rng.normal(size=v.shape) * np.sqrt(.59*(1-.95**2)/mass[:, None])
            x += .04*v
            if i % 4 == 3:
                snapshots.append(np.linalg.norm(_mic(x[b]-x[a], box), axis=1))
        tracks.append(snapshots)
    return np.asarray(tracks).transpose(2, 0, 1)


def prepare(part, stride):
    prepare_static(part, stride)
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    folder = ROOT / part["protein"] / "trajectories" / tid / "hb-screened-heavy"
    folder.mkdir(exist_ok=True)
    key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = folder / f"{stride}-{key}.npz"
    features = None
    if path.exists():
        with np.load(path) as data:
            if np.array_equal(data["pair"], part["pair"]) and np.array_equal(data["origin"], part["origin"]):
                features = data["features"]
    if features is None:
        atoms = reg.atoms(tid).to_pydict()
        charges, mapped = charge_map(atoms, source_plans(part["protein"])[f"R{part['replica']}"])
        rows = []
        for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
            if start == end:
                continue
            origin = int(part["origin"][start])
            block = next(reg.blocks(tid, start=origin, stop=origin+1))
            pair = part["pair"][start:end]
            ideas = []
            for coefficient in (0., 24.):
                rng = np.random.default_rng([36914, part["replica"], origin])
                track = paths(block.positions[0], cell(block.box[0]), pair, atoms, charges, rng, coefficient)
                ideas.append(summary(track, pair[:, 2]))
            rows.append(np.column_stack((*ideas, np.full(len(pair), mapped, dtype=np.float32))))
        features = np.concatenate(rows)
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            np.savez(temp, features=features, pair=part["pair"], origin=part["origin"])
        Path(temp.name).replace(path)
    part["screened_control"] = np.column_stack((features[:, :8], part["explicit_context"]))
    part["screened_charge"] = np.column_stack((features[:, 8:16], features[:, 8:16]-features[:, :8],
                                              features[:, 16:], part["explicit_context"]))


def main():
    training = TRAIN + ADDITIONAL_TRAINING + EXTRA_TRAINING
    training += tuple(json.loads((OUT/"diverse_training_panel.json").read_bytes())["proteins"])
    panel = panel10()
    if len(set(training)) != len(training) or set(training) & set(panel+TEST+HISTORICAL):
        raise ValueError("Training overlaps reserved panels")
    train, valid = (load(group, 100, geometry=True) for group in (training, panel))
    with ThreadPoolExecutor(3) as pool:
        for part, _ in zip(train+valid, pool.map(lambda p: prepare(p, 100), train+valid), strict=True):
            print("SCREENED INPUTS", part["protein"], part["replica"], flush=True)
    for i, name in ((2181, "screened_control"), (2182, "screened_charge")):
        stage = f"ap15_preprocessing/T1I{i}"
        if not (OUT / stage / "selection.json").exists():
            explore(stage, (("xgb-rounds-420", 166, f"hb-origin-stacked-{name}", False, 8, 1.),),
                    i, train, valid)


if __name__ == "__main__":
    main()
