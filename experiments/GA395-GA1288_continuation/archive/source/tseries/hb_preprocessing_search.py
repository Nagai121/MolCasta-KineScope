"""Search static-origin corridor, packing and competitor preprocessing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

from .data import HISTORICAL, TEST, TRAIN, cell, registry, trajectory
from .features import _mic
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_explicit_context import add_explicit_context
from .hb_explicit_hydrogen import add_explicit
from .hb_occupied import add_occupied
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .hb_validation10 import panel10
from .search_events import EXTRA_TRAINING


def frame(position, box, pair, atoms):
    a, b = pair[:, :2].astype(np.intp).T
    delta = _mic(position[b] - position[a], box)
    length = np.linalg.norm(delta, axis=1)
    axis = delta / length[:, None]
    mid = position[a] + .5 * delta
    if box.ndim == 1:
        tree = cKDTree(position % box, boxsize=box)
        _, near = tree.query(mid % box, k=min(36, len(position)))
    else:
        inverse = np.linalg.inv(box)
        tree = cKDTree((position @ inverse) % 1., boxsize=1.)
        _, near = tree.query((mid @ inverse) % 1., k=min(48, len(position)))
    off = _mic(position[near] - mid[:, None], box)
    along = (off * axis[:, None]).sum(axis=2)
    rad = np.linalg.norm(off - along[:, :, None]*axis[:, None], axis=2)
    lo = np.asarray(atoms["residue_index"])
    elem = np.asarray(atoms["element"])
    external = (lo[near] != lo[a, None]) & (lo[near] != lo[b, None])
    third = (near != a[:, None]) & (near != b[:, None])
    blocks = []
    for select in (third, external):
        for fraction in (-.35, 0., .35):
            d = np.sqrt(rad**2 + (along - fraction*length[:, None])**2)
            blocks.extend((np.where(select, d, 9.).min(axis=1),
                           (select * np.exp(-.5*d**2)).sum(axis=1),
                           (select * np.isin(elem[near], ("N", "O", "S")) * np.exp(-.25*d**2)).sum(axis=1)))
    contact = (rad < 2.7) & (np.abs(along) < .5*length[:, None]) & third
    blocks.extend((contact.sum(axis=1), (contact & external).sum(axis=1)))
    return np.column_stack(blocks).astype(np.float32)


def competition(part):
    pair = part["pair"]
    explicit = part["explicit"]
    blocks = []
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        if start == end:
            continue
        p, x = pair[start:end], explicit[start:end]
        values = []
        for quality in (np.exp(-(p[:, 2]-3.5)), x[:, 3],
                        np.sqrt(np.maximum(x[:, 3], 0.))*np.exp(-(p[:, 2]-3.5))):
            affinity = quality.astype(float)
            for _ in range(3):
                for endpoint in (0, 1):
                    group = p[:, endpoint].astype(np.intp)
                    total = np.bincount(group, weights=affinity)[group]
                    values.append(affinity / np.maximum(total, 1e-8))
                    values.append(np.log1p(np.maximum(total-affinity, 0.)))
                    affinity = affinity / np.sqrt(np.maximum(total, .05))
        blocks.append(np.column_stack(values))
    return np.concatenate(blocks).astype(np.float32)


def prepare(part, stride):
    add_cloud(part, stride)
    add_directional(part, stride)
    add_occupied(part, stride)
    add_explicit(part, stride)
    add_explicit_context(part)
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    atoms = reg.atoms(tid).to_pydict()
    folder = ROOT / part["protein"] / "trajectories" / tid / "hb-preprocessing-search"
    folder.mkdir(exist_ok=True)
    key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = folder / f"{stride}-{key}.npz"
    corridor = None
    if path.is_file():
        with np.load(path) as data:
            if np.array_equal(data["pair"], part["pair"]) and np.array_equal(data["origin"], part["origin"]):
                corridor = data["corridor"]
    if corridor is None:
        frames = []
        for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
            if start == end:
                continue
            origin = int(part["origin"][start])
            block = next(reg.blocks(tid, start=origin, stop=origin+1))
            frames.append(frame(block.positions[0], cell(block.box[0]), part["pair"][start:end], atoms))
        corridor = np.concatenate(frames)
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            np.savez(temp, pair=part["pair"], origin=part["origin"], corridor=corridor)
        Path(temp.name).replace(path)
    graph = competition(part)
    static = part["explicit_context"]
    part["corridor"] = np.column_stack((corridor, static))
    part["matching"] = np.column_stack((graph, static))
    part["corridor_matching"] = np.column_stack((corridor, graph, static))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    panel = panel10()
    panel_files = ["diverse_training_panel"]
    if args.wide:
        panel_files += ["volume_training_panel"]
    extra = tuple(p for name in panel_files for p in json.loads((OUT/f"{name}.json").read_bytes())["proteins"])
    train_names = TRAIN + ADDITIONAL_TRAINING + EXTRA_TRAINING + extra
    if len(set(train_names)) != len(train_names) or set(train_names) & set(panel + TEST + HISTORICAL):
        raise ValueError("Training and reserved panels overlap")
    train, valid = (load(names, 100, geometry=True) for names in (train_names, panel))
    with ThreadPoolExecutor(3) as pool:
        for part, _ in zip(train+valid, pool.map(lambda p: prepare(p, 100), train+valid), strict=True):
            print("INPUTS", part["protein"], part["replica"], flush=True)
    first = 2121 if args.wide else 2111
    for i, block in enumerate(("explicit_context", "corridor", "matching", "corridor_matching"), first):
        location = f"ap15_preprocessing/T1I{i}"
        if not (OUT/location/"selection.json").exists():
            explore(location, (("xgb-rounds-420", 166, f"hb-origin-stacked-{block}", False, 8, 1.),),
                    i, train, valid)


if __name__ == "__main__":
    main()
