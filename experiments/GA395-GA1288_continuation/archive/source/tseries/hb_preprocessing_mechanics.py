"""Static-origin mechanical flexibility and solvent access for ROD candidates."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import HISTORICAL, TEST, TRAIN, cell, registry, topology, trajectory
from .features import _mic
from .hb_elastic_paths import elastic_modes
from .hb_heavy_dynamics import neighbor_pairs
from .hb_preprocessing_search import prepare as prepare_static
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .hb_validation10 import panel10
from .search_events import EXTRA_TRAINING


def response(position, box, pair, atoms):
    maps = topology(atoms)[0]
    owner = np.full(len(position), -1, dtype=np.intp)
    for i, residue in enumerate(maps):
        owner[list(residue.values())] = i
    ca = position[[r["CA"] for r in maps]]
    eigenvalues, vectors, _, _, _ = elastic_modes(ca, box)
    a, b = pair[:, :2].astype(np.intp).T
    delta = _mic(position[b]-position[a], box)
    axis = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    diff = vectors[owner[b]]-vectors[owner[a]]
    projection = np.einsum("ncm,nc->nm", diff, axis)
    along = projection**2
    total = np.sum(diff**2, axis=1)
    values = []
    for cutoff in (min(3, len(eigenvalues)), len(eigenvalues)):
        weight = 1. / np.maximum(eigenvalues[:cutoff], .025)
        values += [along[:, :cutoff] @ weight, (total[:, :cutoff]-along[:, :cutoff]) @ weight]
    left, right = neighbor_pairs(position, box, 5.).T
    v = _mic(position[right]-position[left], box)
    distance = np.linalg.norm(v, axis=1)
    v = v / np.maximum(distance[:, None], 1e-8)
    # Directional shape of the surrounding protein. Nearby atoms carry more weight.
    shape = np.exp(-.6*(distance-1.4))[:, None, None]*v[:, :, None]*v[:, None]
    tensor = np.zeros((len(position), 3, 3))
    np.add.at(tensor, left, shape)
    np.add.at(tensor, right, shape)
    inverted = np.linalg.inv(tensor+np.eye(3))
    for index, sign in ((a, 1.), (b, -1.)):
        compliance = np.einsum("ni,nij,nj->n", axis, inverted[index], axis)
        values += [compliance, np.trace(inverted[index], axis1=1, axis2=2)-compliance]
        cohesion = np.zeros((len(position), 3))
        weight = np.exp(-.45*(distance-1.4))
        np.add.at(cohesion, left, v*weight[:, None])
        np.add.at(cohesion, right, -v*weight[:, None])
        values.append(sign*np.einsum("ni,ni->n", axis, cohesion[index]))
    return np.column_stack(values).astype(np.float32)


def prepare(part, stride):
    prepare_static(part, stride)
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    folder = ROOT / part["protein"] / "trajectories" / tid / "hb-directional-compliance"
    folder.mkdir(exist_ok=True)
    key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = folder / f"{stride}-{key}.npz"
    mechanical = None
    if path.is_file():
        with np.load(path) as data:
            if np.array_equal(data["pair"], part["pair"]) and np.array_equal(data["origin"], part["origin"]):
                mechanical = data["mechanical"]
    if mechanical is None:
        atoms = reg.atoms(tid).to_pydict()
        out = []
        for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
            if start == end:
                continue
            origin = int(part["origin"][start])
            block = next(reg.blocks(tid, start=origin, stop=origin+1))
            out.append(response(block.positions[0], cell(block.box[0]), part["pair"][start:end], atoms))
        mechanical = np.concatenate(out)
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            np.savez(temp, pair=part["pair"], origin=part["origin"], mechanical=mechanical)
        Path(temp.name).replace(path)
    angular = part["explicit"][:, [1, 2]]
    coupling = mechanical[:, [0, 1, 4, 6]] * np.maximum(0., np.tile(angular, (1, 2)))
    part["compliance"] = np.column_stack((mechanical, coupling, part["explicit_context"]))
    part["compliance_union"] = np.column_stack((mechanical, coupling, part["corridor_matching"]))


def main():
    extra = tuple(json.loads((OUT/"diverse_training_panel.json").read_bytes())["proteins"])
    training = TRAIN + ADDITIONAL_TRAINING + EXTRA_TRAINING + extra
    panel = panel10()
    if set(training) & set(panel + TEST + HISTORICAL):
        raise ValueError("Training and reserved panels overlap")
    train, valid = (load(group, 100, geometry=True) for group in (training, panel))
    with ThreadPoolExecutor(3) as pool:
        for part, _ in zip(train+valid, pool.map(lambda p: prepare(p, 100), train+valid), strict=True):
            print("MECHANICAL INPUTS", part["protein"], part["replica"], flush=True)
    for trial, block in ((2115, "compliance"), (2116, "compliance_union")):
        stage = f"ap15_preprocessing/T1I{trial}"
        if not (OUT/stage/"selection.json").exists():
            explore(stage, (("xgb-rounds-420", 166, f"hb-origin-stacked-{block}", False, 8, 1.),),
                    trial, train, valid)


if __name__ == "__main__":
    main()
