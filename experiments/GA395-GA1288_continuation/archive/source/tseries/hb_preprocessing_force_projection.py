"""Project static implicit-environment forces onto hydrogen-bond approach axes."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import HISTORICAL, TEST, TRAIN, cell, registry, trajectory
from .features import _mic
from .hb_heavy_dynamics import bonded_terms, neighbor_pairs
from .hb_explicit_hydrogen import source_plans
from .hb_preprocessing_mechanics import prepare as prepare_mechanics
from .hb_preprocessing_screened import charge_map
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT, explore, load
from .hb_validation10 import panel10
from .search_events import EXTRA_TRAINING


def projected(position, box, pair, atoms, charges):
    n = len(position)
    edges, _, stiffness, mass, _ = bonded_terms(position, box, atoms)
    a, b = pair[:, :2].astype(np.intp).T
    axis = _mic(position[b]-position[a], box) / pair[:, 2, None]
    near = neighbor_pairs(position, box, 9.)
    near = near[~np.isin(near[:, 0]*n+near[:, 1], edges[:, 0]*n+edges[:, 1])]
    left, right = near.T
    relative = _mic(position[right]-position[left], box)
    r = np.maximum(np.linalg.norm(relative, axis=1), 1.6)
    u = relative/r[:, None]
    z = np.asarray(atoms["element"])
    radius = np.select([z=="N", z=="O", z=="S"], [1.55, 1.42, 1.75], default=1.68)
    diameter = radius[left]+radius[right]
    polar = np.isin(z, ("N", "O", "S"))
    epsilon = np.where(polar[left] & polar[right], .25, .32)
    contact = np.clip((diameter/r)**6, 0., 10.)
    taper = np.clip((9.-r)/1.5, 0., 1.)
    crowding = np.bincount(np.r_[left, right], weights=np.tile(np.exp(-r*r/21.), 2), minlength=n)
    burial = np.clip(.5*(crowding[left]+crowding[right])/6., 0., 1.)
    shield = np.exp(-r/np.where(burial > .5, 5., 2.5))
    charge_product = charges[left]*charges[right]
    radial_chemistry = -30.*charge_product * shield * (1./r**2 + .3/r)
    radial_contact = 12.*epsilon*(contact-contact**2)/r
    attraction = np.where(polar[left] | polar[right], 0., .10*burial*np.exp(-((r-4.0)/.9)**2))
    ec = _mic(position[edges[:, 1]]-position[edges[:, 0]], box)
    ec /= np.maximum(np.linalg.norm(ec, axis=1, keepdims=True), 1e-8)
    matrices = stiffness[:, None, None] * ec[:, :, None] * ec[:, None, :]
    local = np.broadcast_to(np.eye(3)*4., (n, 3, 3)).copy()
    np.add.at(local, edges[:, 0], matrices)
    np.add.at(local, edges[:, 1], matrices)
    compliance = np.linalg.inv(local)
    values = []
    for radial in (radial_chemistry, radial_contact, attraction, radial_chemistry+radial_contact+attraction):
        f = taper[:, None]*radial[:, None]*u
        force = np.column_stack([np.bincount(left, c, n)-np.bincount(right, c, n) for c in f.T])
        displacement = np.einsum("nij,nj->ni", compliance, force)
        difference = displacement[b]-displacement[a]
        shift = (difference*axis).sum(axis=1)
        acceleration = force[b]/mass[b, None] - force[a]/mass[a, None]
        values.extend((shift, np.linalg.norm(difference-shift[:, None]*axis, axis=1),
                       (acceleration*axis).sum(axis=1),
                       (displacement[a]*axis).sum(axis=1), (displacement[b]*axis).sum(axis=1)))
    return np.column_stack(values).astype(np.float32)


def prepare(part, stride):
    prepare_mechanics(part, stride)
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    folder = ROOT/part["protein"]/"trajectories"/tid/"hb-force-projection"
    folder.mkdir(exist_ok=True)
    stamp = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = folder/f"{stride}-{stamp}.npz"
    values = None
    if path.is_file():
        with np.load(path) as data:
            if np.array_equal(data["pair"], part["pair"]) and np.array_equal(data["origin"], part["origin"]):
                values = data["values"]
    if values is None:
        atoms = reg.atoms(tid).to_pydict()
        plan = source_plans(part["protein"])[f"R{part['replica']}"]
        charges, verified = charge_map(atoms, plan)
        rows = []
        for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
            if start == end:
                continue
            origin = int(part["origin"][start])
            block = next(reg.blocks(tid, start=origin, stop=origin+1))
            pair = part["pair"][start:end]
            physical = projected(block.positions[0], cell(block.box[0]), pair, atoms, charges)
            rows.append(np.column_stack((physical, np.full(len(pair), verified, dtype=np.float32))))
        values = np.concatenate(rows)
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            np.savez(temp, pair=part["pair"], origin=part["origin"], values=values)
        Path(temp.name).replace(path)
    part["force_projection"] = np.column_stack((values, part["explicit_context"]))
    part["force_compliance"] = np.column_stack((values, part["compliance_union"]))


def main():
    training = TRAIN + ADDITIONAL_TRAINING + EXTRA_TRAINING
    training += tuple(json.loads((OUT/"diverse_training_panel.json").read_bytes())["proteins"])
    panel = panel10()
    if set(training) & set(panel+TEST+HISTORICAL):
        raise ValueError("Training overlaps reserved panels")
    train, valid = (load(group, 100, geometry=True) for group in (training, panel))
    with ThreadPoolExecutor(3) as pool:
        for p, _ in zip(train+valid, pool.map(lambda p: prepare(p, 100), train+valid), strict=True):
            print("FORCE INPUTS", p["protein"], p["replica"], flush=True)
    for i, method in ((2271, "force_projection"), (2272, "force_compliance")):
        location = f"ap15_preprocessing/T1I{i}"
        if not (OUT/location/"selection.json").exists():
            explore(location, (("xgb-rounds-420", 166, f"hb-origin-stacked-{method}", False, 8, 1.),),
                    i, train, valid)


if __name__ == "__main__":
    main()
