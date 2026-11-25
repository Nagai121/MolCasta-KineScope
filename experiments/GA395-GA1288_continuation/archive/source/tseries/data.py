"""MiniTopol acquisition, CPU features and reusable replica shards."""
import hashlib
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .features import prepare, structural, feature_graph, enriched_features
from .rod import expand

TRAIN = ("1qw2_A", "2cg7_A", "1bgf_A", "3hvv_A")
VALIDATION = ("1yu5_X", "3mx7_A", "4ftf_A", "1knt_A")
TEST = ("2oxl_B", "4h4n_A", "1vbw_A", "1j2l_A")
HISTORICAL = ("1mix_A", "2qt4_A")
PANELS = dict(train=TRAIN, validation=VALIDATION, test=TEST, historical=HISTORICAL)
REPLICAS = (1, 2, 3)


def registry(root, protein):
    from minitopol import Registry, Workspace
    return Registry(Workspace(Path(root) / protein))


def trajectory(protein, replica):
    return f"atlas-{protein}-analysis-R{replica}"


def acquire(root, protein):
    from minitopol.atlas import AtlasClient, import_atlas
    reg = registry(root, protein)
    if all((Path(reg.workspace.trajectory_path(trajectory(protein, r))) / "A.zarr").is_dir()
           for r in REPLICAS):
        return
    client = AtlasClient(timeout=180)
    client.download(reg.workspace, protein, "analysis")
    result = import_atlas(reg.workspace, protein, "analysis")
    if len(result["replicas"]) != 3 or any(r["A"] != "complete" for r in result["replicas"]):
        raise RuntimeError(result)


def topology(atoms):
    groups = {}
    for atom, residue in enumerate(atoms["residue_index"]):
        groups.setdefault(residue, []).append(atom)
    names = atoms["name"]
    keep = [r for r, aa in sorted(groups.items())
            if {"N", "CA", "C", "O"} <= {names[i] for i in aa}]
    if len(keep) < 2:
        raise ValueError("At least two complete protein residues are required")
    maps = [{names[i]: i for i in groups[r]} for r in keep]
    residue_names = [atoms["residue_name"][groups[r][0]] for r in keep]
    codes = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "SE": 34}
    atom_types = np.array([codes.get(str(e).upper(), 0) for e in atoms["element"]])
    return maps, residue_names, prepare(maps, residue_names, groups, keep, len(names)), atom_types


def cell(raw):
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.shape == (3, 3) and np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-6):
        matrix = np.diag(matrix)
    if matrix.shape == (3,) and np.all(matrix > 0):
        return matrix
    if matrix.shape == (3, 3) and np.isfinite(matrix).all() and np.linalg.det(matrix) > 0:
        return matrix
    raise ValueError(f"Invalid or unsupported ATLAS box {matrix}")


def frame(positions, box, topo, screening_distance, *, future=None, future_box=None):
    maps, residue_names, context, types = topo
    pair, ii, jj, y, counts = expand(positions, box, context, types, screening_distance,
                                    future=future, future_box=future_box)
    base = structural(positions, box, maps, residue_names, context, ii, jj)
    graph = feature_graph(base, pair[:, 0], pair[:, 1], np.zeros(len(pair)), len(positions))
    features = enriched_features(np.column_stack((base, graph)))
    if not np.isfinite(features).all():
        raise ValueError("Non-finite structural features")
    return pair, features, y, counts


def inputs(pair, features, width, representation):
    d = pair[:, 2]
    x = features
    if representation == "competition":
        extra = (x[:, 118] - x[:, 119], x[:, 122] - x[:, 123],
                 np.minimum(x[:, 134], x[:, 135]), np.maximum(x[:, 134], x[:, 135]),
                 x[:, 136] - x[:, 137], x[:, 139] / (x[:, 138] + 0.01),
                 x[:, 141] / (np.exp((3.5 - d) / 0.95) + 0.01),
                 x[:, 140] - x[:, 139])
    elif representation == "environment":
        extra = (x[:, 6] * x[:, 7], x[:, 6] - x[:, 7],
                 d / (x[:, 1] + 0.1), d - x[:, 39],
                 np.log1p(x[:, 12] + x[:, 13]), x[:, 12] - x[:, 13],
                 np.log1p(x[:, 56] + x[:, 57]), x[:, 60] - x[:, 61])
    elif representation == "interaction":
        extra = (x[:, 146] * x[:, 7], x[:, 147] * x[:, 6],
                 x[:, 148] - x[:, 149], x[:, 150] / (1 + x[:, 148] + x[:, 149]),
                 x[:, 152] * x[:, 138], x[:, 153] * x[:, 138],
                 x[:, 162] * x[:, 140], x[:, 165] / (1 + d))
    else:
        raise ValueError(f"Unknown candidate representation {representation}")
    extended = np.column_stack((pair, *extra)).astype(np.float32, copy=False)
    assert extended.shape == (len(pair), 16) and extended.dtype == np.float32
    # Atom identifiers and orientation in the coordinate system remain metadata.
    return np.column_stack((features[:, :width], extended[:, 8:]))


def shard(root, protein, replica, stride, screening_distance):
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    manifest = reg.manifest(tid)
    version = hashlib.sha256()
    for name in ("data.py", "features.py", "rod.py"):
        version.update(Path(__file__).with_name(name).read_bytes())
    signature = dict(trajectory=manifest, stride=stride, screen=screening_distance,
                     equilibrate=49, horizon_ps=1000, version=version.hexdigest())
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(reg.workspace.trajectory_path(tid)) / "molcasta-tseries"
    directory.mkdir(exist_ok=True)
    path = directory / f"{key}.npz"
    if path.is_file():
        return path
    topo = topology(reg.atoms(tid).to_pydict())
    n = manifest["frames"]
    timestamps = np.concatenate([b.time for b in reg.blocks(tid)])
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Non-increasing timestamps in {tid}")
    origins = np.arange(49, n, stride)
    targets = np.searchsorted(timestamps, timestamps[origins] + 1000 - 0.01)
    good = targets < n
    origins, targets = origins[good], targets[good]
    if not len(origins) or not np.allclose(timestamps[targets] - timestamps[origins], 1000,
                                          rtol=0, atol=0.01):
        raise ValueError(f"No exact 1 ns target grid in {tid}")
    pairs, xs, ys, times = [], [], [], []
    total = {}
    start = time.perf_counter()
    cursor = 0
    # Read adjacent origin/target pairs together. Batch nearby origins to reduce I/O.
    while cursor < len(origins):
        end = min(cursor + 8, len(origins)) if stride <= 20 else cursor + 1
        offset = int(origins[cursor])
        stop = int(targets[end - 1]) + 1
        block = next(reg.blocks(tid, start=offset, stop=stop, chunk_frames=stop - offset))
        for t, target in zip(origins[cursor:end], targets[cursor:end]):
            i, j = t - offset, target - offset
            pair, x, y, counts = frame(block.positions[i], cell(block.box[i]), topo,
                                       screening_distance, future=block.positions[j],
                                       future_box=cell(block.box[j]))
            pairs.append(pair)
            xs.append(x)
            ys.append(y)
            times.append(np.full(len(y), t, dtype=np.int32))
            for field, value in counts.items():
                total[field] = total.get(field, 0) + value
        cursor = end
    total.update(origins=len(origins), build_seconds=time.perf_counter() - start,
                 trajectory=tid, horizon_ps=1000, stride=stride, screen=screening_distance)
    tmp = NamedTemporaryFile(dir=directory, suffix=".npz", delete=False)
    try:
        with tmp:
            np.savez(tmp, pair=np.concatenate(pairs), features=np.concatenate(xs),
                     y=np.concatenate(ys), origin=np.concatenate(times), meta=json.dumps(total))
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return path
