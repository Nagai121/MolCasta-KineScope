"""MiniTopol-backed CPU candidate and feature preparation for two phenomena."""
import hashlib
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import cell, registry, trajectory
from .features import AA, _mic
from .legacy import THRESHOLDS, TASKS, build_layout, legacy_scores, snapshot

TASK_NAMES = ("salt", "contact")
BASE_WIDTH = 28


def residue_candidates(s, future, future_box, task, baseline, chunk=65536):
    n = len(s.ca)
    cutoff = THRESHOLDS[TASKS.index(task)]
    box = cell(s.box)
    shape = n * n
    minimum = np.full(shape, np.inf, dtype=np.float32)
    future_min = minimum.copy()
    closest = np.full(shape, -1, dtype=np.int32)
    near = np.zeros(shape, dtype=np.int32)
    medium = near.copy()
    count = near.copy()
    eligible = s.physics_ids[task]
    for start in range(0, len(eligible), chunk):
        chunk_ids = eligible[start:start + chunk]
        i, j = s.i[chunk_ids], s.j[chunk_ids]
        left_id, right_id = s.atom_res[i], s.atom_res[j]
        ids = np.minimum(left_id, right_id) * n + np.maximum(left_id, right_id)
        d = np.linalg.norm(_mic(s.pos[j] - s.pos[i], box), axis=1)
        df = np.linalg.norm(_mic(future[j] - future[i], future_box), axis=1)
        # Stable within each residue pair, then compare the local winners across chunks.
        order = np.lexsort((d, ids))
        first = order[np.r_[True, np.diff(ids[order]) != 0]]
        better = first[d[first] < minimum[ids[first]]]
        closest[ids[better]] = chunk_ids[better]
        minimum[ids[better]] = d[better]
        np.minimum.at(future_min, ids, df)
        np.add.at(count, ids, 1)
        np.add.at(near, ids, d <= cutoff + 1.5)
        np.add.at(medium, ids, d <= cutoff + 3.5)
    keys = np.flatnonzero(np.isfinite(minimum) & (minimum > cutoff))
    left, right = np.divmod(keys, n)
    a, b = s.i[closest[keys]], s.j[closest[keys]]
    swap = s.atom_res[a] != left
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    y = future_min[keys] <= cutoff
    delta = _mic(s.pos[b] - s.pos[a], box)
    d = minimum[keys]
    pair = np.column_stack((a, b, d, delta, s.atom_types[a], s.atom_types[b])).astype(np.float32)
    ca = s.pos[s.ca]
    ca_delta = _mic(ca[right] - ca[left], box)
    ca_d = np.linalg.norm(ca_delta, axis=1)
    va = _mic(s.pos[a] - ca[left], box)
    vb = _mic(s.pos[b] - ca[right], box)
    la, lb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    cos_a = np.einsum("ij,ij->i", va, delta) / np.maximum(la * d, 1e-6)
    cos_b = -np.einsum("ij,ij->i", vb, delta) / np.maximum(lb * d, 1e-6)
    dense = s.density8[s.ca]
    burial = s.density12[s.ca]
    formed = np.flatnonzero(minimum <= cutoff)
    formed_degree = np.bincount(np.r_[formed // n, formed % n], minlength=n)
    aa = np.array([AA.index(r) if r in AA else 20 for r in s.residue_names])
    degree = np.bincount(np.r_[left, right], minlength=n)
    weight = np.exp(np.clip((cutoff - d) / 0.9, -20, 0))
    mass = np.bincount(np.r_[left, right], weights=np.tile(weight, 2), minlength=n)
    near_degree = np.bincount(np.r_[left, right], weights=np.tile(d <= cutoff+3, 2), minlength=n)
    best = np.full(n, np.inf)
    np.minimum.at(best, left, d)
    np.minimum.at(best, right, d)
    baseline_ids, baseline_prob = legacy_scores(s, task, baseline)
    baseline_max = np.full(shape, -1, dtype=np.float32)
    np.maximum.at(baseline_max, baseline_ids, baseline_prob)
    base = np.column_stack((
        d, ca_d, np.log1p(right-left), np.log1p(near[keys]), np.log1p(medium[keys]),
        np.log1p(count[keys]), la+lb, np.abs(la-lb), cos_a+cos_b, np.abs(cos_a-cos_b),
        dense[left]+dense[right], np.abs(dense[left]-dense[right]),
        burial[left]+burial[right], np.abs(burial[left]-burial[right]),
        formed_degree[left]+formed_degree[right], np.abs(formed_degree[left]-formed_degree[right]),
        np.minimum(aa[left], aa[right]), np.maximum(aa[left], aa[right]),
        np.minimum(s.atom_types[a], s.atom_types[b]), np.maximum(s.atom_types[a], s.atom_types[b]),
        np.log1p(degree[left]+degree[right]), near_degree[left]+near_degree[right],
        np.log1p(mass[left]+mass[right]), np.abs(mass[left]-mass[right]),
        d-np.minimum(best[left], best[right]), d-np.maximum(best[left], best[right]),
        d/(ca_d+0.01), np.einsum("ij,ij->i", ca_delta, delta)/np.maximum(ca_d*d, 1e-6),
    )).astype(np.float32)
    assert base.shape == (len(pair), BASE_WIDTH) and np.isfinite(base).all()
    return pair, base, y, baseline_max[keys], {
        "expanded_pairs": int(len(s.i)), "chemical_pairs": int(len(eligible)),
        "initial_residue_pairs": int(len(y)), "positives": int(y.sum()),
    }


def feature_view(pair, base, old, view, width, inherit=False):
    if view == "geometry":
        extras = (base[:, 8]*base[:, 0], base[:, 9]*base[:, 27], base[:, 1]-base[:, 0],
                  base[:, 6]/(base[:, 1]+1), base[:, 3]/(base[:, 5]+1),
                  base[:, 4]-base[:, 3], base[:, 27]*base[:, 26], base[:, 10]*base[:, 26])
    elif view == "competition":
        extras = (base[:, 21]*base[:, 26], base[:, 22]*base[:, 26], base[:, 24]-base[:, 25],
                  base[:, 24]*base[:, 25], base[:, 22]/(base[:, 21]+1),
                  base[:, 15]*base[:, 23], base[:, 14]*base[:, 8], base[:, 11]*base[:, 25])
    elif view == "packing":
        extras = (base[:, 3]-base[:, 4], base[:, 4]/(base[:, 5]+1),
                  base[:, 5]/(base[:, 1]+1), base[:, 10]*base[:, 3],
                  base[:, 14]*base[:, 4], base[:, 13]*base[:, 7],
                  base[:, 25]*base[:, 8], base[:, 23]*base[:, 27])
    else:
        raise ValueError(view)
    extended = np.column_stack((pair, *extras)).astype(np.float32, copy=False)
    assert extended.shape == (len(pair), 16)
    parts = [base[:, :width], extended[:, 8:]]
    if inherit:
        parts.append(old[:, None])
    return np.column_stack(parts)


def shard(root, protein, replica, stride, models):
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    manifest = reg.manifest(tid)
    version = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("legacy.py").read_bytes())
    signature = dict(manifest=manifest, stride=stride, version=version.hexdigest(),
                     models={k: v[-1] for k, v in models.items()}, horizon_ps=1000)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(reg.workspace.trajectory_path(tid)) / "molcasta-events"
    directory.mkdir(exist_ok=True)
    path = directory / f"{key}.npz"
    if path.exists():
        return path
    started = time.perf_counter()
    layout = build_layout(reg.atoms(tid).to_pydict())
    timestamps = np.concatenate([b.time for b in reg.blocks(tid)])
    per_task = {name: [] for name in TASK_NAMES}
    for origin in range(49, len(timestamps), stride):
        target = int(np.searchsorted(timestamps, timestamps[origin]+999.99))
        if target >= len(timestamps):
            break
        if not np.isclose(timestamps[target]-timestamps[origin], 1000, rtol=0, atol=0.01):
            raise ValueError(f"Inexact 1000 ps target at {tid} {origin}")
        block = next(reg.blocks(tid, start=origin, stop=target+1, chunk_frames=target-origin+1))
        s = snapshot(layout, block.positions[0], cell(block.box[0]), protein, origin)
        for task in TASK_NAMES:
            per_task[task].append(residue_candidates(s, block.positions[-1], cell(block.box[-1]),
                                                    task, models))
    result = {}
    metadata = {"trajectory": tid, "horizon_ps": 1000, "stride": stride,
                "build_seconds": time.perf_counter() - started}
    for task, frames in per_task.items():
        if not frames:
            raise ValueError(f"No origins in {tid}")
        for index, label in enumerate(("pair", "base", "y", "old")):
            result[f"{task}_{label}"] = np.concatenate([x[index] for x in frames])
        result[f"{task}_ends"] = np.cumsum([len(x[2]) for x in frames])
        metadata[task] = [x[4] for x in frames]
    result["meta"] = json.dumps(metadata)
    tmp = NamedTemporaryFile(dir=directory, suffix=".npz", delete=False)
    try:
        with tmp:
            np.savez(tmp, **result)
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return path
