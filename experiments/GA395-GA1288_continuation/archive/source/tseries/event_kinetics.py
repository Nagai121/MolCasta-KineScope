"""Causal empirical contact occupancy and transition counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

from .data import PANELS, REPLICAS, cell, registry, trajectory
from .events import TASK_NAMES, shard
from .features import _mic
from .legacy import build_layout, load_models

INTERVAL = 5
OFFSET = 4
MAX_DISTANCE = 7.0
LABEL_STEP = 10 // INTERVAL


def nearby(position, box):
    if box.ndim == 1:
        wrapped = np.remainder(position.astype(np.float64), box)
        wrapped = np.minimum(wrapped, np.nextafter(box.astype(np.float64), 0))
        pairs = cKDTree(wrapped, boxsize=box).query_pairs(MAX_DISTANCE, output_type="ndarray")
    else:
        fractional = np.remainder(position.astype(np.float64) @ np.linalg.inv(box.astype(np.float64)), 1.0)
        fractional = np.minimum(fractional, np.nextafter(1.0, 0.0))
        minimum_scale = np.linalg.svd(box, compute_uv=False)[-1]
        pairs = cKDTree(fractional, boxsize=1.0).query_pairs(
            MAX_DISTANCE/minimum_scale, output_type="ndarray")
    a, b = pairs.T
    d = np.linalg.norm(_mic(position[b]-position[a], box), axis=1)
    keep = d <= MAX_DISTANCE
    return a[keep], b[keep], d[keep]


def distances(layout, position, box):
    n = len(layout.ca)
    a, b, d = nearby(position, box)
    l, r = layout.atom_res[a], layout.atom_res[b]
    key = np.minimum(l, r)*n + np.maximum(l, r)
    cross = np.abs(l-r) >= 2
    charged = ((layout.poscharge[a] & layout.negcharge[b])
               | (layout.negcharge[a] & layout.poscharge[b]))
    output = {}
    for task, keep in (("contact", cross), ("salt", cross & charged)):
        values = np.full(n*n, MAX_DISTANCE + 1, dtype=np.float32)
        np.minimum.at(values, key[keep], d[keep])
        output[task] = values
    return output


def aggregate(past, present, threshold):
    contact = past <= threshold
    condition = (np.abs(past[:-LABEL_STEP] - present) <= 0.5) & ~contact[:-LABEL_STEP]
    similar = (np.abs(past[:-LABEL_STEP] - present) <= 1.0) & ~contact[:-LABEL_STEP]
    formed = contact[LABEL_STEP:]
    unformed = ~contact[:-LABEL_STEP]
    counts = condition.sum(0)
    successes = (condition & formed).sum(0)
    counts_wide = similar.sum(0)
    successes_wide = (similar & formed).sum(0)
    return np.column_stack((
        contact.mean(0), contact[-10:].mean(0),
        np.exp(-np.maximum(past-threshold, 0)).mean(0),
        (unformed & formed).sum(0)/(1+unformed.sum(0)),
        np.log1p(counts), np.log1p(successes),
        np.log1p(counts_wide), np.log1p(successes_wide),
        (successes + .5)/(counts+10),
        (successes_wide + .5)/(counts_wide+10),
        np.log1p(len(past)) * np.ones(len(present)),
        np.min(past, axis=0),
    )).astype(np.float32)


def kinetics_shard(root, protein, replica, stride, models):
    original = shard(root, protein, replica, stride, models)
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = original.with_name(f"kinetics-{original.stem}-{version}.npz")
    if path.exists():
        return path
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    layout = build_layout(reg.atoms(tid).to_pydict())
    n = len(layout.ca)
    with np.load(original) as source:
        rows = {t:(source[f"{t}_pair"],source[f"{t}_ends"]) for t in TASK_NAMES}
    last = 49 + stride*(len(rows["contact"][1])-1)
    recorded = {task: [] for task in TASK_NAMES}
    timestamps = []
    for block in reg.blocks(tid, stop=last, chunk_frames=64):
        for index in range((OFFSET-block.start) % INTERVAL, len(block.time), INTERVAL):
            values = distances(layout, block.positions[index], cell(block.box[index]))
            for task in TASK_NAMES:
                recorded[task].append(values[task])
            timestamps.append(block.time[index])
    time = np.asarray(timestamps)
    if not np.allclose(np.diff(time), INTERVAL*100, rtol=0, atol=.01):
        raise ValueError(f"Inconsistent frames in {tid}")
    result = {}
    for task, (pairs, ends) in rows.items():
        frame_dist = np.stack(recorded[task])
        out = []
        for index, end in enumerate(ends):
            origin = 49 + stride*index
            count = (origin-1-OFFSET)//INTERVAL+1
            portion = pairs[0 if index == 0 else ends[index-1]:end]
            a, b = layout.atom_res[portion[:, :2].astype(np.int32)].T
            key = np.minimum(a,b)*n + np.maximum(a,b)
            out.append(aggregate(frame_dist[:count, key], portion[:, 2],
                                 4.0 if task == "salt" else 4.5))
        result[task] = np.concatenate(out)
        assert np.isfinite(result[task]).all()
    tmp = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    try:
        with tmp:
            np.savez(tmp, **result)
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("atlas-workspace"))
    parser.add_argument("--legacy", type=Path, default=Path("runs/legacy"))
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    models = load_models(args.legacy)

    def run(item):
        protein, replica, stride = item
        path = kinetics_shard(args.data, protein, replica, stride, models)
        print("KINETICS", protein, replica, stride, path.name, flush=True)

    items = [(protein, replica, stride) for panel, stride in (("train", 25), ("validation", 100))
             for protein in PANELS[panel] for replica in REPLICAS]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(run, items))


if __name__ == "__main__":
    main()
