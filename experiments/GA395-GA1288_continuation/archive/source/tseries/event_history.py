"""Causal residue-pair distance histories for the 1 ns event benchmark."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np

from .data import PANELS, REPLICAS, cell, registry, trajectory
from .events import TASK_NAMES, shard
from .features import _mic
from .legacy import TASKS, THRESHOLDS, build_layout, load_models

LAGS = (1, 2, 3, 5, 8, 10, 13, 17, 20, 26, 32, 40, 49)


def summarize(dist, current, threshold):
    delta = current[:, None] - dist
    offsets = np.asarray(LAGS, dtype=np.float32) / 10
    soft = np.exp(-np.maximum(dist-threshold, 0))
    columns = [
        delta[:, 0], delta[:, 3], delta[:, 5], delta[:, 8], delta[:, -1],
        dist.min(1), dist.mean(1), dist.std(1),
        (dist <= threshold).mean(1), soft.mean(1),
        (dist[:, :6] <= threshold).mean(1), soft[:, :6].mean(1),
        (delta * offsets).sum(1) / (offsets ** 2).sum(),
        (dist[:, :6].mean(1) - dist[:, 6:].mean(1)),
        np.min(np.where(dist <= threshold, offsets, 6), axis=1),
        np.abs(np.diff(dist, axis=1)).mean(1),
    ]
    return np.column_stack(columns).astype(np.float32)


def history_shard(root, protein, replica, stride, models):
    original = shard(root, protein, replica, stride, models)
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = original.with_name(f"history-{original.stem}-{version}.npz")
    if path.is_file():
        return path
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    layout = build_layout(reg.atoms(tid).to_pydict())
    n = len(layout.ca)
    indices = {}
    for task in TASK_NAMES:
        rows = layout.physics_ids[task]
        i, j = layout.i[rows], layout.j[rows]
        left, right = layout.atom_res[i], layout.atom_res[j]
        keys = np.minimum(left, right)*n + np.maximum(left, right)
        indices[task] = (i, j, keys)
    with np.load(original) as z:
        rows = {task: (z[f"{task}_pair"], z[f"{task}_ends"]) for task in TASK_NAMES}
    outputs = {task: [] for task in TASK_NAMES}
    for index, origin in enumerate(range(49, 49+len(rows["contact"][1])*stride, stride)):
        start = origin - LAGS[-1]
        block = next(reg.blocks(tid, start=start, stop=origin+1, chunk_frames=LAGS[-1]+1))
        timestamp = block.time[-1] - block.time[-1-np.asarray(LAGS)]
        if not np.allclose(timestamp, 100*np.asarray(LAGS), atol=.01, rtol=0):
            raise ValueError(f"Nonuniform history timestamps in {tid}")
        for task, (pair, ends) in rows.items():
            sub = pair[0 if index == 0 else ends[index-1]:ends[index]]
            current = sub[:, 2]
            ra, rb = layout.atom_res[sub[:, :2].astype(np.int32)].T
            keys = np.minimum(ra, rb)*n + np.maximum(ra, rb)
            lookup = np.full(n*n, -1, dtype=np.int32)
            lookup[keys] = np.arange(len(keys))
            a, b, atom_keys = indices[task]
            targets = lookup[atom_keys]
            keep = targets >= 0
            # Only currently nearby residue pairs receive movement features; all rows remain scoreable.
            if len(current):
                keep &= current[np.maximum(targets, 0)] <= 15
            a, b, targets = a[keep], b[keep], targets[keep]
            history = np.broadcast_to(current[:, None], (len(current), len(LAGS))).copy()
            near = current <= 15
            for column, lag in enumerate(LAGS):
                p = block.positions[-1-lag]
                d = np.linalg.norm(_mic(p[b]-p[a], cell(block.box[-1-lag])), axis=1)
                best = np.full(len(current), np.inf, dtype=np.float32)
                np.minimum.at(best, targets, d)
                history[near, column] = best[near]
            if not np.isfinite(history).all():
                raise ValueError(f"Incomplete history in {tid} {task}")
            outputs[task].append(summarize(history, current, THRESHOLDS[TASKS.index(task)]))
    tmp = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    try:
        with tmp:
            np.savez(tmp, **{task: np.concatenate(parts) for task, parts in outputs.items()})
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("atlas-workspace"))
    parser.add_argument("--legacy", type=Path, default=Path("runs/legacy"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    models = load_models(args.legacy)

    def prepare(item):
        protein, replica, stride = item
        result = history_shard(args.data, protein, replica, stride, models)
        print("HISTORY", protein, replica, stride, result.name, flush=True)

    items = [(protein, replica, stride) for panel, stride in (("train", 25), ("validation", 100))
             for protein in PANELS[panel] for replica in REPLICAS]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(prepare, items))


if __name__ == "__main__":
    main()
