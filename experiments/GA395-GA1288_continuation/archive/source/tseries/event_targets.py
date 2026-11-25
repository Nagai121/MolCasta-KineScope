"""Auxiliary continuous 1 ns distance targets for training trajectories only."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np

from .data import cell, registry, trajectory
from .event_kinetics import distances
from .events import TASK_NAMES, shard
from .legacy import build_layout
from .search_events import EXTRA_TRAINING, TRAIN


def training_distances(root, protein, replica, stride, models):
    if protein not in TRAIN + EXTRA_TRAINING:
        raise ValueError(f"Continuous labels are restricted to training proteins: {protein}")
    original = shard(root, protein, replica, stride, models)
    version = hashlib.sha256(b"".join(Path(__file__).with_name(f).read_bytes()
                                    for f in ("event_kinetics.py", "event_targets.py"))).hexdigest()[:12]
    path = original.with_name(f"targets-{original.stem}-{version}.npz")
    if path.exists():
        return path
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    layout = build_layout(reg.atoms(tid).to_pydict())
    n = len(layout.ca)
    with np.load(original) as source:
        parts = {task: tuple(source[f"{task}_{key}"] for key in ("pair", "ends", "y"))
                 for task in TASK_NAMES}
    outputs = {task: [] for task in TASK_NAMES}
    for index in range(len(parts["contact"][1])):
        origin = 49 + stride*index
        block = next(reg.blocks(tid, start=origin, stop=origin+11, chunk_frames=11))
        if not np.isclose(block.time[10] - block.time[0], 1000, atol=.01, rtol=0):
            raise ValueError(f"Invalid auxiliary target time at {tid} {origin}")
        future = distances(layout, block.positions[10], cell(block.box[10]))
        for task, (pair, ends, labels) in parts.items():
            rows = slice(0 if index == 0 else ends[index-1], ends[index])
            a, b = layout.atom_res[pair[rows, :2].astype(np.int64)].T
            values = future[task][np.minimum(a,b)*n+np.maximum(a,b)]
            threshold = 4.0 if task == "salt" else 4.5
            if not np.array_equal(values <= threshold, labels[rows]):
                raise ValueError(f"Auxiliary targets disagree with event labels in {tid} {task}")
            outputs[task].append(values)
    tmp = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    try:
        with tmp:
            np.savez(tmp, **{task: np.concatenate(out) for task,out in outputs.items()})
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return path
