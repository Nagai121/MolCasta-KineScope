"""Rebuilt for GAm01 (Dosukoi side): observed-history distances.

past() is taken verbatim from the committed GC-series copy
(experiments/GC01_GC23_conditional_history/code/gc_history.py on branch
research/gc01-gc23-codex), which is the same lineage as the GA package.
It returns, for each candidate pair, the pair distances over the 10 frames
before the origin (t-10 .. t-1, 100 ps grid).
"""
import numpy as np

from tseries.data import cell, registry, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import ROOT


def past(part):
    reg = registry(ROOT, part["protein"])
    tid = trajectory(part["protein"], part["replica"])
    chunks = []
    for a, b in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        t = int(part["origin"][a])
        frames = next(reg.blocks(tid, start=t - 10, stop=t + 1, chunk_frames=11))
        if len(frames.time) != 11 or not np.allclose(
                frames.time[-1] - frames.time, np.arange(10, -1, -1) * 100,
                rtol=0, atol=.01):
            raise ValueError("Expected 100 ps history grid")
        pairs = part["pair"][a:b]
        i, j = pairs[:, :2].astype(np.intp).T
        chunks.append(np.column_stack([np.linalg.norm(
            _mic(p[j] - p[i], cell(box)), axis=1)
            for p, box in zip(frames.positions[:-1], frames.box[:-1])]))
    return np.concatenate(chunks)
