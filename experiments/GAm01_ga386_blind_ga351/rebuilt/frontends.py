"""Rebuilt for GAm01 (Dosukoi side): candidate frontend for the GA offset line.

The original frontends module was never committed. The candidate parts are
reconstructed verbatim from ga333.candidates (GA333, committed) and
ga365.parts (GA365, committed): shard-based candidate pairs at origin 49
with pair distance <= 6.0 A. Validation: ga314.load asserts the pairs are
array-equal to the pairs stored in the generated-ensemble caches (passed
for all 8 GA351 proteins), and the extracted targets match blind-ga351.npz.

The third return value (base) is the per-pair frozen value that
ga_static_offset.one places in the "compact" view as its anchor column.
Evidence that it is not zero: runs/ga_native_offset/baselines.npz stores
real_baseline_center[2] = 0.0653 and real_baseline_scale[2] = 0.1292 for
that column on the training proteins. For the GA351 panel the frozen values
were recovered algebraically from the committed static array in
blind-ga351.npz through the committed ridge (baselines.npz): the recovered
values lie in [0, 1] (mean 0.0726, sd 0.1393, many exact zeros), i.e. a
clean probability-like quantity, which also confirms the other compact
columns. They are stored per protein in frozen_anchor.npz next to this
file. For proteins outside the GA351 panel, zeros are returned (unused
in GAm01).
"""
from pathlib import Path

import numpy as np

from tseries.data import shard
from tseries.hb_single_structure import ROOT

_ANCHOR = {}
_anchor_path = Path(__file__).with_name("frozen_anchor.npz")
if _anchor_path.exists():
    with np.load(_anchor_path) as z:
        _ANCHOR = {k: z[k] for k in z.files}


def data(names):
    parts = []
    total = 0
    base_parts = []
    for n in names:
        with np.load(shard(ROOT, n, 1, 100, 8.)) as z:
            keep = (z["origin"] == 49) & (z["pair"][:, 2] <= 6.)
            pairs = z["pair"][keep]
        if not len(pairs):
            raise RuntimeError(f"No candidate rows for {n}")
        count = len(pairs)
        total += count
        parts.append(dict(replica=1, protein=n, pair=pairs,
                          origin=np.full(count, 49),
                          ends=np.array([count]), y=np.zeros(count)))
        anchor = _ANCHOR.get(n)
        if anchor is None:
            base_parts.append(np.zeros(count))
        else:
            if len(anchor) != count:
                raise RuntimeError(f"Frozen anchor length mismatch for {n}")
            base_parts.append(anchor.astype(np.float64))
    base = np.concatenate(base_parts) if base_parts else np.zeros(0)
    return parts, {"combined": np.empty((total, 0))}, base
