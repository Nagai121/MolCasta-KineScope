"""Rebuilt for GAm01 (Dosukoi side): route features placeholder.

The original g2i_path35 module was never committed. In ga_static_offset.one,
paths() feeds only the "local" and "combined" feature views. The GA351 blind
scoring path (ga314.load -> ga348.components/forecasts, ga344.inputs) reads
only the "compact" view, which does not contain route features, so the values
returned here cannot affect the scored quantities. An empty (n, 0) array is
returned so the view assembly keeps the correct row count.
"""
import numpy as np


def paths(positions, box, pair):
    return np.empty((len(pair), 0), dtype=np.float32)
