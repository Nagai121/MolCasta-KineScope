"""Decompose GA305 validation errors by static starting distance."""
from pathlib import Path
import numpy as np

from .explore import record
from .ga309 import predict
from .ga_native_offset import native
from .ga_static_offset import VALID, cohort, estimates, extract

rows = cohort(VALID, 3)
y, x = extract(rows)
s = x["compact"]
h = native(rows)
parent = Path("runs/ga_native_offset")
with np.load(parent/"baselines.npz") as b, np.load(parent/"residual-ridge.npz") as m:
    base = predict(b, "real_baseline_", s)
    pred = base + predict(m, "", h-predict(b, "native_baseline_", s))
out = {}
for lo, hi in zip((0,3.5,4,4.5,5,5.5), (3.5,4,4.5,5,5.5,6.001)):
    keep = (s[:,0]>=lo) & (s[:,0]<hi)
    if keep.sum() > 4:
        out[f"{lo}-{hi}"] = dict(n=int(keep.sum()), **estimates(pred[keep], y[keep], base[keep]))
record(Path("runs/ga309/error_decomposition.json"), out)
print(out)
