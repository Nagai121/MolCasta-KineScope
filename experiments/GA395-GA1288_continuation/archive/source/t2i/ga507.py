"""GA507 measures whether more CHARMM branches stabilize the generated pair level."""
from functools import partial
import json
from pathlib import Path

import numpy as np
import openmm

from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID, estimates

RUNS = Path("runs")
OUT = RUNS / "ga507"


def main():
    OUT.mkdir(exist_ok=True)
    setup = partial(build, force_files=("charmm36_2024.xml", "implicit/gbn2.xml"))
    for name in VALID:
        kwargs = dict(destination=OUT, samples=WINDOW, branches=6, setup_fn=setup,
                      parts_fn=partial(parts, origins=(49, 149, 249)))
        try:
            simulate((name,), 3, step_fs=2, minimize_iterations=60, **kwargs)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            (OUT / f"{name}-recovery.json").write_text(json.dumps({"step_fs": 1, "minimize": 100}), encoding="utf-8")
            simulate((name,), 3, step_fs=1, minimize_iterations=100, **kwargs)
    deltas = []
    rows = []
    for name in VALID:
        for origin in (49, 149, 249):
            cache = f"{name}-{origin}.npz"
            with np.load(OUT / cache) as new, np.load(RUNS / "ga360/charmm" / cache) as old:
                np.testing.assert_array_equal(new["pair"], old["pair"])
                np.testing.assert_array_equal(new["tracks"].shape[:2], (6, len(WINDOW)))
                d2 = new["tracks"][:2].mean((0, 1))
                d6 = new["tracks"].mean((0, 1))
                deltas.append(d6 - d2)
                rows.extend([(name, origin)] * len(d2))
    with np.load(RUNS / "ga360/validation.npz") as data:
        y, base, p, o = [data[key] for key in ("target", "baseline", "parent", "origin")]
        np.testing.assert_array_equal(data["proteins"], [n for n, _ in rows])
        np.testing.assert_array_equal(o, [i for _, i in rows])
    d = np.concatenate(deltas)
    results = {}
    for w in (0., .15, .3, .5, .75):
        q = p + w * d
        results[str(w)] = {label: estimates(q[mask], y[mask], base[mask])
                           for label, mask in (("all", o >= 0), ("first", o == 49), ("later", o != 49))}
    (OUT / "development.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results, flush=True)


if __name__ == "__main__":
    main()
