"""Evaluate six-branch statistics with the already trained GA452 readout."""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from .ga333 import WINDOW
from .ga373 import assemble
from .ga_sequential50 import features
from .ga_static_offset import VALID, estimates

RUNS = Path("runs")
OUT = RUNS / "ga507"


def main():
    torch.set_num_threads(2)
    amber = RUNS / "ga360/amber"
    old = RUNS / "ga360/charmm"
    x, d, g, y, origins, proteins = assemble(VALID, (49, 149, 249), amber, old)
    with np.load(RUNS / "ga360/validation.npz") as z:
        np.testing.assert_array_equal(proteins, z["proteins"])
        np.testing.assert_array_equal(origins, z["origin"])
        np.testing.assert_allclose(y, z["target"], rtol=0, atol=3e-6)
        base, parent = z["baseline"], z["parent"]
    state = torch.load(RUNS / "ga445-494/GA452.pt", map_location="cpu", weights_only=False)
    center, scale = state["center"], state["scale"]
    head = nn.Sequential(nn.Linear(len(center), 96), nn.LayerNorm(96), nn.SiLU(),
                         nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
    head.load_state_dict(state["state"])
    head.eval()
    masks = {"all": origins >= 0, "first": origins == 49, "later": origins != 49}
    variants = {"existing": d}
    for count in (2, 4, 6):
        blocks = []
        for name in VALID:
            for origin in (49, 149, 249):
                tag = f"{name}-{origin}.npz"
                with np.load(amber / tag) as a, np.load(OUT / tag) as b:
                    np.testing.assert_array_equal(a["pair"], b["pair"])
                    start = a["pair"][:, 2]
                    u = a["tracks"][:2, :len(WINDOW)] - start
                    v = b["tracks"][:count, :len(WINDOW)] - start
                    um, vm = u.mean(0).T, v.mean(0).T
                    blocks.append(np.c_[um, vm, um-vm, u.std((0,1)), v.std((0,1)),
                                        u.mean(1).std(0), v.mean(1).std(0)])
        variants[str(count)] = np.concatenate(blocks)
    results = {}
    for key, values in variants.items():
        view = features(x, values, g, state["spec"][0])
        with torch.inference_mode():
            p = head(torch.tensor(((view-center)/scale).clip(-12,12), dtype=torch.float32)).numpy().ravel() + x[:,0]
        final = .65 * parent + .35 * p
        if key == "existing":
            with np.load(RUNS / "ga445-494/GA452.npz") as saved:
                np.testing.assert_allclose(final, saved["prediction"], rtol=0, atol=3e-6)
        results[key] = {label: estimates(final[m], y[m], base[m]) for label, m in masks.items()}
    (OUT / "ga452_replacement.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print({k: {m: s["residual_correlation"] for m, s in v.items()} for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    main()

