"""Confirm one development-selected checkpoint on the separate GA351 panel."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga348 import components, forecasts
from .ga373 import assemble
from .ga_sequential50 import features
from .ga_static_offset import estimates

RUNS = Path("runs")


def main():
    path = RUNS / "ga688-multiwindow"
    best = json.loads((path / "summary.json").read_bytes())
    roster = tuple(row["id"] for row in json.loads((RUNS / "ga351/reserved.json").read_bytes())["proteins"])
    training = json.loads((RUNS / "ga445-494/summary.json").read_bytes())
    assert not set(roster) & (set(training["train_proteins"]) | set(training["validation_proteins"]))
    lock = dict(id=best["id"], validation=best["metrics"], panel="GA351", blend=.35)
    (path / "ga351-confirmation-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    plan = json.loads((RUNS / "ga351/reserved.json").read_bytes())
    roster = tuple(p["id"] for p in plan["proteins"])

    amber, charmm = (RUNS / f"ga351/blind/{name}" for name in ("amber", "charmm"))
    data = components(roster, amber, charmm, ("generated-6",), "all-9")
    parent, _ = forecasts(data, ("generated-6", .25, .2))
    target, base, groups, *_ = data
    x, d, g, y, _, proteins = assemble(roster, (49,), amber, charmm)
    np.testing.assert_array_equal(groups, proteins)
    np.testing.assert_allclose(target, y, rtol=0, atol=3e-6)
    previous = np.load(RUNS / "ga384/blind-ga351.npz")
    np.testing.assert_array_equal(groups, previous["proteins"])
    np.testing.assert_allclose(target, previous["target"], rtol=0, atol=3e-6)
    np.testing.assert_allclose(base, previous["static"], rtol=0, atol=3e-6)
    state = torch.load(path / f"{best['id']}.pt", map_location="cpu", weights_only=False)
    view, _, width = state["spec"]
    a = features(x, d, g, view)
    net = nn.Sequential(nn.Linear(a.shape[1],width),nn.LayerNorm(width),nn.SiLU(),
                        nn.Linear(width,width//2),nn.LayerNorm(width//2),nn.SiLU(),
                        nn.Linear(width//2,7)).eval()
    net.load_state_dict(state["state"])
    with torch.inference_mode():
        z = torch.tensor(((a - state["center"]) / state["scale"]).clip(-10,10), dtype=torch.float32)
        direct = net(z).numpy()[:,0]*state["output_scale"][0]+state["output_center"][0]+x[:,0]
    p = (1 - lock["blend"]) * parent + lock["blend"] * direct
    result = dict(id=best["id"], pairs=len(p), proteins=list(roster),
                  metrics=estimates(p, target, base), parent=estimates(parent, target, base))
    (path / "ga351-confirmation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(path / "ga351-confirmation.npz", prediction=p, parent=parent,
                        target=target, baseline=base, proteins=groups)
    print(result, flush=True)


if __name__ == "__main__":
    main()


