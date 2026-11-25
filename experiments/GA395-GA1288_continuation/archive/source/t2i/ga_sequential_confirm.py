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
    directories = [RUNS / "ga395-444", RUNS / "ga445-494", RUNS / "ga495-506"]
    candidates = [(path, json.loads((path / "summary.json").read_bytes())) for path in directories]
    path, selected = max(candidates, key=lambda item: item[1]["best"]["metrics"]["all"]["residual_correlation"])
    best = selected["best"]
    lock = dict(id=best["id"], validation=best["metrics"], panel="GA351", blend=selected["fixed_blend"])
    (path / "ga351-confirmation-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    plan = json.loads((RUNS / "ga351/reserved.json").read_bytes())
    roster = tuple(p["id"] for p in plan["proteins"])
    assert not set(roster) & (set(selected["train_proteins"]) | set(selected["validation_proteins"]))
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
    if "view" in best:
        from xgboost import XGBRegressor
        from .ga_tree_ablation import transform
        model = XGBRegressor(device="cpu")
        model.load_model(path / f"{best['id']}.ubj")
        direct = model.predict(transform(x, d, g, best["view"])) + x[:, 0]
    else:
        state = torch.load(path / f"{best['id']}.pt", map_location="cpu", weights_only=False)
        spec = state["spec"]
        a = features(x, d, g, spec[0])
        n = a.shape[1]
        heads = {
            "linear": lambda: nn.Linear(n, 1),
            "mlp": lambda: nn.Sequential(nn.Linear(n, 64), nn.SiLU(), nn.Linear(64, 1)),
            "deep": lambda: nn.Sequential(nn.Linear(n, 96), nn.LayerNorm(96), nn.SiLU(),
                                           nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1)),
        }
        net = heads[spec[3]]().eval()
        net.load_state_dict(state["state"])
        with torch.inference_mode():
            z = torch.tensor(((a - state["center"]) / state["scale"]).clip(-12, 12), dtype=torch.float32)
            direct = net(z).numpy().ravel() + x[:, 0]
    p = (1 - lock["blend"]) * parent + lock["blend"] * direct
    result = dict(id=best["id"], pairs=len(p), proteins=list(roster),
                  metrics=estimates(p, target, base), parent=estimates(parent, target, base))
    (path / "ga351-confirmation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(path / "ga351-confirmation.npz", prediction=p, parent=parent,
                        target=target, baseline=base, proteins=groups)
    print(result, flush=True)


if __name__ == "__main__":
    main()

