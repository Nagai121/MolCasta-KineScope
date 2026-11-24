"""GAm01 driver: GA351 blind scoring of GA386 via the rebuilt GA path.

Phase A (sanity gate, mirrors ga384_confirm.main(blind=True) exactly):
  - GA348 parent on the blind panel must reproduce corr 0.6219451585202694 /
    RMSE 0.49967228165511945 (blind-ga351.json), and the static array must
    match blind-ga351.npz.
  - GA384's mixed prediction must reproduce blind-ga351.npz "predicted"
    (corr 0.6221390978666421).
Phase B (only if the gate passes): GA386 scored once, fixed rule
  expanded-1.0-40/0.35, mirroring ga386_confirm.py with the blind panel.
Checkpoints are loaded with weights_only=True (SHA-256 verified beforehand).
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from tseries.data import TEST
from ga.ga344 import inputs
from ga.ga348 import components, forecasts
from ga.ga_native_offset import HOLD
from ga.ga_static_offset import estimates

R = Path("runs")
A = R / "ga351/blind/amber"
B = R / "ga351/blind/charmm"

PARENT_REF = dict(residual_correlation=0.6219451585202694,
                  rmse=0.49967228165511945,
                  residual_sd=0.4994855302878378,
                  bias=-0.01365994463790095)
GA384_REF_CORR = 0.6221390978666421


def head_for(dev_dir, best):
    info = json.loads((dev_dir / "development.json").read_bytes())
    assert info["best"] == best, (info["best"], best)
    return info


def load_head(dev_dir, filename):
    import numpy as _np
    safe = [_np._core.multiarray._reconstruct, _np.ndarray, _np.dtype,
            _np.dtypes.Float32DType, _np.dtypes.Float64DType]
    try:
        from torch.serialization import safe_globals
        ctx = safe_globals(safe)
    except Exception:
        import contextlib
        torch.serialization.add_safe_globals(safe)
        ctx = contextlib.nullcontext()
    with ctx:
        state = torch.load(dev_dir / filename, map_location="cpu", weights_only=True)
    c, s = state["center"], state["scale"]
    head = nn.Sequential(nn.Linear(len(c), 96), nn.LayerNorm(96), nn.SiLU(),
                         nn.Dropout(.12), nn.Linear(96, 48), nn.SiLU(),
                         nn.Linear(48, 2))
    head.load_state_dict(state["state"])
    head.eval()
    return head, c, s


def apply_head(head, c, s, x, d):
    feature = np.c_[x, d]
    with torch.no_grad():
        return head(torch.tensor(((feature - c) / s).clip(-12, 12),
                                 dtype=torch.float32)).numpy()[:, 0] + x[:, 0]


def main():
    plan = json.loads((R / "ga351/reserved.json").read_bytes())
    roster = tuple(p["id"] for p in plan["proteins"])
    reserve = tuple(json.loads((R / "ga321/reserve.json").read_bytes())["proteins"])

    data = components(roster, A, B, ("generated-6",), "all-9")
    parent, _ = forecasts(data, ("generated-6", .25, .2))
    t, base, grp, *_ = data
    x, d, y = inputs(roster, A, B)
    np.testing.assert_allclose(t, y, rtol=0, atol=3e-6)

    with np.load(R / "ga351/blind-ga351.npz") as z:
        ref_pred, ref_t, ref_static, ref_prot = z["predicted"], z["target"], z["static"], z["proteins"]
    np.testing.assert_allclose(t, ref_t, rtol=0, atol=3e-6)
    assert (grp == ref_prot).all()
    base_max_diff = float(np.abs(base - ref_static).max())

    parent_metrics = estimates(parent, t, base)
    print("parent metrics:", parent_metrics, flush=True)
    print("parent ref:    ", PARENT_REF, flush=True)
    print("static(base) max abs diff vs blind npz:", base_max_diff, flush=True)
    gate = (abs(parent_metrics["residual_correlation"] - PARENT_REF["residual_correlation"]) < 1e-6
            and abs(parent_metrics["rmse"] - PARENT_REF["rmse"]) < 1e-6
            and base_max_diff < 1e-6)
    print("PARENT GATE:", "PASS" if gate else "FAIL", flush=True)

    # GA384 cross-check (reproduction of an already published value)
    info384 = head_for(R / "ga384", "joint/False/30/0.2/False")
    assert not set(roster) & (set(TEST + HOLD + reserve) | set(info384["train"]))
    head384, c384, s384 = load_head(R / "ga384", "joint-False-30.pt")
    p384 = apply_head(head384, c384, s384, x, d)
    q384 = 0.8 * parent + 0.2 * p384
    m384 = estimates(q384, t, base)
    q384_max_diff = float(np.abs(q384 - ref_pred).max())
    print("GA384 metrics:", m384, "ref corr:", GA384_REF_CORR, flush=True)
    print("GA384 q max abs diff vs blind npz predicted:", q384_max_diff, flush=True)
    gate384 = abs(m384["residual_correlation"] - GA384_REF_CORR) < 1e-6 and q384_max_diff < 1e-5
    print("GA384 CROSS-CHECK:", "PASS" if gate384 else "FAIL", flush=True)

    result = dict(parent=parent_metrics, parent_ref=PARENT_REF,
                  static_max_abs_diff=base_max_diff,
                  ga384_reproduction=m384, ga384_q_max_abs_diff=q384_max_diff,
                  parent_gate=bool(gate), ga384_gate=bool(gate384))
    if not (gate and gate384):
        Path("gam01_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("STOP: sanity gate failed; GA386 not scored", flush=True)
        return

    # Phase B: GA386, scored once, fixed rule expanded-1.0-40/0.35
    info386 = head_for(R / "ga386", "expanded-1.0-40/0.35")
    assert not set(roster) & (set(TEST + HOLD + reserve) | set(info386["later_proteins"]))
    head386, c386, s386 = load_head(R / "ga386", "expanded-1.0-40.pt")
    p386 = apply_head(head386, c386, s386, x, d)
    q386 = 0.65 * parent + 0.35 * p386
    m386 = estimates(q386, t, base)
    print("GA386 metrics:", m386, flush=True)
    result.update(dict(ga386=m386, pairs=len(t), rule="expanded-1.0-40/0.35"))
    Path("gam01_result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez("gam01_predictions.npz", predicted=q386, target=t, static=base,
             proteins=grp, parent=parent, ga384_reproduction=q384)
    print("GAM01 DONE", flush=True)


if __name__ == "__main__":
    main()
