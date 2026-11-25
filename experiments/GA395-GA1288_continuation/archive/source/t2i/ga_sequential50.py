"""Sequential GA exploration of persistent pair offsets on the GA392 development split."""
from itertools import product
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch import nn

from .ga373 import assemble
from .ga_static_offset import VALID, estimates

RUNS = Path("runs")
OUT = RUNS / "ga395-444"


def read(path):
    return json.loads((RUNS / path).read_bytes())


def pool():
    names = read("ga355/development.json")["successful"]
    sets = [assemble(names, (49,), RUNS / "ga333", RUNS / "ga355")]
    for lane, amber, charmm in [
        ("ga385/lane-0.json", "ga365", "ga385"),
        ("ga392/lane-0.json", "ga392/amber", "ga392/charmm"),
    ]:
        names = read(lane)["proteins"]
        assert not set(names) & set(VALID)
        for origin in (149, 249):
            eligible = [n for n in names if all((RUNS / f"{d}/{n}-{origin}.npz").exists()
                                                for d in (amber, charmm))]
            sets.append(assemble(eligible, (origin,), RUNS / amber, RUNS / charmm))
    x, d, g, y, origins, proteins = [np.concatenate(blocks) for blocks in zip(*sets)]
    assert not set(proteins) & set(VALID)
    v, e, h, t, vo, vg = assemble(VALID, (49, 149, 249), RUNS / "ga360/amber", RUNS / "ga360/charmm")
    with np.load(RUNS / "ga360/validation.npz") as z:
        np.testing.assert_array_equal(vo, z["origin"])
        np.testing.assert_array_equal(vg, z["proteins"])
        np.testing.assert_allclose(t, z["target"], rtol=0, atol=3e-6)
        base, parent = z["baseline"], z["parent"]
    return (x, d, g, y, origins, proteins), (v, e, h, t, vo, vg, base, parent)


def features(x, d, g, mode):
    width = (d.shape[1] - 4) // 3
    a, b = d[:, :width], d[:, width:2 * width]
    level = np.c_[(a + b).mean(1) / 2, a.mean(1), b.mean(1),
                  np.abs(a - b).mean(1), d[:, -4:]]
    if mode == "level":
        return np.c_[x, level]
    if mode == "compact_level":
        return np.c_[x[:, :3], level]
    if mode == "level_graph":
        return np.c_[x, level, g]
    if mode == "level_interaction":
        return np.c_[x, level, level[:, :3] * level[:, 3:4]]
    if mode == "full_response":
        return np.c_[x, d]
    return np.c_[x, d, g]


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(exist_ok=True)
    train, valid = pool()
    x, d, g, y, o, proteins = train
    v, e, h, t, vo, vg, base, parent = valid
    print("assembled", len(y), len(t), flush=True)
    masks = {"all": np.ones(len(t), bool), "first": vo == 49, "later": vo != 49}

    def metrics(p):
        return {k: estimates(p[m], t[m], base[m]) for k, m in masks.items()}

    views = ("level", "compact_level", "level_graph", "level_interaction", "full_response", "full_graph")
    losses = ("huber", "mse", "smooth_mae")
    balances = ("uniform", "later", "protein", "protein_later")
    shapes = ("linear", "mlp", "deep")
    options = list(product(views, losses, balances, shapes))
    weights = {}
    _, inv, counts = np.unique(proteins, return_inverse=True, return_counts=True)
    for balance in balances:
        w = np.ones(len(y))
        if "protein" in balance:
            w *= 1.0 / counts[inv]
        if "later" in balance:
            w *= np.where(o == 49, 1., 4.)
        weights[balance] = torch.tensor(w / w.mean(), dtype=torch.float32, device="cuda")

    target = torch.tensor(y - x[:, 0], dtype=torch.float32, device="cuda")
    cache = {}
    history = []
    champion = None
    for number in range(395, 445):
        start = time.perf_counter()
        available = [spec for spec in options if spec not in {tuple(r["spec"]) for r in history}]
        if not history:
            spec = ("level", "huber", "later", "mlp")
            hypothesis = "Test whether cross-force-field persistent levels carry the useful response without temporal detail."
        elif (number - 395) % 4 == 0:
            tested = {key: sum(r["spec"][0] == key for r in history) for key in views}
            spec = min(available, key=lambda s: (tested[s[0]], sum(a != b for a, b in zip(s[1:], champion["spec"][1:]))))
            hypothesis = f"Test a less sampled representation against {champion['id']} after observing the current best."
        else:
            spec = min(available, key=lambda s: (
                sum(a != b for a, b in zip(s, champion["spec"])),
                sum(r["spec"][j] == s[j] for j in range(4) for r in history)
            ))
            changed = [k for k, a, b in zip(("representation", "loss", "weighting", "architecture"), spec, champion["spec"]) if a != b]
            hypothesis = f"Change {', '.join(changed)} around {champion['id']} after its measured dev score."
        view, loss, balance, shape = spec
        if view not in cache:
            a, b = features(x, d, g, view), features(v, e, h, view)
            c, s = a.mean(0), a.std(0).clip(.001)
            cache[view] = (torch.tensor(((a-c)/s).clip(-12,12), dtype=torch.float32, device="cuda"),
                           torch.tensor(((b-c)/s).clip(-12,12), dtype=torch.float32), c, s)
        a, b, center, scale = cache[view]
        torch.manual_seed(384)
        n = a.shape[1]
        net = {"linear": lambda: nn.Linear(n, 1),
               "mlp": lambda: nn.Sequential(nn.Linear(n, 64), nn.SiLU(), nn.Linear(64, 1)),
               "deep": lambda: nn.Sequential(nn.Linear(n, 96), nn.LayerNorm(96), nn.SiLU(),
                                              nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))}[shape]().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        train_begin = time.perf_counter()
        for _ in range(10):
            for idx in torch.randperm(len(y), device="cuda").split(2048):
                err = net(a[idx]).flatten() - target[idx]
                if loss == "mse":
                    penalty = err.square()
                else:
                    beta = .4 if loss == "huber" else .08
                    penalty = nn.functional.smooth_l1_loss(err, torch.zeros_like(err), beta=beta, reduction="none")
                objective = (weights[balance][idx] * penalty).mean()
                opt.zero_grad(set_to_none=True)
                objective.backward()
                opt.step()
        torch.cuda.synchronize()
        fit = time.perf_counter() - train_begin
        peak = torch.cuda.max_memory_allocated()
        net.cpu().eval()
        infer_start = time.perf_counter()
        with torch.inference_mode():
            direct = net(b).numpy().ravel() + v[:, 0]
        infer = time.perf_counter() - infer_start
        pred = .65 * parent + .35 * direct
        scores = metrics(pred)
        score = scores["all"]["residual_correlation"]
        assert np.isfinite(score)
        model_id = f"GA{number}"
        torch.save(dict(state=net.state_dict(), center=center, scale=scale, spec=spec), OUT / f"{model_id}.pt")
        np.savez_compressed(OUT / f"{model_id}.npz", prediction=pred, readout=direct)
        old = champion["metrics"]["all"]["residual_correlation"] if champion else None
        promoted = old is None or score > old
        interpretation = ("Improved pooled validation over incumbent" if promoted else
                          "Retain incumbent on pooled validation")
        row = dict(id=model_id, spec=spec, hypothesis=hypothesis, metrics=scores,
                   delta_incumbent=None if old is None else score-old, promoted=promoted,
                   interpretation=interpretation, fit_seconds=fit, cpu_seconds=infer,
                   gpu_peak_bytes=peak, elapsed_seconds=time.perf_counter()-start)
        history.append(row)
        if promoted:
            champion = row
        with (OUT / "trials.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(model_id, spec, round(score, 6), f"incumbent={champion['id']}", flush=True)
    summary = dict(best=champion, parent=metrics(parent), rows=len(y),
                   validation_pairs=len(t), train_proteins=sorted(set(proteins)),
                   validation_proteins=sorted(set(vg)), fixed_blend=.35, epochs=10)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
