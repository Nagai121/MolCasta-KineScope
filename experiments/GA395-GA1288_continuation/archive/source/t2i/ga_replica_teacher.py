"""GA508+ learns persistent pair offsets from other replicas and training origins."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from .ga_sequential50 import features, pool
from .ga_static_offset import VALID, estimates
from .ga_force import build
from tseries.data import shard
from tseries.hb_single_structure import ROOT

RUNS = Path("runs")
OUT = RUNS / "ga508-replica-teacher"


def keys(pair):
    atom = pair[:, :2].astype(np.int64)
    return atom[:, 0] * (1 << 24) + atom[:, 1]


def trajectory_rows(name, replica):
    with np.load(shard(ROOT, name, replica, 100, 8.)) as data:
        keep = data["pair"][:, 2] <= 6.
        pair = data["pair"][keep]
        origin = data["origin"][keep]
    return pair, origin


def teachers(x, y, origins, proteins):
    """Exclude the focal origin and blend across independent replicas of the training protein."""
    prior = np.zeros((len(y), 3), np.float32)
    mass = np.zeros_like(prior)
    for name in np.unique(proteins):
        focus = np.flatnonzero(proteins == name)
        pair, time = trajectory_rows(name, 1)
        key = np.empty(len(focus), np.int64)
        for origin in np.unique(origins[focus]):
            at = origins[focus] == origin
            rows = pair[time == origin]
            np.testing.assert_array_equal(rows[:, 2], x[focus[at], 0])
            key[at] = keys(rows)
        observation = []
        for replica in (2, 3):
            cache = RUNS / "ga_force" / f"{name}-R{replica}-4.npz"
            if not cache.exists():
                continue
            from tseries.data import registry, trajectory
            reg = registry(ROOT, name)
            columns = ["residue_index", "name"]
            if reg.atoms(trajectory(name, 1)).select(columns).to_pydict() != reg.atoms(trajectory(name, replica)).select(columns).to_pydict():
                continue
            other, target = build((name, replica, 4))
            candidate, _ = trajectory_rows(name, replica)
            candidate = candidate[:len(target)]
            np.testing.assert_array_equal(candidate[:, 2], other[:, 0])
            observation.append((keys(candidate), other, target))
        observation.append((key, x[focus], y[focus]))
        for group, (otherkey, other, label) in enumerate(observation):
            order = np.argsort(otherkey, kind="stable")
            sorted_keys = otherkey[order]
            left, right = np.searchsorted(sorted_keys, key), np.searchsorted(sorted_keys, key, side="right")
            for offset in range(int((right-left).max(initial=0))):
                matched = np.flatnonzero(left + offset < right)
                j = order[left[matched] + offset]
                if group == len(observation)-1:
                    eligible = origins[focus[matched]] != origins[focus[j]]
                    matched, j = matched[eligible], j[eligible]
                target_rows = focus[matched]
                distance = other[j, 0] - x[target_rows, 0]
                weight = np.exp(-.5 * (distance[:, None] / np.array([.2, .45, .8]))**2)
                prior[target_rows] += weight * (label[j] - other[j, 0])[:, None]
                mass[target_rows] += weight
    return prior / np.maximum(mass, 1e-8), mass


def main():
    OUT.mkdir(exist_ok=True)
    (x, d, g, y, origins, proteins), (v, e, h, t, vo, vg, base, parent) = pool()
    assert not set(proteins) & set(VALID)
    cache = OUT / "training-teacher.npz"
    if cache.exists():
        with np.load(cache) as z:
            priors, support = z["prior"], z["support"]
            np.testing.assert_array_equal(z["original"], y)
    else:
        priors, support = teachers(x, y, origins, proteins)
        np.savez_compressed(cache, prior=priors, support=support, original=y)
    residual = y-x[:, 0]
    oof = np.empty_like(residual)
    for train, val in GroupKFold(5).split(x, residual, proteins):
        oof[val] = Ridge(alpha=10).fit(x[train, :12], residual[train]).predict(x[val, :12])
    reference = Ridge(alpha=10).fit(x[:, :12], residual)
    shift = reference.predict(v[:, :12])
    np.savez(OUT / "ridge.npz", coef=reference.coef_, intercept=reference.intercept_)
    specs = [(0, 0., "level", 6), (1, .65, "level", 6)]
    specs += [(w, k, view, depth) for w in range(3) for k in (.35, .65, .95)
              for view in ("level", "level_interaction", "full_response") for depth in (4, 7)]
    assert len(set(specs)) == len(specs)
    log = OUT / "trials.jsonl"
    history = [json.loads(l) for l in log.read_bytes().splitlines()] if log.exists() else []
    masks = {"all": vo >= 0, "first": vo == 49, "later": vo != 49}
    for i in range(len(history), len(specs)):
        available = [s for s in specs if s not in {tuple(r["spec"]) for r in history}]
        if i < 2:
            spec = specs[i]
        else:
            best = max(history, key=lambda r: r["metrics"]["all"]["residual_correlation"])
            spec = min(available, key=lambda s: (
                sum(a != b for a, b in zip(s, best["spec"])),
                sum(r["spec"][0] == s[0] for r in history),
            ))
        width, fraction, view, depth = spec
        available_weight = fraction * support[:, width] / (1. + support[:, width])
        teacher = (1 - available_weight) * residual + available_weight * priors[:, width]
        model = XGBRegressor(device="cuda", tree_method="hist", n_estimators=400,
                             learning_rate=.04, max_depth=depth, max_bin=96, n_jobs=4,
                             subsample=.88, reg_lambda=12, random_state=508)
        start = perf_counter()
        model.fit(features(x, d, g, view), teacher-oof,
                  sample_weight=np.where(origins == 49, 1., 4.))
        fit = perf_counter()-start
        assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
        model.set_params(device="cpu")
        direct = model.predict(features(v, e, h, view)) + shift + v[:, 0]
        p = .65*parent+.35*direct
        scores = {k: estimates(p[m], t[m], base[m]) for k, m in masks.items()}
        model_id = f"GA{508+i}"
        model.save_model(OUT / f"{model_id}.ubj")
        np.savez_compressed(OUT / f"{model_id}.npz", prediction=p, readout=direct)
        row = dict(id=model_id, spec=spec, metrics=scores, fit_seconds=fit,
                   teacher_fraction_nonzero=float((available_weight > 0).mean()))
        history.append(row)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
        print(model_id, spec, scores["all"]["residual_correlation"], flush=True)
        if scores["all"]["residual_correlation"] >= .700:
            break
    best = max(history, key=lambda r: r["metrics"]["all"]["residual_correlation"])
    (OUT / "summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

