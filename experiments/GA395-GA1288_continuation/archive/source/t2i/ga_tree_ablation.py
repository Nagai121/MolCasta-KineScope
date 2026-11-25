"""Try tree ensembles and force-field ablations on the fixed GA development split."""
from pathlib import Path
import json
from time import perf_counter

import numpy as np
from xgboost import XGBRegressor

from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT = Path("runs/ga495-506")


def transform(x, d, g, view):
    if view == "static":
        return x
    width = (d.shape[1] - 4) // 3
    if view == "amber":
        return np.c_[x, d[:, :width], d[:, [-4, -2]]]
    if view == "charmm":
        return np.c_[x, d[:, width:2 * width], d[:, [-3, -1]]]
    return features(x, d, g, view)


def main():
    OUT.mkdir(exist_ok=True)
    (x, d, g, y, o, proteins), (v, e, h, t, vo, vg, base, parent) = pool()
    _, inverse, counts = np.unique(proteins, return_inverse=True, return_counts=True)
    weights = np.where(o == 49, 1., 4.) / counts[inverse]
    weights = weights / weights.mean()
    masks = {"all": np.ones(len(t), bool), "first": vo == 49, "later": vo != 49}
    specs = [(view, depth) for depth in (3, 6) for view in
             ("static", "amber", "charmm", "level", "level_interaction", "full_response")]
    runs = []
    for i in range(len(specs)):
        if i < 6:
            view, depth = specs.pop(0)
        else:
            rank = {row["view"]: row["metrics"]["all"]["residual_correlation"] for row in runs[:6]}
            view, depth = specs.pop(max(range(len(specs)), key=lambda j: rank[specs[j][0]]))
        model_id = f"GA{495+i}"
        a, b = transform(x, d, g, view), transform(v, e, h, view)
        model = XGBRegressor(device="cuda", tree_method="hist", n_estimators=350,
                             learning_rate=.03, max_depth=depth, max_bin=96, n_jobs=4,
                             subsample=.85, reg_lambda=10, random_state=495)
        start = perf_counter()
        model.fit(a, y - x[:, 0], sample_weight=weights)
        fit = perf_counter() - start
        assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
        model.set_params(device="cpu")
        start = perf_counter()
        direct = model.predict(b) + v[:, 0]
        infer = perf_counter() - start
        pred = .65 * parent + .35 * direct
        scores = {k: estimates(pred[m], t[m], base[m]) for k, m in masks.items()}
        model.save_model(OUT / f"{model_id}.ubj")
        np.savez_compressed(OUT / f"{model_id}.npz", prediction=pred, readout=direct)
        row = dict(id=model_id, view=view, depth=depth, metrics=scores,
                   fit_seconds=fit, cpu_seconds=infer)
        runs.append(row)
        with (OUT / "trials.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        print(model_id, view, depth, scores["all"]["residual_correlation"], flush=True)
    best = max(runs, key=lambda row: row["metrics"]["all"]["residual_correlation"])
    summary = dict(best=best, train_proteins=sorted(set(proteins)),
                   validation_proteins=sorted(set(vg)), fixed_blend=.35)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
