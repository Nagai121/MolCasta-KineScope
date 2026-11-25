"""GA306 development-only sweep of temporal and static residual estimators."""
import itertools
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .explore import record
from .ga_native240 import SAMPLES
from .ga_native_offset import CACHE
from .ga_static_offset import TRAIN_GROUPS, VALID, cohort, extract, estimates, ridge


def cache_path(row):
    name = row["protein"]
    if row.get("replica", 1) != 1:
        name += f"-R{row['replica']}"
    return CACHE / f"{name}-{row['origin']}.npz"


def tracks(rows):
    out = []
    for row in rows:
        with np.load(cache_path(row)) as z:
            np.testing.assert_array_equal(z["pair"], row["pair"])
            out.append(z["tracks"].transpose(2, 0, 1) - row["pair"][:, 2, None, None])
    return np.concatenate(out)


def packets(h, d):
    for begin, end in ((0, 10), (0, 25), (0, 50), (0, 100), (20, 100), (40, 100)):
        take = (np.array(SAMPLES) > begin) & (np.array(SAMPLES) <= end)
        v = h[:, :, take]
        mean = v.mean((1, 2))
        branch = v.mean(2)
        yield f"{begin}-{end}-mean", mean[:, None]
        yield f"{begin}-{end}-distribution", np.c_[
            branch, v.std((1, 2)), np.abs(branch[:, 0]-branch[:, 1]),
            v.min((1, 2)), v.max((1, 2)), v[:, :, -1].mean(1),
            (v+d[:, None, None] <= 3.5).mean((1, 2))]
    yield "all-samples", np.c_[h.mean(1), np.abs(h[:, 0]-h[:, 1])]


def main():
    train = [r for group in TRAIN_GROUPS for r in cohort(group, 1)]
    valid = cohort(VALID, 3)
    yt, xt = extract(train)
    yv, xv = extract(valid)
    st, sv = xt["compact"], xv["compact"]
    groups = np.concatenate([np.repeat(r["protein"], len(r["y"])) for r in train])
    baseline = ridge(st, yt)
    nv = baseline.predict(sv)
    oof = np.empty_like(yt)
    for p in np.unique(groups):
        mask = groups == p
        oof[mask] = ridge(st[~mask], yt[~mask]).predict(st[mask])
    ledger = {}
    for (name, t), (_, v) in zip(packets(tracks(train), st[:, 0]),
                                  packets(tracks(valid), sv[:, 0])):
        proj = ridge(st, t)
        t, v = t-proj.predict(st).reshape(t.shape), v-proj.predict(sv).reshape(v.shape)
        for representation, alpha, weight in itertools.product(
            ("dynamic", "distance", "chemistry", "structural"), (10., 100., 1000.), (False, True)
        ):
            if representation == "distance":
                a = np.c_[t, t * (st[:, :1]-4.5)]
                b = np.c_[v, v * (sv[:, :1]-4.5)]
            elif representation == "chemistry":
                a = np.c_[t, t * st[:, 3:4], t * st[:, 4:5]]
                b = np.c_[v, v * sv[:, 3:4], v * sv[:, 4:5]]
            elif representation == "structural":
                a = np.c_[t, xt["structural"]]
                b = np.c_[v, xv["structural"]]
            else:
                a, b = t, v
            scale = StandardScaler()
            model = Ridge(alpha=alpha)
            cost = np.exp(-0.25*(st[:, 0]-4.5)**2) if weight else None
            model.fit(scale.fit_transform(a), yt-oof, sample_weight=cost)
            key = f"{name}/{representation}/{alpha}/{weight}"
            ledger[key] = estimates(nv+model.predict(scale.transform(b)), yv, nv)
    ranked = sorted(ledger, key=lambda k: ledger[k]["residual_correlation"], reverse=True)
    record(Path("runs/ga306/development.json"), dict(
        trials=ledger, selected=ranked[:10], validation=list(VALID), pairs=len(yv)))
    for key in ranked[:10]:
        print(key, ledger[key], flush=True)


if __name__ == "__main__":
    main()
