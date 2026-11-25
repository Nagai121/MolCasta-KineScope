"""GA308: dynamic relation features along the static ROD candidate graph."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga306 import tracks
from .ga_static_offset import TRAIN_GROUPS, VALID, cohort, estimates, extract, ridge


def relations(rows, trajectory):
    packs = []
    offset = 0
    for row in rows:
        pair = row["pair"]
        count = len(pair)
        d = pair[:, 2]
        h = trajectory[offset:offset+count]
        offset += count
        axes = pair[:, 3:6] / d[:, None]
        x = np.c_[h.mean((1, 2)), h[:, :, -5:].mean((1, 2)),
                  h.std((1, 2)), (h+d[:, None, None] <= 3.5).mean((1, 2))]
        w = np.exp(-0.5 * ((d-3.4)/1.0)**2)
        endpoints = pair[:, :2].astype(int)
        # Pool neighbors across either orientation, then remove the target edge.
        all_atoms = endpoints.T.ravel()
        dirs = np.concatenate((axes, -axes))
        strengths = np.tile(w, 2)
        z = np.tile(x, (2, 1))
        n = all_atoms.max()+1
        mass = np.bincount(all_atoms, weights=strengths, minlength=n)
        scalar = np.zeros((n, x.shape[1]))
        vector = np.zeros((n, x.shape[1], 3))
        np.add.at(scalar, all_atoms, z * strengths[:, None])
        np.add.at(vector, all_atoms, z[:, :, None] * strengths[:, None, None] * dirs[:, None])
        result = []
        for i in range(2):
            ids = endpoints[:, i]
            direction = axes if i == 0 else -axes
            norm = np.maximum(mass[ids] - w, .01)[:, None]
            result.extend(((scalar[ids]-x*w[:, None])/norm,
                           (np.einsum("nfc,nc->nf", vector[ids], direction)-x*w[:, None])/norm))
        packs.append(np.column_stack((x, *result)))
    return np.concatenate(packs).astype(np.float32)


def main():
    train = [r for group in TRAIN_GROUPS for r in cohort(group, 1)]
    valid = cohort(VALID, 3)
    y, x = extract(train)
    v, z = extract(valid)
    s, sv = x["compact"], z["compact"]
    a, b = relations(train, tracks(train)), relations(valid, tracks(valid))
    ref = ridge(s, y)
    nv = ref.predict(sv)
    group = np.concatenate([np.repeat(r["protein"], len(r["y"])) for r in train])
    oof = np.empty_like(y)
    for p in np.unique(group):
        mask = group == p
        oof[mask] = ridge(s[~mask], y[~mask]).predict(s[mask])
    project = ridge(s, a)
    da, db = a-project.predict(s), b-project.predict(sv)
    results = {}
    for n in (4, 8, 12, 16, 20):
        for k in (False, True):
            t, h = da[:, :n], db[:, :n]
            if k:
                t, h = np.c_[t, t*(s[:, :1]-4.5)], np.c_[h, h*(sv[:, :1]-4.5)]
            model = ridge(t, y-oof)
            results[f"{n}/{k}"] = estimates(nv+model.predict(h), v, nv)
    record(Path("runs/ga308/development.json"), results)
    print(max(results.items(),key=lambda item:item[1]["residual_correlation"]), flush=True)


if __name__ == "__main__":
    main()
