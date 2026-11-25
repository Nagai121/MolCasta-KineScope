"""GA564+ fits cross-protein analog retrieval in generated-response space."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT = Path("runs/ga564-analog")


def main():
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True)
    (x, d, g, y, o, proteins), (v, e, h, t, vo, vg, base, parent) = pool()
    assert not set(proteins) & set(vg)
    true = y - x[:, 0]
    oof = np.empty(len(y), np.float32)
    for tr, va in GroupKFold(5).split(x, true, proteins):
        fit = Ridge(alpha=10.).fit(x[tr, :12], true[tr])
        oof[va] = fit.predict(x[va, :12])
    fit = Ridge(alpha=10.).fit(x[:, :12], true)
    shift = fit.predict(v[:, :12]) + v[:, 0]
    _, group = np.unique(proteins, return_inverse=True)
    _, counts = np.unique(group, return_counts=True)
    masks = {"all":vo>=0,"first":vo==49,"later":vo!=49}
    specs = [(view, dim, dyn, k) for view in ("level", "level_interaction", "full_response")
             for dim in (12, 32) for dyn in (1., 3.) for k in (32, 128)]
    cached = {}
    rows = []
    for index in range(len(specs)):
        if index == 0:
            spec = specs[0]
        else:
            best = max(rows, key=lambda r:r["metrics"]["all"]["residual_correlation"])
            done = {tuple(r["spec"]) for r in rows}
            spec = min((s for s in specs if s not in done), key=lambda s:(
                sum(a != b for a,b in zip(s,best["spec"])),
                sum(r["spec"][0]==s[0] for r in rows)))
        view, dim, dyn, k = spec
        tag = (view, dim, dyn)
        start = perf_counter()
        if tag not in cached:
            train, valid = features(x,d,g,view), features(v,e,h,view)
            center, scale = train.mean(0), train.std(0).clip(.001)
            a, b = ((train-center)/scale).clip(-8,8), ((valid-center)/scale).clip(-8,8)
            a[:, x.shape[1]:] *= dyn
            b[:, x.shape[1]:] *= dyn
            # Fit the embedding using CUDA and training proteins only.
            a = torch.tensor(a, dtype=torch.float32, device="cuda")
            ac = a.mean(0)
            acentered = a - ac
            _, eig = torch.linalg.eigh(acentered.T @ acentered)
            kernel = eig[:, -dim:].flip(1).contiguous()
            embedding = (acentered @ kernel).cpu().numpy()
            state = dict(center=center, scale=scale, projection_center=ac.cpu().numpy(),
                         projection=kernel.cpu().numpy(), embedding=embedding)
            cached[tag] = (embedding, (b-ac.cpu().numpy()) @ kernel.cpu().numpy(), state)
        emb, query, state = cached[tag]
        fit_seconds = perf_counter()-start
        start = perf_counter()
        emb_sq = (emb**2).sum(1)
        prediction = []
        for b in np.array_split(query, 48):
            dst = (b**2).sum(1, keepdims=True) + emb_sq - 2*(b @ emb.T)
            ix = np.argpartition(dst, k, axis=1)[:, :k]
            dst = np.take_along_axis(dst, ix, axis=1)
            distance = dst - dst.min(1,keepdims=True)
            weights = np.exp(-distance / np.maximum(np.median(distance,axis=1,keepdims=True), .05))
            weights /= np.sqrt(counts[group[ix]])
            prediction.extend((weights*(true-oof)[ix]).sum(1)/weights.sum(1))
        direct = np.asarray(prediction) + shift
        result = .65*parent+.35*direct
        model_id = f"GA{564+index}"
        scores = {s:estimates(result[m], t[m], base[m]) for s,m in masks.items()}
        row = dict(id=model_id,spec=spec,metrics=scores,fit_seconds=fit_seconds,
                   cpu_seconds=perf_counter()-start)
        rows.append(row)
        base_id = f"{view}-{dim}-{dyn}"
        store = OUT / f"{base_id}.npz"
        if not store.exists():
            np.savez_compressed(store, **state, labels=true-oof, group_count=counts[group],
                                ridge_coef=fit.coef_, ridge_intercept=fit.intercept_)
        np.savez_compressed(OUT / f"{model_id}.npz", prediction=result, readout=direct)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(model_id,spec,scores["all"]["residual_correlation"],flush=True)
        if scores["all"]["residual_correlation"] >= .7:
            break
    best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()


