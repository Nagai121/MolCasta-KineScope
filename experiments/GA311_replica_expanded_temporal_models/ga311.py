"""GA311: fit dynamic residual models on cached extra origins and replicas."""
import time
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga306 import cache_path, tracks
from .ga308 import relations
from .ga309 import predict
from .ga_static_offset import TRAIN_GROUPS, VALID, cohort, estimates, extract, ridge


def main():
    dest = Path("runs/ga311")
    dest.mkdir(parents=True, exist_ok=True)
    train = [r for group in TRAIN_GROUPS for replica in (1, 2)
             for r in cohort(group, 3, replica) if cache_path(r).exists()]
    valid = cohort(VALID, 3)
    assert not {r["protein"] for r in train} & set(VALID)
    print("Training origins", len(train), flush=True)
    y, x = extract(train)
    v, z = extract(valid)
    s, sv = x["compact"], z["compact"]
    with np.load("runs/ga_native_offset/baselines.npz") as fixed:
        nv = predict(fixed,"real_baseline_",sv)
    group = np.concatenate([np.repeat(r["protein"], len(r["y"])) for r in train])
    oof = np.empty_like(y)
    for p in np.unique(group):
        q = group == p
        oof[q] = ridge(s[~q], y[~q]).predict(s[q])
    a, b = tracks(train), tracks(valid)
    profiles = (np.c_[a.mean(1), np.abs(a[:, 0]-a[:, 1])],
                np.c_[b.mean(1), np.abs(b[:, 0]-b[:, 1])])
    graph = relations(train,a), relations(valid,b)
    result = dict(origins=len(train), pairs=len(y), trials={},
                  training_proteins=np.unique(group).tolist(), validation=list(VALID))
    for label, view in (("temporal", profiles), ("graph", graph),
                        ("union", tuple(np.c_[u, w] for u,w in zip(profiles,graph)))):
        model = ridge(s, view[0])
        dt, dv = view[0]-model.predict(s), view[1]-model.predict(sv)
        for context, (t,h) in (
            ("dynamic",(dt,dv)), ("compact",(np.c_[dt,s],np.c_[dv,sv])),
            ("local",(np.c_[dt,x["local"]],np.c_[dv,z["local"]]))):
            for depth in (0,2,4,6):
                started = time.perf_counter()
                name = f"{label}-{context}-{depth}"
                if depth:
                    model = XGBRegressor(device="cuda", tree_method="hist", n_estimators=320,
                                         max_depth=depth, max_bin=96, n_jobs=4, learning_rate=.04,
                                         subsample=.85, colsample_bytree=.85, random_state=311)
                    model.fit(t,y-oof)
                    model.set_params(device="cpu")
                    model.save_model(dest/f"{name}.ubj")
                else:
                    model = ridge(t,y-oof)
                p = nv+model.predict(h)
                result["trials"][name] = dict(seconds=time.perf_counter()-started,
                                              **estimates(p,v,nv))
                record(dest/"development.json",result)
                print(name,result["trials"][name],flush=True)


if __name__ == "__main__":
    main()
