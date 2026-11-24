"""Static-origin prediction of the same pair's preceding ATLAS mean distance."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from tseries.data import TEST,shard
from tseries.hb_single_structure import ROOT
from .explore import record
from .gc_history import past
from .frontends import data
from .g2i_path35 import paths
from tseries.data import registry,trajectory,cell

DEST=Path("runs/ga_static_offset")
VALID=("3mx7_A","3t92_A","4wrp_A")
TRAIN_GROUPS=(
    ("1yu5_X","1knt_A","3txs_C"),
    ("2vm5_A","3fp5_A","1i4m_A","3fyb_A"),
    ("2oxo_A","2qsa_A","5pal_A","1uz3_A"),
    ("4obi_A","5hqh_A","1m9z_A","3fpu_A"),
    ("3po8_A",),("1tov_A",),("3cw3_A",),("7fd1_A",),("1o4k_A",),
)


def cohort(names,max_origins,replica=1):
    parts,views,base=data(names)
    rows=[]
    offset=0
    for p in parts:
        size=len(p["y"])
        if p["replica"]==replica:
            for a,b in list(zip(np.r_[0,p["ends"][:-1]],p["ends"]))[:max_origins]:
                rows.append(dict(protein=p["protein"],replica=replica,origin=int(p["origin"][a]),pair=p["pair"][a:b],
                                 y=p["y"][a:b],frozen=base[offset+a:offset+b],
                                 static=views["combined"][offset+a:offset+b]))
        offset+=size
    return rows


def one(row):
    name,origin=row["protein"],row["origin"]
    replica=row.get("replica",1)
    pair=row["pair"]
    d=pair[:,2]
    real=past(dict(row,replica=replica,origin=np.full(len(d),origin),ends=np.array([len(d)])))
    # Include the origin sample: the same 11 samples used by the reference window.
    target=np.column_stack((real,d)).mean(1)
    with np.load(shard(ROOT,name,replica,100,8.)) as z:
        keep=(z["origin"]==origin)&(z["pair"][:,2]<=6.)
        if not np.array_equal(z["pair"][keep],pair):
            raise ValueError("Pair alignment changed")
        base=z["features"][keep]
    reg=registry(ROOT,name)
    block=next(reg.blocks(trajectory(name,replica),start=origin,stop=origin+1))
    route=paths(block.positions[0],cell(block.box[0]),pair)
    anchor=row["frozen"]
    compact=np.c_[d,d*d,anchor,pair[:,6:8],base[:,[1,3,5]]]
    return target,dict(compact=compact,structural=np.c_[compact,base],
                       local=np.c_[compact,base[:,:118],row["static"],route],
                       combined=np.c_[compact,base,row["static"],route])


def extract(rows):
    with ThreadPoolExecutor(4) as pool:
        items=list(pool.map(one,rows))
    mean=np.concatenate([x for x,_ in items])
    views={k:np.concatenate([d[k] for _,d in items]).astype(np.float32) for k in items[0][1]}
    return mean,views


def ridge(x,y):
    return make_pipeline(StandardScaler(),Ridge(alpha=32.)).fit(x,y)


def estimates(pred,y,nuisance):
    e=pred-y
    true=y-nuisance
    fit=pred-nuisance
    return dict(residual_sd=float(e.std()),rmse=float(np.sqrt(np.mean(e**2))),
                bias=float(e.mean()),residual_correlation=float(np.corrcoef(fit,true)[0,1])
                if fit.std()>1e-7 else None)


def main():
    torch.set_num_threads(3)
    DEST.mkdir(parents=True,exist_ok=True)
    train=[row for group in TRAIN_GROUPS for row in cohort(group,1)]
    valid=cohort(VALID,3)
    test=cohort(TEST,3)+cohort(("4ftf_A","3msw_A"),3)+cohort(("3wx4_A","4wlr_B"),2)
    tg,x=extract(train)
    vg,v=extract(valid)
    yg,z=extract(test)
    group=np.concatenate([np.full(len(r["y"]),r["protein"]) for r in train])
    assert not set(group)&{r["protein"] for r in valid+test}
    nuisance=ridge(x["compact"],tg)
    raw_v=nuisance.predict(v["compact"])
    raw_y=nuisance.predict(z["compact"])
    oof=np.empty(len(tg),dtype=np.float32)
    for name in np.unique(group):
        q=group==name
        oof[q]=ridge(x["compact"][~q],tg[~q]).predict(x["compact"][q])
    report=dict(train=list(dict.fromkeys(group)),validation=VALID,
                test=list(dict.fromkeys(r["protein"] for r in test)),
                rows=[len(tg),len(vg),len(yg)],trials={},criterion=.30,
                baseline=dict(distance=estimates(z["compact"][:,0],yg,raw_y),
                              static_ridge=estimates(raw_y,yg,raw_y)))
    for k in x:
        for depth in (2,4,6):
            for target in ("direct","residual"):
                model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=220,
                                   max_depth=depth,learning_rate=.055,subsample=.85,
                                   colsample_bytree=.8,random_state=431,n_jobs=3)
                model.fit(x[k],tg if target=="direct" else tg-oof)
                if json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]!="cuda:0":
                    raise RuntimeError("Training did not use the selected CUDA device")
                model.set_params(device="cpu")
                ident=f"{k}-{depth}-{target}"
                p=model.predict(v[k])+(raw_v if target=="residual" else 0)
                report["trials"][ident]=dict(validation=estimates(p,vg,raw_v))
                model.save_model(DEST/f"{ident}.ubj")
    best=min(report["trials"],key=lambda k:report["trials"][k]["validation"]["residual_sd"])
    kind,_,target=best.split("-")
    model=XGBRegressor(device="cpu")
    model.load_model(DEST/f"{best}.ubj")
    pred=model.predict(z[kind])+(raw_y if target=="residual" else 0)
    report["selected"]=best
    report["test_metrics"]=estimates(pred,yg,raw_y)
    np.savez(DEST/"heldout.npz",target=yg,predicted=pred,static=raw_y,
             proteins=np.concatenate([np.full(len(r["y"]),r["protein"]) for r in test]),
             origins=np.concatenate([np.full(len(r["y"]),r["origin"]) for r in test]),
             pair=np.concatenate([r["pair"][:,:2] for r in test]))
    standardize,linear=nuisance.steps[0][1],nuisance.steps[1][1]
    np.savez(DEST/"nuisance.npz",mean=standardize.mean_,scale=standardize.scale_,
             weights=linear.coef_,bias=linear.intercept_)
    record(DEST/"results.json",report)
    print(best,report["test_metrics"],flush=True)


if __name__=="__main__":
    main()
