"""GA633+: predict pair offsets with heavy-atom elastic compliance features."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu
from xgboost import XGBRegressor

from tseries.data import registry, trajectory
from tseries.hb_single_structure import ROOT
from .ga_replica_teacher import trajectory_rows
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga633-elastic")


def elastic(coords, pair):
    n=len(coords)
    contacts=cKDTree(coords).query_pairs(5.2,output_type="ndarray")
    i,j=contacts.T
    edge_lengths=np.linalg.norm(coords[i]-coords[j],axis=1)
    springs=1./np.maximum(edge_lengths,1.2)**2
    degree=np.bincount(np.r_[i,j],np.r_[springs,springs],minlength=n)
    lap=sparse.coo_matrix((np.r_[-springs,-springs,degree],
                          (np.r_[i,j,np.arange(n)],np.r_[j,i,np.arange(n)])),shape=(n,n)).tocsc()
    a,b=pair[:,:2].astype(np.intp).T
    fixed=np.eye(1,n,n-1,dtype=np.float32).ravel()
    rng=np.random.default_rng(633)
    probes=rng.choice(np.array([-1.,1.]),size=(n,48))
    estimates=[]
    for damping in (.06,.2,.7):
        h=lap+sparse.diags(damping+fixed,format="csc")
        sampled=splu(h).solve(probes)
        # Hutchinson's bilinear estimate of compliance to an opposing pair impulse.
        diff=sampled[a]-sampled[b]
        load=probes[a]-probes[b]
        compliance=(diff*load).mean(1)
        variance=(diff**2).mean(1)
        endpoint=(sampled*probes).mean(1)
        estimates.extend((compliance,variance,endpoint[a]+endpoint[b],abs(endpoint[a]-endpoint[b])))
    return np.column_stack([degree[a],degree[b],*estimates]).astype(np.float32)


def prepare(x,o,names):
    out=np.empty((len(x),14),dtype=np.float32)
    cache=OUT/"mechanics"
    cache.mkdir(exist_ok=True)
    for k,name in enumerate(np.unique(names),1):
        candidate,time=trajectory_rows(name,1)
        reg=registry(ROOT,name)
        for origin in np.unique(o[names==name]):
            idx=np.flatnonzero((names==name)&(o==origin))
            pair=candidate[time==origin]
            np.testing.assert_array_equal(pair[:,2],x[idx,0])
            f=cache/f"{name}-{origin}.npz"
            if f.exists():
                with np.load(f) as z:
                    np.testing.assert_array_equal(pair,z["pair"])
                    out[idx]=z["elastic"]
            else:
                block=next(reg.blocks(trajectory(name,1),start=int(origin),stop=int(origin)+1))
                q=elastic(block.positions[0],pair)
                out[idx]=q
                np.savez_compressed(f,pair=pair,elastic=q)
        if k%10==0:
            print("elastic preparation",k,flush=True)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(p)&set(vg)
    a,b=prepare(x,o,p),prepare(v,vo,vg)
    mainx,mainv=features(x,d,g,"level"),features(v,e,h,"level")
    outcomes=[]
    for i,(view,depth) in enumerate((s,k) for s in ("mechanics","static","dynamic") for k in (3,5,7)):
        aa,bb={"mechanics":(np.c_[mainx,a],np.c_[mainv,b]),
               "static":(np.c_[x,a],np.c_[v,b]),"dynamic":(mainx,mainv)}[view]
        tic=perf_counter()
        model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=400,max_depth=depth,
                             max_bin=96,learning_rate=.04,n_jobs=4,reg_lambda=10,subsample=.88)
        model.fit(aa,y-x[:,0],sample_weight=np.where(o==49,1.,4.))
        assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
        model.set_params(device="cpu")
        direct=model.predict(bb)+v[:,0]
        pred=.65*parent+.35*direct
        results={k:estimates(pred[m],t[m],base[m]) for k,m in
                 (("all",vo>=0),("first",vo==49),("later",vo!=49))}
        tag=f"GA{633+i}"
        model.save_model(OUT/f"{tag}.ubj")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=pred,readout=direct)
        row=dict(id=tag,view=view,depth=depth,metrics=results,seconds=perf_counter()-tic)
        outcomes.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
        if results["all"]["residual_correlation"]>=.700:
            break
    best=max(outcomes,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
