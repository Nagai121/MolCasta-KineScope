"""GA361 measures pretrained sequence context as a complementary static prior."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Bio.SeqUtils import seq1
import esm
import numpy as np
import torch
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

from tseries.data import registry, shard, trajectory
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_force import build
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga361"

def embed(names):
    torch.set_num_threads(4)
    model, alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    convert=alphabet.get_batch_converter()
    for name in names:
        file=D/f"{name}.npz"
        if file.exists():
            continue
        atoms=registry(ROOT,name).atoms(trajectory(name,1)).to_pydict()
        indices=np.array(atoms["residue_index"])
        ids,first=np.unique(indices,return_index=True)
        sequence="".join(seq1(atoms["residue_name"][i]) for i in first)
        _,_,tokens=convert([(name,sequence)])
        with torch.no_grad():
            encoded=model(tokens,repr_layers=[6])["representations"][6][0,1:1+len(sequence)]
        np.savez(file,ids=ids,values=encoded.numpy())

def features(task, enrich):
    name,rep,count=task
    x,y=build(task)
    with np.load(shard(ROOT,name,rep,100,8.)) as source:
        p=source["pair"]
        ids=p[p[:,2]<=6.][:len(y),:3]
    np.testing.assert_array_equal(ids[:,2],x[:,0])
    atoms=registry(ROOT,name).atoms(trajectory(name,rep)).to_pydict()
    index=np.array(atoms["residue_index"])[ids[:,:2].astype(int)]
    residues,values=enrich[name]
    ab=np.searchsorted(residues,index)
    np.testing.assert_array_equal(residues[ab],index)
    a,b=values[ab].transpose(1,0,2)
    return x,y,np.c_[a,b,np.abs(a-b)]

def main():
    D.mkdir(exist_ok=True)
    train=tuple(json.loads((R/"ga318/development.json").read_bytes())["train"])
    assert not set(train)&set(VALID)
    embed(train+VALID)
    arrays={}
    for name in train+VALID:
        with np.load(D/f"{name}.npz") as p:
            arrays[name]=p["ids"],p["values"]
    reducer=PCA(n_components=16,random_state=361).fit(np.concatenate([arrays[n][1] for n in train]))
    np.savez(D/"projection.npz",mean=reducer.mean_,components=reducer.components_)
    context={k:(i,reducer.transform(v)) for k,(i,v) in arrays.items()}
    jobs=[(n,r,4) for n in train for r in (1,2,3)]
    with ThreadPoolExecutor(4) as pool:
        groups=list(pool.map(lambda t: features(t,context),jobs))
    x,y,e=tuple(np.concatenate(g) for g in zip(*groups))
    del groups
    v,t,h=tuple(np.concatenate(g) for g in zip(*(features((n,1,1),context) for n in VALID)))
    dev=components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    truth,base,*_=dev
    np.testing.assert_allclose(t,truth,rtol=0,atol=3e-6)
    anchor,pm=forecasts(dev,("generated-6",.25,.2))
    results={}
    for label,short in (("embedding",True),("joint",False)):
        xx=np.c_[x[:,:3] if short else x,e]
        vv=np.c_[v[:,:3] if short else v,h]
        for depth in (4,7):
            head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                              learning_rate=.045,n_estimators=360,n_jobs=4,
                              subsample=.85,colsample_bytree=.85)
            head.fit(xx,y-x[:,0])
            if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA training required")
            head.set_params(device="cpu")
            p=head.predict(vv)+v[:,0]
            key=f"{label}-{depth}"
            results[key]={"unmixed":estimates(p,t,base)}
            for w in (.1,.2,.35,.5):
                results[key][str(w)]=estimates((1-w)*anchor+w*p,t,base)
            head.save_model(D/f"{key}.ubj")
            print(key,results[key],flush=True)
    record(D/"development.json",dict(trials=results,parent=pm,proteins=train,rows=len(y)))

if __name__=="__main__":
    main()

