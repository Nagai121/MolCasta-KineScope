"""Audit conditionally matched candidate rankings and the selected GA generator."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from tseries.hb_validation10 import panel10
from .explore import record, score
from .frontends import data, design
from .ga300 import DEST, Generator, add_routes, labels, readout, train_readout, transform
from .gc_assist import matched
from .gc_replica import keys, topology_signature


def replica_pairs(parts, base, observed):
    lengths=np.cumsum([0,*[len(p["y"]) for p in parts]])
    by={(p["protein"],p["replica"]):i for i,p in enumerate(parts)}
    scores=np.log(np.clip(base,1e-6,1-1e-6)/np.clip(1-base,1e-6,1-1e-6))
    indices=[]
    signatures=[topology_signature(p) for p in parts]
    for i,p in enumerate(parts):
        for replica in range(p["replica"]+1,4):
            j=by[(p["protein"],replica)]
            q=parts[j]
            if signatures[i]!=signatures[j]:
                raise ValueError("Replica atom order differs")
            _,a,b=np.intersect1d(keys(p),keys(q),assume_unique=True,return_indices=True)
            ra,rb=a+lengths[i],b+lengths[j]
            keep=(np.abs(p["pair"][a,2]-q["pair"][b,2])<=.15)
            keep &= np.abs(scores[ra]-scores[rb])<=.3
            keep &= np.abs(observed[ra]-observed[rb])>=.19
            indices.append(np.column_stack((ra[keep],rb[keep])))
    return np.concatenate(indices).T


def main():
    torch.set_num_threads(3)
    log=json.loads((DEST/"results.json").read_bytes())
    if len(log["trials"])!=300:
        raise RuntimeError("GA evaluation not complete")
    parts,x,base=data(panel10())
    actual,_=labels(parts)
    x=add_routes(parts,x)
    rows=log["trials"]
    winner=max(rows,key=lambda n:rows[n]["metrics"]["auprc"])
    per_target=[max((n for n in rows if rows[n]["target"]==kind),
                    key=lambda n:rows[n]["metrics"]["auprc"])
                for kind in sorted({r["target"] for r in rows.values()})]
    signals={"frozen_t1":base,"observed_occupancy":actual[:,2]}
    left,right=replica_pairs(parts,base,actual[:,2])
    report={}
    for ident in dict.fromkeys((winner,*per_target)):
        saved=torch.load(DEST/f"{ident}.pt",map_location="cpu",weights_only=True)
        a=torch.from_numpy(design(x[saved["name"]],base))
        model=Generator(a.shape[1],saved["structure"]).eval()
        model.load_state_dict(saved["generator"])
        v=transform(model,((a-saved["center"])/saved["scale"]).clamp(-8,8),saved["mode"])
        head=torch.nn.Sequential(torch.nn.Linear(v.shape[1]+1,20),torch.nn.SiLU(),torch.nn.Linear(20,1)).eval()
        head.load_state_dict(saved["correction"])
        probability=readout(head,saved["rc"],saved["rsd"],v,base)
        signals[ident]=probability
        mode=saved["mode"]
        occupancy={"point":lambda:v[:,0],"moments":lambda:v[:,2],"hurdle":lambda:v[:,1],
                   "distribution":lambda:v[:,:3].sum(1),"joint":lambda:v[:,3]}[mode]()
        order=(occupancy[left]-occupancy[right])*(actual[left,2]-actual[right,2])
        report[ident]=dict(occupancy_rmse=float(np.sqrt(np.mean((occupancy-actual[:,2])**2))),
                           occupancy_correlation=float(np.corrcoef(occupancy,actual[:,2])[0,1]),
                           occupancy_presence_ap=float(average_precision_score(actual[:,2]>0,occupancy)),
                           replica_order=float(((order>0)+.5*(order==0)).mean()) if len(order) else None)
    names=tuple(n for pair in zip(log["generator"],log["readout"]) for n in pair)
    train,xt,bt=data(names)
    xt=add_routes(train,xt)
    measured,_=labels(train)
    y=np.concatenate([p["y"] for p in train])
    group=np.concatenate([np.full(len(p["y"]),p["protein"] in log["generator"]) for p in train])
    rng=np.random.default_rng(300)
    gi,ri=np.flatnonzero(group),np.flatnonzero(~group)
    _,ri=[np.sort(rng.choice(i,min(len(i),65536),replace=False)) for i in (gi,ri)]
    feature=rows[winner]["view"]
    comparators={}
    for name,train_x,valid_x in (("measured_history",measured[:,[0,1,2,3]],actual[:,[0,1,2,3]]),
                                 ("static_direct",xt[feature],x[feature])):
        head,center,scale=train_readout(train_x[ri],bt[ri],y[ri],20000+int(winner[2:]))
        p=readout(head,center,scale,valid_x,base)
        comparators[name]=score(parts,p)
        signals[name]=p
    record(DEST/"audit.json",dict(selected=winner,top_by_target=per_target,
                                 conditional=matched(parts,base,signals,same_frame=True),
                                 replica_pairs=len(left),occupancy=report,
                                 matched_training_controls=comparators))


if __name__=="__main__":
    main()
