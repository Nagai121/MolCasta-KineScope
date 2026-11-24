"""GA346 measures same-pair persistence in reserved proteins, for diagnosis only."""
import json
from pathlib import Path

import numpy as np

from .explore import record
from .ga309 import predict
from .ga_static_offset import cohort, estimates, extract


def assemble(names,replica,count):
    items=cohort(names,count,replica)
    y,x=extract(items)
    with np.load("runs/ga_native_offset/baselines.npz") as m:
        s=predict(m,"real_baseline_",x["compact"])
    offset=0
    out={}
    for r in items:
        n=len(r["y"])
        pack=[(int(a),int(b),float(d),float(z)) for (a,b,d),z in zip(r["pair"][:,:3],(y-s)[offset:offset+n])]
        out.setdefault(r["protein"],[]).append(pack)
        offset+=n
    return out


def main():
    dest=Path("runs/ga346")
    dest.mkdir(parents=True,exist_ok=True)
    names=tuple(json.loads(Path("runs/ga321/reserve.json").read_bytes())["proteins"])
    current=assemble(names,1,1)
    peers=[assemble(names,r,3) for r in (2,3)]
    with np.load("runs/ga332/reserve/evaluation.npz") as e:
        pred=e["predicted"]-e["static"]
        target=e["target"]-e["static"]
        group=e["proteins"]
    rows=[]
    for n in names:
        known={}
        for rep in peers:
            for array in rep[n]:
                for a,b,d,y in array:
                    known.setdefault((a,b),[]).append((d,y))
        for i,(a,b,d,y) in enumerate(current[n][0]):
            if (a,b) not in known:
                continue
            options=np.array(known[a,b])
            j=np.argmin(np.abs(options[:,0]-d))
            val=options[j,1]
            rows.append((n,i,y,val,abs(options[j,0]-d)))
    result={}
    for d in (.2,.4,.8,4.):
        idx=[(n,i,t,q) for n,i,t,q,d0 in rows if d0<=d]
        y=np.array([t for n,i,t,q in idx])
        oracle=np.array([q for n,i,t,q in idx])
        model=np.array([pred[group==n][i] for n,i,t,q in idx])
        aligned=np.array([target[group==n][i] for n,i,t,q in idx])
        np.testing.assert_allclose(y,aligned,atol=2.e-6,rtol=0)
        base=np.zeros_like(y)
        result[str(d)]=dict(pairs=len(y),other_replica=estimates(oracle,y,base),
                            ga332=estimates(model,y,base))
    out=dict(proteins=names,scope="diagnostic; alternate real trajectories excluded from predictors",results=result)
    record(dest/"diagnostics.json",out)
    print(out,flush=True)


if __name__=="__main__":
    main()
