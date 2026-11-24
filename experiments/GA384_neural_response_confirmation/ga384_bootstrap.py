"""Protein-level paired uncertainty of fixed GA384 improvements."""
from pathlib import Path
import numpy as np

R=Path("runs")

def corr(p,y,b):
    return float(np.corrcoef(p-b,y-b)[0,1])

def boot(p,q,y,b,g,seed=384):
    groups=[np.flatnonzero(g==n) for n in np.unique(g)]
    rng=np.random.default_rng(seed)
    draws=[]
    for picked in rng.integers(len(groups),size=(4000,len(groups))):
        i=np.concatenate([groups[k] for k in picked])
        new,old=corr(p[i],y[i],b[i]),corr(q[i],y[i],b[i])
        draws.append((new,old,new-old))
    return dict(n=len(y),proteins=len(groups),ga384=corr(p,y,b),ga348=corr(q,y,b),
                delta=corr(p,y,b)-corr(q,y,b),interval=np.quantile(draws,[.025,.975],axis=0).tolist())

def main():
    from t2i.explore import record
    result={}
    previous=None
    for label in ("previous","reserve"):
        with np.load(R/"ga384"/f"{label}.npz") as a,np.load(R/"ga348"/f"{label}.npz") as z:
            for key in ("target","static","proteins"):
                np.testing.assert_array_equal(a[key],z[key])
            d=(a["predicted"],z["predicted"],a["target"],a["static"],a["proteins"])
        result[label]=boot(*d)
        if label=="previous":
            previous=d
    with np.load(R/"ga384/later-confirmation.npz") as z:
        later=tuple(z[k] for k in ("predicted","parent","target","baseline","proteins"))
    result["later"]=boot(*later)
    result["previous_both_origins"]=boot(*(np.concatenate((a,b)) for a,b in zip(previous,later)))
    record(R/"ga384/paired-bootstrap.json",result)
    print(result,flush=True)

if __name__=="__main__":
    main()
