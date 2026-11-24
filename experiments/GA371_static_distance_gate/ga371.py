"""GA371 gates short-window corrections using only origin structure and generated motion."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga314 import field
from .ga_native240 import SAMPLES
from .ga_static_offset import VALID, cohort, estimates

R=Path("runs")

def main():
    rows=cohort(VALID,3)
    packed={}
    for ff in ("amber","charmm"):
        q=[]
        for r in rows:
            with np.load(R/"ga360"/ff/f"{r['protein']}-{r['origin']}.npz") as z:
                np.testing.assert_array_equal(r["pair"],z["pair"])
                q.append(z["tracks"]-r["pair"][:,2])
        packed[ff]=np.concatenate(q,axis=2)
    with np.load(R/"ga360/validation.npz") as z:
        y,b,parent,origin=z["target"],z["baseline"],z["parent"],z["origin"]
        np.testing.assert_array_equal(z["proteins"],np.concatenate(
            [np.repeat(r["protein"],len(r["pair"])) for r in rows]))
    d=np.concatenate([r["pair"][:,2] for r in rows])
    a,c=packed["amber"],packed["charmm"]
    use=np.array(SAMPLES)<=10
    sa,sc=a[:,use].mean((0,1)),c[:,use].mean((0,1))
    aa,cc=sa-field(a,10,"stable"),sc-c.mean((0,1))
    spread=a[:,use].mean(1).std(0)+c[:,use].mean(1).std(0)
    same=(aa*cc>=0).astype(float)
    agreement=1/(1+np.abs(sa-sc)/.2)
    stable=1/(1+spread/.2)
    near=np.clip((5.5-d)/2,0,1)
    far=1-near
    gates={"constant":np.ones(len(d)),"agreement":agreement,"stable":stable,
           "agree_sign":same,"disagree_sign":1-same,"near":near,"far":far,
           "stable_near":stable*near,"stable_far":stable*far}
    masks={"all":np.ones(len(d),bool),"first":origin==49,"later":origin!=49}
    def metrics(p):
        return {k:estimates(p[m],y[m],b[m]) for k,m in masks.items()}
    trial={}
    baseline=metrics(parent)
    for name,g in gates.items():
        for weight in (.04,.08,.12,.16):
            z=parent+weight*g*(aa+cc)
            trial[f"{name}/{weight}"]=metrics(z)
    def correlation(item,group):
        return item[group]["residual_correlation"]
    eligible=[k for k,v in trial.items() if correlation(v,"first")>=correlation(baseline,"first")-.002]
    best=max(eligible,key=lambda k:correlation(trial[k],"all"),default=None)
    dest=R/"ga371"
    dest.mkdir(exist_ok=True)
    record(dest/"development.json",dict(parent=baseline,best=best,trials=trial,
                                        maximum_first_loss=.002))
    print(best,trial.get(best),flush=True)

if __name__=="__main__":
    main()
