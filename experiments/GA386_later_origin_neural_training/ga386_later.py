"""Score the validation-locked early-window blend on GA368 confirmation."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

from tseries.data import TEST
from .explore import record
from .ga309 import predict as linear
from .ga314 import field
from .ga333 import WINDOW
from .ga_force import build
from .ga348 import predict
from .ga_native240 import SAMPLES
from .ga_native_offset import HOLD
from .ga_static_offset import cohort, estimates, extract

R=Path("runs")
D=R/"ga368"

def main():
    names=TEST+HOLD
    rows=[r for r in cohort(names,2) if r["origin"]==149]
    if tuple(r["protein"] for r in rows)!=names:
        raise RuntimeError("All locked confirmation proteins must contribute")
    y,q=extract(rows)
    compact=q["compact"]
    hold=[]
    for n in names:
        x,t=build((n,1,2))
        first=len(build((n,1,1))[1])
        hold.append((x[first:],t[first:]))
    x,t=tuple(np.concatenate(v) for v in zip(*hold))
    np.testing.assert_allclose(y,t,rtol=0,atol=3e-6)
    np.testing.assert_allclose(x[:,0],compact[:,0],rtol=0,atol=1e-6)
    with np.load(R/"ga_native_offset/baselines.npz") as z:
        b=linear(z,"real_baseline_",compact)
        p=linear(z,"native_baseline_",compact)[:,0]
    cache={}
    for family in ("amber","charmm"):
        ordered=[]
        for r in rows:
            with np.load(D/family/f"{r['protein']}-149.npz") as a:
                np.testing.assert_array_equal(a["pair"],r["pair"])
                ordered.append(a["tracks"]-r["pair"][:,2])
        cache[family]=np.concatenate(ordered,axis=2)
    a,c=cache["amber"],cache["charmm"]
    u,v=a[:2,:len(WINDOW)],c[:2,:len(WINDOW)]
    am,bm=u.mean(0).T,v.mean(0).T
    d=np.c_[am,bm,am-bm,u.std((0,1)),v.std((0,1)),u.mean(1).std(0),v.mean(1).std(0)]
    s1=predict(R/"ga318/all-8.ubj",x)-b
    s2=predict(R/"ga330/all-9.ubj",x)-b
    s3=predict(R/"ga347/all-9.ubj",x)-b
    static=.525*(.75*.5*(s1+s2)+.25*s3)
    long_a,long_c=field(a,10,"stable"),c.mean((0,1))
    d348=.2*(long_a+long_c-2*p)
    parent=.8*(b+static+d348)+.2*predict(R/"ga344/generated-6.ubj",np.c_[x[:,:3],d])
    f=R/"ga386"
    state=torch.load(f/"expanded-1.0-40.pt",map_location="cpu",weights_only=False)
    center,scale=state["center"],state["scale"]
    head=nn.Sequential(nn.Linear(len(center),96),nn.LayerNorm(96),nn.SiLU(),
                       nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,2))
    head.load_state_dict(state["state"])
    head.eval()
    with torch.no_grad():
        inp=torch.tensor(((np.c_[x,d]-center)/scale).clip(-12,12),dtype=torch.float32)
        readout=head(inp).numpy()[:,0]+x[:,0]
    result=.65*parent+.35*readout
    evaluation=dict(proteins=names,origin=149,pairs=len(y),ga348=estimates(parent,y,b),ga386=estimates(result,y,b))
    record(f/"later-confirmation.json",evaluation)
    np.savez(f/"later-confirmation.npz",predicted=result,target=y,baseline=b,parent=parent,
             proteins=np.concatenate([np.repeat(r["protein"],len(r["pair"])) for r in rows]))
    print(evaluation,flush=True)

if __name__=="__main__":
    main()

