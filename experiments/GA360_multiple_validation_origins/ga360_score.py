"""GA360 scores the locked early-origin decisions across three origins per protein."""
from pathlib import Path
import numpy as np
import torch

from .explore import record
from .ga314 import field
from .ga333 import WINDOW
from .ga_force import build
from .ga348 import predict
from .ga309 import predict as linear
from .ga_static_offset import VALID, cohort, estimates, extract

R=Path("runs")
D=R/"ga360"

def main():
    rows=cohort(VALID,3)
    y,views=extract(rows)
    compact=views["compact"]
    x,target=np.concatenate([build((n,1,3))[0] for n in VALID]),np.concatenate([build((n,1,3))[1] for n in VALID])
    np.testing.assert_allclose(target,y,atol=3e-6,rtol=0)
    np.testing.assert_array_equal(x[:,0],compact[:,0])
    with np.load(R/"ga_native_offset/baselines.npz") as z:
        base=linear(z,"real_baseline_",compact)
        proj=linear(z,"native_baseline_",compact)[:,0]
    tracks={}
    for ff in ("amber","charmm"):
        packets=[]
        for row in rows:
            with np.load(D/ff/f"{row['protein']}-{row['origin']}.npz") as z:
                np.testing.assert_array_equal(row["pair"],z["pair"])
                packets.append(z["tracks"]-row["pair"][:,2])
        tracks[ff]=np.concatenate(packets,axis=2)
    a,c=tracks["amber"],tracks["charmm"]
    u,v=a[:2,:len(WINDOW)],c[:2,:len(WINDOW)]
    am,bm=u.mean(0).T,v.mean(0).T
    dyn=np.c_[am,bm,am-bm,u.std((0,1)),v.std((0,1)),u.mean(1).std(0),v.mean(1).std(0)]
    f1=predict(R/"ga318/all-8.ubj",x)-base
    f2=predict(R/"ga330/all-9.ubj",x)-base
    f3=predict(R/"ga347/all-9.ubj",x)-base
    st=.5*(f1+f2)
    static=.525*(.75*st+.25*f3)
    motion=.2*(field(a,10,"stable")+c.mean((0,1))-2*proj)
    short_matrix=np.c_[x[:,:3],dyn]
    parent=.8*(base+motion+static)+.2*predict(R/"ga344/generated-6.ubj",short_matrix)
    o=np.concatenate([np.full(len(row["pair"]),row["origin"]) for row in rows])
    groups={"pooled":np.ones(len(y),dtype=bool),"first":o==49,"later":o!=49}
    alternatives={
        "ga332":base+motion+.525*st,
        "ga348":parent,
        "ga355":.8*(base+motion+static)+.2*predict(R/"ga355/generated-6.ubj",short_matrix),
        "ga357":.9*parent+.1*predict(R/"ga357/distance-6.ubj",short_matrix),
    }
    from .ga359 import Readout
    from .ga363 import Auxiliary
    cp=torch.load(R/"ga359/223-64-correlation-30.pt",map_location="cpu",weights_only=False)
    net=Readout(cp["static"],cp["samples"],cp["width"])
    net.load_state_dict(cp["state"])
    net.eval()
    with torch.no_grad():
        inp=torch.from_numpy(((x-cp["cx"])/cp["sx"]).clip(-12,12))
        dyn_norm=torch.from_numpy((dyn-cp["cd"])/cp["sd"])
        p=net(inp,dyn_norm).numpy()+x[:,0]
    alternatives["ga359"]=.9*parent+.1*p
    cp=torch.load(R/"ga363/223-0.2-10.pt",map_location="cpu",weights_only=False)
    net=Auxiliary(cp["static"],cp["samples"],64)
    net.load_state_dict(cp["state"])
    net.eval()
    with torch.no_grad():
        inp=torch.from_numpy(((x-cp["x_mean"])/cp["x_scale"]).clip(-12,12))
        dyn_norm=torch.from_numpy((dyn-cp["d_mean"])/cp["d_scale"])
        p=net(inp,dyn_norm)[:,0].numpy()+x[:,0]
    alternatives["ga363"]=.9*parent+.1*p
    result={n:dict(pairs=int(mask.sum()),metrics={m:estimates(p[mask],y[mask],base[mask])
                for m,p in alternatives.items()}) for n,mask in groups.items()}
    record(D/"validation.json",result)
    names=np.concatenate([np.repeat(row["protein"],len(row["pair"])) for row in rows])
    np.savez(D/"validation.npz",target=y,baseline=base,parent=parent,origin=o,proteins=names)
    for n,q in result.items():
        print(n,q["pairs"],{m:v["residual_correlation"] for m,v in q["metrics"].items()},flush=True)

if __name__=="__main__":
    main()


