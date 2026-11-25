"""GA1188+: use neighbors outside the scored candidate set as context."""
import numpy as np

from tseries.data import shard
from tseries.hb_single_structure import ROOT


def wide_context(x,origins,proteins):
    out=np.empty((len(x),28),dtype=np.float32)
    for name in np.unique(proteins):
        focus=proteins==name
        with np.load(shard(ROOT,name,1,100,8.)) as data:
            candidates,time=data["pair"],data["origin"]
        for origin in np.unique(origins[focus]):
            mask=focus&(origins==origin)
            rows=candidates[time==origin]
            local=rows[rows[:,2]<=6.]
            np.testing.assert_array_equal(local[:,2],x[mask,0])
            a,b=local[:,:2].astype(np.intp).T
            ids=rows[:,:2].astype(np.intp)
            n=int(ids.max())+1
            repeated=ids.ravel()
            distance=rows[:,2]
            features=[]
            for cut in (5.,6.,7.,8.):
                internal=distance<=cut
                count=np.bincount(repeated,weights=np.repeat(internal,2),minlength=n)
                pressure=np.bincount(repeated,weights=np.repeat(np.exp(-distance/2.)*internal,2),minlength=n)
                features.extend((count[a],count[b],pressure[a],pressure[b]))
            for border in (5.,6.,7.):
                shell=(distance>border)&(distance<=border+1.)
                counts=np.bincount(repeated,weights=np.repeat(shell,2),minlength=n)
                weight=np.bincount(repeated,weights=np.repeat(shell/distance.clip(1.),2),minlength=n)
                features.extend((counts[a],counts[b],weight[a],weight[b]))
            out[mask]=np.column_stack(features)
    return out


if __name__=="__main__":
    from .ga_mixture import main

    main(stability=True,competitive=True,orient_capacity=True,wide_competition=True)
