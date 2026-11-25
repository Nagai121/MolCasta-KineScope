"""Compute competition statistics separately for each simulated branch."""
import numpy as np

from .ga_replica_teacher import trajectory_rows


def branch_capacity(x, raw, index, origins, proteins):
    np.testing.assert_array_equal(proteins,[r[0] for r in index])
    np.testing.assert_array_equal(origins,[r[1] for r in index])
    np.testing.assert_array_equal(x[:,0],[r[2] for r in index])
    result=np.empty((len(x),20),np.float32)
    for protein in np.unique(proteins):
        candidates,frame=trajectory_rows(protein,1)
        by_protein=proteins==protein
        for origin in np.unique(origins[by_protein]):
            mask=by_protein&(origins==origin)
            rows=candidates[frame==origin]
            np.testing.assert_array_equal(rows[:,2],x[mask,0])
            ids=rows[:,:2].astype(np.intp)
            a,b=ids.T
            count=int(ids.max())+1
            shape=x[mask,-34:-26]
            left=np.exp(-.5*((shape[:,1]+.45)/.55)**2)
            right=np.exp(-.5*((shape[:,5]+.45)/.55)**2)
            gate=np.where(rows[:,6]==8,left,1.)*np.where(rows[:,7]==8,right,1.)
            distances=rows[:,2,None,None]+raw[mask]
            weights=np.exp(-.5*((distances-3.25)/.28)**2)*gate[:,None,None]
            congestions=np.empty_like(weights)
            for branch,time in np.ndindex(weights.shape[1:]):
                edge=weights[:,branch,time]
                for iteration in range(5):
                    mass=np.bincount(ids.ravel(),np.repeat(edge,2),minlength=count)
                    if iteration==0:
                        congestions[:,branch,time]=mass[a]*mass[b]
                    cap=np.maximum(mass,1.)
                    edge/=np.sqrt(cap[a]*cap[b])
            values=[]
            for start in (0,2):
                w=weights[:,start:start+2]
                c=congestions[:,start:start+2]
                speed=w[:,:,2:].mean((1,2))-w[:,:,:2].mean((1,2))
                values.extend((w.mean((1,2)),w.std((1,2)),w.mean(2).std(1),
                               w.max((1,2)),w.min((1,2)),speed,
                               (w>.3).mean((1,2)),c.mean((1,2))))
            fir,sec=weights[:,:2],weights[:,2:]
            values.extend((fir.mean((1,2))-sec.mean((1,2)), fir.max((1,2))-sec.max((1,2)),
                           np.abs(fir.mean(1)-sec.mean(1)).mean(1), weights.mean(1).std(1)))
            result[mask]=np.column_stack(values)
    return result


def expanded(x,v,o,vo,p,vg):
    from .ga_branch_fusion import streams
    r,ids,s,dev_ids=streams()
    return branch_capacity(x,r,ids,o,p), branch_capacity(v,s,dev_ids,vo,vg)


if __name__=="__main__":
    from .ga_mixture import main

    main(stability=True,competitive=True,branch_competition=True)
