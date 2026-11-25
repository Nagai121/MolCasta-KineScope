"""GA1116+: candidate-network displacement features from local competing forces."""
import numpy as np

from .ga_replica_teacher import trajectory_rows


def relaxation(x, dynamic, origins, proteins):
    out=np.empty((len(x),24),dtype=np.float32)
    steps=(dynamic.shape[1]-4)//3
    dmean=dynamic[:,:2*steps].mean(1)
    for protein in np.unique(proteins):
        candidate,at=trajectory_rows(protein,1)
        mask=proteins==protein
        for origin in np.unique(origins[mask]):
            rows=mask&(origins==origin)
            pair=candidate[at==origin]
            np.testing.assert_array_equal(pair[:,2],x[rows,0])
            atoms=pair[:,:2].astype(np.intp)
            start=pair[:,3:6]
            degree=np.bincount(atoms.ravel()).clip(1)
            a,b=atoms.T
            orientations=x[rows,-34:-26]
            left=np.exp(-.5*((orientations[:,1]+.45)/.55)**2)
            right=np.exp(-.5*((orientations[:,5]+.45)/.55)**2)
            accept=np.where(pair[:,6]==8,left,1.)*np.where(pair[:,7]==8,right,1.)
            f=[]
            for use_response in (False,True):
                baseline=pair[:,2]+dmean[rows] if use_response else pair[:,2]
                for scale in (.18,.36):
                    delta=np.zeros((len(degree),3),dtype=np.float32)
                    for frame in range(4):
                        axes=start+delta[b]-delta[a]
                        dist=np.linalg.norm(axes,axis=1).clip(.1)
                        proximity=np.exp(-.5*((dist-3.3)/.45)**2)*accept
                        mass=np.bincount(atoms.ravel(),np.repeat(proximity,2),minlength=len(degree)).clip(1)
                        sign=np.clip(baseline-3.3,-.8,.8)
                        transfer=(sign*proximity/np.sqrt(mass[a]*mass[b]))[:,None]*axes/dist[:,None]
                        force=np.zeros_like(delta)
                        np.add.at(force,a,transfer)
                        np.add.at(force,b,-transfer)
                        delta=.40*delta+scale*force/np.sqrt(degree[:,None])
                        if frame in (0,1,3):
                            movement=np.linalg.norm(start+delta[b]-delta[a],axis=1)-pair[:,2]
                            f.extend((movement,proximity))
            out[rows]=np.column_stack(f)
    return np.c_[x,out]


if __name__=="__main__":
    from .ga_mixture import main
    main(stability=True,competitive=True,road_relaxation=True)
