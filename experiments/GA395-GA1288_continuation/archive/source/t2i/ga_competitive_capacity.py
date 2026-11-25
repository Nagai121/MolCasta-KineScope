"""GA1001+: model attraction competition with a bounded per-atom capacity."""
import numpy as np

from .ga_replica_teacher import trajectory_rows


def competition(x,d,origins,proteins,oriented=False,temporal=False):
    response_width=(d.shape[1]-4)//3
    movement=d[:,:2*response_width].mean(1)
    size=(32 if oriented else 16)+(14 if temporal else 0)
    result=np.empty((len(x),size),np.float32)
    for protein in np.unique(proteins):
        by_protein=proteins==protein
        candidates,time=trajectory_rows(protein,1)
        for origin in np.unique(origins[by_protein]):
            mask=by_protein&(origins==origin)
            rows=candidates[time==origin]
            np.testing.assert_array_equal(rows[:,2],x[mask,0])
            atoms=rows[:,:2].astype(np.intp)
            n=int(atoms.max())+1
            left,right=atoms.T
            stat=[]
            orientation=x[mask,-34:-26]
            # Index 1 in each four-column endpoint block is the C reference.
            oxygen_left=np.exp(-.5*((orientation[:,1]+.45)/.55)**2)
            oxygen_right=np.exp(-.5*((orientation[:,5]+.45)/.55)**2)
            gate=np.where(rows[:,6]==8,oxygen_left,1.)*np.where(rows[:,7]==8,oxygen_right,1.)
            for temperature in (.15,.35):
                for distance in (rows[:,2],rows[:,2]+movement[mask]):
                    for angular in ((False,True) if oriented else (False,)):
                        weight=np.exp(-.5*((distance-3.25)/temperature)**2).astype(np.float32)
                        if angular:
                            weight=weight*gate
                        first=None
                        for step in range(6):
                            mass=np.bincount(atoms.ravel(),np.repeat(weight,2),minlength=n)
                            cap=np.maximum(mass,1.)
                            if step==0:
                                first=mass
                            weight=weight/np.sqrt(cap[left]*cap[right])
                        static_left=first[left]
                        static_right=first[right]
                        stat.extend((weight,static_left,static_right,static_left*static_right))
            if temporal:
                levels=[]
                for field in range(2):
                    field_columns=d[mask,field*response_width:(field+1)*response_width]
                    for delta in (field_columns[:,:2].mean(1),field_columns.mean(1),field_columns[:,-1]):
                        weight=np.exp(-.5*((rows[:,2]+delta-3.25)/.25)**2)*gate
                        for _ in range(6):
                            mass=np.bincount(atoms.ravel(),np.repeat(weight,2),minlength=n)
                            cap=np.maximum(mass,1.)
                            weight=weight/np.sqrt(cap[left]*cap[right])
                        stat.extend((weight,mass[left]*mass[right]))
                        levels.append(weight)
                stat.extend((levels[2]-levels[0],levels[5]-levels[3]))
            result[mask]=np.column_stack(stat)
    return np.c_[x,result]


if __name__=="__main__":
    from .ga_mixture import main
    import sys
    main(stability=True,competitive=True,orient_capacity=any(
        flag in sys.argv for flag in ("--orientation","--temporal")),
        temporal_capacity="--temporal" in sys.argv)


