"""GA345 samples directional local motions from elastic-network velocity modes."""
from pathlib import Path

import numpy as np
from openmm import unit
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

from .ga_native240 import SAMPLES, simulate
from .ga_static_offset import VALID


def prepare_modes(topology, initial):
    ca=[a for a in topology.atoms() if a.name=="CA"]
    xyz=np.asarray(initial.value_in_unit(unit.nanometer))[[a.index for a in ca]]
    count=len(ca)
    matrix=np.zeros((count,3,count,3),dtype=float)
    left,right=np.where(np.triu(squareform(pdist(xyz))<1.25,k=1))
    direction=xyz[right]-xyz[left]
    direction/=np.linalg.norm(direction,axis=1,keepdims=True).clip(1.e-8)
    block=direction[:,:,None]*direction[:,None,:]
    matrix[left,:,right,:]=-block
    matrix[right,:,left,:]=-block
    diagonal=np.zeros((count,3,3))
    np.add.at(diagonal,left,block)
    np.add.at(diagonal,right,block)
    i=np.arange(count)
    matrix[i,:,i,:]=diagonal
    spectrum,modes=eigh(matrix.reshape(3*count,3*count),subset_by_index=(0,min(20,3*count-1)))
    modes=modes[:,spectrum>1.e-6]
    if modes.shape[1]<2:
        raise RuntimeError("No usable collective modes")
    origins={a.residue.index:i for i,a in enumerate(ca)}
    order=np.array([origins.get(a.residue.index,-1) for a in topology.atoms()])
    cond=np.zeros((len(order),3,2),dtype=float)
    keep=order>=0
    cond[keep]=modes[:,:2].reshape(count,3,2)[order[keep]]
    cond/=np.sqrt(np.mean(cond**2,axis=(0,1))).clip(1.e-8)
    speed=unit.nanometer/unit.picosecond

    def kick(velocities,index):
        delta=cond[:,:,index//2]*(.075 if index%2==0 else -.075)
        return (velocities.value_in_unit(speed)+delta)*speed

    return kick


if __name__=="__main__":
    dest=Path("runs/ga345")
    dest.mkdir(parents=True,exist_ok=True)
    simulate(VALID,1,destination=dest,samples=tuple(t for t in SAMPLES if t<=25),
             branches=4,step_fs=2,minimize_iterations=60,velocity_hook=prepare_modes)
