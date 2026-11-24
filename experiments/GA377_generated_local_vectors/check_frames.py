from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tseries.data import registry, trajectory
from tseries.hb_single_structure import ROOT
from t2i.ga377_score import minicalc, vector_features
from t2i.ga379 import moving
name="3mx7_A"
mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
atoms=registry(ROOT,name).atoms(trajectory(name,1)).to_pydict()
res=np.array(atoms["residue_index"])
ca=np.flatnonzero(np.array(atoms["name"])=="CA")
rot=Rotation.from_rotvec(np.array([.39,.21,-.25])).as_matrix()
with np.load("runs/ga377/validation/3mx7_A-49.npz") as z:
    initial,xyz,p=z["origin_xyz"],z["generated_xyz"],z["pair"]
for f,args in ((vector_features,(res,ca,mc)),(moving,(res,atoms["name"],mc))):
    expected=f(initial,xyz,p,*args)
    shifted=f(initial@rot+3,xyz@rot+3,p,*args)
    np.testing.assert_allclose(expected,shifted,atol=2e-5,rtol=1e-5)
    print(f.__name__,float(np.max(np.abs(expected-shifted))),flush=True)
