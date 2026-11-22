import sys, json
sys.path.insert(0,"/home/hatch/workspace/minitopol-read-opt/src")
import minitopol
import numpy as np, pandas as pd, zarr
from pathlib import Path

BASE=Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
OUT=Path("/home/hatch/workspace/kinescope/work/data_cache")
PROTEINS=("1bgf_A","1ifg_A","1j8r_A","2b0a_A","1mix_A","2qt4_A")
HORIZON_FRAMES=(20,60,110,310,510,910)

DONOR_N = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"},
           "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}}
DONOR_O = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ACCEPTOR_O = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
              "ASN": {"OD1"}, "GLN": {"OE1"}, "SER": {"OG"},
              "THR": {"OG1"}, "TYR": {"OH"}}
ANCHOR = {("LYS", "NZ"): "CE", ("ARG", "NE"): "CZ",
          ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
          ("ASN", "ND2"): "CG", ("GLN", "NE2"): "CD",
          ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CE1",
          ("TRP", "NE1"): "CE2", ("SER", "OG"): "CB",
          ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
          ("ASP", "OD1"): "CG", ("ASP", "OD2"): "CG",
          ("GLU", "OE1"): "CD", ("GLU", "OE2"): "CD",
          ("ASN", "OD1"): "CG", ("GLN", "OE1"): "CD"}

def distance(v, box):
    box=np.asarray(box,dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths=np.diag(box); v=v-lengths*np.rint(v/lengths)
    else:
        inv=np.linalg.inv(box); fr=v@inv; fr-=np.rint(fr); v=fr@box
    return np.linalg.norm(v,axis=-1)

def layout(df):
    order=list(dict.fromkeys(df["residue_index"].tolist()))
    maps=[]; names=[]
    for r in order:
        idx=np.flatnonzero(df["residue_index"].values==r)
        nm={df["name"].values[i]: int(i) for i in idx}
        if not {"N","CA","C","O"}.issubset(nm): continue
        maps.append(nm); names.append(df["residue_name"].values[idx[0]])
    donors=[]; acceptors=[]
    for k,(m,r) in enumerate(zip(maps,names)):
        if r!="PRO": donors.append((m["N"],k,0,m["CA"]))
        for name in DONOR_N.get(r,()):
            if name in m: donors.append((m[name],k,1,m.get(ANCHOR.get((r,name)),m["CA"])))
        for name in DONOR_O.get(r,()):
            if name in m: donors.append((m[name],k,2,m.get(ANCHOR.get((r,name)),m["CA"])))
        acceptors.append((m["O"],k,0,m["C"]))
        if "OXT" in m: acceptors.append((m["OXT"],k,0,m["C"]))
        for name in ACCEPTOR_O.get(r,()):
            if name in m: acceptors.append((m[name],k,1,m.get(ANCHOR.get((r,name)),m["CA"])))
    return maps,names,donors,acceptors

def pair_series(pos,box,a,b,frames):
    return np.stack([distance(pos[f,a,:]-pos[f,b,:], box[f]) for f in frames])

def build(prot,rep):
    tid=f"atlas-{prot}-analysis-R{rep}"
    d=BASE/tid
    df=pd.read_parquet(d/"atoms.parquet")
    z=zarr.open_group(str(d/"A.zarr"),mode='r')
    pos=z['positions'][:].astype(np.float64)
    box=z['box'][:].astype(np.float64)
    times=z['time'][:]
    assert pos.shape[0]==1001, pos.shape
    maps,names,donors,acceptors=layout(df)
    n=len(maps); ca=np.array([m["CA"] for m in maps])
    sep=np.abs(np.subtract.outer(np.arange(n),np.arange(n)))
    t=10
    ca_d=distance(pos[t,ca[:,None],:]-pos[t,ca[None,:],:],box[t])
    da=np.asarray(donors); ac=np.asarray(acceptors)
    d_at,d_res,d_kind,_=da.T; a_at,a_res,a_kind,_=ac.T
    dd=distance(pos[t,d_at[:,None],:]-pos[t,a_at[None,:],:],box[t])
    ii,jj=np.where((dd>3.5)&(dd<=6.0)&(d_res[:,None]!=a_res[None,:]))
    a=d_at[ii]; b=a_at[jj]; dri=d_res[ii]; arj=a_res[jj]
    frames=list(range(11))
    hist=pair_series(pos,box,a,b,frames)
    x=np.stack([hist[-1],ca_d[dri,arj],np.log1p(sep[dri,arj]),(d_kind[ii]==0),(a_kind[jj]==0),(d_kind[ii]==2),(a_kind[jj]==1)],axis=1).astype(np.float32)
    y=np.column_stack([(pair_series(pos,box,a,b,[f])[0]<=3.5) for f in HORIZON_FRAMES]).astype(np.uint8)
    # K04 series: residue heavy-atom min distance and CA distance
    heavy=[]
    for m in maps:
        heavy.append(np.asarray([v for key,v in m.items() if not key.startswith("H") and not key[0].isdigit()],dtype=int))
    # per candidate residue pair min distance per history frame
    hist_min=np.empty_like(hist); hist_ca=np.empty_like(hist)
    ca_series=pair_series(pos,box,ca[dri],ca[arj],frames)
    hist_ca[:]=ca_series
    for f_i,f in enumerate(frames):
        # group candidates by residue pair to reuse? candidates may repeat residue pairs rarely; do loop chunked
        vals=np.empty(len(a))
        for c in range(len(a)):
            ia=heavy[dri[c]]; ib=heavy[arj[c]]
            dm=distance(pos[f,ia[:,None],:]-pos[f,ib[None,:],:],box[f])
            vals[c]=dm.min()
        hist_min[f_i]=vals
    out=dict(X=x,hist=hist,hist_min=hist_min,hist_ca=hist_ca,y=y,
             donor_atom=a,acceptor_atom=b,donor_res=dri,acceptor_res=arj,
             protein=prot,replica=rep)
    np.savez_compressed(OUT/f"{prot}_R{rep}.npz",**out)
    print(tid,"atoms",pos.shape[1],"res",n,"cand",len(a),"pos per horizon",y.sum(0),flush=True)
    return len(a)

for p in PROTEINS:
    for r in (1,2,3):
        build(p,r)
print("done")
