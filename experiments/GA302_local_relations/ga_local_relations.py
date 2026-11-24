"""GA302: retain individual local relations around each ROD candidate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from tseries.data import TEST, cell, registry, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga_static_offset import TRAIN_GROUPS, VALID, cohort, estimates, extract, ridge

DEST=Path("runs/ga_local_relations")
K=24


def minicalc(path):
    spec=importlib.util.spec_from_file_location("ga302_minicalc_frames",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def relations(row,mc):
    reg=registry(ROOT,row["protein"])
    tid=trajectory(row["protein"],1)
    atoms=reg.atoms(tid).to_pydict()
    block=next(reg.blocks(tid,start=row["origin"],stop=row["origin"]+1))
    p=block.positions[0]
    box=cell(block.box[0])
    a,b=row["pair"][:,:2].astype(int).T
    endpoints=np.stack((a,b),axis=1)
    if box.ndim==1:
        wrapped=p%box
        tree=cKDTree(wrapped,boxsize=box)
        _,near=tree.query(wrapped[endpoints],k=min(K+2,len(p)))
    else:
        wrapped=(p@np.linalg.inv(box))%1.
        tree=cKDTree(wrapped,boxsize=1.)
        radius=7.*np.linalg.norm(np.linalg.inv(box),ord=2)
        neighborhood=tree.query_ball_point(wrapped,radius)
        near=np.empty((*endpoints.shape,K+2),dtype=int)
        for i in np.unique(endpoints):
            adjacent=np.asarray(neighborhood[i],dtype=int)
            distance=np.linalg.norm(_mic(p[adjacent]-p[i],box),axis=1)
            order=adjacent[np.argsort(distance,kind="stable")[:K+2]]
            near[endpoints==i]=np.pad(order,(0,K+2-len(order)),constant_values=i)
    step=_mic(p[b]-p[a],box)
    residue=np.asarray(atoms["residue_index"])
    ca={r:i for i,(n,r) in enumerate(zip(atoms["name"],residue)) if n=="CA"}
    bases=[]
    for pair,vector,neighbors in zip(endpoints,step,near):
        choices=[_mic(p[ca.get(residue[i],i)]-p[i],box) for i in pair]
        choices.extend(_mic(p[neighbors[0]]-p[pair[0]],box))
        axis=mc.unit(vector)
        second=mc.perpendicular(axis,choices)
        bases.append(mc.basis(axis,second))
    basis=np.asarray(bases)
    v=_mic(p[near]-p[a,None,None],box)
    tip=_mic(p[near]-p[b,None,None],box)
    state=mc.FrameState(np.zeros((v.size//3,3)),np.repeat(basis,2*near.shape[2],axis=0),())
    local=mc.to_local_displacement(state,v.reshape(-1,3)).reshape(v.shape)
    da,db=np.linalg.norm(v,axis=-1),np.linalg.norm(tip,axis=-1)
    mask=(near!=a[:,None,None])&(near!=b[:,None,None])&(np.minimum(da,db)<=7.)
    elem=np.asarray(atoms["element"])
    types=[elem[near]==k for k in ("C","N","O","S")]
    rel=[residue[near]==residue[a,None,None],residue[near]==residue[b,None,None]]
    role=np.broadcast_to(np.array([0,1])[None,:,None],da.shape)
    rivals=[]
    best=np.full(len(p),np.inf)
    for i,j in ((a,b),(b,a)):
        np.minimum.at(best,i,row["pair"][:,2])
        rivals.append(np.isfinite(best[near]) & (near!=i[:,None,None]))
        best.fill(np.inf)
    invariant=np.stack([da,db,role,*types,*rel,*rivals],axis=-1)
    def pack(x):
        return x.reshape(len(a),-1,x.shape[-1]).astype(np.float32)
    return dict(local=pack(np.concatenate((local,invariant),axis=-1)),
                global_=pack(np.concatenate((v,invariant),axis=-1)),
                radial=pack(invariant),mask=mask.reshape(len(a),-1))


def gather(rows,mc):
    with ThreadPoolExecutor(4) as pool:
        records=list(pool.map(lambda r:relations(r,mc),rows))
    return {k:np.concatenate([r[k] for r in records]) for k in records[0]}


class Net(nn.Module):
    def __init__(self,n,mode):
        super().__init__()
        self.mode=mode
        self.embed=nn.Sequential(nn.Linear(n,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.weight=nn.Linear(32,1)
        self.head=nn.Sequential(nn.Linear(64+8,48),nn.SiLU(),nn.Linear(48,1))

    def forward(self,x,mask,static):
        h=self.embed(x)
        if self.mode=="attention":
            w=self.weight(h).squeeze(-1).masked_fill(~mask,-1e5).softmax(1)
            mean=(h*w.unsqueeze(-1)).sum(1)
        else:
            mean=(h*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        maxval=h.masked_fill(~mask[:,:,None],-1e5).amax(1)
        maxval=torch.where(mask.any(1,keepdim=True),maxval,0.)
        return self.head(torch.cat((mean,maxval,static),1)).flatten()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--minicalc",type=Path,default=Path("D:/MiniCalc/local_frames.py"))
    args=parser.parse_args()
    mc=minicalc(args.minicalc)
    torch.set_num_threads(3)
    DEST.mkdir(parents=True,exist_ok=True)
    tr=[row for group in TRAIN_GROUPS for row in cohort(group,1)]
    va=cohort(VALID,3)
    te=cohort(TEST,3)+cohort(("4ftf_A","3msw_A"),3)+cohort(("3wx4_A","4wlr_B"),2)
    yt,xt=extract(tr)
    yv,xv=extract(va)
    ye,xe=extract(te)
    train_protein=np.concatenate([np.full(len(r["y"]),r["protein"]) for r in tr])
    oof=np.empty(len(yt),dtype=np.float32)
    for name in np.unique(train_protein):
        mask=train_protein==name
        oof[mask]=ridge(xt["compact"][~mask],yt[~mask]).predict(xt["compact"][mask])
    reference=ridge(xt["compact"],yt)
    pv,pe=[reference.predict(x["compact"]) for x in (xv,xe)]
    print("Building local relational tensors",flush=True)
    a,b,c=[gather(r,mc) for r in (tr,va,te)]
    mean=xt["compact"].mean(0)
    scale=np.maximum(xt["compact"].std(0),.001)
    st,sv,se=[torch.from_numpy(((x["compact"]-mean)/scale).astype(np.float32))
              for x in (xt,xv,xe)]
    y=torch.from_numpy((yt-oof).astype(np.float32)).cuda()
    sm=st.cuda()
    tm=torch.from_numpy(a["mask"]).cuda()
    trials={}
    for kind in ("radial","global_","local"):
        x=torch.from_numpy(a[kind]).cuda()
        for mode in ("pool","attention"):
            for loss in ("squared","robust"):
                torch.manual_seed(302)
                model=Net(x.shape[-1],mode).cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
                for _ in range(9):
                    for idx in torch.randperm(len(y),device="cuda").split(256):
                        p=model(x[idx],tm[idx],sm[idx])
                        error=(p-y[idx]).square().mean() if loss=="squared" else nn.functional.smooth_l1_loss(p,y[idx],beta=.35)
                        opt.zero_grad(set_to_none=True)
                        error.backward()
                        opt.step()
                assert next(model.parameters()).grad.is_cuda
                model.cpu().eval()
                key=f"{kind}-{mode}-{loss}"
                with torch.inference_mode():
                    p=model(torch.from_numpy(b[kind]),torch.from_numpy(b["mask"]),sv).numpy()
                trials[key]=dict(validation=estimates(p+pv,yv,pv))
                torch.save(model.state_dict(),DEST/f"{key}.pt")
                print(key,trials[key]["validation"],flush=True)
    # Select the requested local representation without inspecting the confirmation targets.
    candidates=[k for k in trials if k.startswith("local-")]
    winner=max(candidates,key=lambda k:trials[k]["validation"]["residual_correlation"])
    kind,mode,_=winner.split("-")
    model=Net(c[kind].shape[-1],mode).eval()
    model.load_state_dict(torch.load(DEST/f"{winner}.pt",map_location="cpu",weights_only=True))
    with torch.inference_mode():
        prediction=model(torch.from_numpy(c[kind]),torch.from_numpy(c["mask"]),se).numpy()+pe
    scaler,regressor=reference.steps[0][1],reference.steps[1][1]
    np.savez(DEST/"readout_inputs.npz",mean=mean,scale=scale,ridge_mean=scaler.mean_,
             ridge_scale=scaler.scale_,ridge_weight=regressor.coef_,ridge_bias=regressor.intercept_)
    np.savez(DEST/"evaluation.npz",pred=prediction,target=ye,reference=pe)
    record(DEST/"results.json",dict(selected=winner,validation=VALID,
        training=list(dict.fromkeys(train_protein)),evaluation=list(dict.fromkeys(r["protein"] for r in te)),
        samples=[len(yt),len(yv),len(ye)],trials=trials,test=estimates(prediction,ye,pe),
        ridge=estimates(pe,ye,pe),minicalc_sha256=hashlib.sha256(args.minicalc.read_bytes()).hexdigest()))


if __name__=="__main__":
    main()
