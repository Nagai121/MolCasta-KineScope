"""Short screened-heavy dynamics, distance propagation, and frozen T1 comparison."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
import torch
from torch.nn import functional as F

from tseries.data import TEST, cell, registry, trajectory
from tseries.hb_explicit_hydrogen import source_plans
from tseries.hb_preprocessing_screened import charge_map, paths
from tseries.hb_single_structure import OUT, ROOT
from tseries.hb_validation10 import panel10
from .explore import record, score
from .frontends import data, design
from .ga300 import Generator, ARCHITECTURES

DEST=Path("runs/ga_hybrid300")
VIEWS=("distance","static","heavy_early","heavy_late","heavy_full","heavy_geometry",
       "distance_rod","rod_competition","heavy_and_rod","heavy_seeded_rod",
       "heavy_seeded_competition","all")
LOSSES=("binary","brier","focal","pairwise","local")
BASE="static"


def summary(d,tracks):
    delta=tracks-d[:,None,None]
    return np.column_stack((delta.mean((1,2)),delta.std((1,2)),delta.min((1,2)),
                            delta[:,:,-1].mean(1),(tracks<=3.5).mean((1,2)),
                            (tracks<=4.).mean((1,2)),
                            (tracks<=3.5).any(2).mean(1),delta.mean(2).std(1)))


def rod(d, drift, diffusion, endpoint):
    """Propagate a candidate distance distribution by reduced-time Gaussian updates."""
    a,b=endpoint
    g=drift.copy()
    result=[]
    competition=[]
    for t in (.25,.5,1.,2.):
        p=ndtr((3.5-d-t*g)/np.sqrt(.025+t*diffusion))
        wider=ndtr((4.-d-t*g)/np.sqrt(.025+t*diffusion))
        result.extend((g,np.sqrt(.025+t*diffusion),p,wider))
        occupancies=[]
        for k in (a,b):
            total=np.bincount(k,weights=p)[k]-p
            occupancies.append(total)
            competition.extend((p/(.08+p+total),total))
        # Only the shared-end competition modifies the next drift.
        g=drift+.055*np.log1p(sum(occupancies))
    return np.column_stack(result),np.column_stack(competition)


def frame_features(pair,geometry,tracks):
    d=pair[:,2]
    a,b=pair[:,:2].astype(np.intp).T
    early=summary(d,tracks[:,:,:2])
    late=summary(d,tracks[:,:,4:])
    full=summary(d,tracks)
    local=geometry[:,8:14]
    drift=(local[:,3]-local[:,0]).clip(-.5,.5)
    spread=np.clip(local[:,4]*.25,.003,.2)
    r,c=rod(d,drift,spread,(a,b))
    g=.5*drift+.5*full[:,3]
    rs,cs=rod(d,g,np.clip(.5*spread+.5*full[:,1]**2,.003,.25),(a,b))
    return (early,late,full,r,c,rs,cs)


def dynamics(part,static):
    reg=registry(ROOT,part["protein"])
    tid=trajectory(part["protein"],part["replica"])
    key=hashlib.sha256(part["pair"].tobytes()+part["origin"].tobytes()
                       +Path(__file__).read_bytes()).hexdigest()[:16]
    path=DEST/f"cache-{part['protein']}-{part['replica']}-{key}.npz"
    if path.exists():
        with np.load(path) as z:
            return tuple(z[str(i)] for i in range(7))
    atoms=reg.atoms(tid).to_pydict()
    charges,mapped=charge_map(atoms,source_plans(part["protein"])[f"R{part['replica']}"])
    rows=[]
    for a,b in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        if a==b:
            continue
        origin=int(part["origin"][a])
        block=next(reg.blocks(tid,start=origin,stop=origin+1,chunk_frames=1))
        pair=part["pair"][a:b]
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(
            f"{tid}/{origin}".encode()).digest()[:8],"little"))
        tracks=paths(block.positions[0],cell(block.box[0]),pair,atoms,charges,rng,8.)
        rows.append(frame_features(pair,static[a:b],tracks))
    result=tuple(np.concatenate(x).astype(np.float32) for x in zip(*rows))
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez(f,**{str(i):v for i,v in enumerate(result)},mapped=mapped)
    path.with_suffix(".tmp").replace(path)
    return result


def prepare(names):
    started=time.perf_counter()
    parts,old,base=data(names)
    breaks=np.cumsum([len(p["y"]) for p in parts])[:-1]
    blocks=np.split(old["combined"],breaks)
    with ThreadPoolExecutor(5) as pool:
        files=list(pool.map(dynamics,parts,blocks))
    e,l,h,r,c,s,k=[np.concatenate(x) for x in zip(*files)]
    d=old["distance"][:,:1]
    geo=old["existing_static"]
    views=dict(zip(VIEWS,(d,geo,np.c_[d,e],np.c_[d,l],np.c_[d,h],
               np.c_[h,geo],np.c_[d,r],np.c_[d,r,c],np.c_[h,r],
               np.c_[h,s],np.c_[h,s,k],np.c_[old["combined"],h,r,c,s,k])))
    return parts,views,base,time.perf_counter()-started


def fit(x,y,architecture,loss,seed):
    torch.manual_seed(seed)
    mean=torch.from_numpy(x.mean(0))
    scale=torch.from_numpy(np.maximum(x.std(0),.001))
    tx=((torch.from_numpy(x)-mean)/scale).clamp(-8,8).cuda()
    offset=torch.from_numpy(x[:,0].copy()).cuda()
    target=torch.from_numpy(y.astype(np.float32)).cuda()
    net=Generator(tx.shape[1],architecture).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.0015)
    for _ in range(4):
        for i in torch.randperm(len(tx),device=tx.device).split(4096):
            pred=offset[i]+.35*net(tx[i])[:,0].tanh()
            truth=target[i]
            bce=F.binary_cross_entropy_with_logits(pred,truth,reduction="none")
            p=pred.sigmoid()
            value=bce.mean()
            if loss=="brier":
                value=value+(p-truth).square().mean()
            elif loss=="focal":
                value=(bce*(.35+(p-truth).abs().sqrt())).mean()
            elif loss=="pairwise":
                pos,neg=pred[truth>.5],pred[truth<=.5]
                n=min(len(pos),len(neg))
                if n:
                    value=value+.018*F.softplus(neg[:n]-pos[:n]).mean()
            elif loss=="local":
                value=(bce*(1+.5*(offset[i]>-2.8))).mean()
            opt.zero_grad(set_to_none=True)
            value.backward()
            opt.step()
    assert net.encode.weight.grad.is_cuda
    return net.cpu().eval(),mean,scale


def predict(net,mean,scale,x):
    z=((torch.from_numpy(x)-mean)/scale).clamp(-8,8)
    with torch.inference_mode():
        c=torch.cat([net(b)[:,0].tanh() for b in z.split(32768)])
        return (torch.from_numpy(x[:,0])+.35*c).sigmoid().numpy()


def save_report(path,report):
    for i in range(10):
        try:
            record(path,report)
            return
        except PermissionError:
            if i==9:
                raise
            time.sleep(.1)


def main():
    torch.set_num_threads(3)
    DEST.mkdir(parents=True,exist_ok=True)
    names=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    validation=panel10()
    assert not set(names)&set(validation+TEST)
    tr,a,bt,t=prepare(names)
    print("Prepared training dynamics",t,flush=True)
    va,b,bv,v=prepare(validation)
    print("Prepared validation dynamics",v,flush=True)
    y=np.concatenate([p["y"] for p in tr])
    idx=np.sort(np.random.default_rng(731).choice(len(y),min(len(y),65536),replace=False))
    log=DEST/"results.json"
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if log.exists():
        report=json.loads(log.read_bytes())
        if report["source_sha256"]!=sha:
            raise ValueError("Experiment source changed")
    else:
        report=dict(train=names,validation=validation,source_sha256=sha,training_rows=len(idx),
                    cpu_features_seconds=[t,v],baseline=score(va,bv),trials={})
    for i,view in enumerate(VIEWS):
        x=design(a[view],bt)[idx]
        q=design(b[view],bv)
        for j,loss in enumerate(LOSSES):
            for k,architecture in enumerate(ARCHITECTURES):
                number=i*25+j*5+k+1
                ident=f"hybrid-{number:03}"
                if ident in report["trials"]:
                    continue
                started=time.perf_counter()
                net,mean,scale=fit(x,y[idx],architecture,loss,700+number)
                duration=time.perf_counter()-started
                p=predict(net,mean,scale,q)
                ckpt=DEST/f"{ident}.pt"
                torch.save(dict(net=net.state_dict(),mean=mean,scale=scale,view=view,
                                loss=loss,architecture=architecture),ckpt)
                metrics=score(va,p)
                report["trials"][ident]=dict(view=view,loss=loss,architecture=architecture,
                    metrics=metrics,checkpoint=ckpt.name,fit_seconds=duration,
                    sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())
                save_report(log,report)
                print(ident,view,loss,architecture,metrics["auprc"],flush=True)
    best=max(report["trials"],key=lambda n:report["trials"][n]["metrics"]["auprc"])
    te,c,be,seconds=prepare(TEST)
    saved=torch.load(DEST/report["trials"][best]["checkpoint"],map_location="cpu",weights_only=True)
    z=design(c[saved["view"]],be)
    net=Generator(z.shape[1],saved["architecture"]).eval()
    net.load_state_dict(saved["net"])
    p=predict(net,saved["mean"],saved["scale"],z)
    record(DEST/"test.json",dict(selected=best,proteins=TEST,baseline=score(te,be),
                               corrected=score(te,p),cpu_features_seconds=seconds))


if __name__=="__main__":
    main()
