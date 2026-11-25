"""GA688+: supervise multiple real-history windows from static and generated inputs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .ga_replica_teacher import trajectory_rows
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates
from .gc_history import past

OUT=Path("runs/ga688-multiwindow")


def targets(x,y,o,groups,dest=None,distribution=False,replica=1):
    dest = dest or OUT/"training-history-targets.npz"
    if dest.exists():
        with np.load(dest) as z:
            np.testing.assert_array_equal(y,z["label"])
            return z["multi"]
    jobs=[]
    for n in np.unique(groups):
        allpairs,times=trajectory_rows(n,replica)
        for origin in np.unique(o[groups==n]):
            ids=np.flatnonzero((groups==n)&(o==origin))
            pair=allpairs[times==origin]
            np.testing.assert_array_equal(pair[:,2],x[ids,0])
            jobs.append((ids,dict(protein=n,replica=replica,origin=np.full(len(ids),origin),
                                  ends=np.array([len(ids)]),pair=pair)))
    def make(job):
        idx, part=job
        observed=past(part)
        assert observed.shape==(len(idx),10)
        now=x[idx,0]
        series=np.c_[observed,now]
        check=series.mean(1)
        np.testing.assert_allclose(check,y[idx],rtol=0,atol=3e-6)
        data=np.c_[check-now,series[:,-2:].mean(1)-now,series[:,-3:].mean(1)-now,
                  series[:,-6:].mean(1)-now,series[:,:5].mean(1)-now,
                  series.std(1),np.mean(series<=3.5,axis=1)]
        if distribution:
            quantiles=np.quantile(series,[.05,.2,.35,.5,.65,.8,.95],axis=1).T-now[:,None]
            data=np.c_[check-now,quantiles]
        return idx,data
    output=np.empty((len(y),8 if distribution else 7),dtype=np.float32)
    with ThreadPoolExecutor(4) as executor:
        for i,(idx,data) in enumerate(executor.map(make,jobs),1):
            output[idx]=data
            if i%40==0:
                print("history preparation",i,len(jobs),flush=True)
    np.savez_compressed(dest,label=y,multi=output)
    return output


def main(ablations=False, curriculum=False, seed_check=False, stability=False):
    out = (Path("runs/ga749-window-curriculum") if curriculum else
           Path("runs/ga737-window-ablation") if ablations else OUT)
    if seed_check:
        out = Path("runs/ga789-window-seed-check")
    if stability:
        out = Path("runs/ga817-history-stability")
    out.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,groups),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(groups)&set(vg)
    target=targets(x,y,o,groups)
    centers=target.mean(0)
    spreads=target.std(0).clip(.01)
    label=torch.tensor((target-centers)/spreads,device="cuda",dtype=torch.float32)
    weights=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    if stability:
        with np.load("runs/ga508-replica-teacher/training-teacher.npz") as z:
            np.testing.assert_array_equal(y,z["original"])
            gap=np.abs(z["prior"][:,1]-target[:,0])
            reliable=z["support"][:,1]/(1+z["support"][:,1])
            gap=torch.tensor(gap*reliable,device="cuda",dtype=torch.float32)
    trials=[]
    scorespec=[(view,alpha,width) for view in ("level","full_response") for alpha in (0.,.15,.4,1.) for width in (64,128)]
    if ablations:
        scorespec=[("level",alpha,128,task) for task in ("short","early","spread","occupancy","short_early","short_spread")
                   for alpha in (.35,1.)]
    if curriculum:
        scorespec=[("level",alpha,width,mode)
                   for width in (96,128,192) for mode in ("early_only","late_only","ramp","decay")
                   for alpha in (.5,1.5)]
    if seed_check:
        scorespec=[("level",alpha,128,seed) for seed in (688,773,803,877,941,1009,1337,1901)
                   for alpha in (0.,1.)]
    if stability:
        scorespec=[("level",alpha,128,(strength,seed))
                   for strength in (-2.,-1.,1.,2.) for seed in (688,773,941)
                   for alpha in (0.,1.)]
    cached={}
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    for index in range(len(scorespec)):
        used={tuple(r["spec"]) for r in trials}
        if index==0:
            spec=scorespec[0]
        else:
            best=max(trials,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            spec=min((s for s in scorespec if s not in used),
                     key=lambda s:(sum(a!=b for a,b in zip(s,best["spec"])),
                                   sum(r["spec"][1]==s[1] for r in trials)))
        view,alpha,width,*task=spec
        auxiliary_columns = {"short": (1,2,3), "early": (4,), "spread": (5,), "occupancy": (6,),
                            "short_early": (1,2,3,4), "short_spread": (1,2,3,5)}
        cols = auxiliary_columns[task[0]] if ablations else (1,2,3,4,5,6)
        if view not in cached:
            aa,bb=features(x,d,g,view),features(v,e,h,view)
            mu,sd=aa.mean(0),aa.std(0).clip(.001)
            cached[view]=(torch.tensor(((aa-mu)/sd).clip(-10,10),device="cuda",dtype=torch.float32),
                          torch.tensor(((bb-mu)/sd).clip(-10,10),dtype=torch.float32),mu,sd)
        a,b,mu,sd=cached[view]
        torch.manual_seed(task[0][1] if stability else (task[0] if seed_check else 688))
        net=nn.Sequential(nn.Linear(a.shape[1],width),nn.LayerNorm(width),nn.SiLU(),
                          nn.Linear(width,width//2),nn.LayerNorm(width//2),nn.SiLU(),
                          nn.Linear(width//2,7)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.012)
        begin=perf_counter()
        for epoch in range(18):
            coefficient = alpha
            if curriculum:
                mode = task[0]
                multiplier = {"early_only": float(epoch<9), "late_only": float(epoch>=9),
                              "ramp": epoch/17., "decay": 1.-epoch/17.}[mode]
                coefficient *= multiplier
            for idx in torch.randperm(len(a),device="cuda").split(1536):
                error=(net(a[idx])-label[idx]).square()
                w=weights[idx]
                if stability:
                    w=w*torch.exp((task[0][0]*gap[idx]).clamp(-2.,2.))
                    w=w/w.mean()
                value=(error[:,0]+coefficient*error[:,cols].mean(1))*w
                opt.zero_grad(set_to_none=True)
                value.mean().backward()
                opt.step()
        torch.cuda.synchronize()
        cost=perf_counter()-begin
        cpu=deepcopy(net).cpu().eval()
        with torch.inference_mode():
            direct=cpu(b).numpy()[:,0]*spreads[0]+centers[0]+v[:,0]
        result=.65*parent+.35*direct
        metrics={k:estimates(result[m],t[m],base[m]) for k,m in masks.items()}
        model_id=f"GA{(817 if stability else (789 if seed_check else (749 if curriculum else (737 if ablations else 688))))+index}"
        torch.save(dict(state=cpu.state_dict(),center=mu,scale=sd,output_center=centers,
                        output_scale=spreads,spec=spec),out/f"{model_id}.pt")
        np.savez_compressed(out/f"{model_id}.npz",prediction=result,readout=direct)
        row=dict(id=model_id,spec=spec,metrics=metrics,fit_seconds=cost)
        trials.append(row)
        with (out/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(model_id,spec,metrics["all"]["residual_correlation"],flush=True)
        if metrics["all"]["residual_correlation"]>=.700:
            break
    best=max(trials,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (out/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    import sys
    main(ablations="--aux-sets" in sys.argv, curriculum="--curriculum" in sys.argv,
         seed_check="--seed-check" in sys.argv, stability="--stability" in sys.argv)







