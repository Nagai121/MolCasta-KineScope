"""GA1104+: compare directed atom competition on protein-held folds."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch
from torch import nn

from .ga_competitive_capacity import competition
from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga1104-grouped-capacity")


def main(fold_ensemble=False, expanded=False):
    torch.set_num_threads(3)
    out=Path("runs/ga1164-fold-ensemble") if fold_ensemble else OUT
    if expanded:
        out=Path("runs/ga1200-expanded-crossfold")
    out.mkdir(exist_ok=True)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    label=targets(x,y,o,p)[:,0]

    with np.load("runs/ga508-replica-teacher/training-teacher.npz") as z:
        np.testing.assert_array_equal(z["original"],y)
        gap=np.abs(z["prior"][:,1]-label)*z["support"][:,1]/(1+z["support"][:,1])
    if expanded:
        with np.load("runs/ga1029-later-expansion/train-extra.npz") as extra:
            assert not set(p)&set(extra["proteins"])
            assert not set(vg)&set(extra["proteins"])
            x,d,g,y,o,p,label=[np.concatenate((a,extra[key][:,0] if key=="label" else extra[key]))
                for a,key in zip((x,d,g,y,o,p,label),("x","d","g","y","origins","proteins","label"))]
        gap=np.pad(gap,(0,len(y)-len(gap)))
    x0=x[:,0]
    first_id=1200 if expanded else (1164 if fold_ensemble else 1104)
    views={
        "undirected":(features(competition(x,d,o,p),d,g,"level"),
                      features(competition(v,e,vo,vg),e,h,"level")),
        "direction":(features(competition(x,d,o,p,oriented=True),d,g,"level"),
                     features(competition(v,e,vo,vg,oriented=True),e,h,"level")),
    }
    cv=list(GroupKFold(4).split(x,y,p))
    reference=np.empty_like(y)
    for tr,va in cv:
        reference[va]=Ridge(alpha=10.).fit(x[tr,:12],y[tr]).predict(x[va,:12])

    def run(tr,val,start,spec):
        view,seed,robust=spec
        source=views[view][0][tr]
        center,spread=source.mean(0),source.std(0).clip(.001)
        mu,sd=label[tr].mean(),label[tr].std().clip(.01)
        a=torch.tensor(((source-center)/spread).clip(-10,10),device="cuda",dtype=torch.float32)
        b=torch.tensor(((val-center)/spread).clip(-10,10),dtype=torch.float32)
        yy=torch.tensor((label[tr]-mu)/sd,device="cuda",dtype=torch.float32)
        w=torch.tensor(np.where(o[tr]==49,1.,4.)*np.exp(np.clip(robust*gap[tr],-2.,2.)),
                       device="cuda",dtype=torch.float32)
        torch.manual_seed(seed)
        k=5
        model=nn.Sequential(nn.Linear(a.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                            nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,3*k)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.01)
        for _ in range(18):
            for j in torch.randperm(len(tr),device="cuda").split(1536):
                z=model(a[j])
                logits=z[:,:k].log_softmax(1)
                mean=z[:,k:2*k]
                std=nn.functional.softplus(z[:,2*k:])+.15
                log_prob=torch.logsumexp(logits-.5*((yy[j,None]-mean)/std).square()-std.log(),dim=1)
                mse=((logits.exp()*mean).sum(1)-yy[j]).square()
                weight=w[j]
                loss=((-log_prob+.5*mse)*weight/weight.mean()).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net=model.cpu().eval()
        with torch.inference_mode():
            pred=net(b)
            predicted=(pred[:,:k].softmax(1)*pred[:,k:2*k]).sum(1).numpy()*sd+mu+start
        return predicted,dict(weights=net.state_dict(),center=center,scale=spread,target_mean=mu,target_scale=sd,spec=spec)

    rows=[]
    fold_development={}
    for idx,(view,seed,robust) in enumerate((view,seed,robust)
           for seed in (688,872,1009) for view in views for robust in (0.,-2.)):
        spec=(view,seed,robust)
        prediction=np.empty_like(y)
        external=[]
        for fold,(tr,va) in enumerate(cv):
            if fold_ensemble:
                query=np.concatenate((views[view][0][va],views[view][1]))
                positions=np.concatenate((x0[va],v[:,0]))
                values,state=run(tr,query,positions,spec)
                prediction[va]=values[:len(va)]
                external.append(values[len(va):])
                torch.save(state,out/f"GA{first_id+idx}-fold{fold}.pt")
            else:
                prediction[va],_=run(tr,views[view][0][va],x0[va],spec)
        fold=[estimates(prediction[va],y[va],reference[va])["residual_correlation"] for _,va in cv]
        model_id=f"GA{first_id+idx}"
        row=dict(id=model_id,spec=spec,folds=fold,cv_mean=float(np.mean(fold)))
        rows.append(row)
        if external:
            fold_development[model_id]=np.mean(external,axis=0)
        np.savez_compressed(out/f"{model_id}-cv.npz",prediction=prediction,baseline=reference,target=y,proteins=p)
        with (out/"trials.jsonl").open("a",encoding="utf-8") as file:
            file.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
    groups={}
    for row in rows:
        view,seed,robust=row["spec"]
        groups.setdefault((view,robust),[]).append(row)
    best=max(groups,key=lambda k:np.mean([r["cv_mean"] for r in groups[k]]))
    direct=[]
    for row in groups[best]:
        if fold_ensemble:
            direct.append(fold_development[row["id"]])
        else:
            pred,state=run(np.arange(len(y)),views[best[0]][1],v[:,0],row["spec"])
            direct.append(pred)
            torch.save(state,out/f"{row['id']}.pt")
    estimate=.65*parent+.35*np.mean(direct,axis=0)
    summary=dict(best_family=best,cv={str(k):float(np.mean([r["cv_mean"] for r in v])) for k,v in groups.items()},
                 development={k:estimates(estimate[m],t[m],base[m]) for k,m in
                              (("all",vo>=0),("first",vo==49),("later",vo!=49))})
    np.savez_compressed(out/"development.npz",prediction=estimate)
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(summary,flush=True)


if __name__=="__main__":
    import sys
    main(fold_ensemble="--fold-ensemble" in sys.argv,expanded="--expanded" in sys.argv)


