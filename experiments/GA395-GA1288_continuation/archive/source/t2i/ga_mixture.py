"""GA872+: fit multimodal conditional distributions of pair residuals."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_multiwindow import targets
from .ga_sequential50 import features, pool
from .ga_static_offset import estimates

OUT=Path("runs/ga872-mixture")


def main(stability=False, replica2=False, seedset=False, new_proteins=False, domain_robust=False, competitive=False, capacity_audit=False, late_expansion=False, orient_capacity=False, orientation_audit=False, temporal_capacity=False, reliability_width=False, late_combination=False, road_relaxation=False, branch_competition=False, wide_competition=False, motif_heads=False, field_augmentation=False, scale_146=False):
    first_id, out_dir = (890, Path("runs/ga890-robust-mixture")) if stability else (872, OUT)
    for enabled, model_id, folder in (
        (replica2, 909, "ga909-r2-augmentation"),
        (seedset, 929, "ga929-mixture-seeds"),
        (new_proteins, 963, "ga963-protein-expansion"),
        (domain_robust, 971, "ga971-domain-robust"),
        (competitive, 1001, "ga1001-competitive-capacity"),
        (capacity_audit, 1013, "ga1013-competition-audit"),
        (late_expansion, 1029, "ga1029-later-expansion"),
        (orient_capacity, 1037, "ga1037-oriented-capacity"),
        (orientation_audit, 1049, "ga1049-orientation-audit"),
        (temporal_capacity, 1065, "ga1065-temporal-capacity"),
        (reliability_width, 1077, "ga1077-reliability-width"),
        (late_combination, 1095, "ga1095-late-combination"),
        (road_relaxation, 1116, "ga1116-rod-relaxation"),
        (branch_competition, 1146, "ga1146-branch-capacity"),
        (wide_competition, 1188, "ga1188-wide-capacity"),
        (motif_heads, 1212, "ga1212-motif-heads"),
        (field_augmentation, 1265, "ga1265-field-augmentation"),
        (scale_146, 1289, "ga1289-scale-146"),
    ):
        if enabled:
            first_id, out_dir = model_id, Path("runs") / folder
    out_dir.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    assert not set(p)&set(vg)
    label=targets(x,y,o,p)
    r1_count=len(y)
    if road_relaxation:
        from .ga_rod_relaxation import relaxation
        relaxation_x=relaxation(x,d,o,p)
        relaxation_v=relaxation(v,e,vo,vg)
    if branch_competition:
        from .ga_branch_capacity import expanded
        branch_x,branch_v=expanded(x,v,o,vo,p,vg)
    if motif_heads:
        from .ga_motif_heads import MotifOutput,routing
        mot_train,mot_dev=x,v
    if competitive or capacity_audit or reliability_width:
        from .ga_competitive_capacity import competition
        extra_x,extra_v=competition(x,d,o,p,oriented=orient_capacity or branch_competition,temporal=temporal_capacity),competition(v,e,vo,vg,oriented=orient_capacity or branch_competition,temporal=temporal_capacity)
        if competitive:
            x,v=extra_x,extra_v
    if orientation_audit:
        from .ga_competitive_capacity import competition
        extra_x,extra_v=competition(x,d,o,p),competition(v,e,vo,vg)
        angle_x,angle_v=competition(x,d,o,p,oriented=True),competition(v,e,vo,vg,oriented=True)
    if reliability_width:
        angle_x,angle_v=competition(x,d,o,p,oriented=True),competition(v,e,vo,vg,oriented=True)
    if late_combination:
        with np.load("runs/ga1029-later-expansion/train-extra.npz") as extra:
            assert not set(extra["proteins"])&set(vg)
            x,d,g,y,o,p,label=[np.concatenate([a,extra[key]]) for a,key in
                zip((x,d,g,y,o,p,label),("x","d","g","y","origins","proteins","label"))]
        from .ga_competitive_capacity import competition
        angle_x,angle_v=competition(x,d,o,p,oriented=True),competition(v,e,vo,vg,oriented=True)
    if wide_competition:
        from .ga_wide_competition import wide_context
        x=np.c_[x,wide_context(x,o,p)]
        v=np.c_[v,wide_context(v,vo,vg)]
    if branch_competition:
        x,v=np.c_[x,branch_x],np.c_[v,branch_v]
    if road_relaxation:
        x,v=np.c_[x,relaxation_x[:,-24:]],np.c_[v,relaxation_v[:,-24:]]
    if replica2 or new_proteins or late_expansion or scale_146:
        entry="train-extra.npz" if (new_proteins or late_expansion or scale_146) else "train-r2.npz"
        with np.load(out_dir/entry) as r2:
            assert not set(r2["proteins"])&set(vg)
            x,d,g,y,o,label=[np.concatenate([a,r2[key]]) for a,key in
                zip((x,d,g,y,o,label),("x","d","g","y","origins","label"))]
    lm,ls=label.mean(0),label.std(0).clip(.01)
    z=torch.tensor((label-lm)/ls,device="cuda",dtype=torch.float32)
    weight=torch.tensor(np.where(o==49,1.,4.),device="cuda",dtype=torch.float32)
    if stability:
        with np.load("runs/ga508-replica-teacher/training-teacher.npz") as q:
            np.testing.assert_array_equal(q["original"],y[:r1_count])
            confidence=np.abs(q["prior"][:,1]-label[:r1_count,0])*q["support"][:,1]/(1+q["support"][:,1])
            confidence=np.pad(confidence,(0,len(y)-r1_count))
            if reliability_width:
                gaps=np.abs(q["prior"]-label[:r1_count,0:1])*q["support"]/(1+q["support"])
                all_confidence=torch.tensor(gaps,device="cuda",dtype=torch.float32)
            confidence=torch.tensor(confidence,device="cuda",dtype=torch.float32)
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    options=[(view,k,aux) for view in ("level","full_response") for k in (1,3,5) for aux in (0.,.4,1.)]
    if stability:
        options=[("level",count,1.,factor,seed)
                 for count in (3,5,8) for factor in (-.5,-1.,-2.) for seed in (688,872)]
    if replica2 or new_proteins or late_expansion or scale_146:
        options=[("level",count,1.,factor,seed)
                 for count in (3,5) for factor in (-1.,-2.) for seed in (688,872)]
    if seedset:
        options=[("level",3,1.,factor,seed) for seed in (688,773,803,872,941,1009,1337,1901)
                 for factor in (0.,-2.)]
    if domain_robust:
        options=[("level",3,1.,factor,seed,temp)
                 for factor in (-1.,-2.) for seed in (688,872,1009) for temp in (.15,.5,1.)]
        groups,numbers,counts=np.unique(p,return_inverse=True,return_counts=True)
        domains=torch.tensor(numbers,device="cuda")
        domain_counts=torch.tensor(counts,device="cuda",dtype=torch.float32)
    if competitive:
        options=[("level",count,aux,-2.,seed) for count in (1,3,5)
                 for aux in (0.,1.) for seed in (688,872)]
    if capacity_audit:
        options=[(view,5,1.,-2.,seed) for seed in (688,773,872,941,1009,1337,1901,2026)
                 for view in ("level","capacity")]
    if orientation_audit:
        options=[(view,5,0.,-2.,seed) for seed in (688,773,872,941,1009,1337,1901,2026)
                 for view in ("capacity","angles")]
    if reliability_width:
        options=[(view,5,0.,-2.,seed,col) for view in ("capacity","angles") for col in (0,1,2)
                 for seed in (688,872,1009)]
    if late_combination:
        prior=json.loads(Path("runs/ga1029-later-expansion/summary.json").read_bytes())
        score=prior["metrics"]["all"]["residual_correlation"]
        if score>=.640:
            options=[(view,k,1.,-2.,seed,1.) for view in ("level","angles")
                     for k in (3,5) for seed in (688,872)]
        else:
            options=[("angles",5,0.,-2.,seed,weight) for seed in (688,872,1009) for weight in (.3,.6,1.)]
        (out_dir/"decision.json").write_text(json.dumps(
            {"from":prior["id"],"development_score":score,"options":options},indent=2),encoding="utf-8")
    if branch_competition:
        options=[("level",count,aux,-2.,seed) for count in (3,5,8)
                 for aux in (0.,1.) for seed in (688,872,1009)]
    if motif_heads:
        options=[("level",count,0.,-2.,seed,kind) for count in (3,5)
                 for seed in (688,872) for kind in ("chemistry","separation","distance")]
    if field_augmentation:
        options=[("level",5,0.,-2.,seed,swapping) for seed in (688,872,1009)
                 for swapping in (.10,.25,.40,.50)]
        def swap_fields(t):
            width=(t.shape[1]-4)//3
            assert t.shape[1]==3*width+4
            return np.c_[t[:,width:2*width],t[:,:width],-t[:,2*width:3*width],t[:,[-3,-4,-1,-2]]]
        swapped_train=swap_fields(d)
    cache={}
    swap_cache={}
    rows=[]
    for i in range(len(options)):
        if i==0:
            spec=options[0]
        else:
            teacher=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            used={tuple(r["spec"]) for r in rows}
            spec=min((s for s in options if s not in used),
                     key=lambda s:(sum(a!=b for a,b in zip(s,teacher["spec"])),
                                   sum(r["spec"][1]==s[1] for r in rows)))
        view,num,aux,*confidence_spec=spec
        if view not in cache:
            if view=="capacity":
                aa,bb=features(extra_x,d,g,"level"),features(extra_v,e,h,"level")
            elif view=="angles":
                aa,bb=features(angle_x,d,g,"level"),features(angle_v,e,h,"level")
            else:
                aa,bb=features(x,d,g,view),features(v,e,h,view)
            center,scale=aa.mean(0),aa.std(0).clip(.001)
            cache[view]=(torch.tensor(((aa-center)/scale).clip(-10,10),device="cuda",dtype=torch.float32),
                         torch.tensor(((bb-center)/scale).clip(-10,10),dtype=torch.float32),center,scale)
        a,b,center,scale=cache[view]
        if field_augmentation and view not in swap_cache:
            swapped=features(x,swapped_train,g,view)
            swap_cache[view]=torch.tensor(((swapped-center)/scale).clip(-10,10),device="cuda",dtype=torch.float32)
        torch.manual_seed(confidence_spec[1] if stability else 872)
        head=nn.Sequential(nn.Linear(a.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                           nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,num*3+6)).cuda()
        if motif_heads:
            train_route,n_groups=routing(mot_train,confidence_spec[2])
            dev_route,_=routing(mot_dev,confidence_spec[2])
            train_route=torch.tensor(train_route,device="cuda")
            dev_route=torch.tensor(dev_route,dtype=torch.long)
            head=MotifOutput(a.shape[1],num*3+6,n_groups).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.0018,weight_decay=.01)
        if domain_robust:
            domain_loss=torch.ones(len(groups),device="cuda")
        for _ in range(18):
            for j in torch.randperm(len(y),device="cuda").split(1536):
                inputs=a[j]
                if field_augmentation:
                    change=torch.rand((len(j),1),device="cuda")<confidence_spec[2]
                    inputs=torch.where(change,swap_cache[view][j],inputs)
                out=head(inputs,train_route[j]) if motif_heads else head(inputs)
                logits=out[:,:num].log_softmax(1)
                means=out[:,num:2*num]
                sigma=nn.functional.softplus(out[:,2*num:3*num])+.15
                residual=z[j,0:1]-means
                likelihood=torch.logsumexp(logits-.5*(residual/sigma).square()-sigma.log(),1)
                mean=(logits.exp()*means).sum(1)
                calibrate=(mean-z[j,0]).square()
                auxiliary=(out[:,3*num:]-z[j,1:]).square().mean(1)
                w=weight[j]
                if stability:
                    gap=all_confidence[j,confidence_spec[2]] if reliability_width else confidence[j]
                    w=w*torch.exp((confidence_spec[0]*gap).clamp(-2.,2.))
                    w=w/w.mean()
                if late_combination:
                    w=w*torch.where(j>=r1_count,confidence_spec[2],1.)
                    w=w/w.mean()
                per_pair=-likelihood+.5*calibrate+aux*auxiliary
                if domain_robust:
                    group=domains[j]
                    with torch.no_grad():
                        one=torch.bincount(group,minlength=len(groups))
                        subtotal=torch.bincount(group,weights=per_pair.detach().clip(0,8),minlength=len(groups))
                        domain_loss=torch.where(one>0,.97*domain_loss+.03*subtotal/one.clamp_min(1),domain_loss)
                        tilt=(confidence_spec[2]*(domain_loss[group]-domain_loss.mean())).clamp(-1.2,1.2).exp()
                        w=w*tilt*domain_counts[group].rsqrt()
                        w=w/w.mean()
                objective=per_pair*w
                opt.zero_grad(set_to_none=True)
                objective.mean().backward()
                opt.step()
        model=deepcopy(head).cpu().eval()
        with torch.inference_mode():
            out=model(b,dev_route) if motif_heads else model(b)
            expected=(out[:,:num].softmax(1)*out[:,num:2*num]).sum(1).numpy()
        direct=expected*ls[0]+lm[0]+v[:,0]
        pred=.65*parent+.35*direct
        metrics={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        name=f"GA{first_id+i}"
        torch.save(dict(state=model.state_dict(),static_center=center,static_scale=scale,
                        target_center=lm,target_scale=ls,spec=spec),out_dir/f"{name}.pt")
        np.savez_compressed(out_dir/f"{name}.npz",prediction=pred,readout=direct)
        row=dict(id=name,spec=spec,metrics=metrics)
        rows.append(row)
        with (out_dir/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(name,spec,metrics["all"]["residual_correlation"],flush=True)
        if metrics["all"]["residual_correlation"]>=.700:
            break
    best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (out_dir/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
    if late_expansion and best["metrics"]["all"]["residual_correlation"]<.700:
        main(stability=True,late_combination=True)


if __name__=="__main__":
    import sys
    main(stability=any(arg in sys.argv for arg in ("--stability","--seed-check","--domain-robust","--capacity-audit","--orientation-audit","--reliability-width")),
         seedset="--seed-check" in sys.argv, domain_robust="--domain-robust" in sys.argv,
         capacity_audit="--capacity-audit" in sys.argv, orientation_audit="--orientation-audit" in sys.argv, reliability_width="--reliability-width" in sys.argv)





















