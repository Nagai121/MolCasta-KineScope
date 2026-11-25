"""GA1224+: learn a shared atom displacement from the candidate network."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ga_competitive_capacity import competition
from .ga_pair_graph import indexed_graphs,pack
from .ga_replica_teacher import trajectory_rows
from .ga_sequential50 import features,pool
from .ga_static_offset import estimates

OUT=Path("runs/ga1224-atom-displacement")


class AtomDisplacement(nn.Module):
    def __init__(self,dim,width,cycles,vector_fraction):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(dim,width),nn.LayerNorm(width),nn.SiLU())
        self.kick=nn.Sequential(nn.Linear(width+1,width),nn.SiLU(),nn.Linear(width,2))
        self.residual=nn.Sequential(nn.Linear(width+1,width),nn.SiLU(),nn.Linear(width,1))
        self.cycles=cycles
        self.vector_fraction=vector_fraction

    def forward(self,x,edges,degree,coords,size):
        embedding=self.encoder(x)
        length=coords.norm(dim=1).clamp_min(.1)
        a,b=edges.T
        disturbance=coords.new_zeros((size,3))
        shift=length.new_zeros(length.shape)
        for _ in range(self.cycles):
            rel=coords+disturbance[b]-disturbance[a]
            unit=rel/rel.norm(dim=1,keepdim=True).clamp_min(.1)
            control=self.kick(torch.cat([embedding,shift[:,None]],1)).tanh()
            update=coords.new_zeros((size,3))
            update.index_add_(0,a,unit*control[:,:1])
            update.index_add_(0,b,-unit*control[:,1:2])
            disturbance=.50*disturbance+.75*update/degree.sqrt()
            new=coords+disturbance[b]-disturbance[a]
            shift=new.norm(dim=1)-length
        correction=self.residual(torch.cat([embedding,shift[:,None]],1)).flatten()
        return self.vector_fraction*shift+(1.-self.vector_fraction)*correction


def geometries(graphs,names):
    groups={}
    for name in np.unique(names):
        groups[name]=trajectory_rows(name,1)
    result=[]
    for row in graphs:
        ix,origin=row[3:5]
        pair,frames=groups[names[ix[0]]]
        vectors=pair[frames==origin,3:6]
        assert len(vectors)==len(ix)
        result.append(torch.tensor(vectors,dtype=torch.float32))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    xc,vc=competition(x,d,o,p,oriented=True),competition(v,e,vo,vg,oriented=True)
    xx=features(xc,d,g,"level")
    mean,scale=xx.mean(0),xx.std(0).clip(.001)
    tr=indexed_graphs(xc,d,g,y,o,p,mean,scale)
    va=indexed_graphs(vc,e,h,t,vo,vg,mean,scale)
    tr_coord=geometries(tr,p)
    va_coord=torch.cat(geometries(va,vg))
    a,ve,degree,_,size=pack(va,"cpu")
    expected=np.concatenate([r[3] for r in va])
    masks={"all":vo>=0,"first":vo==49,"later":vo!=49}
    options=[(width,cycles,vector) for width in (48,96)
             for cycles in (1,3) for vector in (.3,.7,1.)]
    rows=[]
    for i in range(len(options)):
        if rows:
            leader=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
            tried={tuple(r["spec"]) for r in rows}
            spec=min((s for s in options if s not in tried),
                     key=lambda s:sum(a!=b for a,b in zip(s,leader["spec"])))
        else:
            spec=options[0]
        width,cycles,vector=spec
        torch.manual_seed(1224)
        model=AtomDisplacement(xx.shape[1],width,cycles,vector).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.012)
        rng=np.random.default_rng(1224)
        for _ in range(18):
            for batch in np.array_split(rng.permutation(len(tr)),32):
                x0,edges,nodes,targets,n=pack([tr[j] for j in batch],"cuda")
                coords=torch.cat([tr_coord[j] for j in batch]).cuda()
                w=torch.tensor(np.concatenate([np.full(len(tr[j][2]),1. if tr[j][4]==49 else 4.,np.float32)
                                               for j in batch]),device="cuda")
                opt.zero_grad(set_to_none=True)
                difference=model(x0,edges,nodes,coords,n)-targets
                (w*difference.square()).mean().backward()
                opt.step()
        net=deepcopy(model).cpu().eval()
        direct=np.empty_like(t)
        with torch.inference_mode():
            direct[expected]=net(a,ve,degree,va_coord,size).numpy()+v[expected,0]
        pred=.65*parent+.35*direct
        metrics={k:estimates(pred[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{1224+i}"
        torch.save(dict(state=net.state_dict(),mean=mean,scale=scale,spec=spec),OUT/f"{tag}.pt")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=pred,readout=direct)
        row=dict(id=tag,spec=spec,metrics=metrics)
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,spec,metrics["all"]["residual_correlation"],flush=True)
        if metrics["all"]["residual_correlation"]>=.700:
            break
    best=max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"])
    (OUT/"summary.json").write_text(json.dumps(best,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
