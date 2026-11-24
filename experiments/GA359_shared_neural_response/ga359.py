"""GA359 tests a shared neural encoder for the two force-field responses."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_static_offset import VALID, estimates

R=Path("runs")

class Readout(nn.Module):
    def __init__(self, static, samples, width):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(samples,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(static+2*width+4,width),nn.LayerNorm(width),
                                nn.SiLU(),nn.Dropout(.1),nn.Linear(width,1))
    def forward(self,x,d):
        a,b,_=torch.chunk(d[:,:-4],3,dim=1)
        u,v=self.encode(a),self.encode(b)
        return self.head(torch.cat((x,(u+v)*.5,u-v,d[:,-4:]),dim=1)).flatten()

def main():
    dest=R/"ga359"
    dest.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    batches=[inputs((n,),R/"ga333",R/"ga355") for n in names]
    x,d,y=tuple(np.concatenate(z) for z in zip(*batches))
    v,e,t=inputs(VALID,R/"ga310",R/"ga313")
    ref=components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    actual,base,*_=ref
    np.testing.assert_allclose(actual,t,atol=3e-6,rtol=0)
    parent,baseline=forecasts(ref,("generated-6",.25,.2))
    y=y-x[:,0]
    idx=np.arange(len(y))
    group=np.repeat(np.arange(len(batches)),[len(a[2]) for a in batches])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results={}
    for static,width,mode in ((3,48,"huber"),(3,48,"correlation"),(3,96,"correlation"),
                              (x.shape[1],64,"huber"),(x.shape[1],64,"correlation")):
        torch.manual_seed(359)
        rng=np.random.default_rng(359)
        cx, sx=x[:,:static].mean(0),x[:,:static].std(0).clip(.001)
        tx=torch.tensor(((x[:,:static]-cx)/sx).clip(-12,12),device="cuda")
        q=torch.tensor(((v[:,:static]-cx)/sx).clip(-12,12))
        cd,sd=d.mean(0),d.std(0).clip(.001)
        td=torch.tensor((d-cd)/sd,device="cuda")
        h=torch.tensor((e-cd)/sd)
        label=torch.tensor(y,device="cuda")
        model=Readout(static,(d.shape[1]-4)//3,width).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        for epoch in range(1,41):
            model.train()
            steps=[]
            if mode=="correlation":
                for k in rng.permutation(len(batches)):
                    loc=rng.permutation(idx[group==k])
                    steps.extend(np.array_split(loc,max(1,len(loc)//256)))
            else:
                steps=np.array_split(rng.permutation(idx),max(1,len(idx)//512))
            for ii in steps:
                i=torch.tensor(ii,device="cuda")
                p=model(tx[i],td[i])
                truth=label[i]
                loss=nn.functional.smooth_l1_loss(p,truth,beta=.3)
                if mode=="correlation" and len(ii)>3:
                    u,w=p-p.mean(),truth-truth.mean()
                    corr=(u*w).mean()/torch.sqrt((u.square().mean()+1e-6)*(w.square().mean()+1e-6))
                    loss=loss-.10*corr
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if epoch not in (5,10,20,30,40):
                continue
            model.eval()
            cpu=Readout(static,(d.shape[1]-4)//3,width)
            cpu.load_state_dict({k:s.detach().cpu() for k,s in model.state_dict().items()})
            cpu.eval()
            with torch.no_grad():
                p=cpu(q,h).numpy()+v[:,0]
            key=f"{static}-{width}-{mode}-{epoch}"
            for w in (.1,.2,.35,.5):
                results[f"{key}/{w}"]=estimates((1-w)*parent+w*p,t,base)
            torch.save(dict(state=cpu.state_dict(),cx=cx,sx=sx,cd=cd,sd=sd,
                            static=static,width=width,samples=(d.shape[1]-4)//3),dest/f"{key}.pt")
    best=max(results,key=lambda n:results[n]["residual_correlation"])
    out=dict(best=best,metrics=results[best],parent=baseline,results=results,
             train_proteins=names,device=torch.cuda.get_device_name())
    record(dest/"development.json",out)
    print(best,results[best],baseline,flush=True)

if __name__=="__main__":
    main()
