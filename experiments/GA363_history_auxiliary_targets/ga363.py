"""GA363 uses separate history segments as auxiliary labels during training."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga359 import Readout
from .ga_static_offset import VALID, estimates
from .gc_history import past

R=Path("runs")
D=R/"ga363"

def truth(name):
    with np.load(R/"ga355"/f"{name}-49.npz") as z:
        pair=z["pair"]
    n=len(pair)
    samples=past(dict(protein=name,replica=1,pair=pair,origin=np.full(n,49),ends=np.array([n])))
    np.testing.assert_equal(samples.shape,(n,10))
    start=pair[:,2]
    early=samples[:,:5].mean(1)
    late=samples[:,5:].mean(1)
    mean=(5*early+5*late+start)/11.
    return np.column_stack((mean-start,early-start,late-start,
                            (samples<=3.5).mean(1))).astype(np.float32)

class Auxiliary(Readout):
    def __init__(self,static,samples,width):
        super().__init__(static,samples,width)
        self.head[-1]=nn.Linear(width,4)
    def forward(self,x,d):
        a,b,_=torch.chunk(d[:,:-4],3,dim=1)
        u,v=self.encode(a),self.encode(b)
        return self.head(torch.cat((x,(u+v)*.5,u-v,d[:,-4:]),dim=1))

def main():
    D.mkdir(exist_ok=True)
    roster=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,y=inputs(roster,R/"ga333",R/"ga355")
    with ThreadPoolExecutor(4) as executor:
        target=np.concatenate(list(executor.map(truth,roster)))
    np.testing.assert_allclose(target[:,0],y-x[:,0],rtol=0,atol=3e-6)
    v,e,t=inputs(VALID,R/"ga310",R/"ga313")
    ref=components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    actual,base,*_=ref
    np.testing.assert_allclose(actual,t,atol=3e-6,rtol=0)
    parent,pm=forecasts(ref,("generated-6",.25,.2))
    data=torch.from_numpy(target).cuda()
    mean,scale=d.mean(0),d.std(0).clip(.001)
    td=torch.tensor((d-mean)/scale,device="cuda")
    vd=torch.tensor((e-mean)/scale)
    slots={}
    for static,weight in ((3,.2),(3,.6),(x.shape[1],.2),(x.shape[1],.6)):
        torch.manual_seed(363)
        xm,xs=x[:,:static].mean(0),x[:,:static].std(0).clip(.001)
        tx=torch.tensor(((x[:,:static]-xm)/xs).clip(-12,12),device="cuda")
        vx=torch.tensor(((v[:,:static]-xm)/xs).clip(-12,12))
        samples=(d.shape[1]-4)//3
        head=Auxiliary(static,samples,64).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        for epoch in range(1,31):
            head.train()
            for i in torch.randperm(len(y),device="cuda").split(512):
                p=head(tx[i],td[i])
                target_y=data[i]
                loss=nn.functional.smooth_l1_loss(p[:,0],target_y[:,0],beta=.3)
                aux=nn.functional.smooth_l1_loss(p[:,1:3],target_y[:,1:3],beta=.3)
                occ=nn.functional.binary_cross_entropy_with_logits(p[:,3],target_y[:,3])
                opt.zero_grad(set_to_none=True)
                (loss+weight*(aux+.15*occ)).backward()
                opt.step()
            if epoch not in (10,20,30):
                continue
            cpu=Auxiliary(static,samples,64)
            cpu.load_state_dict({k:z.detach().cpu() for k,z in head.state_dict().items()})
            cpu.eval()
            with torch.no_grad():
                p=cpu(vx,vd)[:,0].numpy()+v[:,0]
            key=f"{static}-{weight}-{epoch}"
            torch.save(dict(state=cpu.state_dict(),static=static,samples=samples,
                            x_mean=xm,x_scale=xs,d_mean=mean,d_scale=scale),D/f"{key}.pt")
            for w in (.1,.2,.35):
                slots[f"{key}/{w}"]=estimates((1-w)*parent+w*p,t,base)
    best=max(slots,key=lambda k:slots[k]["residual_correlation"])
    record(D/"development.json",dict(best=best,metrics=slots[best],trials=slots,parent=pm,
                                     train=roster,rows=len(y),device=torch.cuda.get_device_name()))
    print(best,slots[best],pm,flush=True)

if __name__=="__main__":
    main()
